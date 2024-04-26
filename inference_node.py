import argparse
import collections
import pickle
import threading
import time

import numpy as np
import zmq

DEFAULT_INFERENCE_PORT = 4321


class ZMQInferenceServer:
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_INFERENCE_PORT):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        self._socket.bind(f"tcp://*:{port}")
        self._stop_event = threading.Event()

    def get_obs(self):
        state_dict = None
        while True:
            try:
                # check for a message, this will not block
                message = self._socket.recv(flags=zmq.NOBLOCK)

            except zmq.Again as e:
                # print("observation queue exhausted")
                break
            else:
                state_dict = pickle.loads(message)
        if state_dict is None:  # block until an observation is recieved
            while True:
                message = self._socket.recv()
                state_dict = pickle.loads(message)
                if "obs" not in state_dict and "t" not in state_dict:
                    if "num_diffusion_iters" in state_dict:  # ignore and send success
                        self._socket.send_string("success")
                    continue
                break

        return state_dict["obs"], state_dict["t"]

    def infer(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, obs):
        raise NotImplementedError

    def serve(self):
        # self._socket.setsockopt(zmq.RCVTIMEO, 10000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            obs, t = self.get_obs()  # get obs from the client

            print(f"Recieved observation at time {t}. Inference start!")
            pred = self.act(obs)
            print(f"Inference ended.")

            message = pickle.dumps({"acts": pred, "t": t})
            self._socket.send(message)  # send the action back to the client

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQInferenceClient:
    """A class representing a ZMQ client for a leader robot."""

    def __init__(
        self,
        port: int = DEFAULT_INFERENCE_PORT,
        host: str = "111.11.111.11",
        default_action=None,
        queue_size=32,
        ensemble_mode="new",
        act_tau=0.5,
    ):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        self._socket.connect(f"tcp://{host}:{port}")
        print(f"connected -- tcp://{host}:{port}")

        self.act_q = collections.deque(maxlen=queue_size)
        self.t = 0
        self.last_act = default_action
        self.ensemble_mode = ensemble_mode
        self.act_tau = act_tau

    def act(self, obs):
        self.t += 1
        final_act = self.last_act

        # send the observation
        message = pickle.dumps({"obs": obs, "t": self.t})
        self._socket.send(message)

        # process the incoming message queue
        while True:
            try:
                # check for a message, this will not block
                message = self._socket.recv(flags=zmq.NOBLOCK)

            except zmq.Again as e:
                # print("action queue exhausted")
                break
            else:
                state_dict = pickle.loads(message)
                acts, pt = state_dict["acts"], state_dict["t"]

                while len(self.act_q) > 0 and self.act_q[0][1] < self.t:
                    self.act_q.popleft()
                while pt < self.t and len(acts) > 0:
                    pt += 1
                    acts = acts[1:]
                for c_acts, ct in self.act_q:
                    if ct == pt:
                        c_acts.append(acts[0])
                        pt += 1
                        acts = acts[1:]
                        if len(acts) == 0:
                            break
                # for
                # push all the new actions in
                for i, act in enumerate(acts):
                    self.act_q.append(([act], pt + i))

        # now searching for the matching time stamp
        while len(self.act_q) > 0:
            c_acts, tt = self.act_q.popleft()
            if tt == self.t:
                if self.ensemble_mode == "act":
                    z_act = c_acts[0]
                    for act in c_acts[1:]:
                        z_act = z_act * self.act_tau + act * (1.0 - self.act_tau)
                    final_act = z_act
                elif self.ensemble_mode == "avg":
                    final_act = np.mean(np.array(c_acts), axis=0)
                elif self.ensemble_mode == "old":
                    final_act = c_acts[0]
                elif self.ensemble_mode == "new":
                    final_act = c_acts[-1]

                break
        print("action queue (dt):", [t - self.t for a, t in self.act_q])
        print("action queue (size):", [len(a) for a, t in self.act_q])
        self.last_act = final_act
        # print("action:", final_act)
        return final_act


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--type", type=str, default="client")
    args.add_argument("--freq", type=float, default=10)
    args.add_argument("--inft", type=float, default=0.2)
    args = args.parse_args()

    action_dim = 24

    class DummyAgentServer(ZMQInferenceServer):
        def infer(self, obs):
            print("start inference")
            time.sleep(args.inft)
            print("stop inference")
            return np.zeros((16, action_dim))

    if args.type == "server":
        server = DummyAgentServer()
        server.serve()

    elif args.type == "client":
        client = ZMQInferenceClient(
            default_action=np.zeros((action_dim,)), queue_size=32
        )
        obs = {
            "img": np.zeros((4, 240, 360), dtype=np.uint16),
            "eef": np.zeros((24,), dtype=np.float32),
        }

        while True:
            time.sleep(1 / args.freq)
            action = client.act(obs)
