import collections
import json
import os
from typing import Any, Dict

import numpy as np
import quaternion
import torch
import time

import pickle

from learning.dp.pipeline import Agent as DPAgent
from inference_node import (
    ZMQInferenceClient,
    ZMQInferenceServer,
    DEFAULT_INFERENCE_PORT,
)


def from_numpy(data, device, unsqueeze=True):
    return {
        key: torch.from_numpy(value).to(device=device)[None]
        if unsqueeze
        else torch.from_numpy(value).to(device=device)
        for key, value in data.items()
        if key != "activated"
    }


UR_IDX = list(range(6)) + list(range(12, 18))
LEFT_HAND_IDX = list(range(6, 12))
RIGHT_HAND_IDX = list(range(18, 24))


def get_reset_joints(ur_eef=False):
    if ur_eef:
        # these are EEF pose
        arm_joints_left = [
            -0.10244499252760966,
            -0.7492784625293504,
            0.14209881911585326,
            -0.3622358797572402,
            -1.4347279978985925,
            0.8691789808786153,
        ]

        arm_joints_right = [
            0.2313341406775527,
            -0.7512396951283128,
            0.06337444935928868,
            0.6512089317940273,
            1.3246649009637026,
            0.5471978290474188,
        ]
    else:
        arm_joints_left = [-80, -140, -80, -85, -10, 80]
        arm_joints_right = [-270, -30, 70, -85, 10, 0]
    hand_joints = [0, 0, 0, 0, 0.5, 0.5]
    reset_joints_left = np.concatenate([np.deg2rad(arm_joints_left), hand_joints])
    reset_joints_right = np.concatenate([np.deg2rad(arm_joints_right), hand_joints])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    return reset_joints


def get_eef_pose(eef_pose, eef_delta):
    pos_delta = eef_delta[:3]
    rot_delta = eef_delta[3:]
    pos = eef_pose[:3] + pos_delta
    # quaternion multiplication
    rot = quaternion.as_rotation_vector(
        quaternion.from_rotation_vector(rot_delta)
        * quaternion.from_rotation_vector(eef_pose[3:])
    )
    return np.concatenate((pos, rot))


def parse_txt_to_json(input_file_path, output_file_path):
    # Initialize an empty dictionary to store the parsed key-value pairs
    data = {}

    # Open and read the input text file line by line
    with open(input_file_path, "r") as file:
        for line in file:
            kv = line.strip().split(": ", 1)
            if len(kv) != 2:
                continue
            key, value = kv
            if key == "camera_indices":
                data[key] = list(map(int, value))
            elif key == "representation_type":
                data[key] = value.split("-")
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value == "True":
                            value = True
                        elif value == "False":
                            value = False
                        elif value == "None":
                            value = None

                data[key] = value

    with open(output_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    return data


class BimanualDPAgentServer(ZMQInferenceServer):
    def __init__(
        self,
        ckpt_path,
        dp_args=None,
        binarize_finger_action=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if dp_args is None:
            dp_args = self.get_default_dp_args()

        # rewrite dp_args based on saved args
        args_txt = os.path.join(os.path.dirname(ckpt_path), "args_log.txt")
        args_json = os.path.join(os.path.dirname(ckpt_path), "args_log.json")
        args = parse_txt_to_json(args_txt, args_json)
        for k in dp_args.keys():
            if k == "output_sizes":
                dp_args[k]["img"] = args["image_output_size"]
            else:
                if k in args:
                    dp_args[k] = args[k]

        # save dp args in ckpt path as json
        ckpt_dir = os.path.dirname(ckpt_path)
        args_path = os.path.join(ckpt_dir, "dp_args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                dp_args = json.load(f)
        else:
            with open(args_path, "w") as f:
                json.dump(dp_args, f)

        torch.cuda.set_device(0)
        self.dp = DPAgent(
            output_sizes=dp_args["output_sizes"],
            representation_type=dp_args["representation_type"],
            identity_encoder=dp_args["identity_encoder"],
            camera_indices=dp_args["camera_indices"],
            pred_horizon=dp_args["pred_horizon"],
            obs_horizon=dp_args["obs_horizon"],
            action_horizon=dp_args["action_horizon"],
            without_sampling=dp_args["without_sampling"],
            predict_eef_delta=dp_args["predict_eef_delta"],
            predict_pos_delta=dp_args["predict_pos_delta"],
            use_ddim=dp_args["use_ddim"],
        )
        self.dp_args = dp_args
        self.obsque = collections.deque(maxlen=dp_args["obs_horizon"])
        self.dp.load(ckpt_path)
        self.action_queue = collections.deque(maxlen=dp_args["action_horizon"])
        self.max_length = 100
        self.count = 0
        self.except_thumb_hand_indices = np.array([6, 7, 8, 9, 18, 19, 20, 21])
        self.binaraize_finger_action = binarize_finger_action
        self.clip_far = dp_args["clip_far"]
        self.predict_eef_delta = dp_args["predict_eef_delta"]
        self.predict_pos_delta = dp_args["predict_pos_delta"]
        assert not (self.predict_eef_delta and self.predict_pos_delta)
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)

        self.num_diffusion_iters = dp_args["num_diffusion_iters"]

        self.hand_uppers = np.array([110.0, 110.0, 110.0, 110.0, 90.0, 120.0])
        self.hand_lowers = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

        # TODO: remove hack
        self.hand_new_uppers = np.array([75] * 4 + [90.0, 120.0])

        self.trigger_state = {"l": True, "r": True}

    @staticmethod
    def get_default_dp_args():
        return {
            "output_sizes": {
                "eef": 64,
                "hand_pos": 64,
                "img": 128,
                "pos": 128,
                "touch": 64,
            },
            "representation_type": ["img", "pos", "touch", "depth"],
            "identity_encoder": False,
            "camera_indices": [0, 1, 2],
            "obs_horizon": 4,
            "pred_horizon": 16,
            "action_horizon": 8,
            "num_diffusion_iters": 15,
            "without_sampling": False,
            "clip_far": False,
            "predict_eef_delta": False,
            "predict_pos_delta": False,
            "use_ddim": False,
        }

    def compile_inference(self, precision="high"):
        message = self._socket.recv()
        start_time = time.time()
        state_dict = pickle.loads(message)
        self.num_diffusion_iters = state_dict["num_diffusion_iters"]
        example_obs = state_dict["example_obs"]
        print(
            f"received compilation request: # diff iters = {state_dict['num_diffusion_iters']}"
        )

        torch.set_float32_matmul_precision(precision)
        self.dp.policy.forward = torch.compile(torch.no_grad(self.dp.policy.forward))

        for i in range(25):  # burn in
            self.act(example_obs)
        print("success, compile time: " + str(time.time() - start_time))
        self._socket.send_string("success")

    def infer(self, obs: Dict[str, Any]) -> np.ndarray:
        return self.dp.predict([obs], num_diffusion_iters=self.num_diffusion_iters)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        curr_joint_pos = obs["joint_positions"]
        curr_eef_pose = obs["ee_pos_quat"]
        obs = self.dp.get_observation([obs], load_img=True)
        if "img" in obs:
            obs["img"] = self.dp.eval_transform(obs["img"].squeeze(0))
        return self.infer(obs)


class BimanualDPAgent(ZMQInferenceClient):
    def __init__(
        self,
        ckpt_path,
        dp_args=None,
        binarize_finger_action=False,
        port=DEFAULT_INFERENCE_PORT,
        host="127.0.0.1",
        temporal_ensemble_mode="new",
        temporal_ensemble_act_tau=0.5,
    ):
        super().__init__(
            default_action=get_reset_joints(),
            port=port,
            host=host,
            ensemble_mode=temporal_ensemble_mode,
            act_tau=temporal_ensemble_act_tau,
        )

        if dp_args is None:
            dp_args = self.get_default_dp_args()

        # rewrite dp_args based on saved args
        args_txt = os.path.join(os.path.dirname(ckpt_path), "args_log.txt")
        args_json = os.path.join(os.path.dirname(ckpt_path), "args_log.json")
        args = parse_txt_to_json(args_txt, args_json)
        for k in dp_args.keys():
            if k == "output_sizes":
                dp_args[k]["img"] = args["image_output_size"]
            else:
                if k in args:
                    dp_args[k] = args[k]

        self.dp_args = dp_args
        self.obsque = collections.deque(maxlen=dp_args["obs_horizon"])
        # self.dp.load(ckpt_path)
        self.action_queue = collections.deque(maxlen=dp_args["action_horizon"])
        self.max_length = 100
        self.count = 0
        self.except_thumb_hand_indices = np.array([6, 7, 8, 9, 18, 19, 20, 21])
        self.binaraize_finger_action = binarize_finger_action
        self.clip_far = dp_args["clip_far"]
        self.predict_eef_delta = dp_args["predict_eef_delta"]
        self.predict_pos_delta = dp_args["predict_pos_delta"]
        assert not (self.predict_eef_delta and self.predict_pos_delta)
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)

        self.num_diffusion_iters = dp_args["num_diffusion_iters"]

        self.hand_uppers = np.array([110.0, 110.0, 110.0, 110.0, 90.0, 120.0])
        self.hand_lowers = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

        # TODO: remove hack
        self.hand_new_uppers = np.array([75] * 4 + [90.0, 120.0])

        self.trigger_state = {"l": True, "r": True}

    @staticmethod
    def get_default_dp_args():
        return {
            "output_sizes": {
                "eef": 64,
                "hand_pos": 64,
                "img": 128,
                "pos": 128,
                "touch": 64,
            },
            "representation_type": ["img", "pos", "touch", "depth"],
            "identity_encoder": False,
            "camera_indices": [0, 1, 2],
            "obs_horizon": 4,
            "pred_horizon": 16,
            "action_horizon": 8,
            "num_diffusion_iters": 15,
            "without_sampling": False,
            "clip_far": False,
            "predict_eef_delta": False,
            "predict_pos_delta": False,
            "use_ddim": False,
        }

    def compile_inference(self, example_obs, num_diffusion_iters):
        message = pickle.dumps(
            {"example_obs": example_obs, "num_diffusion_iters": num_diffusion_iters}
        )
        self._socket.send(message)

        message = self._socket.recv()
        assert message == b"success"

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        curr_joint_pos = obs["joint_positions"]
        curr_eef_pose = obs["ee_pos_quat"]
        act = super().act(obs)

        if self.predict_pos_delta:
            self.control[UR_IDX] = curr_joint_pos[UR_IDX]
            self.control = self.control + act
            act = self.control
            # act = curr_joint_pos + act

        if self.predict_eef_delta:
            left_arm_act = get_eef_pose(curr_eef_pose[:6], act[:6])
            left_hand_act = act[6:12]
            right_arm_act = get_eef_pose(curr_eef_pose[6:], act[12:18])
            right_hand_act = act[18:24]
            act = np.concatenate(
                [left_arm_act, left_hand_act, right_arm_act, right_hand_act],
                axis=-1,
            )

        # if binarize_finger_action is True, binarize the finger action

        if self.binaraize_finger_action:
            mean_act = np.mean(act[self.except_thumb_hand_indices])
            if mean_act > 0.5:
                act[self.except_thumb_hand_indices] = 1.0
            else:
                act[self.except_thumb_hand_indices] = 0.0
        else:
            left_hand = (
                act[LEFT_HAND_IDX] * (self.hand_uppers - self.hand_lowers)
                + self.hand_lowers
            )
            act[LEFT_HAND_IDX] = (left_hand - self.hand_lowers) / (
                self.hand_new_uppers - self.hand_lowers
            )
            right_hand = (
                act[RIGHT_HAND_IDX] * (self.hand_uppers - self.hand_lowers)
                + self.hand_lowers
            )
            act[RIGHT_HAND_IDX] = (right_hand - self.hand_lowers) / (
                self.hand_new_uppers - self.hand_lowers
            )
            act[list(range(6, 12)) + list(range(18, 24))] = np.clip(
                act[list(range(6, 12)) + list(range(18, 24))], 0, 1
            )

        return act
