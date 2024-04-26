import datetime
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import tyro

from agents.agent import SafetyWrapper
from camera_node import ZMQClientCamera
from env import EvalRobotEnv
from robot_node import ZMQClientRobot


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    activated=True,
) -> None:
    obs["activated"] = activated
    obs["control"] = action  # add action to obs
    recorded_file = folder / (timestamp.isoformat() + ".pkl")
    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)


@dataclass
class Args:
    robot_port: int = 6000
    wrist_camera_port: int = 5001
    base_camera_port: int = 5000
    hostname: str = "127.0.0.1"
    hz: int = 100
    show_camera_view: bool = True

    agent: str = "dp"
    robot_type: str = "ur5"
    hand_type: str = "ability"
    save_data: bool = False
    data_dir: str = "/shared/data/bc_data"
    verbose: bool = False
    safe: bool = False
    use_vel_ik: bool = False

    traj_path: str = "/shared/data/test_data"
    dp_ckpt_path: str = "best.ckpt"


def main(args):
    camera_clients = {
        "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
    }
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = EvalRobotEnv(
        robot_client,
        traj_path=args.traj_path,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
    )
    if args.agent.startswith("dp"):
        from agents.dp_agent import BimanualDPAgent

        agent = BimanualDPAgent(ckpt_path=args.dp_ckpt_path)
    else:
        raise ValueError(f"Invalid agent name: {args.agent}")

    if args.hand_type == "ability":
        arm_joints_left = [-80, -140, -80, -85, -10, 80]
        arm_joints_right = [-270, -30, 70, -85, 10, 0]
        hand_joints = [0, 0, 0, 0, 0.5, 0.5]
    else:
        raise ValueError(f"Invalid hand type: {args.hand_type}")
    reset_joints_left = np.concatenate([np.deg2rad(arm_joints_left), hand_joints])
    reset_joints_right = np.concatenate([np.deg2rad(arm_joints_right), hand_joints])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_real_obs()["joint_positions"]
    if args.hand_type == "ability":
        curr_joints[6:12] = hand_joints
        curr_joints[18:] = hand_joints
    print("Current joints:", curr_joints)
    print("Reset joints:", reset_joints)
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.01), 20)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        obs = env.step(jnt)

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_real_obs())
    obs = env.get_real_obs()
    joints = obs["joint_positions"]

    # if args.hand_type == "ability":
    ur_idx = list(range(0, 6)) + list(range(12, 18))
    hand_idx = list(range(6, 12)) + list(range(18, 24))

    if args.safe:
        max_joint_delta = 0.5
        max_hand_delta = 0.1
        safety_wrapper = SafetyWrapper(
            ur_idx, hand_idx, agent, delta=max_joint_delta, hand_delta=max_hand_delta
        )

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    for step in range(3):
        print("Countdown step", step)
        time.sleep(0.5)

    print_color("\nReady to go 🚀🚀🚀", color="green", attrs=("bold",))

    start_time = time.time()

    if args.save_data:
        time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        save_path = (
            Path(args.data_dir).expanduser()
            / (args.traj_path.split("/")[-1] + "_openloop")
            / time_str
        )
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {save_path}")

    while obs is not None:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        if args.safe:
            action = safety_wrapper.act_safe(
                agent, obs, eef=(args.agent.endswith("_eef"))
            )
        else:
            action = agent.act(obs)
        dt = datetime.datetime.now()
        img, depth = camera_clients["base"].read()
        if args.save_data:
            obs["base_rgb"] = img
            obs["base_depth"] = depth
            save_frame(save_path, dt, obs, action, activated=agent.trigger_state)
        # input("Press Enter to continue...")

        if args.agent.endswith("_eef"):
            obs = env.step_eef(action)
        else:
            obs = env.step(action)

    # save eval video
    import glob
    import os

    from moviepy.editor import ImageSequenceClip

    episode_dir = save_path

    # find all the pkl files in the episode directory
    pkls = sorted(glob.glob(os.path.join(episode_dir, "*.pkl")))

    # read all images
    frames = []
    for pkl in pkls:
        with open(pkl, "rb") as f:
            try:
                data = pickle.load(f)
            except:
                continue
            rgb = data["base_rgb"]
            rgb = np.concatenate([rgb[i] for i in range(rgb.shape[0])], axis=1)
            frames.append(rgb)

    # Create a video clip
    clip = ImageSequenceClip(frames, fps=10)
    ckpt_path = os.path.dirname(args.dp_ckpt_path)
    clip.write_videofile(os.path.join(ckpt_path, f"{time_str}_openloop.mp4"))


if __name__ == "__main__":
    main(tyro.cli(Args))
