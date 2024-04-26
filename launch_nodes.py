from dataclasses import dataclass
from multiprocessing import Process
from typing import List, Optional, Tuple

import tyro

from camera_node import ZMQServerCamera, ZMQServerCameraFaster
from robot_node import ZMQServerRobot
from robots.robot import BimanualRobot


@dataclass
class Args:
    robot: str = "bimanual_ur"
    hand_type: str = ""
    hostname: str = "127.0.0.1"
    robot_ip: str = "111.111.1.1"
    faster: bool = True
    cam_names: Tuple[str, ...] = "435"
    ability_gripper_grip_range: int = 110
    img_size: Optional[Tuple[int, int]] = None  # (320, 240)


def launch_server_cameras(port: int, camera_id: List[str], args: Args):
    from cameras.realsense_camera import RealSenseCamera

    camera = RealSenseCamera(camera_id, img_size=args.img_size)

    if args.faster:
        server = ZMQServerCameraFaster(camera, port=port, host=args.hostname)
    else:
        server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


def launch_robot_server(port: int, args: Args):
    if args.robot == "ur":
        from robots.ur import URRobot

        robot = URRobot(robot_ip=args.robot_ip)
    elif args.robot == "bimanual_ur":
        from robots.ur import URRobot

        if args.hand_type == "ability":
            # 6 DoF Ability Hand
            # robot_l - right hand; robot_r - left hand
            _robot_l = URRobot(
                robot_ip="111.111.1.3",
                no_gripper=False,
                gripper_type="ability",
                grip_range=args.ability_gripper_grip_range,
                port_idx=1,
            )
            _robot_r = URRobot(
                robot_ip="111.111.2.3",
                no_gripper=False,
                gripper_type="ability",
                grip_range=args.ability_gripper_grip_range,
                port_idx=2,
            )
        else:
            # Robotiq gripper
            _robot_l = URRobot(robot_ip="111.111.1.3", no_gripper=False)
            _robot_r = URRobot(robot_ip="111.111.2.3", no_gripper=False)
        robot = BimanualRobot(_robot_l, _robot_r)
    else:
        raise NotImplementedError(f"Robot {args.robot} not implemented")
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"Starting robot server on port {port}")
    server.serve()


CAM_IDS = {
    "435": "000000000000",
}


def create_camera_server(args: Args) -> List[Process]:
    ids = [CAM_IDS[name] for name in args.cam_names]
    camera_port = 5000
    # start a single python process for all cameras
    print(f"Launching cameras {ids} on port {camera_port}")
    server = Process(target=launch_server_cameras, args=(camera_port, ids, args))
    return server


def main(args):
    camera_server = create_camera_server(args)
    print("Starting camera server process")
    camera_server.start()

    launch_robot_server(6000, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
