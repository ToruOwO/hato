from typing import Dict

import numpy as np

from robots.robot import Robot


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
        self,
        robot_ip: str = "111.111.1.11",
        no_gripper: bool = False,
        gripper_type="",
        grip_range=110,
        port_idx=-1,
    ):
        import rtde_control
        import rtde_receive

        [print("in ur robot:", robot_ip) for _ in range(3)]
        self.robot = rtde_control.RTDEControlInterface(robot_ip)
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            if gripper_type == "ability":
                from robots.ability_gripper import AbilityGripper

                self.gripper = AbilityGripper(port_idx=port_idx, grip_range=grip_range)
                self.gripper.connect()
            else:
                from robots.robotiq_gripper import RobotiqGripper

                self.gripper = RobotiqGripper()
                self.gripper.connect(hostname=robot_ip, port=63352)

        [print("connect") for _ in range(3)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper
        self.gripper_type = gripper_type

        self.velocity = 0.5
        self.acceleration = 0.5

        # EEF
        self.velocity_l = 0.3
        self.acceleration_l = 0.3
        self.dt = 1.0 / 500  # 2ms
        self.lookahead_time = 0.2  # [0.03, 0.2]s smoothens the trajectory
        self.gain = 100  # [100, 2000] proportional gain for following target position

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            if self.gripper_type == "ability":
                return 12
            else:
                return 7
        return 6

    def _get_gripper_pos(self) -> float:
        if self.gripper_type in ["ability"]:
            gripper_pos = self.gripper.get_current_position()
            return gripper_pos
        else:
            gripper_pos = self.gripper.get_current_position()
            assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
            return gripper_pos / 255

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def get_joint_velocities(self) -> np.ndarray:
        return self.r_inter.getActualQd()

    def get_eef_speed(self) -> np.ndarray:
        return self.r_inter.getActualTCPSpeed()

    def get_eef_pose(self) -> np.ndarray:
        """Get the current pose of the leader robot's end effector.

        Returns:
            T: The current pose of the leader robot's end effector.
        """
        return self.r_inter.getActualTCPPose()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints,
            self.velocity,
            self.acceleration,
            self.dt,
            self.lookahead_time,
            self.gain,
        )
        if self._use_gripper:
            if self.gripper_type == "ability":
                assert (
                    max(joint_state[6:]) <= 1 and min(joint_state[6:]) >= 0
                ), f"Gripper position must be between 0 and 1:{joint_state[6:]}"
                self.gripper.move(joint_state[6:], debug=False)
            elif self.gripper_type == "allegro":
                self.gripper.move(joint_state[6:])
            elif self.gripper_type == "dummy":
                pass
            else:
                gripper_pos = joint_state[-1] * 255
                # print(f"gripper move command: {gripper_pos}")
                self.gripper.move(int(gripper_pos), 255, 10)
        self.robot.waitPeriod(t_start)

    def command_eef_pose(self, eef_pose: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            eef_pose (np.ndarray): The EEF pose to command the leader robot to.
        """
        pose_command = eef_pose[:6]
        # print("current TCP:", self.r_inter.getActualTCPPose())
        # print("pose_command:", pose_command)
        # input("press enter to continue")
        t_start = self.robot.initPeriod()
        self.robot.servoL(
            pose_command,
            self.velocity_l,
            self.acceleration_l,
            self.dt,
            self.lookahead_time,
            self.gain,
        )
        if self._use_gripper:
            if self.gripper_type == "ability":
                assert (
                    max(eef_pose[6:]) <= 1 and min(eef_pose[6:]) >= 0
                ), f"Gripper position must be between 0 and 1:{eef_pose[6:]}"
                self.gripper.move(eef_pose[6:])
            else:
                gripper_pos = eef_pose[-1] * 255
                # print(f"gripper move command: {gripper_pos}")
                self.gripper.move(int(gripper_pos), 255, 10)
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        joint_vels = self.get_joint_velocities()
        eef_speed = self.get_eef_speed()
        pos_quat = self.get_eef_pose()
        gripper_pos = np.array([joints[-1]])
        if self._use_gripper and self.gripper_type == "ability":
            # include Ability hand touch data
            touch = self.gripper.get_current_touch()
        else:
            touch = np.zeros(30)
        return {
            "joint_positions": joints,
            "joint_velocities": joint_vels,
            "eef_speed": eef_speed,
            "ee_pos_quat": pos_quat,  # TODO: this is pos_rot actually
            "gripper_position": gripper_pos,
            "touch": touch,
        }


