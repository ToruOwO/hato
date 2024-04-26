import glob
import os
import pickle
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from natsort import natsorted

from cameras.camera import CameraDriver
from robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class EvalRobotEnv:
    def __init__(
        self,
        robot: Robot,
        traj_path: str,
        control_rate_hz: float,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self.traj_path = traj_path

        self.pkls = natsorted(
            glob.glob(os.path.join(self.traj_path, "*.pkl"), recursive=True)
        )
        print("Finished reading dir", self.traj_path)
        print("No. of files:", len(self.pkls))
        self.traj_len = len(self.pkls)
        self.count = 0

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        # Return positive integer for batched envs.
        return self.traj_len

    def step_eef(self, eef_pose: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            eef_pose: end effector pose command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(eef_pose) == self._robot.num_dofs(), f"input:{len(eef_pose)}"
        self._robot.command_eef_pose(eef_pose)
        self._rate.sleep()
        return self.get_obs()

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_real_obs(self) -> Dict[str, Any]:
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        for k, v in robot_obs.items():
            observations[k] = v
        return observations

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        if self.count >= self.traj_len:
            return None
        pkl = self.pkls[self.count]
        with open(pkl, "rb") as f:
            observations = pickle.load(f)
        self.count += 1
        return observations


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        show_camera_view: bool = True,
        save_depth: bool = True,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        print("RobotEnv: control_rate_hz", control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

        self._show_camera_view = show_camera_view
        if self._show_camera_view:
            for name in list(self._camera_dict.keys()):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        self._save_depth = save_depth

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        # Return positive integer for batched envs.
        return 0

    def step_eef(self, eef_pose: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            eef_pose: end effector pose command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(eef_pose) == self._robot.num_dofs(), f"input:{len(eef_pose)}"
        self._robot.command_eef_pose(eef_pose)
        self._rate.sleep()
        return self.get_obs()

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            if self._save_depth:
                observations[f"{name}_depth"] = depth

            if self._show_camera_view:
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                image_depth = cv2.hconcat([image[:, :, ::-1], depth])
                cv2.imshow(name, image_depth)
                cv2.waitKey(1)

        robot_obs = self._robot.get_observations()
        for k, v in robot_obs.items():
            observations[k] = v
        return observations
