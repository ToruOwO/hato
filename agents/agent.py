from typing import Any, Dict, Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        left_obs = {}
        right_obs = {}
        for key, val in obs.items():
            L = val.shape[0]
            half_dim = L // 2
            if key.endswith("rgb") or key.endswith("depth"):
                left_obs[key] = val
                right_obs[key] = val
            else:
                assert L == half_dim * 2, f"{key} must be even, something is wrong"
                left_obs[key] = val[:half_dim]
                right_obs[key] = val[half_dim:]
        return np.concatenate(
            [self.agent_left.act(left_obs), self.agent_right.act(right_obs)]
        )


class SafetyWrapper:
    def __init__(self, ur_idx, hand_idx, agent, delta=0.5, hand_delta=0.1):
        self.ur_idx = ur_idx
        self.hand_idx = hand_idx
        self.agent = agent
        self.delta = delta
        self.hand_delta = hand_delta

        # Ability Hand ranges
        self.num_hand_dofs = 12
        self.upper_ranges = [110, 110, 110, 110, 90, 120] * 2
        self.lower_ranges = [5] * self.num_hand_dofs

    def _hand_pos_to_cmd(self, pos):
        """
        pos: desired hand degrees for Ability Hands
        """
        assert len(pos) == self.num_hand_dofs
        cmd = [0] * self.num_hand_dofs
        for i in range(self.num_hand_dofs):
            if i in [5, 11]:
                pos[i] = -pos[i]
            cmd[i] = (pos[i] - self.lower_ranges[i]) / (
                self.upper_ranges[i] - self.lower_ranges[i]
            )
        return cmd

    def act_safe(self, agent, obs, eef=False):
        joints = obs["joint_positions"]
        action = agent.act(obs)
        if eef:
            eef_pose = obs["ee_pos_quat"]
            left_eef_pos = eef_pose[:3]
            right_eef_pos = eef_pose[6:9]
            left_eef_target = action[:3]
            right_eef_target = action[12:15]
            if np.linalg.norm(left_eef_pos - left_eef_target) > 0.5:
                print("Left EEF action is too big")
                print(
                    f"Left EEF pos: {left_eef_pos}, target: {left_eef_target}, diff: {left_eef_pos - left_eef_target}"
                )
            if np.linalg.norm(right_eef_pos - right_eef_target) > 0.5:
                print("Right EEF action is too big")
                print(
                    f"Right EEF pos: {right_eef_pos}, target: {right_eef_target}, diff: {right_eef_pos - right_eef_target}"
                )

            left_eef_target = np.clip(
                left_eef_target,
                left_eef_pos - self.delta,
                left_eef_pos + self.delta,
            )
            right_eef_target = np.clip(
                right_eef_target,
                right_eef_pos - self.delta,
                right_eef_pos + self.delta,
            )
            action[:3] = left_eef_target
            action[12:15] = right_eef_target
        else:
            # check if action is too big
            if (np.abs(action[self.ur_idx] - joints[self.ur_idx]) > self.delta).any():
                print("Action is too big")

                # print which joints are too big
                joint_index = np.where(np.abs(action - joints) > self.delta)[0]
                for j in joint_index:
                    if j in self.ur_idx:
                        print(
                            f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                        )
            action[self.ur_idx] = np.clip(
                action[self.ur_idx],
                joints[self.ur_idx] - self.delta,
                joints[self.ur_idx] + self.delta,
            )

        if self.hand_idx is not None:
            joint_cmd = self._hand_pos_to_cmd(joints[self.hand_idx])
            action[self.hand_idx] = joint_cmd + np.clip(
                action[self.hand_idx] - joint_cmd, -self.hand_delta, self.hand_delta
            )
            action[self.hand_idx] = np.clip(action[self.hand_idx], 0, 1)
        return action
