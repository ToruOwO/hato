from typing import Dict

import numpy as np
import quaternion
from oculus_reader.reader import OculusReader
from agents.agent import Agent


mj2ur = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
ur2mj = np.linalg.inv(mj2ur)

trigger_state = {"l": False, "r": False}


def apply_transfer(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    # xyz can be 3dim or 4dim (homogeneous) or can be a rotation matrix
    if len(xyz) == 3:
        xyz = np.append(xyz, 1)
    return np.matmul(mat, xyz)[:3]


quest2isaac = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
left2isaac = np.array(
    [
        [0, -1, 0, 0],
        [-1 / 2 * np.sqrt(2), 0, -1 / 2 * np.sqrt(2), 0],
        [1 / 2 * np.sqrt(2), 0, -1 / 2 * np.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
)
right2isaac = np.array(
    [
        [0, -1, 0, 0],
        [-1 / 2 * np.sqrt(2), 0, 1 / 2 * np.sqrt(2), 0],
        [-1 / 2 * np.sqrt(2), 0, -1 / 2 * np.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
)
isaac2left = np.linalg.inv(left2isaac)
isaac2right = np.linalg.inv(right2isaac)

quest2left = np.matmul(isaac2left, quest2isaac)
quest2right = np.matmul(isaac2right, quest2isaac)
left2quest = np.linalg.inv(quest2left)
right2quest = np.linalg.inv(quest2right)

translation_scaling_factor = 1.0


class SingleArmQuestAgent(Agent):
    def __init__(
        self,
        robot_type: str,
        which_hand: str,
        eef_control_mode: int = 0,
        verbose: bool = False,
    ) -> None:
        """Interact with the robot using the quest controller.

        leftTrig: press to start control (also record the current position as the home position)
        leftJS: a tuple of (x,y) for the joystick, only need y to control the gripper
        """
        self.which_hand = which_hand
        self.eef_control_mode = eef_control_mode
        assert self.which_hand in ["l", "r"]

        self.oculus_reader = OculusReader()
        self.control_active = False
        self.reference_quest_pose = None
        self.reference_ee_rot_ur = None
        self.reference_ee_pos_ur = None
        self.reference_js = [0.5, 0.5]
        self.js_speed_scale = 0.1

        self.robot_type = robot_type
        self._verbose = verbose

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if self.robot_type == "ur5":
            num_dof = 6
        current_eef_pose = obs["ee_pos_quat"]

        # pos and rot in robot base frame
        ee_pos = current_eef_pose[:3]
        ee_rot = current_eef_pose[3:]
        ee_rot = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(ee_rot))

        if self.which_hand == "l":
            pose_key = "l"
            trigger_key = "leftTrig"
            grip_key = "leftGrip"
            joystick_key = "leftJS"
            # left yx
            gripper_open_key = "Y"
            gripper_close_key = "X"
        elif self.which_hand == "r":
            pose_key = "r"
            trigger_key = "rightTrig"
            grip_key = "rightGrip"
            joystick_key = "rightJS"
            # right ba for the key
            gripper_open_key = "B"
            gripper_close_key = "A"
        else:
            raise ValueError(f"Unknown hand: {self.which_hand}")
        # check the trigger button state
        (
            pose_data,
            button_data,
        ) = self.oculus_reader.get_transformations_and_buttons()
        if len(pose_data) == 0 or len(button_data) == 0:
            print("no data, quest not yet ready")
            return np.concatenate(
                [current_eef_pose, obs["joint_positions"][num_dof:] * 0.0]
            )

        if self.eef_control_mode == 0:
            new_gripper_angle = [button_data[grip_key][0]]
        elif self.eef_control_mode == 1:
            new_gripper_angle = [button_data[grip_key][0]] * 6  # [0, 1]
        else:
            # (x, y) position of joystick, range (-1.0, 1.0)
            if self.which_hand == "r":
                js_y = button_data[joystick_key][0] * -1
                js_x = button_data[joystick_key][1] * -1
            else:
                js_y = button_data[joystick_key][0]
                js_x = button_data[joystick_key][1] * -1

            if self.eef_control_mode == 2:
                # convert js_x, js_y from range (-1.0, 1.0) to (0.0, 1.0)
                js_x = (js_x + 1) / 2
                js_y = (js_y + 1) / 2
                # control absolute position using joystick
                self.reference_js = [js_x, js_y]
            else:
                # control relative position using joystick
                self.reference_js = [
                    max(0, min(self.reference_js[0] + js_x * self.js_speed_scale, 1)),
                    max(0, min(self.reference_js[1] + js_y * self.js_speed_scale, 1)),
                ]
            new_gripper_angle = [
                button_data[grip_key][0],
                button_data[grip_key][0],
                button_data[grip_key][0],
                button_data[grip_key][0],
                self.reference_js[0],
                self.reference_js[1],
            ]  # [0, 1]
        arm_not_move_return = np.concatenate([current_eef_pose, new_gripper_angle])
        if len(pose_data) == 0:
            print("no data, quest not yet ready")
            return arm_not_move_return

        global trigger_state
        trigger_state[self.which_hand] = button_data[trigger_key][0] > 0.5
        if trigger_state[self.which_hand]:
            if self.control_active is True:
                if self._verbose:
                    print("controlling the arm")
                current_pose = pose_data[pose_key]
                delta_rot = current_pose[:3, :3] @ np.linalg.inv(
                    self.reference_quest_pose[:3, :3]
                )
                delta_pos = current_pose[:3, 3] - self.reference_quest_pose[:3, 3]
                if self.which_hand == "l":
                    t_mat = quest2left
                    t_mat_inv = left2quest
                    ur2isaac = left2isaac
                else:
                    t_mat = quest2right
                    t_mat_inv = right2quest
                    ur2isaac = right2isaac
                delta_pos_ur = (
                    apply_transfer(t_mat, delta_pos) * translation_scaling_factor
                )
                delta_rot_ur = t_mat[:3, :3] @ delta_rot @ t_mat_inv[:3, :3]
                if self._verbose:
                    print(f"delta pos and rot in VR space: \n{delta_pos}, {delta_rot}")
                    print(
                        f"delta pos and rot in ur space: \n{delta_pos_ur}, {delta_rot_ur}"
                    )
                    delta_pos_isaac = apply_transfer(quest2isaac, delta_pos)
                    delta_pos_isaac_ur = apply_transfer(ur2isaac, delta_pos_ur)
                    print("delta pos in isaac", delta_pos_isaac)
                    print("delta pos in ur in isaac", delta_pos_isaac_ur)
                    print("delta", delta_pos_isaac - delta_pos_isaac_ur)

                next_ee_rot_ur = delta_rot_ur @ self.reference_ee_rot_ur  # [3, 3]
                next_ee_pos_ur = delta_pos_ur + self.reference_ee_pos_ur
                next_ee_rot_ur = quaternion.as_rotation_vector(
                    quaternion.from_rotation_matrix(next_ee_rot_ur)
                )
                new_eef_pose = np.concatenate([next_ee_pos_ur, next_ee_rot_ur])
                command = np.concatenate([new_eef_pose, new_gripper_angle])
                return command

            else:  # last state is not in active
                self.control_active = True
                if self._verbose:
                    print("control activated!")
                self.reference_quest_pose = pose_data[pose_key]

                # reference in their local TCP frames
                self.reference_ee_rot_ur = ee_rot
                self.reference_ee_pos_ur = ee_pos
                return arm_not_move_return
        else:
            if self._verbose:
                print("deactive control")
            self.control_active = False
            self.reference_quest_pose = None
            return arm_not_move_return


class DualArmQuestAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right
        global trigger_state
        self.trigger_state = trigger_state

    def act(self, obs: Dict) -> np.ndarray:
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


if __name__ == "__main__":
    oculus_reader = OculusReader()
    while True:
        """
        example output:
        ({'l': array([[-0.828395 ,  0.541667 , -0.142682 ,  0.219646 ],
        [-0.107737 ,  0.0958919,  0.989544 , -0.833478 ],
        [ 0.549685 ,  0.835106 , -0.0210789, -0.892425 ],
        [ 0.       ,  0.       ,  0.       ,  1.       ]]), 'r': array([[-0.328058,  0.82021 ,  0.468652, -1.8288  ],
        [ 0.070887,  0.516083, -0.8536  , -0.238691],
        [-0.941994, -0.246809, -0.227447, -0.370447],
        [ 0.      ,  0.      ,  0.      ,  1.      ]])},
        {'A': False, 'B': False, 'RThU': True, 'RJ': False, 'RG': False, 'RTr': False, 'X': False, 'Y': False, 'LThU': True, 'LJ': False, 'LG': False, 'LTr': False, 'leftJS': (0.0, 0.0), 'leftTrig': (0.0,), 'leftGrip': (0.0,), 'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (0.0,)})

        """
        pose_data, button_data = oculus_reader.get_transformations_and_buttons()
        if len(pose_data) == 0:
            print("no data")
            continue
        else:
            print(pose_data, button_data)
