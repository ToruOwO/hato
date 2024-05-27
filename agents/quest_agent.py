from typing import Dict

import numpy as np
import quaternion
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils.inverse_kinematics import qpos_from_site_pose, nullspace_method
from oculus_reader.reader import OculusReader
from agents.agent import Agent

mjlib = mjbindings.mjlib

mj2ur = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
ur2mj = np.linalg.inv(mj2ur)

trigger_state = {"l": False, "r": False}


def apply_transfer(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    # xyz can be 3dim or 4dim (homogeneous) or can be a rotation matrix
    if len(xyz) == 3:
        xyz = np.append(xyz, 1)
    return np.matmul(mat, xyz)[:3]


quest2ur = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
# 45 deg CCW
ur2left = np.array(
    [
        [1 / 2 * np.sqrt(2), 1 / 2 * np.sqrt(2), 0, 0],
        [-1 / 2 * np.sqrt(2), 1 / 2 * np.sqrt(2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
# 45 deg CW
ur2right = np.array(
    [
        [1 / 2 * np.sqrt(2), -1 / 2 * np.sqrt(2), 0, 0],
        [1 / 2 * np.sqrt(2), 1 / 2 * np.sqrt(2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
ur2quest = np.linalg.inv(quest2ur)


def velocity_ik(physics,
    site_name,
    delta_rot_mat,
    delta_pos,
    joint_names):

  dtype = physics.data.qpos.dtype

  # Convert site name to index.
  site_id = physics.model.name2id(site_name, 'site')


  jac = np.empty((6, physics.model.nv), dtype=dtype)
  err = np.zeros(6, dtype=dtype)
  jac_pos, jac_rot = jac[:3], jac[3:]
  err_pos, err_rot = err[:3], err[3:]

  err_pos[:] = delta_pos[:]

  delta_rot_quat = np.empty(4, dtype=dtype)
  mjlib.mju_mat2Quat(delta_rot_quat, delta_rot_mat)
  mjlib.mju_quat2Vel(err_rot, delta_rot_quat, 1)

  mjlib.mj_jacSite(
      physics.model.ptr, physics.data.ptr, jac_pos, jac_rot, site_id)

  dof_indices = []
  for jn in joint_names:
    dof_idx = physics.model.joint(jn).id
    dof_indices.append(dof_idx)

  jac_joints = jac[:, dof_indices]

  update_joints = nullspace_method(
      jac_joints, err, regularization_strength=0.03)

  return update_joints


class SingleArmQuestAgent(Agent):
    def __init__(
        self,
        robot_type: str,
        which_hand: str,
        eef_control_mode: int = 0,
        verbose: bool = False,
        use_vel_ik: bool = False,
        vel_ik_speed_scale: float = 0.95,
    ) -> None:
        """Interact with the robot using the quest controller.

        leftTrig: press to start control (also record the current position as the home position)
        leftJS: a tuple of (x,y) for the joystick, only need y to control the gripper
        """
        self.which_hand = which_hand
        self.eef_control_mode = eef_control_mode
        assert self.which_hand in ["l", "r"]

        self.oculus_reader = OculusReader()
        if robot_type == "ur5":
            mjcf_model = mjcf.from_path("universal_robots_ur5e/ur5e.xml")
            mjcf_model.name = robot_type
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
        self.physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        self.control_active = False
        self.reference_quest_pose = None
        self.reference_ee_rot_ur = None
        self.reference_ee_pos_ur = None
        self.reference_js = [0.5, 0.5]
        self.js_speed_scale = 0.1

        self.use_vel_ik = use_vel_ik
        self.vel_ik_speed_scale = vel_ik_speed_scale

        if use_vel_ik:
            self.translation_scaling_factor = 1.0
        else:
            self.translation_scaling_factor = 2.0

        self.robot_type = robot_type
        self._verbose = verbose

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if self.robot_type == "ur5":
            num_dof = 6
        current_qpos = obs["joint_positions"][:num_dof]  # last one dim is the gripper
        # run the fk
        self.physics.data.qpos[:num_dof] = current_qpos
        self.physics.step()

        ee_rot_mj = np.array(
            self.physics.named.data.site_xmat["attachment_site"]
        ).reshape(3, 3)
        ee_pos_mj = np.array(self.physics.named.data.site_xpos["attachment_site"])
        if self.which_hand == "l":
            pose_key = "l"
            trigger_key = "leftTrig"
            grip_key = "leftGrip"
            joystick_key = "leftJS"
        elif self.which_hand == "r":
            pose_key = "r"
            trigger_key = "rightTrig"
            grip_key = "rightGrip"
            joystick_key = "rightJS"
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
                [current_qpos, obs["joint_positions"][num_dof:] * 0.0]
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
        arm_not_move_return = np.concatenate([current_qpos, new_gripper_angle])
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
                    t_mat = np.matmul(ur2left, quest2ur)
                else:
                    t_mat = np.matmul(ur2right, quest2ur)
                delta_pos_ur = (
                    apply_transfer(t_mat, delta_pos) * self.translation_scaling_factor
                )
                # ? is this the case?
                delta_rot_ur = quest2ur[:3, :3] @ delta_rot @ ur2quest[:3, :3]
                if self._verbose:
                    print(f"delta pos and rot in VR space: \n{delta_pos}, {delta_rot}")
                    print(
                        f"delta pos and rot in ur space: \n{delta_pos_ur}, {delta_rot_ur}"
                    )
                next_ee_rot_ur = delta_rot_ur @ self.reference_ee_rot_ur
                next_ee_pos_ur = delta_pos_ur + self.reference_ee_pos_ur

                if self.use_vel_ik:
                    next_ee_pos_mj = apply_transfer(ur2mj, next_ee_pos_ur)
                    next_ee_rot_mj = ur2mj[:3, :3] @ next_ee_rot_ur

                    err_rot_mj = next_ee_rot_mj @ np.linalg.inv(ee_rot_mj)
                    err_pos_mj = next_ee_pos_mj - ee_pos_mj

                    print(err_pos_mj)

                    delta_qpos = velocity_ik(
                        self.physics,
                        "attachment_site",
                        err_rot_mj.flatten(),
                        err_pos_mj,
                        joint_names=[
                            "shoulder_pan_joint",
                            "shoulder_lift_joint",
                            "elbow_joint",
                            "wrist_1_joint",
                            "wrist_2_joint",
                            "wrist_3_joint",
                        ],
                    )

                    new_qpos = current_qpos + delta_qpos * self.vel_ik_speed_scale

                else:
                    target_quat = quaternion.as_float_array(
                        quaternion.from_rotation_matrix(ur2mj[:3, :3] @ next_ee_rot_ur)
                    )
                    ik_result = qpos_from_site_pose(
                        self.physics,
                        "attachment_site",
                        target_pos=apply_transfer(ur2mj, next_ee_pos_ur),
                        target_quat=target_quat,
                        tol=1e-14,
                        max_steps=400,
                    )
                    self.physics.reset()
                    if ik_result.success:
                        new_qpos = ik_result.qpos[:num_dof]
                    else:
                        print("ik failed, using the original qpos")
                        return arm_not_move_return
                command = np.concatenate([new_qpos, new_gripper_angle])
                return command

            else:  # last state is not in active
                self.control_active = True
                if self._verbose:
                    print("control activated!")
                self.reference_quest_pose = pose_data[pose_key]

                self.reference_ee_rot_ur = mj2ur[:3, :3] @ ee_rot_mj
                self.reference_ee_pos_ur = apply_transfer(mj2ur, ee_pos_mj)
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
