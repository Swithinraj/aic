"""
Deploy a locally-trained ACT policy for the AIC cable-insertion task.

ROS param (required):
    checkpoint_path — absolute path to the pretrained_model/ folder, e.g.
        /home/ibrahim/ros2_ws/src/aic/outputs/train/aic_act_run_001/checkpoints/100000/pretrained_model

State (33D, must match training):
    tcp_pose(7) + tcp_velocity(6) + tcp_error(6) + joint_positions(7) + joint_velocity(7)

Action (6D delta TCP at 10 Hz):
    [dx, dy, dz, drx, dry, drz]  — position delta (m) + axis-angle rotation delta (rad)

Usage:
    pixi run ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=team_policy.run_act \\
        -p checkpoint_path:=/absolute/path/to/pretrained_model
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from std_msgs.msg import Header

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file

# ImageNet normalisation — lerobot default for video features
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Safety clamps per 10 Hz step (100 ms)
_MAX_DELTA_POS_M   = 0.15   # 15 cm — covers p95 of training deltas (mean=5.5cm, p95=13cm)
_MAX_DELTA_ROT_RAD = 0.20   # ~11 deg max rotation per step


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float64)


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / angle
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, c])


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        checkpoint_path = str(
            parent_node.declare_parameter("checkpoint_path", "").value
        )
        if not checkpoint_path:
            raise ValueError(
                "RunACT requires the 'checkpoint_path' ROS parameter.\n"
                "  -p checkpoint_path:=/absolute/path/to/pretrained_model"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(checkpoint_path)

        # --- Load config + weights ---
        with open(path / "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)  # lerobot adds "type"; ACTConfig doesn't expect it

        import draccus
        config = draccus.decode(ACTConfig, cfg_dict)
        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        # --- Load normalisation stats ---
        # Input normaliser (observation.state stats)
        pre_path  = path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        # Output unnormaliser (action stats)
        post_path = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

        pre_stats  = load_file(str(pre_path))  if pre_path.exists()  else {}
        post_stats = load_file(str(post_path)) if post_path.exists() else {}

        def _get(d: dict, key: str, shape: tuple, default: float) -> torch.Tensor:
            if key in d:
                return d[key].to(self.device).float()
            self.get_logger().warning(f"Stat key '{key}' not found — using default {default}")
            return torch.full(shape, default, device=self.device)

        STATE_DIM  = 33
        ACTION_DIM = 6

        self.state_mean  = _get(pre_stats,  "observation.state.mean", (STATE_DIM,),  0.0).view(1, -1)
        self.state_std   = _get(pre_stats,  "observation.state.std",  (STATE_DIM,),  1.0).view(1, -1)
        self.action_mean = _get(post_stats, "action.mean",            (ACTION_DIM,), 0.0).view(1, -1)
        self.action_std  = _get(post_stats, "action.std",             (ACTION_DIM,), 1.0).view(1, -1)

        self._img_mean = _IMAGENET_MEAN.to(self.device)
        self._img_std  = _IMAGENET_STD.to(self.device)

        self.get_logger().info(
            f"RunACT loaded:\n"
            f"  path       = {path}\n"
            f"  device     = {self.device}\n"
            f"  state      = {STATE_DIM}D\n"
            f"  action     = {ACTION_DIM}D delta-TCP\n"
            f"  pre_stats  = {'found' if pre_stats  else 'MISSING (using defaults)'}\n"
            f"  post_stats = {'found' if post_stats else 'MISSING (using defaults)'}"
        )

    # ----------------------------------------------------------------
    # Observation → model input
    # ----------------------------------------------------------------

    def _img_to_tensor(self, img_msg) -> torch.Tensor:
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )
        t = (
            torch.from_numpy(arr.copy())
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        return (t - self._img_mean) / self._img_std

    def _build_state(self, obs_msg) -> torch.Tensor:
        """33D state: tcp_pose(7)+tcp_vel(6)+tcp_err(6)+jpos(7)+jvel(7)."""
        cs  = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js  = obs_msg.joint_states

        raw = np.array([
            tcp.position.x,    tcp.position.y,    tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x,  vel.linear.y,  vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *list(cs.tcp_error),
            *list(js.position[:7]),
            *list(js.velocity[:7]),
        ], dtype=np.float32)

        t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        return (t - self.state_mean) / self.state_std

    def _to_batch(self, obs_msg) -> dict:
        return {
            "observation.images.left":   self._img_to_tensor(obs_msg.left_image),
            "observation.images.center": self._img_to_tensor(obs_msg.center_image),
            "observation.images.right":  self._img_to_tensor(obs_msg.right_image),
            "observation.state":         self._build_state(obs_msg),
        }

    # ----------------------------------------------------------------
    # Action → motion command
    # ----------------------------------------------------------------

    def _delta_to_motion(self, obs_msg, action_6d: np.ndarray) -> MotionUpdate:
        """Apply 6D delta to current TCP pose → absolute MODE_POSITION command."""
        cs  = obs_msg.controller_state
        tcp = cs.tcp_pose

        cur_pos  = np.array([tcp.position.x,    tcp.position.y,    tcp.position.z],   dtype=np.float64)
        cur_quat = np.array([tcp.orientation.x, tcp.orientation.y,
                              tcp.orientation.z, tcp.orientation.w], dtype=np.float64)

        # Clamp for safety
        dp = np.clip(action_6d[:3].astype(np.float64), -_MAX_DELTA_POS_M, _MAX_DELTA_POS_M)
        dr = action_6d[3:6].astype(np.float64)
        dr_norm = np.linalg.norm(dr)
        if dr_norm > _MAX_DELTA_ROT_RAD:
            dr = dr / dr_norm * _MAX_DELTA_ROT_RAD

        new_pos  = cur_pos + dp
        dq       = _axis_angle_to_quat(dr)
        # new_quat = dq * cur_quat  (matches how deltas were computed in training)
        new_quat = _quat_multiply(dq, cur_quat)
        nrm = np.linalg.norm(new_quat)
        if nrm > 1e-9:
            new_quat /= nrm

        target = Pose(
            position=Point(x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])),
            orientation=Quaternion(
                x=float(new_quat[0]), y=float(new_quat[1]),
                z=float(new_quat[2]), w=float(new_quat[3]),
            ),
        )

        mu = MotionUpdate()
        mu.header = Header(
            frame_id="base_link",
            stamp=self._parent_node.get_clock().now().to_msg(),
        )
        mu.pose = target
        mu.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        )
        # Match the stiffness/damping used by CheatCode during training
        mu.target_stiffness = np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()
        mu.target_damping   = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()
        mu.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        mu.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        return mu

    # ----------------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.policy.reset()
        self.get_logger().info(f"RunACT.insert_cable() start — task: {task}")

        TIME_LIMIT_S    = 120.0          # doubled: give model more time for final approach
        STEP_S          = 1.0 / 10.0   # 10 Hz matches training
        FORCE_LIMIT_N   = 80.0         # raised: scoring only penalises >20N for >1s
        TORQUE_LIMIT_NM = 15.0         # raised: allow insertion torques
        EMERGENCY_FORCE = 150.0        # hard abort only at dangerous levels
        step_count      = 0
        force_exceeded_count = 0
        start           = time.time()

        while time.time() - start < TIME_LIMIT_S:
            t0 = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warning("No observation — skipping step")
                time.sleep(STEP_S)
                continue

            # Force/torque safety check — soft pause instead of hard stop
            wrench = obs_msg.wrist_wrench.wrench
            force_norm  = math.sqrt(wrench.force.x**2  + wrench.force.y**2  + wrench.force.z**2)
            torque_norm = math.sqrt(wrench.torque.x**2 + wrench.torque.y**2 + wrench.torque.z**2)

            # Emergency hard stop at extreme force
            if force_norm > EMERGENCY_FORCE:
                self.get_logger().warning(
                    f"RunACT EMERGENCY force — {force_norm:.1f}N — aborting"
                )
                break

            # Soft limit: skip action but keep looping (robot holds position)
            if force_norm > FORCE_LIMIT_N or torque_norm > TORQUE_LIMIT_NM:
                force_exceeded_count += 1
                if force_exceeded_count % 10 == 1:
                    self.get_logger().warning(
                        f"RunACT force soft-limit — force={force_norm:.1f}N torque={torque_norm:.2f}Nm — pausing motion"
                    )
                time.sleep(STEP_S)
                step_count += 1
                continue

            batch = self._to_batch(obs_msg)

            with torch.inference_mode():
                norm_action = self.policy.select_action(batch)   # (1, 6) normalised

            raw_action = (norm_action * self.action_std + self.action_mean)
            action_np  = raw_action[0].cpu().numpy().astype(np.float64)

            if step_count < 5:
                self.get_logger().info(
                    f"step={step_count} norm={norm_action[0].cpu().numpy().round(3).tolist()} "
                    f"raw={action_np.round(4).tolist()}"
                )

            mu = self._delta_to_motion(obs_msg, action_np)
            move_robot(motion_update=mu)

            step_count += 1
            if step_count % 10 == 0:
                send_feedback(f"RunACT step={step_count} elapsed={time.time()-start:.1f}s force={force_norm:.1f}N")

            time.sleep(max(0.0, STEP_S - (time.time() - t0)))

        self.get_logger().info(f"RunACT finished — {step_count} steps")
        return True


# aic_model resolves the class by the last component of the module path.
# -p policy:=team_policy.run_act  →  looks for `run_act` attribute
run_act = RunACT
