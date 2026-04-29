"""
Deploy a locally-trained ACT policy for the AIC cable-insertion task.

ROS param (required):
    checkpoint_path — absolute path to the pretrained_model/ folder, e.g.
        /home/ibrahim/ros2_ws/src/aic/outputs/train/aic_act_run_001/checkpoints/100000/pretrained_model

State (30D, must match training):
    tcp_pose(7) + tcp_velocity(6) + joint_positions(7) + joint_velocity(7) + port_xyz(3)
    port_xyz — GT during training (privileged_tf), YOLO detection at inference (same base_link frame)

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
import threading
import time
from pathlib import Path

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from std_msgs.msg import String

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

# Force safety thresholds (competition penalty triggers at 20N sustained > 1s)
_FORCE_HOLD_N = 19.0   # hold current position above this
_FORCE_LOG_N  = 12.0   # start logging force readouts above this

# Impedance control params (matching Rocky's RunACT reference implementation)
_STIFFNESS = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]).flatten().tolist()
_DAMPING   = np.diag([ 40.0,  40.0,  40.0, 15.0, 15.0, 15.0]).flatten().tolist()
# Lateral XYZ compliance; zero rotation compliance to keep plug aligned
_WRENCH_GAINS = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]


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

        STATE_DIM  = 30   # tcp_pose(7)+tcp_vel(6)+jpos(7)+jvel(7)+port_xyz(3)
        ACTION_DIM = 6

        self.state_mean  = _get(pre_stats,  "observation.state.mean", (STATE_DIM,),  0.0).view(1, -1)
        self.state_std   = _get(pre_stats,  "observation.state.std",  (STATE_DIM,),  1.0).view(1, -1)
        self.action_mean = _get(post_stats, "action.mean",            (ACTION_DIM,), 0.0).view(1, -1)
        self.action_std  = _get(post_stats, "action.std",             (ACTION_DIM,), 1.0).view(1, -1)

        self._img_mean = _IMAGENET_MEAN.to(self.device)
        self._img_std  = _IMAGENET_STD.to(self.device)

        # --- YOLO port pose (filled by subscription, used in _build_state) ---
        self._yolo_lock        = threading.Lock()
        self._port_xyz         = np.zeros(3, dtype=np.float32)  # port position in base_link
        self._target_port_name = ""                              # set in insert_cable per task

        parent_node.create_subscription(
            String, "/fused_yolo/detections_json",
            self._cb_fused_yolo, 10,
        )

        self.get_logger().info(
            f"RunACT loaded:\n"
            f"  path       = {path}\n"
            f"  device     = {self.device}\n"
            f"  state      = {STATE_DIM}D (proprioception + port_xyz from YOLO)\n"
            f"  action     = {ACTION_DIM}D delta-TCP\n"
            f"  pre_stats  = {'found' if pre_stats  else 'MISSING (using defaults)'}\n"
            f"  post_stats = {'found' if post_stats else 'MISSING (using defaults)'}"
        )

    # ----------------------------------------------------------------
    # Observation → model input
    # ----------------------------------------------------------------

    def _cb_fused_yolo(self, msg: String) -> None:
        """Update port_xyz from the YOLO fused detections topic."""
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        with self._yolo_lock:
            target = self._target_port_name.lower()
        for det in dets:
            name = str(det.get("instance_name", "")).lower()
            cls  = str(det.get("class_name",    "")).lower()
            if target and not (target in name or target in cls):
                continue
            pose = det.get("pose_base_link", {}).get("position", {})
            if not pose:
                continue
            xyz = np.array([
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("z", 0.0)),
            ], dtype=np.float32)
            with self._yolo_lock:
                self._port_xyz = xyz
            return  # take first match

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
        """30D state: tcp_pose(7)+tcp_vel(6)+jpos(7)+jvel(7)+port_xyz(3).

        port_xyz comes from YOLO at inference (same base_link frame as GT used during training).
        Falls back to zeros if YOLO hasn't detected the port yet.
        """
        cs  = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js  = obs_msg.joint_states

        with self._yolo_lock:
            port_xyz = self._port_xyz.copy()

        raw = np.array([
            tcp.position.x,    tcp.position.y,    tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x,  vel.linear.y,  vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *list(js.position[:7]),
            *list(js.velocity[:7]),
            *port_xyz.tolist(),
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

    def _make_motion_update(self, obs_msg, pos: np.ndarray, quat: np.ndarray) -> MotionUpdate:
        """Build a MODE_POSITION MotionUpdate with impedance control and wrench compliance."""
        mu = MotionUpdate()
        mu.pose = Pose(
            position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            orientation=Quaternion(
                x=float(quat[0]), y=float(quat[1]),
                z=float(quat[2]), w=float(quat[3]),
            ),
        )
        mu.header.frame_id = "base_link"
        mu.header.stamp = self.get_clock().now().to_msg()
        mu.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        mu.target_stiffness = _STIFFNESS
        mu.target_damping   = _DAMPING
        mu.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        mu.wrench_feedback_gains_at_tip = _WRENCH_GAINS
        return mu

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

        return self._make_motion_update(obs_msg, new_pos, new_quat)

    def _hold_motion(self, obs_msg) -> MotionUpdate:
        """Hold current TCP pose (zero delta) — used during force safety hold."""
        tcp = obs_msg.controller_state.tcp_pose
        pos  = np.array([tcp.position.x,    tcp.position.y,    tcp.position.z],   dtype=np.float64)
        quat = np.array([tcp.orientation.x, tcp.orientation.y,
                          tcp.orientation.z, tcp.orientation.w], dtype=np.float64)
        return self._make_motion_update(obs_msg, pos, quat)

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
        with self._yolo_lock:
            self._target_port_name = str(task.port_name).strip().lower()
            self._port_xyz         = np.zeros(3, dtype=np.float32)  # reset until YOLO sees it
        self.get_logger().info(f"RunACT.insert_cable() start — task: {task}")

        TIME_LIMIT_S = 60.0
        STEP_S       = 1.0 / 10.0   # 10 Hz matches training
        step_count   = 0
        start        = time.time()

        # --- Tare: measure resting force (cable weight + gravity on sensor) ---
        # The F/T sensor reads ~20N at rest due to cable + gripper mass.
        # We subtract this baseline so the safety threshold is relative to contact force only.
        force_baseline = 0.0
        for _ in range(10):
            obs0 = get_observation()
            if obs0 is not None:
                w0 = obs0.wrist_wrench.wrench.force
                force_baseline = math.sqrt(w0.x*w0.x + w0.y*w0.y + w0.z*w0.z)
                break
            time.sleep(0.05)
        self.get_logger().info(f"RunACT force baseline (tare) = {force_baseline:.1f}N")

        while time.time() - start < TIME_LIMIT_S:
            t0 = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warning("No observation — skipping step")
                time.sleep(STEP_S)
                continue

            # --- Force safety check (relative to resting baseline) ---
            w = obs_msg.wrist_wrench.wrench.force
            force_mag = math.sqrt(w.x*w.x + w.y*w.y + w.z*w.z)
            contact_force = max(0.0, force_mag - force_baseline)
            if contact_force > _FORCE_LOG_N or step_count % 10 == 0:
                self.get_logger().info(f"step={step_count} force={force_mag:.1f}N contact={contact_force:.1f}N")
            if contact_force > _FORCE_HOLD_N:
                self.get_logger().warning(
                    f"Contact force {contact_force:.1f}N > {_FORCE_HOLD_N}N — holding position"
                )
                move_robot(motion_update=self._hold_motion(obs_msg))
                time.sleep(max(0.0, STEP_S - (time.time() - t0)))
                continue

            batch = self._to_batch(obs_msg)

            with torch.inference_mode():
                norm_action = self.policy.select_action(batch)   # (1, 6) normalised

            raw_action = (norm_action * self.action_std + self.action_mean)
            action_np  = raw_action[0].cpu().numpy().astype(np.float64)

            if step_count < 5 or step_count % 10 == 0:
                with self._yolo_lock:
                    pxyz = self._port_xyz.tolist()
                self.get_logger().info(
                    f"step={step_count} port_xyz={[round(v,3) for v in pxyz]} "
                    f"raw={action_np.round(4).tolist()}"
                )

            mu = self._delta_to_motion(obs_msg, action_np)
            move_robot(motion_update=mu)

            step_count += 1
            if step_count % 10 == 0:
                send_feedback(f"RunACT step={step_count} elapsed={time.time()-start:.1f}s")

            time.sleep(max(0.0, STEP_S - (time.time() - t0)))

        self.get_logger().info(f"RunACT finished — {step_count} steps")
        return True


# aic_model resolves the class by the last component of the module path.
# -p policy:=team_policy.run_act  →  looks for `run_act` attribute
run_act = RunACT
