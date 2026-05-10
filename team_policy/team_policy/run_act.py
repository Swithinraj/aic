"""
Deploy a locally-trained ACT policy for the AIC cable-insertion task.

CLEAN HYBRID ROLLBACK
---------------------
This is intentionally close to the older policy that behaved well:

  Phase 1 — ACT model with closed-loop replanning.
  Phase 2 — force-guided insertion along recent ACT motion.

The only V2-specific addition is the 30D observation layout:

  tcp_pose(7) + tcp_velocity(6) + joint_positions(7) + joint_velocity(7)
  + held YOLO port_xyz(3)

YOLO is primarily used as an input to the trained policy, matching the training
pipeline:

  * before first matching detection: [0, 0, 0]
  * when detection is valid: current YOLO xyz in base_link
  * when YOLO temporarily drops: hold the last valid xyz

After ACT reaches the port, YOLO is used only for a bounded lateral fine-align
guard so the insertion phase does not start from an obvious side offset.
"""
from __future__ import annotations

import json
import math
import sys
import time
import types
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from std_msgs.msg import Header, String

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

# ---------------------------------------------------------------------------
# LeRobot ACT import workaround (bypass GROOT dataclass error)
# ---------------------------------------------------------------------------
import lerobot as _lerobot_pkg

_lerobot_root = Path(_lerobot_pkg.__file__).resolve().parent
_policies_dir = _lerobot_root / "policies"
_act_dir = _policies_dir / "act"

_policies_pkg = types.ModuleType("lerobot.policies")
_policies_pkg.__path__ = [str(_policies_dir)]
sys.modules["lerobot.policies"] = _policies_pkg

_act_pkg = types.ModuleType("lerobot.policies.act")
_act_pkg.__path__ = [str(_act_dir)]
sys.modules["lerobot.policies.act"] = _act_pkg

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_IMG_H, _IMG_W = 480, 640

_FORCE_HARD_N = 80.0
_TORQUE_HARD_NM = 12.0

_TIME_LIMIT_S = 170.0
_ACT_TIMEOUT_S = 80.0
_STEP_HZ = 10.0
_STEP_S = 1.0 / _STEP_HZ
_PACE_WALL_CAP_S = 1.0

_REPLAN_EVERY = 10
_EMA_ALPHA = 0.7

_MAX_TRANSLATION_DELTA_M = 0.15
_MAX_ROTATION_DELTA_RAD = 0.35

_STIFFNESS_APPROACH = np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()
_DAMPING_APPROACH = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()
_STIFFNESS_INSERT = _STIFFNESS_APPROACH
_DAMPING_INSERT = _DAMPING_APPROACH

_INSERT_STEP_M = 0.001
_INSERT_DEPTH_M = 0.040
_INSERT_ACTUAL_DEPTH_M = 0.025
_INSERT_TARGET_LEAD_M = 0.014
_INSERT_FORCE_THRESH = 5.0
_HOLD_AFTER_INSERT_S = 3.0

_STALL_WINDOW = 30
_STALL_VEL_THRESH = 0.0003
_MIN_ACT_STEPS = 100
_INSERT_AXIS_ACTIONS = 10
_ACTION_DIR_MIN_M = 1e-5
_ACT_NEAR_PORT_MIN_STEPS = 30
_ACT_NEAR_PORT_HOLD_STEPS = 6
_ACT_NEAR_PORT_SFP_M = 0.085
_ACT_NEAR_PORT_SC_M = 0.120
_ACT_PORT_WORSEN_M = 0.020
_ACT_PORT_WORSEN_STEPS = 4
_ACT_YOLO_ASSIST_MIN_STEPS = 10
_ACT_YOLO_ASSIST_MAX_PORT_DIST_M = 0.22
_ACT_YOLO_ASSIST_GAIN = 0.35
_ACT_YOLO_ASSIST_STEP_M = 0.004

_YOLO_FINE_MAX_STEPS = 25
_YOLO_FINE_TOL_M = 0.006
_YOLO_FINE_MAX_PORT_DIST_M = 0.35
_YOLO_FINE_STEP_M = 0.008
_YOLO_FINE_GAIN = 0.6
_YOLO_FINE_TOTAL_CAP_M = 0.080
_YOLO_FINE_NO_IMPROVE_STEPS = 6
_YOLO_FINE_IMPROVE_EPS_M = 0.0015
_YOLO_INSERT_LATERAL_MAX_SFP_M = 0.018
_YOLO_INSERT_LATERAL_MAX_SC_M = 0.025

_INSERT_STALL_WINDOW = 30
_INSERT_STALL_PROGRESS_M = 0.0005
_INSERT_STALL_MIN_CMD_M = 0.020
_INSERT_SEARCH_RADIUS_M = 0.002
_INSERT_SEARCH_MAX_ATTEMPTS = 12

_MIN_STATE_STD = 1e-6

_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)

_SCHEMA_V2_30D = "v2_30d_with_port_xyz"
_SCHEMA_CHUNK50_33D = "legacy_33d_with_tcp_error"
_SCHEMA_EXTENDED_ACT = "extended_act_subclass_schema"


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / angle
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def detect_state_schema(state_dim: int, *_unused, override: str = "auto") -> str:
    """Minimal schema detection kept for checkpoint compatibility."""
    if override == "auto" and _unused and isinstance(_unused[-1], str):
        override = _unused[-1]
    if override == "false":
        return _SCHEMA_V2_30D
    if override == "true":
        return _SCHEMA_CHUNK50_33D
    if state_dim == 30:
        return _SCHEMA_V2_30D
    if state_dim == 33:
        return _SCHEMA_CHUNK50_33D
    if state_dim in {63, 68, 75, 77}:
        return _SCHEMA_EXTENDED_ACT
    raise ValueError(f"Unsupported observation.state dim: {state_dim}. Expected 30 or 33.")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.node = parent_node

        checkpoint_path = self._resolve_checkpoint_path(
            str(parent_node.declare_parameter("checkpoint_path", "").value)
        )

        decl = parent_node.declare_parameter
        self.action_scale = float(decl("action_scale", 1.0).value)
        self.rotation_gain = float(decl("rotation_gain", 1.0).value)
        self.max_translation_delta_m = float(
            decl("max_translation_delta_m", _MAX_TRANSLATION_DELTA_M).value
        )
        self.max_rotation_delta_rad = float(
            decl("max_rotation_delta_rad", _MAX_ROTATION_DELTA_RAD).value
        )
        self.ema_alpha = float(decl("ema_alpha", _EMA_ALPHA).value)
        self.replan_every = int(decl("replan_every", _REPLAN_EVERY).value)
        self.delta_pose_scale = float(decl("delta_pose_scale", -1.0).value)

        self.insert_step_m = float(decl("insert_step_m", _INSERT_STEP_M).value)
        self.insert_depth_m = float(decl("insert_depth_m", _INSERT_DEPTH_M).value)
        self.insert_actual_depth_m = float(
            decl("insert_actual_depth_m", _INSERT_ACTUAL_DEPTH_M).value
        )
        self.insert_target_lead_m = float(
            decl("insert_target_lead_m", _INSERT_TARGET_LEAD_M).value
        )
        self.insert_rotation_gain = float(decl("insert_rotation_gain", 1.0).value)
        self.require_yolo_insert_confirm = bool(
            decl("require_yolo_insert_confirm", False).value
        )
        self.insert_force_thresh_n = float(
            decl("insert_force_thresh_n", _INSERT_FORCE_THRESH).value
        )
        self.hold_after_insert_s = float(decl("hold_after_insert_s", _HOLD_AFTER_INSERT_S).value)
        self.time_limit_s = float(decl("time_limit_s", _TIME_LIMIT_S).value)
        self.act_timeout_s = float(decl("act_timeout_s", _ACT_TIMEOUT_S).value)
        self.force_hard_stop_n = float(decl("force_hard_stop_n", _FORCE_HARD_N).value)
        self.torque_hard_stop_nm = float(decl("torque_hard_stop_nm", _TORQUE_HARD_NM).value)
        self.schema_override = str(decl("prev_action_in_state_override", "auto").value).lower()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = checkpoint_path

        with open(path / "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)

        import draccus

        config = draccus.decode(ACTConfig, cfg_dict)
        self.config = config
        self.state_dim = int(config.input_features["observation.state"].shape[0])
        self.action_dim = int(config.output_features["action"].shape[0])
        if self.action_dim != 6:
            raise ValueError(f"RunACT expects 6D Cartesian actions, got {self.action_dim}")

        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        pre_path = path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        post_path = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        pre_stats = load_file(str(pre_path)) if pre_path.exists() else {}
        post_stats = load_file(str(post_path)) if post_path.exists() else {}

        def _get(stats, key, shape, default):
            if key in stats:
                return stats[key].to(self.device).float()
            self.get_logger().warning(f"Stat key '{key}' missing; using {default}")
            return torch.full(shape, default, device=self.device)

        self.state_mean = _get(
            pre_stats, "observation.state.mean", (self.state_dim,), 0.0
        ).view(1, -1)
        self.state_std = torch.clamp(
            _get(pre_stats, "observation.state.std", (self.state_dim,), 1.0).view(1, -1),
            min=_MIN_STATE_STD,
        )
        self.action_mean = _get(post_stats, "action.mean", (self.action_dim,), 0.0).view(1, -1)
        self.action_std = _get(post_stats, "action.std", (self.action_dim,), 1.0).view(1, -1)

        self._img_mean = _IMAGENET_MEAN.to(self.device)
        self._img_std = _IMAGENET_STD.to(self.device)
        self.schema = detect_state_schema(
            self.state_dim, pre_stats, post_stats, override=self.schema_override
        )
        if self.delta_pose_scale < 0.0:
            # V2 was trained on frame-to-frame delta poses. Legacy 33D checkpoints
            # were commonly run as velocity-like commands integrated over 0.1s.
            self.delta_pose_scale = _STEP_S if self.schema == _SCHEMA_CHUNK50_33D else 1.0

        self._prev_action: Optional[np.ndarray] = None
        self._target_port_name = ""
        self._target_port_type = ""
        self._target_module_name = ""
        self._target_plug_type = ""
        self._is_sc_task = False

        self._yolo_lock = threading.Lock()
        self._yolo_port_xyz = np.zeros(3, dtype=np.float32)
        self._yolo_seen_target = False
        self._yolo_last_det_time: Optional[float] = None
        self._yolo_reported_once = False
        parent_node.create_subscription(String, "/fused_yolo/detections_json", self._cb_fused_yolo, 10)

        n_action_steps = int(getattr(config, "n_action_steps", 10) or 10)
        chunk_size = int(getattr(config, "chunk_size", 100) or 100)
        self.get_logger().info(
            "RunACT loaded (clean hybrid rollback):\n"
            f"  path             = {path}\n"
            f"  device           = {self.device}\n"
            f"  state_dim/schema = {self.state_dim} / {self.schema}\n"
            f"  action_dim       = {self.action_dim}\n"
            f"  chunk/actions    = {chunk_size} / {n_action_steps}\n"
            f"  replan_every     = {self.replan_every}\n"
            f"  action_scale     = {self.action_scale}\n"
            f"  delta_pose_scale = {self.delta_pose_scale}\n"
            f"  rotation_gain    = {self.rotation_gain}\n"
            f"  max_delta        = {self.max_translation_delta_m}m / {self.max_rotation_delta_rad}rad\n"
            f"  insert           = step {self.insert_step_m*1000:.1f}mm, "
            f"cmd {self.insert_depth_m*1000:.1f}mm, "
            f"actual {self.insert_actual_depth_m*1000:.1f}mm, "
            f"lead {self.insert_target_lead_m*1000:.1f}mm\n"
            f"  time             = {self.time_limit_s}s total, {self.act_timeout_s}s ACT"
        )

    # ------------------------------------------------------------------
    # Setup and observation helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint_path(self, raw_path: str) -> Path:
        raw = raw_path.strip()
        if not raw:
            raise ValueError("RunACT requires -p checkpoint_path:=/path/to/pretrained_model")

        path = Path(raw).expanduser()
        candidates = [path, path / "pretrained_model"]
        for candidate in candidates:
            if all((candidate / filename).exists() for filename in _REQUIRED_FILES):
                return candidate
        missing = [name for name in _REQUIRED_FILES if not (path / name).exists()]
        raise FileNotFoundError(
            f"Checkpoint path is not a valid pretrained_model directory: {path}. "
            f"Missing at top level: {missing}"
        )

    def _now(self) -> float:
        return self._parent_node.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _pad(values, length: int) -> list[float]:
        out = [float(v) for v in list(values)[:length]]
        while len(out) < length:
            out.append(0.0)
        return out

    def _ros_image_to_rgb(self, img_msg) -> np.ndarray:
        arr = np.frombuffer(img_msg.data, dtype=np.uint8)
        if arr.size < img_msg.height * img_msg.width * 3:
            raise ValueError("image buffer too small for RGB/BGR image")
        arr = arr[: img_msg.height * img_msg.width * 3].reshape(img_msg.height, img_msg.width, 3)
        encoding = str(getattr(img_msg, "encoding", "rgb8")).lower()
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr).copy()

    def _img_to_tensor(self, img_msg) -> torch.Tensor:
        arr = self._ros_image_to_rgb(img_msg)
        t = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        if t.shape[2] != _IMG_H or t.shape[3] != _IMG_W:
            t = F.interpolate(t, size=(_IMG_H, _IMG_W), mode="bilinear", align_corners=False)
        return (t - self._img_mean) / self._img_std

    def _current_port_state(self) -> tuple[np.ndarray, bool, float, bool]:
        with self._yolo_lock:
            xyz = self._yolo_port_xyz.copy()
            seen_target = bool(self._yolo_seen_target)
            last_det_time = self._yolo_last_det_time
        if not seen_target or last_det_time is None:
            return xyz, False, 10.0, False
        age_s = self._now() - last_det_time
        return xyz, bool(age_s < 0.15), float(min(10.0, max(0.0, age_s))), True

    def _current_port_xyz(self) -> tuple[np.ndarray, bool]:
        xyz, fresh_valid, _, _ = self._current_port_state()
        return xyz, fresh_valid

    def _build_state(self, obs_msg) -> torch.Tensor:
        cs = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js = obs_msg.joint_states

        tcp_pose = [
            tcp.position.x,
            tcp.position.y,
            tcp.position.z,
            tcp.orientation.x,
            tcp.orientation.y,
            tcp.orientation.z,
            tcp.orientation.w,
        ]
        tcp_vel = [
            vel.linear.x,
            vel.linear.y,
            vel.linear.z,
            vel.angular.x,
            vel.angular.y,
            vel.angular.z,
        ]
        joint_pos = self._pad(js.position, 7)
        joint_vel = self._pad(js.velocity, 7)

        if self.schema == _SCHEMA_V2_30D:
            port_xyz, _ = self._current_port_xyz()
            raw = np.array([*tcp_pose, *tcp_vel, *joint_pos, *joint_vel, *port_xyz], dtype=np.float32)
        else:
            tcp_error = self._pad(getattr(cs, "tcp_error", []), 6)
            raw = np.array([*tcp_pose, *tcp_vel, *tcp_error, *joint_pos, *joint_vel], dtype=np.float32)

        if raw.shape[0] != self.state_dim:
            raise ValueError(
                f"Built state dim {raw.shape[0]}, checkpoint expects {self.state_dim}. "
                f"schema={self.schema}"
            )
        t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        return (t - self.state_mean) / self.state_std

    def _to_batch(self, obs_msg) -> dict:
        return {
            "observation.images.left": self._img_to_tensor(obs_msg.left_image),
            "observation.images.center": self._img_to_tensor(obs_msg.center_image),
            "observation.images.right": self._img_to_tensor(obs_msg.right_image),
            "observation.state": self._build_state(obs_msg),
        }

    # ------------------------------------------------------------------
    # YOLO hold-last input for V2 state
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_name(value: object) -> str:
        return str(value).strip().lower()

    def _target_match_rank(self, det: dict) -> int | None:
        target_port = self._target_port_name
        target_type = self._target_port_type
        target_module = self._target_module_name
        names = {
            self._norm_name(det.get("instance_name", "")),
            self._norm_name(det.get("class_name", "")),
        }
        names.discard("")
        if not names:
            return None

        exact_aliases = {target_port} if target_port else set()
        if any(name in exact_aliases for name in names):
            return 0
        if target_type == "sfp" and any(name == "sfp_port" or name.startswith("sfp_port_") for name in names):
            return 1
        if target_type == "sc" and any(name == "sc_port" or name.startswith("sc_port_") for name in names):
            return 1
        if target_port and any(target_port in name or name in target_port for name in names):
            return 2
        return None

    def _cb_fused_yolo(self, msg: String) -> None:
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(dets, list):
            return

        best_rank = None
        best_conf = float("-inf")
        best_xyz = None
        for det in dets:
            if not isinstance(det, dict):
                continue
            rank = self._target_match_rank(det)
            if rank is None:
                continue
            pos = det.get("pose_base_link", {}).get("position", {})
            if not isinstance(pos, dict):
                continue
            try:
                xyz = np.array(
                    [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))],
                    dtype=np.float32,
                )
                conf = float(det.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if best_rank is None or rank < best_rank or (rank == best_rank and conf > best_conf):
                best_rank = rank
                best_conf = conf
                best_xyz = xyz

        # Hold-last semantics: if there is no match in this message, do nothing.
        if best_xyz is not None:
            with self._yolo_lock:
                self._yolo_port_xyz = best_xyz
                self._yolo_seen_target = True
                self._yolo_last_det_time = self._now()
                should_report = not self._yolo_reported_once
                self._yolo_reported_once = True
            if should_report:
                self.get_logger().info(
                    "YOLO target latched: "
                    f"rank={best_rank} conf={best_conf:.3f} "
                    f"xyz=({best_xyz[0]:+.4f},{best_xyz[1]:+.4f},{best_xyz[2]:+.4f})"
                )

    def _reset_task_target(self, task: Task) -> None:
        with self._yolo_lock:
            self._target_port_name = self._norm_name(getattr(task, "port_name", ""))
            self._target_port_type = self._norm_name(getattr(task, "port_type", ""))
            self._target_module_name = self._norm_name(getattr(task, "target_module_name", ""))
            self._target_plug_type = self._norm_name(getattr(task, "plug_type", ""))
            self._yolo_port_xyz = np.zeros(3, dtype=np.float32)
            self._yolo_seen_target = False
            self._yolo_last_det_time = None
            self._yolo_reported_once = False
        self._prev_action = None
        self._is_sc_task = self._target_plug_type == "sc"

    # ------------------------------------------------------------------
    # Action and motion helpers
    # ------------------------------------------------------------------

    def _smooth_action(self, action_np: np.ndarray) -> np.ndarray:
        if self._prev_action is None:
            self._prev_action = action_np.copy()
            return action_np
        smoothed = self.ema_alpha * action_np + (1.0 - self.ema_alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def _apply_action_shaping(
        self,
        action_6d: np.ndarray,
        port_dist_m: Optional[float] = None,
    ) -> np.ndarray:
        del port_dist_m  # Kept for test/smoke compatibility; no YOLO controller here.
        action = np.nan_to_num(
            action_6d.astype(np.float64) * self.action_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        action[3:6] *= self.rotation_gain

        t_norm = float(np.linalg.norm(action[:3]))
        if self.max_translation_delta_m > 0.0 and t_norm > self.max_translation_delta_m:
            action[:3] *= self.max_translation_delta_m / t_norm

        r_norm = float(np.linalg.norm(action[3:6]))
        if self.max_rotation_delta_rad > 0.0 and r_norm > self.max_rotation_delta_rad:
            action[3:6] *= self.max_rotation_delta_rad / r_norm
        return action

    def _select_model_action(self, obs_msg) -> np.ndarray:
        batch = self._to_batch(obs_msg)
        with torch.inference_mode():
            norm_action = self.policy.select_action(batch)
        raw_action = (norm_action * self.action_std + self.action_mean)
        action_np = raw_action[0].cpu().numpy().astype(np.float64)
        action_np = self._smooth_action(action_np)
        return self._apply_action_shaping(action_np)

    def _make_motion(self, target_pose: Pose, stiffness, damping, wrench_gains=None) -> MotionUpdate:
        mu = MotionUpdate()
        mu.header = Header(
            frame_id="base_link",
            stamp=self._parent_node.get_clock().now().to_msg(),
        )
        mu.pose = target_pose
        mu.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION
        )
        mu.target_stiffness = stiffness
        mu.target_damping = damping
        mu.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        mu.wrench_feedback_gains_at_tip = wrench_gains or [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        return mu

    def _delta_to_pose(self, obs_msg, action_6d: np.ndarray) -> Pose:
        tcp = obs_msg.controller_state.tcp_pose
        cur_pos = np.array([tcp.position.x, tcp.position.y, tcp.position.z], dtype=np.float64)
        cur_quat = np.array(
            [tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w],
            dtype=np.float64,
        )

        new_pos = cur_pos + action_6d[:3] * self.delta_pose_scale
        dq = _axis_angle_to_quat(action_6d[3:6] * self.delta_pose_scale)
        new_quat = _quat_multiply(dq, cur_quat)
        nrm = np.linalg.norm(new_quat)
        if nrm > 1e-9:
            new_quat /= nrm

        return Pose(
            position=Point(x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])),
            orientation=Quaternion(
                x=float(new_quat[0]),
                y=float(new_quat[1]),
                z=float(new_quat[2]),
                w=float(new_quat[3]),
            ),
        )

    def _get_tcp_state(self, obs_msg):
        tcp = obs_msg.controller_state.tcp_pose
        pos = np.array([tcp.position.x, tcp.position.y, tcp.position.z], dtype=np.float64)
        quat = np.array(
            [tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w],
            dtype=np.float64,
        )
        wrench = obs_msg.wrist_wrench.wrench
        force_n = math.sqrt(wrench.force.x**2 + wrench.force.y**2 + wrench.force.z**2)
        torque_n = math.sqrt(wrench.torque.x**2 + wrench.torque.y**2 + wrench.torque.z**2)
        return pos, quat, force_n, torque_n

    def _pace_to_step(self, t0_sim: float) -> None:
        wall_start = time.monotonic()
        while self._now() - t0_sim < _STEP_S:
            if time.monotonic() - wall_start > _PACE_WALL_CAP_S:
                return
            time.sleep(0.01)

    def _near_port_threshold_m(self) -> float:
        if self._target_plug_type == "sc" or self._target_port_type == "sc":
            return _ACT_NEAR_PORT_SC_M
        return _ACT_NEAR_PORT_SFP_M

    def _insert_lateral_threshold_m(self) -> float:
        if self._target_plug_type == "sc" or self._target_port_type == "sc":
            return _YOLO_INSERT_LATERAL_MAX_SC_M
        return _YOLO_INSERT_LATERAL_MAX_SFP_M

    def _apply_yolo_approach_assist(
        self,
        target_pose: Pose,
        tcp_pos: np.ndarray,
        tcp_quat: np.ndarray,
        recent_actions,
        step_count: int,
    ) -> tuple[float | None, float]:
        if step_count < _ACT_YOLO_ASSIST_MIN_STEPS or not recent_actions or len(recent_actions) < 5:
            return None, 0.0
        port_xyz, port_valid = self._current_port_xyz()
        if not port_valid:
            return None, 0.0
        port_dist = float(np.linalg.norm(port_xyz.astype(np.float64) - tcp_pos))
        if port_dist > _ACT_YOLO_ASSIST_MAX_PORT_DIST_M:
            return None, 0.0

        axis, _ = self._pick_insertion_axis(tcp_quat, recent_actions)
        lateral = self._lateral_error_to_port(tcp_pos, port_xyz, axis)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= self._insert_lateral_threshold_m():
            return lateral_norm, 0.0

        correction = lateral * _ACT_YOLO_ASSIST_GAIN
        corr_norm = float(np.linalg.norm(correction))
        if corr_norm > _ACT_YOLO_ASSIST_STEP_M:
            correction *= _ACT_YOLO_ASSIST_STEP_M / corr_norm
            corr_norm = _ACT_YOLO_ASSIST_STEP_M

        target_pose.position.x += float(correction[0])
        target_pose.position.y += float(correction[1])
        target_pose.position.z += float(correction[2])
        return lateral_norm, corr_norm

    # ------------------------------------------------------------------
    # Phase 1: ACT approach
    # ------------------------------------------------------------------

    def _run_act_phase(self, get_observation, move_robot, send_feedback, start_time):
        self.policy.reset()
        self._prev_action = None
        step_count = 0
        start_pos = None
        pos_history = deque(maxlen=_STALL_WINDOW)
        recent_actions = deque(maxlen=30)
        near_port_steps = 0
        port_worsen_steps = 0
        best_port_dist = float("inf")
        saw_near_port = False

        self.get_logger().info("=== PHASE 1: ACT approach (clean hybrid) ===")

        while self._now() - start_time < self.act_timeout_s:
            if self._now() - start_time > self.time_limit_s - 30.0:
                self.get_logger().info("ACT phase: reserving final 30s for insertion")
                break

            t0_sim = self._now()
            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(_STEP_S)
                continue

            pos, quat, force_n, torque_n = self._get_tcp_state(obs_msg)
            if force_n > self.force_hard_stop_n or torque_n > self.torque_hard_stop_nm:
                self.get_logger().error(f"HARD STOP — F={force_n:.1f}N T={torque_n:.2f}Nm")
                return None, None, None, recent_actions, step_count

            if start_pos is None:
                start_pos = pos.copy()
            pos_history.append(pos.copy())

            if step_count > 0 and step_count % max(1, self.replan_every) == 0:
                self.policy._action_queue.clear()

            shaped = self._select_model_action(obs_msg)
            recent_actions.append(shaped.copy())

            target_pose = self._delta_to_pose(obs_msg, shaped)
            assist_lateral, assist_corr = self._apply_yolo_approach_assist(
                target_pose,
                pos,
                quat,
                recent_actions,
                step_count,
            )
            mu = self._make_motion(target_pose, _STIFFNESS_APPROACH, _DAMPING_APPROACH)
            move_robot(motion_update=mu)

            port_xyz, port_valid = self._current_port_xyz()
            port_d = None
            lateral_norm = None
            if port_valid:
                port_d = float(np.linalg.norm(port_xyz.astype(np.float64) - pos))
                if len(recent_actions) >= 5:
                    latch_axis, _ = self._pick_insertion_axis(quat, recent_actions)
                    lateral_norm = float(
                        np.linalg.norm(self._lateral_error_to_port(pos, port_xyz, latch_axis))
                    )
                if port_d < best_port_dist:
                    best_port_dist = port_d
                    port_worsen_steps = 0
                elif saw_near_port and port_d > best_port_dist + _ACT_PORT_WORSEN_M:
                    port_worsen_steps += 1
                else:
                    port_worsen_steps = 0

                near_and_aligned = (
                    lateral_norm is not None
                    and port_d <= self._near_port_threshold_m()
                    and lateral_norm <= self._insert_lateral_threshold_m()
                )
                if near_and_aligned:
                    near_port_steps += 1
                    saw_near_port = True
                else:
                    near_port_steps = 0

                if step_count >= _ACT_NEAR_PORT_MIN_STEPS:
                    if near_port_steps >= _ACT_NEAR_PORT_HOLD_STEPS:
                        self.get_logger().info(
                            "ACT phase: YOLO near-port latch — switching to insertion "
                            f"port_d={port_d*1000:.1f}mm "
                            f"lateral={lateral_norm*1000:.1f}mm"
                        )
                        break
                    if (
                        lateral_norm is not None
                        and lateral_norm <= self._insert_lateral_threshold_m()
                        and port_worsen_steps >= _ACT_PORT_WORSEN_STEPS
                    ):
                        self.get_logger().info(
                            "ACT phase: port distance is worsening after near pass — "
                            f"switching to insertion now best={best_port_dist*1000:.1f}mm "
                            f"now={port_d*1000:.1f}mm lateral={lateral_norm*1000:.1f}mm"
                        )
                        break

            if len(pos_history) >= _STALL_WINDOW and step_count >= _MIN_ACT_STEPS:
                positions = np.array(list(pos_history))
                per_step_vel = np.linalg.norm(np.diff(positions, axis=0), axis=1).mean()
                if per_step_vel < _STALL_VEL_THRESH:
                    self.get_logger().info(
                        f"Stall detected at step {step_count} "
                        f"(vel={per_step_vel*1000:.3f}mm/step) — switching to insertion"
                    )
                    break

            if step_count < 5 or step_count % 20 == 0:
                dist = np.linalg.norm(pos - start_pos)
                port_s = ""
                if port_d is not None:
                    port_s = f" | port_d={port_d*1000:.1f}mm"
                    if lateral_norm is not None:
                        port_s += f" lateral={lateral_norm*1000:.1f}mm"
                if assist_lateral is not None:
                    port_s += (
                        f" assist_lat={assist_lateral*1000:.1f}mm"
                        f" assist={assist_corr*1000:.1f}mm"
                    )
                self.get_logger().info(
                    f"ACT step={step_count:4d} | "
                    f"tcp=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) | "
                    f"Δ=({shaped[0]:+.5f},{shaped[1]:+.5f},{shaped[2]:+.5f},"
                    f"{shaped[3]:+.5f},{shaped[4]:+.5f},{shaped[5]:+.5f}) | "
                    f"F={force_n:.1f}N | travel={dist:.3f}m{port_s}"
                )

            step_count += 1
            if step_count % 50 == 0:
                elapsed = self._now() - start_time
                send_feedback(
                    f"ACT step={step_count} t={elapsed:.0f}s F={force_n:.1f}N "
                    f"tcp=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                )

            self._pace_to_step(t0_sim)

        obs_msg = get_observation()
        if obs_msg is None:
            return None, None, None, recent_actions, step_count
        pos, quat, _, _ = self._get_tcp_state(obs_msg)
        return obs_msg, pos, quat, recent_actions, step_count

    # ------------------------------------------------------------------
    # Phase 2: force-guided insertion
    # ------------------------------------------------------------------

    def _pick_insertion_axis(self, init_quat: np.ndarray, recent_actions) -> tuple[np.ndarray, str]:
        if recent_actions and len(recent_actions) >= 5:
            actions_arr = np.array(list(recent_actions)[-_INSERT_AXIS_ACTIONS:])
            avg_delta_pos = actions_arr[:, :3].mean(axis=0)
            dir_norm = float(np.linalg.norm(avg_delta_pos))
            if dir_norm > _ACTION_DIR_MIN_M:
                return avg_delta_pos / dir_norm, f"last_{len(actions_arr)}_model_actions"
        R = _quat_to_rotation_matrix(init_quat)
        return R[:, 2], "gripper_z"

    @staticmethod
    def _lateral_error_to_port(
        tcp_pos: np.ndarray,
        port_xyz: np.ndarray,
        insertion_axis: np.ndarray,
    ) -> np.ndarray:
        axis = np.asarray(insertion_axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            return np.zeros(3, dtype=np.float64)
        axis = axis / axis_norm
        port_vec = np.asarray(port_xyz, dtype=np.float64) - np.asarray(tcp_pos, dtype=np.float64)
        return port_vec - axis * float(np.dot(port_vec, axis))

    @staticmethod
    def _perpendicular_search_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = axis / axis_norm
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(axis, ref))) > 0.85:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u = np.cross(axis, ref)
        u /= max(float(np.linalg.norm(u)), 1e-9)
        v = np.cross(axis, u)
        v /= max(float(np.linalg.norm(v)), 1e-9)
        return u, v

    @staticmethod
    def _search_offset(attempt: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        pattern = (
            u,
            -u,
            v,
            -v,
            u + v,
            u - v,
            -u + v,
            -u - v,
        )
        direction = np.asarray(pattern[attempt % len(pattern)], dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        radius = _INSERT_SEARCH_RADIUS_M * (1 + attempt // len(pattern))
        return direction * min(radius, _INSERT_SEARCH_RADIUS_M * 2.0)

    def _run_yolo_fine_align(
        self,
        get_observation,
        move_robot,
        start_time,
        init_pos: np.ndarray,
        init_quat: np.ndarray,
        recent_actions,
    ) -> tuple[object | None, np.ndarray | None, np.ndarray | None, np.ndarray, float | None]:
        insert_dir, axis_source = self._pick_insertion_axis(init_quat, recent_actions)
        port_xyz, port_valid = self._current_port_xyz()
        if not port_valid:
            self.get_logger().info(
                "YOLO fine align skipped: no held target; using "
                f"{axis_source} insertion axis"
            )
            return None, init_pos, init_quat, insert_dir, None

        self.get_logger().info(
            "=== PHASE 1.5: YOLO lateral fine align === "
            f"axis={axis_source} dir=({insert_dir[0]:+.3f},{insert_dir[1]:+.3f},{insert_dir[2]:+.3f})"
        )

        last_obs = None
        last_pos = init_pos.copy()
        last_quat = init_quat.copy()
        total_correction = 0.0
        best_lateral = float("inf")
        no_improve_steps = 0
        last_lateral_norm = None

        for step in range(_YOLO_FINE_MAX_STEPS):
            if self._now() - start_time > self.time_limit_s - 20.0:
                break
            t0_sim = self._now()
            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(_STEP_S)
                continue

            pos, quat, force_n, torque_n = self._get_tcp_state(obs_msg)
            last_obs, last_pos, last_quat = obs_msg, pos, quat
            if force_n > self.force_hard_stop_n or torque_n > self.torque_hard_stop_nm:
                self.get_logger().error(
                    f"YOLO FINE HARD STOP — F={force_n:.1f}N T={torque_n:.2f}Nm"
                )
                break

            port_xyz, port_valid = self._current_port_xyz()
            if not port_valid:
                break

            port_dist = float(np.linalg.norm(port_xyz.astype(np.float64) - pos))
            if port_dist > _YOLO_FINE_MAX_PORT_DIST_M:
                self.get_logger().info(
                    f"YOLO fine align skipped: port_d={port_dist*1000:.1f}mm too large"
                )
                break

            lateral = self._lateral_error_to_port(pos, port_xyz, insert_dir)
            lateral_norm = float(np.linalg.norm(lateral))
            last_lateral_norm = lateral_norm
            if lateral_norm <= _YOLO_FINE_TOL_M:
                self.get_logger().info(
                    f"YOLO fine aligned at step {step}: lateral={lateral_norm*1000:.1f}mm "
                    f"port_d={port_dist*1000:.1f}mm"
                )
                break

            if lateral_norm < best_lateral - _YOLO_FINE_IMPROVE_EPS_M:
                best_lateral = lateral_norm
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if total_correction >= _YOLO_FINE_TOTAL_CAP_M:
                self.get_logger().info(
                    "YOLO fine align stopped: correction cap reached "
                    f"({total_correction*1000:.1f}mm), lateral={lateral_norm*1000:.1f}mm"
                )
                break
            if no_improve_steps >= _YOLO_FINE_NO_IMPROVE_STEPS:
                self.get_logger().info(
                    "YOLO fine align stopped: lateral error not improving "
                    f"(best={best_lateral*1000:.1f}mm, now={lateral_norm*1000:.1f}mm)"
                )
                break

            correction = lateral * _YOLO_FINE_GAIN
            corr_norm = float(np.linalg.norm(correction))
            if corr_norm > _YOLO_FINE_STEP_M:
                correction *= _YOLO_FINE_STEP_M / corr_norm
                corr_norm = _YOLO_FINE_STEP_M
            total_correction += corr_norm

            target_pose = Pose(
                position=Point(
                    x=float(pos[0] + correction[0]),
                    y=float(pos[1] + correction[1]),
                    z=float(pos[2] + correction[2]),
                ),
                orientation=Quaternion(
                    x=float(quat[0]),
                    y=float(quat[1]),
                    z=float(quat[2]),
                    w=float(quat[3]),
                ),
            )
            mu = self._make_motion(target_pose, _STIFFNESS_APPROACH, _DAMPING_APPROACH)
            move_robot(motion_update=mu)

            if step < 3 or step % 5 == 0:
                self.get_logger().info(
                    f"YOLO_FINE step={step:2d} | lateral={lateral_norm*1000:.1f}mm | "
                    f"corr={corr_norm*1000:.1f}mm | total={total_correction*1000:.1f}mm | "
                    f"port_d={port_dist*1000:.1f}mm"
                )

            self._pace_to_step(t0_sim)

        return last_obs, last_pos, last_quat, insert_dir, last_lateral_norm

    def _run_insertion_phase(
        self,
        get_observation,
        move_robot,
        send_feedback,
        start_time,
        init_pos: np.ndarray,
        init_quat: np.ndarray,
        recent_actions,
        insert_dir: np.ndarray,
    ) -> int:
        self.get_logger().info("=== PHASE 2: Force-guided insertion ===")

        insert_dir = np.asarray(insert_dir, dtype=np.float64)
        dir_norm = float(np.linalg.norm(insert_dir))
        if dir_norm < 1e-9:
            insert_dir = _quat_to_rotation_matrix(init_quat)[:, 2]
        else:
            insert_dir = insert_dir / dir_norm

        target_pos = init_pos.copy()
        last_target_pose = None
        commanded_push = 0.0
        insert_steps = 0
        contact_detected = False
        budget_logged = False
        progress_history = deque(maxlen=_INSERT_STALL_WINDOW)
        best_actual_progress = 0.0
        search_attempts = 0
        search_exhausted_logged = False
        search_offset = np.zeros(3, dtype=np.float64)
        search_u, search_v = self._perpendicular_search_basis(insert_dir)
        progress_reached_but_unaligned = False

        self.get_logger().info(
            f"Insertion direction: ({insert_dir[0]:+.3f},{insert_dir[1]:+.3f},{insert_dir[2]:+.3f})"
        )

        avg_rot_delta = np.zeros(3, dtype=np.float64)
        if recent_actions and len(recent_actions) >= 5:
            actions_arr = np.array(list(recent_actions))
            avg_rot_delta = actions_arr[:, 3:6].mean(axis=0) * 0.3 * self.insert_rotation_gain

        while self._now() - start_time < self.time_limit_s:
            t0_sim = self._now()
            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(_STEP_S)
                continue

            pos, quat, force_n, torque_n = self._get_tcp_state(obs_msg)
            if force_n > self.force_hard_stop_n or torque_n > self.torque_hard_stop_nm:
                self.get_logger().error(
                    f"INSERT HARD STOP — F={force_n:.1f}N T={torque_n:.2f}Nm"
                )
                break
            actual_progress = max(0.0, float(np.dot(pos - init_pos, insert_dir)))
            progress_history.append(actual_progress)

            if force_n > self.insert_force_thresh_n and not contact_detected:
                contact_detected = True
                self.get_logger().info(
                    f"Contact detected at F={force_n:.1f}N — continuing insertion"
                )

            if actual_progress >= self.insert_actual_depth_m:
                port_xyz, port_valid = self._current_port_xyz()
                lateral_norm = None
                if port_valid:
                    lateral_norm = float(
                        np.linalg.norm(self._lateral_error_to_port(pos, port_xyz, insert_dir))
                    )
                if not port_valid and self.require_yolo_insert_confirm:
                    if not progress_reached_but_unaligned:
                        progress_reached_but_unaligned = True
                        self.get_logger().info(
                            "Insertion progress reached but YOLO target is not held; "
                            "continuing instead of returning success "
                            f"actual={actual_progress*1000:.1f}mm"
                        )
                elif lateral_norm is None or lateral_norm <= self._insert_lateral_threshold_m():
                    self.get_logger().info(
                        f"Actual insertion progress reached: {actual_progress*1000:.1f}mm "
                        f"(commanded={commanded_push*1000:.1f}mm)"
                    )
                    break
                if not progress_reached_but_unaligned:
                    progress_reached_but_unaligned = True
                    self.get_logger().info(
                        "Insertion progress reached but YOLO lateral is still high; "
                        f"continuing instead of returning success "
                        f"actual={actual_progress*1000:.1f}mm "
                        f"lateral={lateral_norm*1000:.1f}mm "
                        f"limit={self._insert_lateral_threshold_m()*1000:.1f}mm"
                    )

            if actual_progress > best_actual_progress + 0.001:
                best_actual_progress = actual_progress
                search_offset[:] = 0.0
                progress_history.clear()

            if len(progress_history) >= _INSERT_STALL_WINDOW:
                progress_gain = float(progress_history[-1] - progress_history[0])
                stalled = (
                    progress_gain < _INSERT_STALL_PROGRESS_M
                    and force_n > self.insert_force_thresh_n
                    and commanded_push >= min(_INSERT_STALL_MIN_CMD_M, self.insert_depth_m * 0.5)
                )
                if stalled:
                    if search_attempts >= _INSERT_SEARCH_MAX_ATTEMPTS:
                        if not search_exhausted_logged:
                            search_exhausted_logged = True
                            self.get_logger().info(
                                "Insertion search exhausted; holding until timeout "
                                "because insertion is not visually confirmed: "
                                f"actual={actual_progress*1000:.1f}mm, "
                                f"best={best_actual_progress*1000:.1f}mm, "
                                f"cmd={commanded_push*1000:.1f}mm"
                            )
                        progress_history.clear()
                        search_offset[:] = 0.0
                    else:
                        search_offset = self._search_offset(search_attempts, search_u, search_v)
                        search_attempts += 1
                        progress_history.clear()
                        self.get_logger().info(
                            "Insertion stalled; applying search offset "
                            f"attempt={search_attempts}/{_INSERT_SEARCH_MAX_ATTEMPTS} "
                            f"offset=({search_offset[0]*1000:+.1f},"
                            f"{search_offset[1]*1000:+.1f},"
                            f"{search_offset[2]*1000:+.1f})mm "
                            f"actual={actual_progress*1000:.1f}mm"
                        )

            if commanded_push >= self.insert_depth_m and not budget_logged:
                budget_logged = True
                self.get_logger().info(
                    f"Commanded insertion target reached: {commanded_push*1000:.1f}mm; "
                    f"holding target until actual progress reaches "
                    f"{self.insert_actual_depth_m*1000:.1f}mm or time expires"
                )

            step_size = self.insert_step_m
            if force_n > 40.0:
                step_size *= 0.3
            elif force_n > 25.0:
                step_size *= 0.6

            commanded_push = min(self.insert_depth_m, commanded_push + step_size)
            target_progress = min(
                self.insert_depth_m,
                max(commanded_push, actual_progress + self.insert_target_lead_m),
            )
            target_pos = init_pos + insert_dir * target_progress + search_offset
            rot_scale = 0.1 if force_n > 15.0 else 0.3
            dq = _axis_angle_to_quat(avg_rot_delta * rot_scale)
            new_quat = _quat_multiply(dq, quat)
            nrm = np.linalg.norm(new_quat)
            if nrm > 1e-9:
                new_quat /= nrm

            target_pose = Pose(
                position=Point(
                    x=float(target_pos[0]),
                    y=float(target_pos[1]),
                    z=float(target_pos[2]),
                ),
                orientation=Quaternion(
                    x=float(new_quat[0]),
                    y=float(new_quat[1]),
                    z=float(new_quat[2]),
                    w=float(new_quat[3]),
                ),
            )
            last_target_pose = target_pose
            mu = self._make_motion(
                target_pose,
                _STIFFNESS_INSERT,
                _DAMPING_INSERT,
                wrench_gains=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            )
            move_robot(motion_update=mu)

            insert_steps += 1

            if insert_steps % 10 == 0 or insert_steps < 3:
                tcp_travel = np.linalg.norm(pos - init_pos)
                self.get_logger().info(
                    f"INSERT step={insert_steps:3d} | "
                    f"tcp=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) | "
                    f"tgt=({target_pos[0]:+.4f},{target_pos[1]:+.4f},{target_pos[2]:+.4f}) | "
                    f"F={force_n:.1f}N | cmd={commanded_push*1000:.1f}mm | "
                    f"actual={actual_progress*1000:.1f}mm | moved={tcp_travel*1000:.1f}mm"
                )

            if insert_steps % 20 == 0:
                send_feedback(
                    f"INSERT step={insert_steps} F={force_n:.1f}N "
                    f"cmd={commanded_push*1000:.1f}mm actual={actual_progress*1000:.1f}mm"
                )

            self._pace_to_step(t0_sim)

        self.get_logger().info(
            f"Holding position for connector stabilization ({self.hold_after_insert_s:.1f}s)..."
        )
        hold_start = self._now()
        while (
            self._now() - hold_start < self.hold_after_insert_s
            and self._now() - start_time < self.time_limit_s
        ):
            obs_msg = get_observation()
            if last_target_pose is not None:
                mu = self._make_motion(last_target_pose, _STIFFNESS_INSERT, _DAMPING_INSERT)
                move_robot(motion_update=mu)
            elif obs_msg is not None:
                pos, quat, _, _ = self._get_tcp_state(obs_msg)
                hold_pose = Pose(
                    position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                    orientation=Quaternion(
                        x=float(quat[0]),
                        y=float(quat[1]),
                        z=float(quat[2]),
                        w=float(quat[3]),
                    ),
                )
                mu = self._make_motion(hold_pose, _STIFFNESS_INSERT, _DAMPING_INSERT)
                move_robot(motion_update=mu)
            time.sleep(_STEP_S)

        return insert_steps

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.get_logger().info(f"RunACT.insert_cable() clean hybrid start — task: {task}")
        start = self._now()
        self._reset_task_target(task)
        self.get_logger().info(
            "RunACT target reset: "
            f"plug={self._target_plug_type or '?'} "
            f"port={self._target_port_name or '?'} "
            f"module={self._target_module_name or '?'}"
        )

        pos = None
        quat = None
        recent_actions = deque(maxlen=30)
        insert_dir = None
        act_steps_total = 0
        aligned_for_insert = False

        for approach_pass in range(3):
            if self._now() - start > self.time_limit_s - 35.0:
                break

            obs_msg, pos, quat, recent_actions, act_steps = self._run_act_phase(
                get_observation, move_robot, send_feedback, start
            )
            act_steps_total += act_steps
            self.get_logger().info(
                f"ACT phase pass {approach_pass + 1} complete: "
                f"{act_steps} steps in {self._now() - start:.1f}s"
            )
            if pos is None or quat is None:
                break

            fine_lateral = None
            if self._now() - start < self.time_limit_s - 20.0:
                _, fine_pos, fine_quat, insert_dir, fine_lateral = self._run_yolo_fine_align(
                    get_observation,
                    move_robot,
                    start,
                    pos,
                    quat,
                    recent_actions,
                )
                if fine_pos is not None and fine_quat is not None:
                    pos, quat = fine_pos, fine_quat

            if fine_lateral is None or fine_lateral <= self._insert_lateral_threshold_m():
                aligned_for_insert = True
                break

            self.get_logger().info(
                "Fine align is still outside insertion gate; running another "
                "model approach pass instead of inserting "
                f"lateral={fine_lateral*1000:.1f}mm "
                f"limit={self._insert_lateral_threshold_m()*1000:.1f}mm"
            )

        insert_steps = 0
        if aligned_for_insert and pos is not None and self._now() - start < self.time_limit_s - 5.0:
            if insert_dir is None:
                insert_dir, _ = self._pick_insertion_axis(quat, recent_actions)
            insert_steps = self._run_insertion_phase(
                get_observation,
                move_robot,
                send_feedback,
                start,
                pos,
                quat,
                recent_actions,
                insert_dir,
            )
        elif pos is not None:
            self.get_logger().info(
                "Insertion skipped because YOLO alignment never reached the gate; "
                "holding until trial budget instead of returning early"
            )
            hold_pose = Pose(
                position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=Quaternion(
                    x=float(quat[0]),
                    y=float(quat[1]),
                    z=float(quat[2]),
                    w=float(quat[3]),
                ),
            )
            while self._now() - start < self.time_limit_s - 1.0:
                mu = self._make_motion(hold_pose, _STIFFNESS_APPROACH, _DAMPING_APPROACH)
                move_robot(motion_update=mu)
                time.sleep(_STEP_S)

        elapsed = self._now() - start
        self.get_logger().info(
            f"RunACT clean hybrid finished — ACT:{act_steps_total} + INSERT:{insert_steps} "
            f"in {elapsed:.1f}s"
        )
        return True


# aic_model resolves the class by the last component of the module path.
run_act = RunACT
