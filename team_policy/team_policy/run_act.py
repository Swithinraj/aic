"""
Deploy the trained_model_v3 ACT policy for the AIC cable-insertion task.

PIPELINE
--------
    Phase 1  : ACT closed-loop approach, close to the older clean hybrid that
               behaved well.
    Phase 1.5: bounded YOLO lateral fine-align *between* approach and insertion.
    Phase 2  : force-guided axial insertion along recent ACT motion.

V3 OBSERVATION LAYOUT (77D — must match convert_to_lerobot_v2._build_state_77d)
------------------------------------------------------------------------------
    [ 0: 7]  tcp_pose                  (x y z qx qy qz qw)
    [ 7:13]  tcp_velocity              (vx vy vz wx wy wz)
    [13:19]  tcp_error                 (controller tracking error)
    [19:26]  joint_positions
    [26:33]  joint_velocity
    [33:36]  yolo_port_xyz             (held fused target xyz in base_link)
    [36:37]  yolo_valid                (1.0 if last fused detection < 0.15s ago)
    [37:38]  yolo_age                  (seconds, clamped to 10.0)
    [38:41]  port_delta_tcp            (yolo_port_xyz - tcp_position)
    [41:47]  tared_wrist_force_torque  (fx fy fz tx ty tz, baseline subtracted)
    [47:54]  yolo_left   feature 7D    [conf,cx,cy,w,h,valid,age]
    [54:61]  yolo_center feature 7D
    [61:68]  yolo_right  feature 7D
    [68:70]  plug_type_onehot          [is_sfp, is_sc]
    [70:77]  target_module_onehot      see _TARGET_MODULE_NAMES order

YOLO TARGET LOCK
----------------
We lock onto the FIRST detection that matches the requested plug/port and
never swap instances afterwards. YOLO is primarily a model input. During
approach it is only used as a small lateral guard when already close, then as
a bounded fine-align guard before insertion.

ACTION
------
6D delta TCP pose [dx, dy, dz, drx, dry, drz] applied directly (delta_pose
storage convention from convert_to_lerobot_v2).
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
# LeRobot ACT import workaround (bypass GROOT dataclass error on package import)
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

# Hard safety cutoffs.
_FORCE_HARD_N = 80.0
_TORQUE_HARD_NM = 12.0

# Time budget.
_TIME_LIMIT_S = 190.0
_ACT_TIMEOUT_S = 100.0
_STEP_HZ = 10.0
_STEP_S = 1.0 / _STEP_HZ
_PACE_WALL_CAP_S = 1.0

# ACT inference shaping.
_REPLAN_EVERY = 10
_EMA_ALPHA = 0.7
_MAX_TRANSLATION_DELTA_M = 0.15
_MAX_ROTATION_DELTA_RAD = 0.35

# V3 trained-state constants.
_YOLO_FRESH_S = 0.15
_YOLO_MAX_AGE_S = 10.0
_CAMERAS = ("left", "center", "right")
_ZERO_YOLO_FEATURE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _YOLO_MAX_AGE_S], dtype=np.float32)
_TARGET_MODULE_NAMES = (
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
    "sc_port_0",
    "sc_port_1",
)

# Impedance (same gains for approach and insertion — the controller wins
# alignment via low stiffness rather than aggressive position commands).
_STIFFNESS = np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()
_DAMPING = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()
_WRENCH_GAINS = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

# Insertion geometry.
_INSERT_STEP_M = 0.001
_INSERT_DEPTH_M = 0.040
_INSERT_ACTUAL_DEPTH_M = 0.025
_INSERT_TARGET_LEAD_M = 0.014
_INSERT_FORCE_THRESH = 5.0
_HOLD_AFTER_INSERT_S = 3.0

# Approach phase: when do we stop ACT and go to insertion.
_STALL_WINDOW = 30
_STALL_VEL_THRESH = 0.0003
_MIN_ACT_STEPS = 100
_ACT_NEAR_PORT_MIN_STEPS = 30
_ACT_NEAR_PORT_HOLD_STEPS = 6
_ACT_NEAR_PORT_SFP_M = 0.085
_ACT_NEAR_PORT_SC_M = 0.120
_ACT_PORT_WORSEN_M = 0.020
_ACT_PORT_WORSEN_STEPS = 4

# Small lateral assist while ACT is already near the port. This mirrors the
# clean hybrid rollback: no global homing controller, no overriding the model's
# approach direction.
_ACT_YOLO_ASSIST_MIN_STEPS = 10
_ACT_YOLO_ASSIST_MAX_PORT_DIST_M = 0.22
_ACT_YOLO_ASSIST_GAIN = 0.35
_ACT_YOLO_ASSIST_STEP_M = 0.004

# Optional fine-align step (skipped when approach already inside the gate).
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

# Pixel-space pre-insert alignment.  This is a lightweight version of the
# separate visual-servo policy: after ACT/fine-align gets close, use live
# per-camera YOLO bbox centers to learn a local XY pixel Jacobian and correct
# the final side offset before any insertion push.
_PIXEL_ALIGN_MAX_STEPS = 40
_PIXEL_ALIGN_TOL_PX = 6.0
_PIXEL_ALIGN_STABLE_FRAMES = 3
_PIXEL_ALIGN_MAX_STEP_M = 0.0015
_PIXEL_ALIGN_PROBE_M = 0.0010
_PIXEL_ALIGN_LAMBDA = 0.55
_PIXEL_ALIGN_FRESH_S = 1.0
_PIXEL_ALIGN_SETTLE_STEPS = 3
_PIXEL_ALIGN_MAX_TOTAL_M = 0.025
_PIXEL_ALIGN_CAMERA_ORDER = ("center", "left", "right")
_PIXEL_ALIGN_ENTRY_SFP_M = _ACT_NEAR_PORT_SFP_M
_PIXEL_ALIGN_ENTRY_SC_M = _ACT_NEAR_PORT_SC_M

# Insertion stall + lateral-search.
_INSERT_STALL_WINDOW = 30
_INSERT_STALL_PROGRESS_M = 0.0005
_INSERT_STALL_MIN_CMD_M = 0.020
_INSERT_SEARCH_RADIUS_M = 0.002
_INSERT_SEARCH_MAX_ATTEMPTS = 12

_INSERT_AXIS_ACTIONS = 10
_ACTION_DIR_MIN_M = 1e-5
_MIN_STATE_STD = 1e-6

_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)

_SCHEMA_V3_77D = "v3_77d_yolo_force_task"
_SCHEMA_V2_30D = "v2_30d_with_port_xyz"
_SCHEMA_LEGACY_33D = "legacy_33d_with_tcp_error"
_SCHEMA_CHUNK50_33D = _SCHEMA_LEGACY_33D


# ---------------------------------------------------------------------------
# Math helpers
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


def detect_state_schema(state_dim: int, *legacy_args, override: str = "auto") -> str:
    """Pick the observation layout for the loaded checkpoint.

    The current deployment target is V3 (77D).  Older 30D/33D layouts are
    still recognised so old checkpoints fail loudly with a useful error
    rather than silently scrambling state.
    """
    if legacy_args and isinstance(legacy_args[-1], str):
        override = legacy_args[-1]
    override = (override or "auto").lower()
    if state_dim == 77:
        return _SCHEMA_V3_77D
    if override == "true":
        return _SCHEMA_LEGACY_33D
    if override == "false":
        return _SCHEMA_V2_30D
    if state_dim == 30:
        return _SCHEMA_V2_30D
    if state_dim == 33:
        return _SCHEMA_LEGACY_33D
    raise ValueError(
        f"Unsupported observation.state dim: {state_dim}. "
        "trained_model_v3 should be 77D — verify --policy.input_features matches "
        "convert_to_lerobot_v2.STATE_DIM."
    )


def _pose_from_arrays(pos: np.ndarray, quat: np.ndarray) -> Pose:
    return Pose(
        position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation=Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3]),
        ),
    )


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

        # ---- Load checkpoint ----------------------------------------------
        with open(checkpoint_path / "config.json") as f:
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
        self.policy.load_state_dict(load_file(checkpoint_path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        # ---- Normalizer stats ---------------------------------------------
        pre_path = checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        post_path = checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        pre_stats = load_file(str(pre_path)) if pre_path.exists() else {}
        post_stats = load_file(str(post_path)) if post_path.exists() else {}

        def _get(stats, key, shape, default):
            if key in stats:
                return stats[key].to(self.device).float()
            self.get_logger().warning(f"Stat key '{key}' missing; using {default}")
            return torch.full(shape, default, device=self.device)

        self.state_mean = _get(pre_stats, "observation.state.mean", (self.state_dim,), 0.0).view(1, -1)
        self.state_std = torch.clamp(
            _get(pre_stats, "observation.state.std", (self.state_dim,), 1.0).view(1, -1),
            min=_MIN_STATE_STD,
        )
        self.action_mean = _get(post_stats, "action.mean", (self.action_dim,), 0.0).view(1, -1)
        self.action_std = _get(post_stats, "action.std", (self.action_dim,), 1.0).view(1, -1)

        # Image normalization is per-camera (B, 3, 1, 1).  We *prefer* the values
        # baked into the checkpoint preprocessor; fall back to ImageNet if any
        # camera key is missing.  trained_model_v3 was trained with
        # use_imagenet_stats=true so these should match — we verify and warn
        # if they don't.
        self._img_mean_per_cam: dict[str, torch.Tensor] = {}
        self._img_std_per_cam: dict[str, torch.Tensor] = {}
        for cam in _CAMERAS:
            m_key = f"observation.images.{cam}.mean"
            s_key = f"observation.images.{cam}.std"
            if m_key in pre_stats and s_key in pre_stats:
                m = pre_stats[m_key].to(self.device).float().view(1, 3, 1, 1)
                s = pre_stats[s_key].to(self.device).float().view(1, 3, 1, 1)
            else:
                self.get_logger().warning(
                    f"Image stats for {cam} missing in preprocessor; using ImageNet defaults"
                )
                m = _IMAGENET_MEAN.to(self.device)
                s = _IMAGENET_STD.to(self.device)
            self._img_mean_per_cam[cam] = m
            self._img_std_per_cam[cam] = s
        # Sanity: warn if cameras disagree (would imply a corrupted checkpoint).
        m0 = self._img_mean_per_cam[_CAMERAS[0]]
        s0 = self._img_std_per_cam[_CAMERAS[0]]
        for cam in _CAMERAS[1:]:
            if not torch.allclose(self._img_mean_per_cam[cam], m0, atol=1e-4) or \
               not torch.allclose(self._img_std_per_cam[cam], s0, atol=1e-4):
                self.get_logger().warning(
                    f"Image normalization stats differ across cameras (left vs {cam})"
                )
        # Backwards-compatible aliases for tests / older code paths.
        self._img_mean = m0
        self._img_std = s0
        self.schema = detect_state_schema(self.state_dim, override=self.schema_override)
        if self.delta_pose_scale < 0.0:
            # V2/V3 trained on frame-to-frame target deltas; legacy 33D ran as integrated velocity.
            self.delta_pose_scale = 1.0 if self.schema in (_SCHEMA_V3_77D, _SCHEMA_V2_30D) else _STEP_S

        n_action_steps = int(getattr(config, "n_action_steps", 10) or 10)
        chunk_size = int(getattr(config, "chunk_size", 100) or 100)
        if self.schema == _SCHEMA_V3_77D and self.replan_every == _REPLAN_EVERY:
            self.replan_every = max(1, n_action_steps)

        # ---- Per-task / per-rollout state ---------------------------------
        self._prev_action: Optional[np.ndarray] = None

        self._target_port_name = ""
        self._target_port_type = ""
        self._target_module_name = ""
        self._target_plug_type = ""
        self._is_sc_task = False

        self._yolo_lock = threading.Lock()
        self._yolo_port_xyz = np.zeros(3, dtype=np.float32)
        self._yolo_port_valid = False
        self._yolo_port_stamp_s = float("-inf")
        self._yolo_locked_instance = ""           # FIRST-SIGHT lock: instance_name
        self._yolo_locked_class = ""              # ...class_name fallback
        self._yolo_lock_announced = False
        self._cam_lock = threading.Lock()
        self._cam_last_det_time: dict[str, Optional[float]] = {cam: None for cam in _CAMERAS}
        self._cam_last_conf: dict[str, float] = {cam: 0.0 for cam in _CAMERAS}
        self._cam_last_bbox: dict[str, Optional[list[float]]] = {cam: None for cam in _CAMERAS}
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        self._wrist_force_tare_ready = False
        self._plug_type_onehot = np.zeros(2, dtype=np.float32)
        self._target_module_onehot = np.zeros(len(_TARGET_MODULE_NAMES), dtype=np.float32)
        self._locked_insert_axis: Optional[np.ndarray] = None
        self._locked_insert_axis_source = ""

        parent_node.create_subscription(
            String, "/fused_yolo/detections_json", self._cb_fused_yolo, 10
        )
        parent_node.create_subscription(
            String, "/left_camera/yolo/detections_json",
            lambda msg: self._cb_per_camera_yolo(msg, "left"), 10,
        )
        parent_node.create_subscription(
            String, "/center_camera/yolo/detections_json",
            lambda msg: self._cb_per_camera_yolo(msg, "center"), 10,
        )
        parent_node.create_subscription(
            String, "/right_camera/yolo/detections_json",
            lambda msg: self._cb_per_camera_yolo(msg, "right"), 10,
        )

        self.get_logger().info(
            "RunACT loaded:\n"
            f"  path             = {checkpoint_path}\n"
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
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint_path(self, raw_path: str) -> Path:
        raw = raw_path.strip()
        if not raw:
            raise ValueError("RunACT requires -p checkpoint_path:=/path/to/pretrained_model")

        path = Path(raw).expanduser()
        candidates = [path, path / "pretrained_model"]
        if path.is_dir():
            step_dirs = []
            for child in path.iterdir():
                if child.is_dir() and (child / "pretrained_model").is_dir():
                    step_dirs.append(child)
            step_dirs.sort(
                key=lambda p: (int(p.name) if p.name.isdigit() else -1, p.stat().st_mtime),
                reverse=True,
            )
            candidates.extend(step / "pretrained_model" for step in step_dirs)

        for candidate in candidates:
            if all((candidate / filename).exists() for filename in _REQUIRED_FILES):
                return candidate
        missing = [name for name in _REQUIRED_FILES if not (path / name).exists()]
        raise FileNotFoundError(
            f"Checkpoint path is not a valid pretrained_model directory: {path}. "
            f"Missing at top level: {missing}. Also checked nested */pretrained_model directories."
        )

    def _now(self) -> float:
        return self._parent_node.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _pad(values, length: int) -> list[float]:
        out = [float(v) for v in list(values)[:length]]
        while len(out) < length:
            out.append(0.0)
        return out

    @staticmethod
    def _norm_name(value: object) -> str:
        return str(value).strip().lower()

    @staticmethod
    def _make_plug_type_onehot(plug_type: object) -> np.ndarray:
        value = str(plug_type).strip().lower()
        if value == "sfp":
            return np.array([1.0, 0.0], dtype=np.float32)
        if value == "sc":
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    @staticmethod
    def _make_target_module_onehot(target_module_name: object) -> np.ndarray:
        value = str(target_module_name).strip().lower()
        out = np.zeros(len(_TARGET_MODULE_NAMES), dtype=np.float32)
        try:
            out[_TARGET_MODULE_NAMES.index(value)] = 1.0
        except ValueError:
            pass
        return out

    @staticmethod
    def _wrench_6d(obs_msg) -> np.ndarray:
        wrench = obs_msg.wrist_wrench.wrench
        return np.array(
            [
                wrench.force.x, wrench.force.y, wrench.force.z,
                wrench.torque.x, wrench.torque.y, wrench.torque.z,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _build_yolo_feature(
        confidence: float,
        bbox_xyxy: Optional[list[float]],
        img_h: int,
        img_w: int,
        last_det_time: Optional[float],
        now: float,
    ) -> np.ndarray:
        age = min(_YOLO_MAX_AGE_S, now - last_det_time) if last_det_time is not None else _YOLO_MAX_AGE_S
        valid = 1.0 if age < _YOLO_FRESH_S else 0.0
        if bbox_xyxy is not None and len(bbox_xyxy) == 4 and img_h > 0 and img_w > 0:
            x1, y1, x2, y2 = bbox_xyxy
            cx = float((x1 + x2) / 2.0) / float(img_w)
            cy = float((y1 + y2) / 2.0) / float(img_h)
            bw = float(x2 - x1) / float(img_w)
            bh = float(y2 - y1) / float(img_h)
        else:
            confidence = 0.0
            cx = cy = bw = bh = 0.0
        return np.array([confidence, cx, cy, bw, bh, valid, age], dtype=np.float32)

    # ------------------------------------------------------------------
    # Image / state encoding
    # ------------------------------------------------------------------

    def _ros_image_to_rgb(self, img_msg) -> np.ndarray:
        arr = np.frombuffer(img_msg.data, dtype=np.uint8)
        if arr.size < img_msg.height * img_msg.width * 3:
            raise ValueError("image buffer too small for RGB/BGR image")
        arr = arr[: img_msg.height * img_msg.width * 3].reshape(img_msg.height, img_msg.width, 3)
        encoding = str(getattr(img_msg, "encoding", "rgb8")).lower()
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr).copy()

    def _img_to_tensor(self, img_msg, cam: str | None = None) -> torch.Tensor:
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
        per_cam_mean = getattr(self, "_img_mean_per_cam", {})
        per_cam_std = getattr(self, "_img_std_per_cam", {})
        mean = per_cam_mean.get(cam, self._img_mean) if cam else self._img_mean
        std = per_cam_std.get(cam, self._img_std) if cam else self._img_std
        return (t - mean) / std

    def _build_state(self, obs_msg) -> torch.Tensor:
        cs = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js = obs_msg.joint_states

        tcp_pose = [
            tcp.position.x, tcp.position.y, tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
        ]
        tcp_vel = [
            vel.linear.x, vel.linear.y, vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
        ]
        joint_pos = self._pad(js.position, 7)
        joint_vel = self._pad(js.velocity, 7)

        if self.schema == _SCHEMA_V3_77D:
            tcp_xyz = np.array([tcp.position.x, tcp.position.y, tcp.position.z], dtype=np.float32)
            tcp_error = self._pad(getattr(cs, "tcp_error", []), 6)
            port_xyz, port_seen = self._current_port_xyz()
            if not port_seen:
                port_xyz = np.zeros(3, dtype=np.float32)
                yolo_age = _YOLO_MAX_AGE_S
                yolo_valid = 0.0
                port_delta_tcp = np.zeros(3, dtype=np.float32)
            else:
                yolo_age = min(_YOLO_MAX_AGE_S, self._current_port_age_s())
                yolo_valid = 1.0 if yolo_age < _YOLO_FRESH_S else 0.0
                port_delta_tcp = port_xyz.astype(np.float32) - tcp_xyz
            tared_wrench = self._wrench_6d(obs_msg) - self._wrist_force_tare
            yolo_left = self._camera_yolo_feature("left", obs_msg.left_image)
            yolo_center = self._camera_yolo_feature("center", obs_msg.center_image)
            yolo_right = self._camera_yolo_feature("right", obs_msg.right_image)
            raw = np.array(
                [
                    *tcp_pose,
                    *tcp_vel,
                    *tcp_error,
                    *joint_pos,
                    *joint_vel,
                    *port_xyz,
                    yolo_valid,
                    yolo_age,
                    *port_delta_tcp,
                    *tared_wrench,
                    *yolo_left,
                    *yolo_center,
                    *yolo_right,
                    *self._plug_type_onehot,
                    *self._target_module_onehot,
                ],
                dtype=np.float32,
            )
        elif self.schema == _SCHEMA_V2_30D:
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
            "observation.images.left":   self._img_to_tensor(obs_msg.left_image,   cam="left"),
            "observation.images.center": self._img_to_tensor(obs_msg.center_image, cam="center"),
            "observation.images.right":  self._img_to_tensor(obs_msg.right_image,  cam="right"),
            "observation.state":         self._build_state(obs_msg),
        }

    def _camera_yolo_feature(self, cam: str, img_msg) -> np.ndarray:
        now = time.time()
        with self._cam_lock:
            confidence = self._cam_last_conf.get(cam, 0.0)
            bbox = self._cam_last_bbox.get(cam)
            last_det_time = self._cam_last_det_time.get(cam)
        return self._build_yolo_feature(
            confidence=confidence,
            bbox_xyxy=bbox,
            img_h=int(getattr(img_msg, "height", _IMG_H) or _IMG_H),
            img_w=int(getattr(img_msg, "width", _IMG_W) or _IMG_W),
            last_det_time=last_det_time,
            now=now,
        )

    def _camera_pixel_measurement(self, cam: str, img_msg) -> Optional[dict]:
        """Return current target-port pixel error for one camera.

        Error is bbox center minus image center.  This intentionally avoids
        image processing dependencies and uses the same per-camera YOLO stream
        already feeding the 77D state.
        """
        now = time.time()
        with self._cam_lock:
            bbox = self._cam_last_bbox.get(cam)
            conf = float(self._cam_last_conf.get(cam, 0.0))
            last_det_time = self._cam_last_det_time.get(cam)
        if bbox is None or len(bbox) != 4 or last_det_time is None:
            return None
        age = now - float(last_det_time)
        if age > _PIXEL_ALIGN_FRESH_S:
            return None
        img_w = float(getattr(img_msg, "width", _IMG_W) or _IMG_W)
        img_h = float(getattr(img_msg, "height", _IMG_H) or _IMG_H)
        if img_w <= 0.0 or img_h <= 0.0:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
        desired = np.array([img_w * 0.5, img_h * 0.5], dtype=np.float64)
        error = center - desired
        return {
            "camera": cam,
            "bbox": [x1, y1, x2, y2],
            "center": center,
            "desired": desired,
            "error": error,
            "err_px": float(np.linalg.norm(error)),
            "conf": conf,
            "age": age,
        }

    def _best_pixel_measurement(self, obs_msg) -> Optional[dict]:
        for cam in _PIXEL_ALIGN_CAMERA_ORDER:
            img_msg = getattr(obs_msg, f"{cam}_image", None)
            if img_msg is None:
                continue
            meas = self._camera_pixel_measurement(cam, img_msg)
            if meas is not None:
                return meas
        return None

    @staticmethod
    def _pixel_xy_correction(jacobian_px_per_m: np.ndarray, error_px: np.ndarray) -> np.ndarray:
        jac = np.asarray(jacobian_px_per_m, dtype=np.float64).reshape(2, 2)
        err = np.asarray(error_px, dtype=np.float64).reshape(2)
        try:
            delta_xy = -_PIXEL_ALIGN_LAMBDA * np.linalg.pinv(jac, rcond=1e-3) @ err
        except np.linalg.LinAlgError:
            return np.zeros(2, dtype=np.float64)
        norm = float(np.linalg.norm(delta_xy))
        if norm > _PIXEL_ALIGN_MAX_STEP_M:
            delta_xy *= _PIXEL_ALIGN_MAX_STEP_M / norm
        return delta_xy.astype(np.float64)

    # ------------------------------------------------------------------
    # YOLO target lock — FIRST-SIGHT semantics
    # ------------------------------------------------------------------

    def _current_port_xyz(self) -> tuple[np.ndarray, bool]:
        with self._yolo_lock:
            return self._yolo_port_xyz.copy(), bool(self._yolo_port_valid)

    def _current_port_age_s(self) -> float:
        with self._yolo_lock:
            if not self._yolo_port_valid:
                return float("inf")
            stamp = float(self._yolo_port_stamp_s)
        return max(0.0, time.time() - stamp)

    def _det_match_rank(self, det: dict) -> tuple[int | None, str, str]:
        """Return (rank, instance_name, class_name) or (None,...) if no match.

        rank 0  : exact target_port_name
        rank 1  : same port_type family (sfp_port_*, sc_port_*)
        rank 2  : substring overlap with target_port

        Deliberately no target-module fallback here. The 77D model gets
        target_module_onehot separately; using module/card detections as port
        detections would poison the target-port state.
        """
        instance = self._norm_name(det.get("instance_name", ""))
        class_name = self._norm_name(det.get("class_name", ""))
        names = {n for n in (instance, class_name) if n}
        if not names:
            return None, instance, class_name

        target_port = self._target_port_name
        target_type = self._target_port_type

        exact_aliases = {target_port} if target_port else set()
        if any(n in exact_aliases for n in names):
            return 0, instance, class_name
        if target_type == "sfp" and any(n == "sfp_port" or n.startswith("sfp_port_") for n in names):
            return 1, instance, class_name
        if target_type == "sc" and any(n == "sc_port" or n.startswith("sc_port_") for n in names):
            return 1, instance, class_name
        if target_port and any(target_port in n or n in target_port for n in names):
            return 2, instance, class_name
        return None, instance, class_name

    def _cb_fused_yolo(self, msg: String) -> None:
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(dets, list):
            return

        # Pre-extract usable detections.
        scored = []
        for det in dets:
            if not isinstance(det, dict):
                continue
            rank, instance, class_name = self._det_match_rank(det)
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
            scored.append((rank, instance, class_name, conf, xyz))

        if not scored:
            return

        with self._yolo_lock:
            locked_instance = getattr(self, "_yolo_locked_instance", "")
            locked_class = getattr(self, "_yolo_locked_class", "")

        chosen: tuple[str, str, float, np.ndarray, int] | None = None  # (inst, cls, conf, xyz, rank)
        if locked_instance:
            # We already locked. Only update from the SAME instance we locked
            # onto — never let a higher-conf rival steal the goal channel.
            for rank, inst, cls, conf, xyz in scored:
                if inst and inst == locked_instance:
                    chosen = (inst, cls, conf, xyz, rank)
                    break
                if not inst and cls and cls == locked_class:
                    chosen = (inst, cls, conf, xyz, rank)
                    break
            # If the locked instance is missing in this frame, hold-last (no-op).
        else:
            # FIRST-SIGHT: lock onto the best-rank match in this frame.
            scored.sort(key=lambda s: (s[0], -s[3]))  # rank asc, conf desc
            rank, inst, cls, conf, xyz = scored[0]
            chosen = (inst, cls, conf, xyz, rank)

        if chosen is None:
            return

        inst, cls, conf, xyz, rank = chosen
        announce = False
        with self._yolo_lock:
            if not getattr(self, "_yolo_locked_instance", "") and not getattr(self, "_yolo_locked_class", ""):
                self._yolo_locked_instance = inst
                self._yolo_locked_class = cls
                announce = not getattr(self, "_yolo_lock_announced", False)
                self._yolo_lock_announced = True
            self._yolo_port_xyz = xyz
            self._yolo_port_valid = True
            self._yolo_port_stamp_s = time.time()

        if announce:
            self.get_logger().info(
                "YOLO LOCKED on first sight: "
                f"instance='{inst or '?'}' class='{cls or '?'}' "
                f"rank={rank} conf={conf:.3f} "
                f"xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})"
            )

    def _cb_per_camera_yolo(self, msg: String, cam: str) -> None:
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(dets, list) or cam not in _CAMERAS:
            return

        best_rank = None
        best_conf = float("-inf")
        best_bbox: Optional[list[float]] = None
        for det in dets:
            if not isinstance(det, dict):
                continue
            rank, _, _ = self._det_match_rank(det)
            if rank is None:
                continue
            bbox = det.get("bbox_xyxy")
            if bbox is None or len(bbox) != 4:
                continue
            try:
                conf = float(det.get("confidence", 0.0))
                bbox_f = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            if best_rank is None or rank < best_rank or (rank == best_rank and conf > best_conf):
                best_rank = rank
                best_conf = conf
                best_bbox = bbox_f

        if best_bbox is not None:
            with self._cam_lock:
                self._cam_last_det_time[cam] = time.time()
                self._cam_last_conf[cam] = best_conf
                self._cam_last_bbox[cam] = best_bbox

    def _reset_task_target(self, task: Task) -> None:
        with self._yolo_lock:
            self._target_port_name = self._norm_name(getattr(task, "port_name", ""))
            self._target_port_type = self._norm_name(getattr(task, "port_type", ""))
            self._target_module_name = self._norm_name(getattr(task, "target_module_name", ""))
            self._target_plug_type = self._norm_name(getattr(task, "plug_type", ""))
            self._yolo_port_xyz = np.zeros(3, dtype=np.float32)
            self._yolo_port_valid = False
            self._yolo_port_stamp_s = float("-inf")
            self._yolo_locked_instance = ""
            self._yolo_locked_class = ""
            self._yolo_lock_announced = False
        with self._cam_lock:
            for cam in _CAMERAS:
                self._cam_last_det_time[cam] = None
                self._cam_last_conf[cam] = 0.0
                self._cam_last_bbox[cam] = None
        self._prev_action = None
        self._is_sc_task = self._target_plug_type == "sc"
        self._plug_type_onehot = self._make_plug_type_onehot(self._target_plug_type)
        self._target_module_onehot = self._make_target_module_onehot(self._target_module_name)
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        self._wrist_force_tare_ready = False
        self._clear_insertion_axis_lock()

    def _tare_wrist_force(self, get_observation) -> None:
        if self.schema != _SCHEMA_V3_77D:
            return
        for _ in range(10):
            obs_msg = get_observation()
            if obs_msg is not None:
                self._wrist_force_tare = self._wrench_6d(obs_msg)
                self._wrist_force_tare_ready = True
                force_n = float(np.linalg.norm(self._wrist_force_tare[:3]))
                self.get_logger().info(
                    "Wrist wrench tare captured for 77D state: "
                    f"force_baseline={force_n:.1f}N "
                    f"tare={[round(float(v), 3) for v in self._wrist_force_tare.tolist()]}"
                )
                return
            time.sleep(0.05)
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        self._wrist_force_tare_ready = False
        self.get_logger().warning("Wrist wrench tare unavailable; using zero tare for 77D state")

    # ------------------------------------------------------------------
    # Action shaping & motion command
    # ------------------------------------------------------------------

    def _smooth_action(self, action_np: np.ndarray) -> np.ndarray:
        if self._prev_action is None:
            self._prev_action = action_np.copy()
            return action_np
        smoothed = self.ema_alpha * action_np + (1.0 - self.ema_alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def _apply_action_shaping(self, action_6d: np.ndarray) -> np.ndarray:
        action = np.nan_to_num(
            action_6d.astype(np.float64) * self.action_scale,
            nan=0.0, posinf=0.0, neginf=0.0,
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

    def _make_motion(self, target_pose: Pose) -> MotionUpdate:
        mu = MotionUpdate()
        mu.header = Header(
            frame_id="base_link",
            stamp=self._parent_node.get_clock().now().to_msg(),
        )
        mu.pose = target_pose
        mu.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION
        )
        mu.target_stiffness = _STIFFNESS
        mu.target_damping = _DAMPING
        mu.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        mu.wrench_feedback_gains_at_tip = list(_WRENCH_GAINS)
        return mu

    def _send_pose(self, move_robot, target_pose: Pose) -> None:
        move_robot(motion_update=self._make_motion(target_pose))

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

        return _pose_from_arrays(new_pos, new_quat)

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

    # ------------------------------------------------------------------
    # Insertion axis lock
    # ------------------------------------------------------------------

    def _near_port_threshold_m(self) -> float:
        is_sc = self._is_sc_task or self._target_plug_type == "sc" or self._target_port_type == "sc"
        return _ACT_NEAR_PORT_SC_M if is_sc else _ACT_NEAR_PORT_SFP_M

    def _axis_lock_port_dist_m(self) -> float:
        return max(0.120, self._near_port_threshold_m() * 1.5)

    def _insert_lateral_threshold_m(self) -> float:
        is_sc = self._is_sc_task or self._target_plug_type == "sc" or self._target_port_type == "sc"
        return _YOLO_INSERT_LATERAL_MAX_SC_M if is_sc else _YOLO_INSERT_LATERAL_MAX_SFP_M

    def _pixel_align_entry_threshold_m(self) -> float:
        is_sc = self._is_sc_task or self._target_plug_type == "sc" or self._target_port_type == "sc"
        return _PIXEL_ALIGN_ENTRY_SC_M if is_sc else _PIXEL_ALIGN_ENTRY_SFP_M

    def _port_distance_to_tcp_m(self, tcp_pos: np.ndarray) -> Optional[float]:
        port_xyz, port_valid = self._current_port_xyz()
        if not port_valid:
            return None
        return float(np.linalg.norm(port_xyz.astype(np.float64) - np.asarray(tcp_pos, dtype=np.float64)))

    def _top_of_port_reached(self, tcp_pos: np.ndarray) -> tuple[bool, Optional[float], float]:
        port_dist = self._port_distance_to_tcp_m(tcp_pos)
        threshold = self._pixel_align_entry_threshold_m()
        return port_dist is not None and port_dist <= threshold, port_dist, threshold

    def _pick_insertion_axis(self, init_quat: np.ndarray, recent_actions) -> tuple[np.ndarray, str]:
        if recent_actions and len(recent_actions) >= 5:
            actions_arr = np.array(list(recent_actions)[-_INSERT_AXIS_ACTIONS:])
            avg_delta_pos = actions_arr[:, :3].mean(axis=0)
            dir_norm = float(np.linalg.norm(avg_delta_pos))
            if dir_norm > _ACTION_DIR_MIN_M:
                return avg_delta_pos / dir_norm, f"last_{len(actions_arr)}_model_actions"
        return _quat_to_rotation_matrix(init_quat)[:, 2], "gripper_z"

    def _clear_insertion_axis_lock(self) -> None:
        self._locked_insert_axis = None
        self._locked_insert_axis_source = ""

    def _locked_or_pick_insertion_axis(
        self,
        init_quat: np.ndarray,
        recent_actions,
        *,
        lock: bool = False,
        reason: str = "",
    ) -> tuple[np.ndarray, str]:
        if self._locked_insert_axis is not None:
            return self._locked_insert_axis.copy(), f"locked_{self._locked_insert_axis_source}"

        axis, source = self._pick_insertion_axis(init_quat, recent_actions)
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            axis = _quat_to_rotation_matrix(init_quat)[:, 2]
            axis_norm = float(np.linalg.norm(axis))
            source = "gripper_z"
        axis = axis / max(axis_norm, 1e-9)

        if lock:
            self._locked_insert_axis = axis.copy()
            self._locked_insert_axis_source = source
            suffix = f" reason={reason}" if reason else ""
            self.get_logger().info(
                "Insertion axis locked: "
                f"source={source} dir=({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f}){suffix}"
            )
            return axis.copy(), f"locked_{source}"
        return axis, source

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
        pattern = (u, -u, v, -v, u + v, u - v, -u + v, -u - v)
        direction = np.asarray(pattern[attempt % len(pattern)], dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        radius = _INSERT_SEARCH_RADIUS_M * (1 + attempt // len(pattern))
        return direction * min(radius, _INSERT_SEARCH_RADIUS_M * 2.0)

    # ------------------------------------------------------------------
    # Phase 1: ACT approach with small near-port lateral assist.
    # ------------------------------------------------------------------

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

    def _run_act_phase(self, get_observation, move_robot, send_feedback, start_time):
        self.policy.reset()
        self._prev_action = None
        step_count = 0
        start_pos: Optional[np.ndarray] = None
        pos_history: deque = deque(maxlen=_STALL_WINDOW)
        recent_actions: deque = deque(maxlen=30)
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
                target_pose, pos, quat, recent_actions, step_count,
            )
            self._send_pose(move_robot, target_pose)

            port_xyz, port_valid = self._current_port_xyz()
            port_d: Optional[float] = None
            lateral_norm: Optional[float] = None
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
                            "ACT phase: near-port latch — switching to insertion "
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
                            "ACT phase: port distance worsening after near pass — "
                            f"switching to insertion best={best_port_dist*1000:.1f}mm "
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
                send_feedback(
                    f"ACT step={step_count} t={self._now() - start_time:.0f}s F={force_n:.1f}N "
                    f"tcp=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                )

            self._pace_to_step(t0_sim)

        obs_msg = get_observation()
        if obs_msg is None:
            return None, None, None, recent_actions, step_count
        pos, quat, _, _ = self._get_tcp_state(obs_msg)
        return obs_msg, pos, quat, recent_actions, step_count

    # ------------------------------------------------------------------
    # Phase 1.5: optional bounded YOLO fine-align
    # ------------------------------------------------------------------

    def _run_yolo_fine_align(
        self,
        get_observation,
        move_robot,
        start_time,
        init_pos: np.ndarray,
        init_quat: np.ndarray,
        recent_actions,
    ) -> tuple[object | None, np.ndarray, np.ndarray, np.ndarray, float | None]:
        insert_dir, axis_source = self._pick_insertion_axis(init_quat, recent_actions)
        port_xyz, port_valid = self._current_port_xyz()
        if not port_valid:
            self.get_logger().info(
                f"YOLO fine align skipped: no held target; using {axis_source} insertion axis"
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
        last_lateral_norm: Optional[float] = None

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
                self.get_logger().error(f"FINE HARD STOP — F={force_n:.1f}N T={torque_n:.2f}Nm")
                break

            port_xyz, port_valid = self._current_port_xyz()
            if not port_valid:
                break
            port_dist = float(np.linalg.norm(port_xyz.astype(np.float64) - pos))
            if port_dist > _YOLO_FINE_MAX_PORT_DIST_M:
                self.get_logger().info(f"YOLO fine align skipped: port_d={port_dist*1000:.1f}mm too large")
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
                    f"YOLO fine stopped: cap reached ({total_correction*1000:.1f}mm), "
                    f"lateral={lateral_norm*1000:.1f}mm"
                )
                break
            if no_improve_steps >= _YOLO_FINE_NO_IMPROVE_STEPS:
                self.get_logger().info(
                    f"YOLO fine stopped: not improving (best={best_lateral*1000:.1f}mm, "
                    f"now={lateral_norm*1000:.1f}mm)"
                )
                break

            correction = lateral * _YOLO_FINE_GAIN
            corr_norm = float(np.linalg.norm(correction))
            if corr_norm > _YOLO_FINE_STEP_M:
                correction *= _YOLO_FINE_STEP_M / corr_norm
                corr_norm = _YOLO_FINE_STEP_M
            total_correction += corr_norm
            target_pos = pos + correction

            self._send_pose(move_robot, _pose_from_arrays(target_pos, quat))

            if step < 3 or step % 5 == 0:
                self.get_logger().info(
                    f"YOLO_FINE step={step:2d} | lateral={lateral_norm*1000:.1f}mm | "
                    f"corr={corr_norm*1000:.1f}mm | total={total_correction*1000:.1f}mm | "
                    f"port_d={port_dist*1000:.1f}mm"
                )

            self._pace_to_step(t0_sim)

        return last_obs, last_pos, last_quat, insert_dir, last_lateral_norm

    # ------------------------------------------------------------------
    # Phase 1.6: pixel-space YOLO alignment before insertion
    # ------------------------------------------------------------------

    def _send_relative_xy(
        self,
        get_observation,
        move_robot,
        dx: float,
        dy: float,
        start_time: float,
    ) -> tuple[object | None, np.ndarray | None, np.ndarray | None]:
        obs_msg = get_observation()
        if obs_msg is None:
            return None, None, None
        pos, quat, force_n, torque_n = self._get_tcp_state(obs_msg)
        if force_n > self.force_hard_stop_n or torque_n > self.torque_hard_stop_nm:
            self.get_logger().error(f"PIXEL ALIGN HARD STOP — F={force_n:.1f}N T={torque_n:.2f}Nm")
            return None, pos, quat

        target_pos = pos + np.array([float(dx), float(dy), 0.0], dtype=np.float64)
        target_pose = _pose_from_arrays(target_pos, quat)
        for _ in range(_PIXEL_ALIGN_SETTLE_STEPS):
            if self._now() - start_time > self.time_limit_s - 10.0:
                break
            self._send_pose(move_robot, target_pose)
            time.sleep(_STEP_S)

        obs_msg = get_observation()
        if obs_msg is None:
            return None, target_pos, quat
        pos, quat, _, _ = self._get_tcp_state(obs_msg)
        return obs_msg, pos, quat

    def _estimate_pixel_jacobian(
        self,
        get_observation,
        move_robot,
        camera: str,
        start_time: float,
        base_error: np.ndarray,
    ) -> Optional[np.ndarray]:
        cols = []
        for axis_idx, (dx, dy) in enumerate((
            (_PIXEL_ALIGN_PROBE_M, 0.0),
            (0.0, _PIXEL_ALIGN_PROBE_M),
        )):
            obs_probe, _, _ = self._send_relative_xy(
                get_observation, move_robot, dx, dy, start_time
            )
            if obs_probe is None:
                return None
            img_msg = getattr(obs_probe, f"{camera}_image", None)
            meas_probe = self._camera_pixel_measurement(camera, img_msg)
            # Return to the pre-probe XY before trying the next axis.
            self._send_relative_xy(get_observation, move_robot, -dx, -dy, start_time)
            if meas_probe is None:
                return None
            col = (meas_probe["error"] - base_error) / _PIXEL_ALIGN_PROBE_M
            if float(np.linalg.norm(col)) < 1e-6:
                return None
            cols.append(col)
            self.get_logger().info(
                "PIXEL_ALIGN probe "
                f"axis={axis_idx} d=({dx*1000:.1f},{dy*1000:.1f})mm "
                f"de=({col[0]:+.1f},{col[1]:+.1f})px/m"
            )

        jacobian = np.column_stack(cols)
        if not np.all(np.isfinite(jacobian)):
            return None
        return jacobian

    def _run_pixel_yolo_align(
        self,
        get_observation,
        move_robot,
        send_feedback,
        start_time: float,
        init_pos: np.ndarray,
        init_quat: np.ndarray,
    ) -> tuple[bool, np.ndarray, np.ndarray, float | None]:
        obs_msg = get_observation()
        if obs_msg is None:
            return False, init_pos, init_quat, None

        meas = self._best_pixel_measurement(obs_msg)
        if meas is None:
            self.get_logger().info("PIXEL_ALIGN required but no fresh per-camera port bbox is available")
            send_feedback("PIXEL_ALIGN unavailable reason=no_fresh_port_bbox")
            return False, init_pos, init_quat, None

        camera = str(meas["camera"])
        jacobian: Optional[np.ndarray] = None
        stable_frames = 0
        total_xy = 0.0
        last_err_px: Optional[float] = None
        pos = init_pos.copy()
        quat = init_quat.copy()

        self.get_logger().info(
            "=== PHASE 1.6: YOLO pixel alignment === "
            f"camera={camera} err={meas['err_px']:.1f}px "
            f"center=({meas['center'][0]:.1f},{meas['center'][1]:.1f}) "
            f"target=({meas['desired'][0]:.1f},{meas['desired'][1]:.1f})"
        )
        send_feedback(f"PIXEL_ALIGN start camera={camera} err={meas['err_px']:.1f}px")

        for step in range(_PIXEL_ALIGN_MAX_STEPS):
            if self._now() - start_time > self.time_limit_s - 10.0:
                break

            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(_STEP_S)
                continue
            current = self._camera_pixel_measurement(camera, getattr(obs_msg, f"{camera}_image", None))
            if current is None:
                current = self._best_pixel_measurement(obs_msg)
                if current is None:
                    self.get_logger().info("PIXEL_ALIGN stopped: lost per-camera port bbox")
                    return False, pos, quat, last_err_px
                camera = str(current["camera"])
                jacobian = None

            err_px = float(current["err_px"])
            last_err_px = err_px
            if err_px <= _PIXEL_ALIGN_TOL_PX:
                stable_frames += 1
                if stable_frames >= _PIXEL_ALIGN_STABLE_FRAMES:
                    pos, quat, _, _ = self._get_tcp_state(obs_msg)
                    self.get_logger().info(
                        f"PIXEL_ALIGN complete step={step} err={err_px:.1f}px "
                        f"stable={stable_frames}"
                    )
                    send_feedback(f"PIXEL_ALIGN complete err={err_px:.1f}px")
                    return True, pos, quat, err_px
            else:
                stable_frames = 0

            if jacobian is None:
                jacobian = self._estimate_pixel_jacobian(
                    get_observation, move_robot, camera, start_time, current["error"]
                )
                if jacobian is None:
                    self.get_logger().info("PIXEL_ALIGN stopped: could not estimate local pixel Jacobian")
                    return False, pos, quat, err_px
                continue

            delta_xy = self._pixel_xy_correction(jacobian, current["error"])
            step_norm = float(np.linalg.norm(delta_xy))
            if step_norm < 1e-6:
                self.get_logger().info(f"PIXEL_ALIGN stopped: tiny correction err={err_px:.1f}px")
                break
            if total_xy + step_norm > _PIXEL_ALIGN_MAX_TOTAL_M:
                self.get_logger().info(
                    f"PIXEL_ALIGN stopped: total XY cap reached "
                    f"total={(total_xy + step_norm)*1000:.1f}mm err={err_px:.1f}px"
                )
                break

            before_err = err_px
            obs_after, pos_after, quat_after = self._send_relative_xy(
                get_observation, move_robot, float(delta_xy[0]), float(delta_xy[1]), start_time
            )
            if obs_after is None or pos_after is None or quat_after is None:
                return False, pos, quat, last_err_px
            pos, quat = pos_after, quat_after
            total_xy += step_norm

            after = self._camera_pixel_measurement(camera, getattr(obs_after, f"{camera}_image", None))
            after_err = float(after["err_px"]) if after is not None else float("inf")
            if after_err > before_err + 2.0:
                # The local mapping changed or the sign was bad. Refit once
                # instead of repeatedly walking in the wrong direction.
                jacobian = None

            if step < 3 or step % 5 == 0:
                self.get_logger().info(
                    "PIXEL_ALIGN "
                    f"step={step} camera={camera} err={before_err:.1f}->{after_err:.1f}px "
                    f"dxy=({delta_xy[0]*1000:+.2f},{delta_xy[1]*1000:+.2f})mm "
                    f"total={total_xy*1000:.1f}mm"
                )
                send_feedback(
                    f"PIXEL_ALIGN step={step} err={before_err:.1f}->{after_err:.1f}px "
                    f"dxy=({delta_xy[0]*1000:+.2f},{delta_xy[1]*1000:+.2f})mm"
                )

        return False, pos, quat, last_err_px

    # ------------------------------------------------------------------
    # Phase 2: force-guided insertion
    # ------------------------------------------------------------------

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
        last_target_pose: Optional[Pose] = None
        commanded_push = 0.0
        insert_steps = 0
        contact_detected = False
        budget_logged = False
        progress_history: deque = deque(maxlen=_INSERT_STALL_WINDOW)
        best_actual_progress = 0.0
        search_attempts = 0
        search_exhausted_logged = False
        search_offset = np.zeros(3, dtype=np.float64)
        search_u, search_v = self._perpendicular_search_basis(insert_dir)
        progress_reached_but_unaligned = False

        # Use the model's average rotation hint to keep orientation consistent
        # with the trained behaviour during the push.
        avg_rot_delta = np.zeros(3, dtype=np.float64)
        if recent_actions and len(recent_actions) >= 5:
            actions_arr = np.array(list(recent_actions))
            avg_rot_delta = actions_arr[:, 3:6].mean(axis=0) * 0.3

        self.get_logger().info(
            f"Insertion direction: ({insert_dir[0]:+.3f},{insert_dir[1]:+.3f},{insert_dir[2]:+.3f})"
        )

        while self._now() - start_time < self.time_limit_s:
            t0_sim = self._now()
            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(_STEP_S)
                continue

            pos, quat, force_n, torque_n = self._get_tcp_state(obs_msg)
            if force_n > self.force_hard_stop_n or torque_n > self.torque_hard_stop_nm:
                self.get_logger().error(f"INSERT HARD STOP — F={force_n:.1f}N T={torque_n:.2f}Nm")
                break

            actual_progress = max(0.0, float(np.dot(pos - init_pos, insert_dir)))
            progress_history.append(actual_progress)

            if force_n > self.insert_force_thresh_n and not contact_detected:
                contact_detected = True
                self.get_logger().info(f"Contact detected at F={force_n:.1f}N — continuing")

            if actual_progress >= self.insert_actual_depth_m:
                # Verify lateral alignment hasn't drifted before declaring success.
                port_xyz, port_valid = self._current_port_xyz()
                lateral_norm = None
                if port_valid:
                    lateral_norm = float(
                        np.linalg.norm(self._lateral_error_to_port(pos, port_xyz, insert_dir))
                    )
                if lateral_norm is None or lateral_norm <= self._insert_lateral_threshold_m():
                    self.get_logger().info(
                        f"Actual insertion progress reached: {actual_progress*1000:.1f}mm "
                        f"(commanded={commanded_push*1000:.1f}mm)"
                    )
                    break
                if not progress_reached_but_unaligned:
                    progress_reached_but_unaligned = True
                    self.get_logger().info(
                        "Insertion progress reached but YOLO lateral is still high; "
                        "continuing instead of returning success "
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
                                f"actual={actual_progress*1000:.1f}mm "
                                f"best={best_actual_progress*1000:.1f}mm "
                                f"cmd={commanded_push*1000:.1f}mm"
                            )
                        progress_history.clear()
                        search_offset[:] = 0.0
                        continue
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
                    f"Commanded budget reached: {commanded_push*1000:.1f}mm; "
                    f"holding until actual reaches {self.insert_actual_depth_m*1000:.1f}mm"
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

            target_pose = _pose_from_arrays(target_pos, new_quat)
            last_target_pose = target_pose
            self._send_pose(move_robot, target_pose)

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

        # Hold to let the connector seat.
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
                self._send_pose(move_robot, last_target_pose)
            elif obs_msg is not None:
                pos, quat, _, _ = self._get_tcp_state(obs_msg)
                self._send_pose(move_robot, _pose_from_arrays(pos, quat))
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
        self._tare_wrist_force(get_observation)
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
                    get_observation, move_robot, start, pos, quat, recent_actions
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
            hold_pose = _pose_from_arrays(pos, quat)
            while self._now() - start < self.time_limit_s - 1.0:
                self._send_pose(move_robot, hold_pose)
                time.sleep(_STEP_S)

        elapsed = self._now() - start
        self.get_logger().info(
            f"RunACT clean hybrid finished — ACT:{act_steps_total} + INSERT:{insert_steps} "
            f"in {elapsed:.1f}s"
        )
        return True


# aic_model resolves the class by the last component of the module path.
run_act = RunACT
