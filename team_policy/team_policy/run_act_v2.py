"""
Deploy an ACT policy trained with the v2 63D state (Schema v6).

State layout at inference (must exactly match training in convert_to_lerobot_v2.py):
  [0:7]   tcp_pose        7D  x y z qx qy qz qw
  [7:13]  tcp_velocity    6D  vx vy vz wx wy wz
  [13:19] tcp_error       6D  controller tracking error (from controller_state)
  [19:26] joint_positions 7D
  [26:33] joint_velocity  7D
  [33:36] port_xyz        3D  fused YOLO xyz in base_link (hold-last, zeros before first)
  [36:42] wrist_force     6D  fx fy fz tx ty tz
  [42:49] yolo_left       7D  conf cx_norm cy_norm w_norm h_norm valid age
  [49:56] yolo_center     7D
  [56:63] yolo_right      7D

Inherits all motion control logic from RunACT (Phase 1 ACT, Phase 1.5 YOLO align,
Phase 2 force-guided insertion).  Only _build_state() and the YOLO subscriptions
are overridden.

Usage:
    ros2 run aic_model aic_model --ros-args \\
        -p policy:=team_policy.run_act_v2 \\
        -p checkpoint_path:=/path/to/pretrained_model_57d
"""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, Optional

import numpy as np
import torch
from std_msgs.msg import String

from team_policy.run_act import RunACT  # type: ignore[import-unresolved]

# Schema constant for 57D
_SCHEMA_V6_57D = "v6_57d_per_cam_yolo_force"

_CAMERAS = ("left", "center", "right")
_CAM_TOPICS: Dict[str, str] = {
    "left":   "/left_camera/yolo/detections_json",
    "center": "/center_camera/yolo/detections_json",
    "right":  "/right_camera/yolo/detections_json",
}

# Per-camera YOLO feature constants (must match episode_recorder_v2.py)
_AGE_VALID_S = 0.15
_MAX_AGE_S   = 10.0


class RunACTV2(RunACT):
    """ACT policy runner for the 57D state schema (v6 per-camera YOLO + force/torque)."""

    def __init__(self, parent_node):
        super().__init__(parent_node)

        # Validate that the loaded checkpoint is actually 63D
        if self.state_dim != 63:
            raise ValueError(
                f"RunACTV2 expects a 63D checkpoint but got state_dim={self.state_dim}. "
                "Use run_act.py for 30D checkpoints."
            )

        # Override schema
        self.schema = _SCHEMA_V6_57D

        # Per-camera YOLO state (lock-protected)
        self._cam_lock: threading.Lock = threading.Lock()
        self._cam_last_det_time: Dict[str, Optional[float]] = {
            cam: None for cam in _CAMERAS
        }
        self._cam_last_conf: Dict[str, float] = {cam: 0.0 for cam in _CAMERAS}
        self._cam_last_bbox: Dict[str, Optional[list]] = {cam: None for cam in _CAMERAS}

        # Image dimensions for bbox normalisation.
        # Use the _IMG_H/_IMG_W used during training (480×640 after resize in converter).
        # At inference the images are resized to the same size by _img_to_tensor().
        from team_policy.run_act import _IMG_H, _IMG_W  # type: ignore[import-unresolved]
        self._inf_img_h = _IMG_H   # 480
        self._inf_img_w = _IMG_W   # 640

        for cam in _CAMERAS:
            parent_node.create_subscription(
                String,
                _CAM_TOPICS[cam],
                lambda msg, c=cam: self._cb_per_cam_yolo(msg, c),
                10,
            )

        self.get_logger().info(
            f"RunACTV2 ready — state_dim=63, schema={_SCHEMA_V6_57D}"
        )

    # ------------------------------------------------------------------
    # Per-camera YOLO callbacks
    # ------------------------------------------------------------------

    def _cb_per_cam_yolo(self, msg: String, cam: str) -> None:
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(dets, list):
            return

        best_rank = None
        best_conf = float("-inf")
        best_bbox: Optional[list] = None

        for det in dets:
            if not isinstance(det, dict):
                continue
            rank = self._target_match_rank(det)
            if rank is None:
                continue
            conf = float(det.get("confidence", 0.0))
            bbox = det.get("bbox_xyxy")
            if bbox is None or len(bbox) != 4:
                continue
            if best_rank is None or rank < best_rank or (rank == best_rank and conf > best_conf):
                best_rank = rank
                best_conf = conf
                best_bbox = [float(v) for v in bbox]

        if best_bbox is not None:
            with self._cam_lock:
                self._cam_last_det_time[cam] = time.time()
                self._cam_last_conf[cam]     = best_conf
                self._cam_last_bbox[cam]     = best_bbox

    def _reset_per_cam_state(self) -> None:
        with self._cam_lock:
            for cam in _CAMERAS:
                self._cam_last_det_time[cam] = None
                self._cam_last_conf[cam]     = 0.0
                self._cam_last_bbox[cam]     = None

    # ------------------------------------------------------------------
    # Per-camera feature vector
    # ------------------------------------------------------------------

    def _build_cam_feature(self, cam: str, now: float) -> np.ndarray:
        """Return 7D [conf, cx, cy, w, h, valid, age] for one camera at inference."""
        with self._cam_lock:
            last_time = self._cam_last_det_time[cam]
            conf      = self._cam_last_conf[cam]
            bbox      = self._cam_last_bbox[cam]

        age   = min(_MAX_AGE_S, now - last_time) if last_time is not None else _MAX_AGE_S
        valid = 1.0 if age < _AGE_VALID_S else 0.0

        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx = float((x1 + x2) / 2) / self._inf_img_w
            cy = float((y1 + y2) / 2) / self._inf_img_h
            bw = float(x2 - x1) / self._inf_img_w
            bh = float(y2 - y1) / self._inf_img_h
        else:
            cx = cy = bw = bh = 0.0
            conf  = 0.0

        return np.array([conf, cx, cy, bw, bh, valid, age], dtype=np.float32)

    # ------------------------------------------------------------------
    # Override _build_state → 57D
    # ------------------------------------------------------------------

    def _build_state(self, obs_msg) -> torch.Tensor:
        cs  = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js  = obs_msg.joint_states
        w   = obs_msg.wrist_wrench.wrench

        tcp_pose = [
            tcp.position.x, tcp.position.y, tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
        ]
        tcp_vel = [
            vel.linear.x, vel.linear.y, vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
        ]
        tcp_error = self._pad(getattr(cs, "tcp_error", []), 6)
        joint_pos = self._pad(js.position, 7)
        joint_vel = self._pad(js.velocity, 7)

        # Fused YOLO port_xyz (hold-last, zeros before first detection)
        port_xyz, _ = self._current_port_xyz()

        # Force/torque (6D)
        wrist_force = [
            w.force.x,  w.force.y,  w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ]

        # Per-camera YOLO features (7D × 3 cameras)
        now = time.time()
        feat_left   = self._build_cam_feature("left",   now)
        feat_center = self._build_cam_feature("center", now)
        feat_right  = self._build_cam_feature("right",  now)

        raw = np.array([
            *tcp_pose, *tcp_vel, *tcp_error, *joint_pos, *joint_vel, *port_xyz,
            *wrist_force,
            *feat_left, *feat_center, *feat_right,
        ], dtype=np.float32)

        if raw.shape[0] != self.state_dim:
            raise ValueError(
                f"RunACTV2 built state dim {raw.shape[0]}, "
                f"checkpoint expects {self.state_dim}"
            )
        t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        return (t - self.state_mean) / self.state_std

    # ------------------------------------------------------------------
    # Reset per-episode state
    # ------------------------------------------------------------------

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        # Reset per-camera state at the start of each trial
        self._reset_per_cam_state()
        return super().insert_cable(task, get_observation, move_robot, send_feedback)


# aic_model entry point
# Usage: -p policy:=team_policy.run_act_v2
run_act_v2 = RunACTV2


def main():
    raise SystemExit(
        "RunACTV2 is not a standalone node.\n"
        "Run via aic_model:\n"
        "  ros2 run aic_model aic_model --ros-args \\\n"
        "    -p policy:=team_policy.run_act_v2 \\\n"
        "    -p checkpoint_path:=/path/to/pretrained_model_57d"
    )
