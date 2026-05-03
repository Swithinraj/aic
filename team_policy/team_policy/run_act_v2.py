"""
Deploy an ACT policy trained with the v2 63D, 68D, 75D, or 77D state.

77D state layout at inference (must exactly match training in convert_to_lerobot_v2.py):
  [0:7]   tcp_pose        7D  x y z qx qy qz qw
  [7:13]  tcp_velocity    6D  vx vy vz wx wy wz
  [13:19] tcp_error       6D  controller tracking error (from controller_state)
  [19:26] joint_positions 7D
  [26:33] joint_velocity  7D
  [33:36] yolo_port_xyz   3D  fused YOLO xyz in base_link (hold-last, zeros before first)
  [36:37] yolo_valid      1D  fresh detection flag, not hold-last existence
  [37:38] yolo_age        1D  seconds since last valid target detection
  [38:41] port_delta_tcp  3D  yolo_port_xyz - tcp position in base_link
  [41:47] tared_wrist_force_torque 6D  tare-subtracted fx fy fz tx ty tz
  [47:54] yolo_left       7D  conf cx_norm cy_norm w_norm h_norm valid age
  [54:61] yolo_center     7D
  [61:68] yolo_right      7D
  [68:70] plug_type_onehot 2D  [is_sfp, is_sc]
  [70:77] target_module_onehot 7D exact target module identity

Legacy 63D checkpoints are still supported; they use the older state layout
without port_delta_tcp and plug_type_onehot, and they keep raw wrist_force.

Inherits all motion control logic from RunACT (Phase 1 ACT, Phase 1.5 YOLO align,
Phase 2 force-guided insertion).  Only _build_state() and the YOLO subscriptions
are overridden.

Usage:
    ros2 run aic_model aic_model --ros-args \\
        -p policy:=team_policy.run_act_v2 \\
        -p checkpoint_path:=/path/to/pretrained_model_77d
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
from team_policy.training_robot.episode_recorder_v2 import (
    TARGET_MODULE_NAMES,
    build_yolo_feature,
    build_plug_type_onehot,
    build_target_module_onehot,
)

# Schema constants
_SCHEMA_V6_63D = "v6_63d_per_cam_yolo_force"
_SCHEMA_V7_68D = "v7_68d_port_delta_tared_force_plug_type"
_SCHEMA_V8_75D = "v8_75d_target_module_conditioned"
_SCHEMA_V9_77D = "v9_77d_fresh_fused_yolo"

_CAMERAS = ("left", "center", "right")
_CAM_TOPICS: Dict[str, str] = {
    "left":   "/left_camera/yolo/detections_json",
    "center": "/center_camera/yolo/detections_json",
    "right":  "/right_camera/yolo/detections_json",
}

class RunACTV2(RunACT):
    """ACT policy runner for the 63D, 68D, 75D, or 77D V2 state schema."""

    def __init__(self, parent_node):
        super().__init__(parent_node)

        # Accept legacy 63D, synced 68D, target-conditioned 75D, and fresh-fused 77D checkpoints.
        if self.state_dim not in {63, 68, 75, 77}:
            raise ValueError(
                f"RunACTV2 expects a 63D, 68D, 75D, or 77D checkpoint but got state_dim={self.state_dim}. "
                "Use run_act.py for 30D checkpoints."
            )
        self._use_extended_state = self.state_dim >= 68
        self._use_target_module_state = self.state_dim >= 75
        self._use_fused_freshness_state = self.state_dim == 77

        # Override schema
        if self._use_fused_freshness_state:
            self.schema = _SCHEMA_V9_77D
        elif self._use_target_module_state:
            self.schema = _SCHEMA_V8_75D
        elif self._use_extended_state:
            self.schema = _SCHEMA_V7_68D
        else:
            self.schema = _SCHEMA_V6_63D

        # Per-camera YOLO state (lock-protected)
        self._cam_lock: threading.Lock = threading.Lock()
        self._cam_last_det_time: Dict[str, Optional[float]] = {
            cam: None for cam in _CAMERAS
        }
        self._cam_last_conf: Dict[str, float] = {cam: 0.0 for cam in _CAMERAS}
        self._cam_last_bbox: Dict[str, Optional[list]] = {cam: None for cam in _CAMERAS}

        # Cache the live observation image size per camera.
        # Per-camera YOLO boxes arrive in the detector's image pixel space, so the
        # online feature builder must normalize with the same live image size used
        # during collection rather than the resized video export resolution.
        self._cam_last_img_hw: Dict[str, tuple[int, int]] = {
            cam: (0, 0) for cam in _CAMERAS
        }

        # Last-resort fallback only; normal execution should use the observation dims.
        from team_policy.run_act import _IMG_H, _IMG_W  # type: ignore[import-unresolved]
        self._fallback_img_h = _IMG_H
        self._fallback_img_w = _IMG_W
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        self._plug_type_onehot = np.zeros(2, dtype=np.float32)
        self._target_module_onehot = np.zeros(len(TARGET_MODULE_NAMES), dtype=np.float32)

        for cam in _CAMERAS:
            parent_node.create_subscription(
                String,
                _CAM_TOPICS[cam],
                lambda msg, c=cam: self._cb_per_cam_yolo(msg, c),
                10,
            )

        self.get_logger().info(
            f"RunACTV2 ready — state_dim={self.state_dim}, schema={self.schema}"
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
                self._cam_last_img_hw[cam]   = (0, 0)

    # ------------------------------------------------------------------
    # Per-camera feature vector
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_image_hw(obs_msg, cam: str) -> tuple[int, int]:
        image_msg = getattr(obs_msg, f"{cam}_image", None)
        if image_msg is None:
            return 0, 0
        return int(getattr(image_msg, "height", 0)), int(getattr(image_msg, "width", 0))

    def _build_cam_feature(self, cam: str, now: float, img_h: int, img_w: int) -> np.ndarray:
        """Return 7D [conf, cx, cy, w, h, valid, age] for one camera at inference."""
        with self._cam_lock:
            last_time = self._cam_last_det_time[cam]
            conf      = self._cam_last_conf[cam]
            bbox      = self._cam_last_bbox[cam]
            cached_h, cached_w = self._cam_last_img_hw[cam]

        use_h = img_h if img_h > 0 else cached_h if cached_h > 0 else self._fallback_img_h
        use_w = img_w if img_w > 0 else cached_w if cached_w > 0 else self._fallback_img_w

        return build_yolo_feature(
            confidence=conf,
            bbox_xyxy=bbox,
            img_h=use_h,
            img_w=use_w,
            last_det_time=last_time,
            now=now,
        )

    # ------------------------------------------------------------------
    # Override _build_state → 63D, 68D, 75D, or 77D depending on checkpoint
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
        port_xyz, port_valid, port_age, port_seen = self._current_port_state()
        tcp_xyz = np.array(tcp_pose[:3], dtype=np.float32)

        # Force/torque (6D)
        wrist_force = np.array([
            w.force.x,  w.force.y,  w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ], dtype=np.float32)
        tared_wrist_force = wrist_force - self._wrist_force_tare
        port_delta_tcp = (
            port_xyz.astype(np.float32) - tcp_xyz
            if port_seen else np.zeros(3, dtype=np.float32)
        )

        # Per-camera YOLO features (7D × 3 cameras)
        now = time.time()
        image_hw = {cam: self._obs_image_hw(obs_msg, cam) for cam in _CAMERAS}
        with self._cam_lock:
            for cam, (img_h, img_w) in image_hw.items():
                if img_h > 0 and img_w > 0:
                    self._cam_last_img_hw[cam] = (img_h, img_w)
        feat_left   = self._build_cam_feature("left",   now, *image_hw["left"])
        feat_center = self._build_cam_feature("center", now, *image_hw["center"])
        feat_right  = self._build_cam_feature("right",  now, *image_hw["right"])

        if self.state_dim == 77:
            raw = np.array([
                *tcp_pose, *tcp_vel, *tcp_error, *joint_pos, *joint_vel, *port_xyz,
                float(port_valid), float(port_age),
                *port_delta_tcp, *tared_wrist_force,
                *feat_left, *feat_center, *feat_right,
                *self._plug_type_onehot, *self._target_module_onehot,
            ], dtype=np.float32)
        elif self._use_extended_state:
            raw = np.array([
                *tcp_pose, *tcp_vel, *tcp_error, *joint_pos, *joint_vel, *port_xyz,
                *port_delta_tcp, *tared_wrist_force,
                *feat_left, *feat_center, *feat_right,
                *self._plug_type_onehot,
            ], dtype=np.float32)
            if self._use_target_module_state:
                raw = np.concatenate(
                    [raw, self._target_module_onehot.astype(np.float32)],
                    axis=0,
                )
        else:
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
        self._plug_type_onehot = build_plug_type_onehot(getattr(task, "plug_type", ""))
        self._target_module_onehot = build_target_module_onehot(
            getattr(task, "target_module_name", "")
        )
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        for _ in range(10):
            obs = get_observation()
            if obs is not None:
                wrench = obs.wrist_wrench.wrench
                self._wrist_force_tare = np.array([
                    wrench.force.x, wrench.force.y, wrench.force.z,
                    wrench.torque.x, wrench.torque.y, wrench.torque.z,
                ], dtype=np.float32)
                break
            time.sleep(0.05)
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
        "    -p checkpoint_path:=/path/to/pretrained_model_77d"
    )
