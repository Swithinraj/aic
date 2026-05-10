"""
Deploy an ACT policy trained on the stripped V3 63D state.

This runner is model-only:
  * build the 63D observation.state expected by the checkpoint
  * run ACT to get a 6D delta TCP action
  * send that action directly as a Cartesian delta pose command

No deterministic guide, YOLO gate, depth gate, stall detector, forced z motion,
action shaping, action scaling, smoothing, hybrid insertion, search, or hold
controller is used. YOLO is used only as input features to the network.

V3 63D state layout:
  [0:7]   tcp_pose        7D  x y z qx qy qz qw
  [7:13]  tcp_velocity    6D  vx vy vz wx wy wz
  [13:20] joint_positions 7D
  [20:27] joint_velocity  7D
  [27:33] tared_wrist_force_torque 6D  fx fy fz tx ty tz
  [33:40] yolo_left       7D  conf cx_norm cy_norm w_norm h_norm valid age
  [40:47] yolo_center     7D
  [47:54] yolo_right      7D
  [54:56] plug_type_onehot 2D  [is_sfp, is_sc]
  [56:63] target_module_onehot 7D exact target module identity

Usage:
    ros2 run aic_model aic_model --ros-args \\
        -p policy:=team_policy.run_act_v3 \\
        -p checkpoint_path:=/path/to/pretrained_model_63d
"""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, Optional

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import String

from team_policy.run_act import (  # type: ignore[import-unresolved]
    RunACT,
    _DAMPING_APPROACH,
    _STEP_S,
    _STIFFNESS_APPROACH,
    _axis_angle_to_quat,
    _quat_multiply,
)
from team_policy.training_robot.episode_recorder_v2 import (
    TARGET_MODULE_NAMES,
    build_plug_type_onehot,
    build_target_module_onehot,
    build_yolo_feature,
)

_SCHEMA_V3_63D = "v3_63d_stripped_from_v9_77d"
_V3_REPLAN_EVERY = 10
_V3_MODEL_ROLLOUT_S = 0.0  # <= 0 means unbounded; external cancel/stop ends the rollout.

_CAMERAS = ("left", "center", "right")
_CAM_TOPICS: Dict[str, str] = {
    "left": "/left_camera/yolo/detections_json",
    "center": "/center_camera/yolo/detections_json",
    "right": "/right_camera/yolo/detections_json",
}


class RunACTV3(RunACT):
    """ACT policy runner for the stripped 63D V3 state schema."""

    def __init__(self, parent_node):
        super().__init__(parent_node)

        if self.state_dim != 63:
            raise ValueError(
                f"RunACTV3 expects a stripped 63D checkpoint but got state_dim={self.state_dim}. "
                "Use run_act_v2.py for 77D/75D/68D checkpoints or run_act.py for 30D checkpoints."
            )

        self.schema = _SCHEMA_V3_63D
        if self.replan_every == 10:
            self.replan_every = _V3_REPLAN_EVERY
        self.v3_model_rollout_s = float(
            parent_node.declare_parameter(
                "v3_model_rollout_s",
                _V3_MODEL_ROLLOUT_S,
            ).value
        )
        self._cam_lock: threading.Lock = threading.Lock()
        self._cam_last_det_time: Dict[str, Optional[float]] = {cam: None for cam in _CAMERAS}
        self._cam_last_conf: Dict[str, float] = {cam: 0.0 for cam in _CAMERAS}
        self._cam_last_bbox: Dict[str, Optional[list]] = {cam: None for cam in _CAMERAS}
        self._cam_last_img_hw: Dict[str, tuple[int, int]] = {cam: (0, 0) for cam in _CAMERAS}

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

        rollout = "unbounded" if self.v3_model_rollout_s <= 0.0 else f"{self.v3_model_rollout_s:.1f}s"
        self.get_logger().info(
            "RunACTV3 ready - "
            f"state_dim={self.state_dim}, schema={self.schema}, "
            f"replan_every={self.replan_every}, "
            f"model_rollout={rollout}, "
            "mode=model_only_raw_direct_action"
        )

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
            bbox = det.get("bbox_xyxy")
            if bbox is None or len(bbox) != 4:
                continue
            conf = float(det.get("confidence", 0.0))
            if best_rank is None or rank < best_rank or (rank == best_rank and conf > best_conf):
                best_rank = rank
                best_conf = conf
                best_bbox = [float(v) for v in bbox]

        if best_bbox is not None:
            with self._cam_lock:
                self._cam_last_det_time[cam] = time.time()
                self._cam_last_conf[cam] = best_conf
                self._cam_last_bbox[cam] = best_bbox

    def _reset_per_cam_state(self) -> None:
        with self._cam_lock:
            for cam in _CAMERAS:
                self._cam_last_det_time[cam] = None
                self._cam_last_conf[cam] = 0.0
                self._cam_last_bbox[cam] = None
                self._cam_last_img_hw[cam] = (0, 0)

    @staticmethod
    def _obs_image_hw(obs_msg, cam: str) -> tuple[int, int]:
        image_msg = getattr(obs_msg, f"{cam}_image", None)
        if image_msg is None:
            return 0, 0
        return int(getattr(image_msg, "height", 0)), int(getattr(image_msg, "width", 0))

    def _build_cam_feature(self, cam: str, now: float, img_h: int, img_w: int) -> np.ndarray:
        """Return 7D [conf, cx, cy, w, h, valid, age] for one camera."""
        with self._cam_lock:
            last_time = self._cam_last_det_time[cam]
            conf = self._cam_last_conf[cam]
            bbox = self._cam_last_bbox[cam]
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

    def _select_model_action(self, obs_msg) -> np.ndarray:
        batch = self._to_batch(obs_msg)
        with torch.inference_mode():
            norm_action = self.policy.select_action(batch)
        raw_action = norm_action * self.action_std + self.action_mean
        return raw_action[0].cpu().numpy().astype(np.float64)

    def _delta_to_pose(self, obs_msg, action_6d: np.ndarray) -> Pose:
        tcp = obs_msg.controller_state.tcp_pose
        cur_pos = np.array(
            [tcp.position.x, tcp.position.y, tcp.position.z],
            dtype=np.float64,
        )
        cur_quat = np.array(
            [tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w],
            dtype=np.float64,
        )

        action = action_6d.astype(np.float64)
        new_pos = cur_pos + action[:3]
        dq = _axis_angle_to_quat(action[3:6])
        new_quat = _quat_multiply(dq, cur_quat)
        nrm = np.linalg.norm(new_quat)
        if nrm > 1e-9:
            new_quat /= nrm

        return Pose(
            position=Point(
                x=float(new_pos[0]),
                y=float(new_pos[1]),
                z=float(new_pos[2]),
            ),
            orientation=Quaternion(
                x=float(new_quat[0]),
                y=float(new_quat[1]),
                z=float(new_quat[2]),
                w=float(new_quat[3]),
            ),
        )

    def _build_state(self, obs_msg) -> torch.Tensor:
        cs = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js = obs_msg.joint_states
        w = obs_msg.wrist_wrench.wrench

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

        wrist_force = np.array(
            [
                w.force.x,
                w.force.y,
                w.force.z,
                w.torque.x,
                w.torque.y,
                w.torque.z,
            ],
            dtype=np.float32,
        )
        tared_wrist_force = wrist_force - self._wrist_force_tare

        now = time.time()
        image_hw = {cam: self._obs_image_hw(obs_msg, cam) for cam in _CAMERAS}
        with self._cam_lock:
            for cam, (img_h, img_w) in image_hw.items():
                if img_h > 0 and img_w > 0:
                    self._cam_last_img_hw[cam] = (img_h, img_w)

        feat_left = self._build_cam_feature("left", now, *image_hw["left"])
        feat_center = self._build_cam_feature("center", now, *image_hw["center"])
        feat_right = self._build_cam_feature("right", now, *image_hw["right"])

        raw = np.array(
            [
                *tcp_pose,
                *tcp_vel,
                *joint_pos,
                *joint_vel,
                *tared_wrist_force,
                *feat_left,
                *feat_center,
                *feat_right,
                *self._plug_type_onehot,
                *self._target_module_onehot,
            ],
            dtype=np.float32,
        )

        if raw.shape[0] != self.state_dim:
            raise ValueError(
                f"RunACTV3 built state dim {raw.shape[0]}, checkpoint expects {self.state_dim}"
            )
        t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        return (t - self.state_mean) / self.state_std

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        start_wall = time.monotonic()
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
                self._wrist_force_tare = np.array(
                    [
                        wrench.force.x,
                        wrench.force.y,
                        wrench.force.z,
                        wrench.torque.x,
                        wrench.torque.y,
                        wrench.torque.z,
                    ],
                    dtype=np.float32,
                )
                break
            time.sleep(0.05)

        self._reset_task_target(task)
        self.policy.reset()
        self._prev_action = None

        rollout = "unbounded" if self.v3_model_rollout_s <= 0.0 else f"{self.v3_model_rollout_s:.1f}s"
        self.get_logger().info(
            f"RunACTV3 raw model-only rollout start - duration={rollout}; "
            "no force-z, no scaling, no smoothing, no clamp, no hybrid insert/search"
        )

        step_count = 0
        prev_target_pos = None
        while True:
            elapsed = time.monotonic() - start_wall
            if self.v3_model_rollout_s > 0.0 and elapsed >= self.v3_model_rollout_s:
                break
            t0_sim = self._now()
            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(_STEP_S)
                continue

            if step_count > 0 and step_count % max(1, self.replan_every) == 0:
                self.policy._action_queue.clear()

            action = self._select_model_action(obs_msg)
            target_pose = self._delta_to_pose(obs_msg, action)
            target_pos = np.array(
                [target_pose.position.x, target_pose.position.y, target_pose.position.z],
                dtype=np.float64,
            )
            mu = self._make_motion(target_pose, _STIFFNESS_APPROACH, _DAMPING_APPROACH)

            move_robot(motion_update=mu)

            pos, _, force_n, _ = self._get_tcp_state(obs_msg)

            if step_count < 5 or step_count % 20 == 0:
                elapsed = time.monotonic() - start_wall
                prev_err = (
                    float(np.linalg.norm(pos - prev_target_pos))
                    if prev_target_pos is not None
                    else 0.0
                )
                self.get_logger().info(
                    f"V3 MODEL step={step_count:4d} t={elapsed:6.1f}s | "
                    f"tcp=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) | "
                    f"target=({target_pos[0]:+.4f},{target_pos[1]:+.4f},{target_pos[2]:+.4f}) | "
                    f"prev_err={prev_err*1000:.1f}mm | "
                    f"action=({action[0]:+.5f},{action[1]:+.5f},{action[2]:+.5f},"
                    f"{action[3]:+.5f},{action[4]:+.5f},{action[5]:+.5f}) | "
                    f"F={force_n:.1f}N"
                )
            prev_target_pos = target_pos

            step_count += 1
            if step_count % 50 == 0:
                elapsed = time.monotonic() - start_wall
                send_feedback(f"V3 model rollout step={step_count} t={elapsed:.0f}s")

            self._pace_to_step(t0_sim)

        self.get_logger().info("RunACTV3 raw model-only rollout finished; returning True")
        return True


# aic_model entry point
# Usage: -p policy:=team_policy.run_act_v3
run_act_v3 = RunACTV3


def main():
    raise SystemExit(
        "RunACTV3 is not a standalone node.\n"
        "Run via aic_model:\n"
        "  ros2 run aic_model aic_model --ros-args \\\n"
        "    -p policy:=team_policy.run_act_v3 \\\n"
        "    -p checkpoint_path:=/path/to/pretrained_model_63d"
    )