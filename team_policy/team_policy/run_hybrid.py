"""
New run_hybrid.py
Phase 1: ACT coarse approach.
Phase 2: Multiview Left/Right Edge Matching.
"""
from __future__ import annotations

import json
import math
import sys
import time
import threading
import types
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from geometry_msgs.msg import Point, Pose, Quaternion, Wrench
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    cv2 = None

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import Policy

# ---------------------------------------------------------------------------
# ACT / LeRobot Imports
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

from team_policy.training_robot.episode_recorder_v2 import (
    TARGET_MODULE_NAMES,
    build_plug_type_onehot,
    build_target_module_onehot,
    build_yolo_feature,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMG_H, _IMG_W = 480, 640
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_CAMERAS = ("left", "center", "right")

_STIFFNESS_APPROACH = np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()
_DAMPING_APPROACH = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()
_STEP_S = 1.0 / 10.0


# ---------------------------------------------------------------------------
# Math Helpers
# ---------------------------------------------------------------------------
def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / angle
    s, c = math.sin(angle / 2.0), math.cos(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)


# ---------------------------------------------------------------------------
# Policy Class
# ---------------------------------------------------------------------------
class RunHybrid(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.node = parent_node

        decl = parent_node.declare_parameter
        # ACT params
        act_path = str(decl("act_checkpoint_path", "").value)
        fallback_path = str(decl("checkpoint_path", "").value)
        self.checkpoint_path = act_path if act_path else fallback_path
        self.act_enable = bool(decl("act_enable", True).value)
        self.act_max_steps = int(decl("act_max_steps", 600).value)
        self.act_timeout_s = float(decl("act_timeout_s", 80.0).value)
        self.act_min_duration_s = float(decl("act_min_duration_s", 40.0).value)
        if self.act_timeout_s < self.act_min_duration_s:
            self.act_timeout_s = self.act_min_duration_s
        self.act_max_translation_delta_m = float(decl("act_max_translation_delta_m", 0.15).value)
        self.act_max_rotation_delta_rad = float(decl("act_max_rotation_delta_rad", 0.35).value)
        self.act_action_scale = float(decl("act_action_scale", 1.0).value)
        
        # Edge Align params
        self.edge_align_enable = bool(decl("edge_align_enable", True).value)
        self.edge_align_timeout_s = float(decl("edge_align_timeout_s", 70.0).value)
        self.edge_align_min_duration_s = float(decl("edge_align_min_duration_s", 50.0).value)
        if self.edge_align_timeout_s < self.edge_align_min_duration_s:
            self.edge_align_timeout_s = self.edge_align_min_duration_s
        self.edge_align_tol_px = float(decl("edge_align_tol_px", 8.0).value)
        self.edge_align_stable_frames = int(decl("edge_align_stable_frames", 5).value)
        self.edge_probe_step_m = float(decl("edge_probe_step_m", 0.0015).value)
        self.edge_servo_gain = float(decl("edge_servo_gain", 0.3).value)
        self.edge_servo_damping = float(decl("edge_servo_damping", 1.0).value)
        self.edge_max_xy_step_m = float(decl("edge_max_xy_step_m", 0.002).value)
        
        self.edge_pair_lock_enable = bool(decl("edge_pair_lock_enable", True).value)
        self.edge_pair_reacquire_after_bad_frames = int(decl("edge_pair_reacquire_after_bad_frames", 8).value)
        self.edge_pair_bad_px = float(decl("edge_pair_bad_px", 120.0).value)
        
        self._target_port_anchor = {"left": None, "right": None}
        self._plug_anchor = {"left": None, "right": None}
        self._edge_pair_anchor = {"left": None, "right": None}
        
        # Safety/Misc
        self.force_hard_stop_n = float(decl("force_hard_stop_n", 35.0).value)
        self.hybrid_time_limit_s = float(decl("hybrid_time_limit_s", 170.0).value)
        self.debug_overlay_enable = bool(decl("debug_overlay_enable", False).value)

        self.debug_edge_overlay_enable = bool(decl("debug_edge_overlay_enable", True).value)
        self.debug_edge_overlay_save_dir = str(decl("debug_edge_overlay_save_dir", "").value)
        self.debug_edge_overlay_save_every = int(decl("debug_edge_overlay_save_every", 0).value)
        left_debug_topic = str(decl("left_debug_image_input_topic", "/left_camera/yolo/annotated_image").value)
        right_debug_topic = str(decl("right_debug_image_input_topic", "/right_camera/yolo/annotated_image").value)

        self.get_logger().info(
            f"HYBRID_MINIMAL_INIT mode=act_then_edge_match act_min={self.act_min_duration_s:.1f}s edge_min={self.edge_align_min_duration_s:.1f}s"
        )
        self.get_logger().info("HYBRID_REMOVED old=cad,gru,learned_vs,triangulation,force_insert,old_ibvs,bbox_tip")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_act(self.checkpoint_path)

        # State vars
        self._task = None
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        self._plug_type_onehot = np.zeros(2, dtype=np.float32)
        self._target_module_onehot = np.zeros(len(TARGET_MODULE_NAMES), dtype=np.float32)
        
        # YOLO subscribers and state
        self.latest_dets: Dict[str, list] = {"left": [], "right": []}
        self.latest_dets_time: Dict[str, float] = {"left": 0.0, "right": 0.0}
        self._yolo_lock = threading.Lock()

        parent_node.create_subscription(String, "/left_camera/yolo/detections_json", lambda msg: self._cb_yolo(msg, "left"), 10)
        parent_node.create_subscription(String, "/right_camera/yolo/detections_json", lambda msg: self._cb_yolo(msg, "right"), 10)
        # We also need center camera for ACT feature
        parent_node.create_subscription(String, "/center_camera/yolo/detections_json", lambda msg: self._cb_yolo(msg, "center"), 10)
        
        # Debug overlay state
        self._last_edge_pair = {"left": None, "right": None}
        self._annotated_images = {"left": None, "right": None}
        self._annotated_lock = threading.Lock()
        self._debug_pub_left = parent_node.create_publisher(Image, "/hybrid/debug/left_edge_match_image", 1)
        self._debug_pub_right = parent_node.create_publisher(Image, "/hybrid/debug/right_edge_match_image", 1)
        self._debug_save_count = 0

        parent_node.create_subscription(Image, left_debug_topic, lambda msg: self._cb_annotated(msg, "left"), 1)
        parent_node.create_subscription(Image, right_debug_topic, lambda msg: self._cb_annotated(msg, "right"), 1)
        self.latest_dets["center"] = []
        self.latest_dets_time["center"] = 0.0
        
        # We track stable detections for ACT early stopping
        self._act_stable_frames = 0
        
        missing = []
        for name in ["_check_early_stop", "_select_best_target_strictly", "_run_act_coarse", "_run_edge_align", "_measure_errors"]:
            if not hasattr(self, name):
                missing.append(name)
        if missing:
            raise RuntimeError(f"RunHybrid missing required methods: {missing}")
            
        if hasattr(self, "_find_port_and_plug"):
            self.get_logger().warning("HYBRID_STALE_METHOD_PRESENT name=_find_port_and_plug")
        
    def _cb_yolo(self, msg: String, cam: str):
        try:
            dets = json.loads(msg.data)
            if isinstance(dets, list):
                with self._yolo_lock:
                    self.latest_dets[cam] = dets
                    self.latest_dets_time[cam] = time.time()
        except Exception:
            pass

    def _cb_annotated(self, msg: Image, cam: str):
        with self._annotated_lock:
            self._annotated_images[cam] = msg

    def _load_act(self, raw_path: str):
        path = Path(raw_path).expanduser()
        if path.name != "pretrained_model" and (path / "pretrained_model").is_dir():
            path = path / "pretrained_model"
        
        self.get_logger().info(f"HYBRID_ACT_LOAD checkpoint={path}")
        
        with open(path / "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)
        import draccus
        self.act_config = draccus.decode(ACTConfig, cfg_dict)
        self.state_dim = int(self.act_config.input_features["observation.state"].shape[0])
        self.action_dim = int(self.act_config.output_features["action"].shape[0])
        self.get_logger().info(f"HYBRID_ACT_CONFIG state_dim={self.state_dim} action_dim={self.action_dim}")
        
        self.policy = ACTPolicy(self.act_config)
        self.policy.load_state_dict(load_file(path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        pre_stats = load_file(str(path / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
        post_stats = load_file(str(path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))

        def _get(stats, key, shape, default):
            return stats[key].to(self.device).float() if key in stats else torch.full(shape, default, device=self.device)

        self.state_mean = _get(pre_stats, "observation.state.mean", (self.state_dim,), 0.0).view(1, -1)
        self.state_std = torch.clamp(_get(pre_stats, "observation.state.std", (self.state_dim,), 1.0).view(1, -1), min=1e-6)
        self.action_mean = _get(post_stats, "action.mean", (self.action_dim,), 0.0).view(1, -1)
        self.action_std = _get(post_stats, "action.std", (self.action_dim,), 1.0).view(1, -1)

        # Extract image keys
        self.act_image_keys = []
        self.act_image_hw = {}
        for k, v in self.act_config.input_features.items():
            if str(k).startswith("observation.images."):
                self.act_image_keys.append(str(k))
                shape = list(getattr(v, "shape", []))
                self.act_image_hw[str(k)] = (int(shape[-2]), int(shape[-1])) if len(shape) >= 3 else (_IMG_H, _IMG_W)
        
        # Verify mapping
        self.image_key_to_obs_attr = {}
        for key in self.act_image_keys:
            if key.endswith("left") or key.endswith("left_camera"):
                self.image_key_to_obs_attr[key] = "left_image"
            elif key.endswith("center") or key.endswith("center_camera"):
                self.image_key_to_obs_attr[key] = "center_image"
            elif key.endswith("right") or key.endswith("right_camera"):
                self.image_key_to_obs_attr[key] = "right_image"
            else:
                raise ValueError(f"Cannot map ACT image key to obs_msg attribute: {key}")
                
        self.img_mean = _IMAGENET_MEAN.to(self.device)
        self.img_std = _IMAGENET_STD.to(self.device)

    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds / 1e9

    def _pad(self, values, length: int) -> list[float]:
        out = [float(v) for v in list(values)[:length]]
        while len(out) < length:
            out.append(0.0)
        return out

    def _ros_image_to_rgb(self, img_msg) -> np.ndarray:
        arr = np.frombuffer(img_msg.data, dtype=np.uint8)
        arr = arr[: img_msg.height * img_msg.width * 3].reshape(img_msg.height, img_msg.width, 3)
        encoding = str(getattr(img_msg, "encoding", "rgb8")).lower()
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr).copy()

    def _img_to_tensor(self, img_msg, target_h, target_w, image_mean, image_std) -> torch.Tensor:
        arr = self._ros_image_to_rgb(img_msg)
        t = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        if t.shape[2] != target_h or t.shape[3] != target_w:
            t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return (t - image_mean) / image_std

    def _get_tcp_state(self, obs_msg):
        tcp = obs_msg.controller_state.tcp_pose
        pos = np.array([tcp.position.x, tcp.position.y, tcp.position.z], dtype=np.float64)
        quat = np.array([tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w], dtype=np.float64)
        wrench = obs_msg.wrist_wrench.wrench
        force_n = math.sqrt(wrench.force.x**2 + wrench.force.y**2 + wrench.force.z**2)
        return pos, quat, force_n

    def _build_act_cam_feature(self, cam: str, now: float, img_h: int, img_w: int) -> np.ndarray:
        with self._yolo_lock:
            dets = self.latest_dets.get(cam, [])
            last_time = self.latest_dets_time.get(cam, None)

        best_rank = None
        best_conf = 0.0
        best_bbox = None
        for det in dets:
            rank = self._target_match_rank(det)
            if rank is None: continue
            conf = float(det.get("confidence", 0.0))
            if best_rank is None or rank < best_rank or (rank == best_rank and conf > best_conf):
                best_rank = rank
                best_conf = conf
                bbox = det.get("bbox_xyxy")
                if bbox and len(bbox) == 4:
                    best_bbox = [float(v) for v in bbox]

        use_h = img_h if img_h > 0 else _IMG_H
        use_w = img_w if img_w > 0 else _IMG_W
        return build_yolo_feature(best_conf, best_bbox, use_h, use_w, last_time, now)

    def _target_match_rank(self, det: dict) -> Optional[int]:
        names = {str(det.get("instance_name", "")).strip().lower(), str(det.get("class_name", "")).strip().lower()}
        target_port = getattr(self._task, "port_name", "").strip().lower()
        target_type = getattr(self._task, "port_type", "").strip().lower()
        
        if target_port and any(target_port in n for n in names): return 0
        if target_type == "sfp" and any("sfp_port" in n for n in names): return 1
        if target_type == "sc" and any("sc_port" in n for n in names): return 1
        return None

    def _build_state_63d(self, obs_msg) -> torch.Tensor:
        cs = obs_msg.controller_state
        tcp, vel, js, w = cs.tcp_pose, cs.tcp_velocity, obs_msg.joint_states, obs_msg.wrist_wrench.wrench
        tcp_pose = [tcp.position.x, tcp.position.y, tcp.position.z, tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w]
        tcp_vel = [vel.linear.x, vel.linear.y, vel.linear.z, vel.angular.x, vel.angular.y, vel.angular.z]
        joint_pos = self._pad(js.position, 7)
        joint_vel = self._pad(js.velocity, 7)
        tared_force = np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z], dtype=np.float32) - self._wrist_force_tare

        now = time.time()
        img_hw = {}
        for cam in _CAMERAS:
            im = getattr(obs_msg, f"{cam}_image", None)
            img_hw[cam] = (int(im.height), int(im.width)) if im else (0, 0)
            
        feat_left = self._build_act_cam_feature("left", now, *img_hw["left"])
        feat_center = self._build_act_cam_feature("center", now, *img_hw["center"])
        feat_right = self._build_act_cam_feature("right", now, *img_hw["right"])

        raw = np.array([
            *tcp_pose, *tcp_vel, *joint_pos, *joint_vel, *tared_force,
            *feat_left, *feat_center, *feat_right,
            *self._plug_type_onehot, *self._target_module_onehot
        ], dtype=np.float32)
        
        t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        return (t - self.state_mean) / self.state_std

    def _to_act_batch(self, obs_msg) -> dict:
        state_norm = self._build_state_63d(obs_msg)
        batch = {"observation.state": state_norm}
        for key in self.act_image_keys:
            attr = self.image_key_to_obs_attr[key]
            img_msg = getattr(obs_msg, attr)
            tgt_h, tgt_w = self.act_image_hw[key]
            batch[key] = self._img_to_tensor(img_msg, tgt_h, tgt_w, self.img_mean, self.img_std)
        return batch

    def _select_model_action(self, obs_msg) -> np.ndarray:
        batch = self._to_act_batch(obs_msg)
        with torch.inference_mode():
            norm_action = self.policy.select_action(batch)
        raw_action = norm_action * self.action_std + self.action_mean
        act_np = raw_action[0].cpu().numpy().astype(np.float64) * self.act_action_scale
        
        t_norm = float(np.linalg.norm(act_np[:3]))
        if self.act_max_translation_delta_m > 0 and t_norm > self.act_max_translation_delta_m:
            act_np[:3] *= self.act_max_translation_delta_m / t_norm
            
        r_norm = float(np.linalg.norm(act_np[3:6]))
        if self.act_max_rotation_delta_rad > 0 and r_norm > self.act_max_rotation_delta_rad:
            act_np[3:6] *= self.act_max_rotation_delta_rad / r_norm
            
        return act_np

    def _delta_to_pose(self, cur_pos, cur_quat, action_6d: np.ndarray) -> Pose:
        new_pos = cur_pos + action_6d[:3]
        dq = _axis_angle_to_quat(action_6d[3:6])
        new_quat = _quat_multiply(dq, cur_quat)
        nrm = np.linalg.norm(new_quat)
        if nrm > 1e-9: new_quat /= nrm
        return Pose(
            position=Point(x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])),
            orientation=Quaternion(x=float(new_quat[0]), y=float(new_quat[1]), z=float(new_quat[2]), w=float(new_quat[3]))
        )

    def _make_motion(self, target_pose: Pose) -> MotionUpdate:
        mu = MotionUpdate()
        mu.header = Header(frame_id="base_link", stamp=self.node.get_clock().now().to_msg())
        mu.pose = target_pose
        mu.trajectory_generation_mode = TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION)
        mu.target_stiffness = _STIFFNESS_APPROACH
        mu.target_damping = _DAMPING_APPROACH
        mu.feedforward_wrench_at_tip = Wrench()
        mu.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        return mu

    # -----------------------------------------------------------------------
    # Task Execution
    # -----------------------------------------------------------------------
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._task = task
        self._plug_type_onehot = build_plug_type_onehot(getattr(task, "plug_type", ""))
        self._target_module_onehot = build_target_module_onehot(getattr(task, "target_module_name", ""))
        
        # Tare
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        for _ in range(5):
            obs = get_observation()
            if obs:
                w = obs.wrist_wrench.wrench
                self._wrist_force_tare = np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z], dtype=np.float32)
                break
            time.sleep(0.05)

        start_time = time.monotonic()
        
        self._target_port_anchor = {"left": None, "right": None}
        self._plug_anchor = {"left": None, "right": None}
        self._edge_pair_anchor = {"left": None, "right": None}
        self._act_stable_frames = 0

        if self.act_enable:
            success = self._run_act_coarse(get_observation, move_robot, send_feedback, start_time)
            if not success: return False
            
        if self.edge_align_enable:
            success = self._run_edge_align(get_observation, move_robot, send_feedback, start_time)
            if not success: return False
            
        return True

    def _check_early_stop(self) -> bool:
        with self._yolo_lock:
            left = list(self.latest_dets.get("left", []))
            right = list(self.latest_dets.get("right", []))
            
        def has_both(dets):
            port, plug = self._select_best_target_strictly(dets, self._task)
            return port is not None and plug is not None

        if has_both(left) or has_both(right):
            self._act_stable_frames += 1
        else:
            self._act_stable_frames = 0
            
        return self._act_stable_frames >= 5

    def _run_act_coarse(self, get_observation, move_robot, send_feedback, start_time):
        self.get_logger().info("HYBRID_PHASE ACT_COARSE")
        try:
            self.policy.reset()
        except Exception:
            pass

        self._act_stable_frames = 0
        phase_start = time.monotonic()
        step = 0

        while True:
            now = time.monotonic()
            total_elapsed = now - start_time
            phase_elapsed = now - phase_start
            min_done = phase_elapsed >= self.act_min_duration_s

            if total_elapsed > self.hybrid_time_limit_s:
                self.get_logger().info("HYBRID_ACT_DONE reason=global_timeout")
                return False
            if phase_elapsed > self.act_timeout_s and min_done:
                self.get_logger().info("HYBRID_ACT_DONE reason=act_timeout")
                break
            if step >= self.act_max_steps and min_done:
                self.get_logger().info("HYBRID_ACT_DONE reason=max_steps")
                break

            obs = get_observation()
            if not obs:
                time.sleep(0.05)
                continue

            pos, quat, force_n = self._get_tcp_state(obs)
            if force_n > self.force_hard_stop_n:
                self.get_logger().error("HYBRID_ACT_DONE reason=force_limit")
                return False

            if step > 5:
                try:
                    early_stop = self._check_early_stop()
                except Exception as e:
                    self.get_logger().error(f"HYBRID_ACT_EARLY_STOP_ERROR reason={e}")
                    early_stop = False
                    
                if early_stop:
                    if min_done:
                        self.get_logger().info("HYBRID_ACT_DONE reason=detections_stable")
                        break
                    if step % 20 == 0:
                        remaining = max(0.0, self.act_min_duration_s - phase_elapsed)
                        self.get_logger().info(f"HYBRID_ACT_MIN_HOLD remaining={remaining:.1f}s")

            if step == 0:
                probe_batch = self._to_act_batch(obs)
                self.get_logger().info(f"HYBRID_ACT_BATCH_KEYS keys={list(probe_batch.keys())}")
                self.get_logger().info(f"HYBRID_ACT_IMAGE_KEYS keys={self.act_image_keys}")
                self.get_logger().info(f"HYBRID_ACT_STATE shape={probe_batch['observation.state'].shape}")
                for k in self.act_image_keys:
                    self.get_logger().info(f"HYBRID_ACT_IMAGE key={k} shape={probe_batch[k].shape}")

            try:
                action = self._select_model_action(obs)
            except Exception as e:
                self.get_logger().error(f"HYBRID_ACT_ERROR reason={e}")
                return False

            target_pose = self._delta_to_pose(pos, quat, action)
            move_robot(motion_update=self._make_motion(target_pose))

            if step % 20 == 0:
                self.get_logger().info(
                    f"HYBRID_ACT_STEP step={step} elapsed={phase_elapsed:.1f}s action={action[:3]} force={force_n:.1f}"
                )
                send_feedback(f"ACT step {step}")

            step += 1
            time.sleep(_STEP_S)

        obs = get_observation()
        if obs:
            pos, quat, _ = self._get_tcp_state(obs)
            target = Pose(
                position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])),
            )
            move_robot(motion_update=self._make_motion(target))
            time.sleep(0.5)

        return True

    def _select_best_target_strictly(self, dets, task):
        req_port = getattr(task, "port_name", "").strip()
        plug_type = getattr(task, "plug_type", "").strip()
        
        best_port = None
        best_port_score = -1
        
        best_plug = None
        best_plug_score = -1
        
        for d in dets:
            inst = str(d.get("instance_name", "")).strip()
            cls = str(d.get("class_name", "")).strip()
            conf = float(d.get("confidence", 0.0))
            
            port_score = -1
            if req_port and (inst == req_port or cls == req_port):
                port_score = 100.0 + conf
            elif "port" in inst or "port" in cls:
                if plug_type == "sc" and ("sc_port" in inst or "sc_port" in cls):
                    port_score = 10.0 + conf
                elif plug_type != "sc" and ("sfp_port" in inst or "sfp_port" in cls):
                    port_score = 10.0 + conf
                    
            if port_score > best_port_score:
                best_port_score = port_score
                best_port = d
                
            plug_score = -1
            if plug_type == "sc" and ("sc_plug" in inst or "sc_plug" in cls):
                plug_score = 10.0 + conf
            elif plug_type != "sc" and ("sfp_module" in inst or "sfp_module" in cls):
                plug_score = 10.0 + conf
                
            if plug_score > best_plug_score:
                best_plug_score = plug_score
                best_plug = d
                
        return best_port, best_plug

    def _bbox_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
        return iou

    def _update_anchor(self, camera, anchor, detections, kind):
        if anchor is None: return None
        best_det, best_score, best_iou = None, -1.0, 0.0
        for d in detections:
            bbox = d.get("bbox_xyxy")
            if not bbox or len(bbox) != 4: continue
            bbox = [float(v) for v in bbox]
            iou = self._bbox_iou(anchor["bbox_xyxy"], bbox)
            inst = str(d.get("instance_name", "")).strip()
            cls = str(d.get("class_name", "")).strip()
            conf = float(d.get("confidence", 0.0))
            score = -1.0
            if kind == "port":
                if inst == anchor["instance_name"]: score = 10.0 + iou + conf
                elif cls == anchor["class_name"] and iou > 0.25: score = 2.0 * iou + conf
            elif kind == "plug":
                if cls == anchor["class_name"] and iou > 0.20: score = 2.0 * iou + conf
            if score > best_score:
                best_score = score
                best_det = d
                best_iou = iou
        if best_det:
            return {
                "camera": camera,
                "instance_name": str(best_det.get("instance_name", "")),
                "class_name": str(best_det.get("class_name", "")),
                "bbox_xyxy": [float(v) for v in best_det.get("bbox_xyxy")],
                "confidence": float(best_det.get("confidence", 0.0)),
                "last_seen_time": time.time(),
                "missed_count": 0,
                "iou": best_iou
            }
        else:
            anchor["missed_count"] += 1
            return anchor

    def _initialize_edge_anchors(self):
        self.get_logger().info("HYBRID_EDGE_ALIGN initializing anchors...")
        start_t = time.monotonic()
        while time.monotonic() - start_t < 1.0:
            time.sleep(0.1)
        with self._yolo_lock:
            left_dets = self.latest_dets.get("left", [])
            right_dets = self.latest_dets.get("right", [])
        for cam, dets in [("left", left_dets), ("right", right_dets)]:
            port, plug = self._select_best_target_strictly(dets, self._task)
            if port:
                self._target_port_anchor[cam] = {
                    "camera": cam, "instance_name": str(port.get("instance_name", "")),
                    "class_name": str(port.get("class_name", "")),
                    "bbox_xyxy": [float(v) for v in port.get("bbox_xyxy", [0]*4)],
                    "confidence": float(port.get("confidence", 0.0)),
                    "last_seen_time": time.time(), "missed_count": 0, "iou": 1.0
                }
            if plug:
                self._plug_anchor[cam] = {
                    "camera": cam, "instance_name": str(plug.get("instance_name", "")),
                    "class_name": str(plug.get("class_name", "")),
                    "bbox_xyxy": [float(v) for v in plug.get("bbox_xyxy", [0]*4)],
                    "confidence": float(plug.get("confidence", 0.0)),
                    "last_seen_time": time.time(), "missed_count": 0, "iou": 1.0
                }
            if not port or not plug:
                self.get_logger().info(f"HYBRID_EDGE_ALIGN missing port/plug initially for camera={cam}")

    def _compute_edges(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        return {
            "left": {"mid": np.array([x1, cy]), "dir": "v", "pts": ((x1, y1), (x1, y2))},
            "right": {"mid": np.array([x2, cy]), "dir": "v", "pts": ((x2, y1), (x2, y2))},
            "top": {"mid": np.array([cx, y1]), "dir": "h", "pts": ((x1, y1), (x2, y1))},
            "bottom": {"mid": np.array([cx, y2]), "dir": "h", "pts": ((x1, y2), (x2, y2))},
        }

    def _match_nearest_edges(self, plug_bbox, port_bbox, cam: str):
        plug_edges = self._compute_edges(plug_bbox)
        port_edges = self._compute_edges(port_bbox)
        
        locked_pair = self._edge_pair_anchor.get(cam)
        if locked_pair and "bad_frames" in locked_pair:
            if locked_pair["bad_frames"] >= self.edge_pair_reacquire_after_bad_frames:
                self.get_logger().info(f"HYBRID_EDGE_PAIR_REACQUIRE camera={cam} old={locked_pair['plug_edge']}-{locked_pair['port_edge']} new=auto reason=bad_for_{locked_pair['bad_frames']}_frames")
                locked_pair = None
                self._edge_pair_anchor[cam] = None
        
        if locked_pair and self.edge_pair_lock_enable:
            pname = locked_pair["plug_edge"]
            tname = locked_pair["port_edge"]
            pedge = plug_edges[pname]
            tedge = port_edges[tname]
            dist = float(np.linalg.norm(pedge["mid"] - tedge["mid"]))
            match = {
                "plug_edge": pname, "port_edge": tname,
                "plug_mid": pedge["mid"], "port_mid": tedge["mid"],
                "plug_pts": pedge["pts"], "port_pts": tedge["pts"],
                "err_uv": tedge["mid"] - pedge["mid"], "err_px": dist, "reason": "locked"
            }
            if dist > self.edge_pair_bad_px:
                locked_pair["bad_frames"] = locked_pair.get("bad_frames", 0) + 1
            else:
                locked_pair["bad_frames"] = 0
            self.get_logger().info(f"HYBRID_EDGE_PAIR_LOCKED camera={cam} plug_edge={pname} port_edge={tname} err_px={dist:.1f}")
            return match
            
        opposites = [("top", "bottom"), ("bottom", "top"), ("left", "right"), ("right", "left")]
        best_match = None
        best_dist = float('inf')
        for pname, tname in opposites:
            pedge = plug_edges[pname]
            tedge = port_edges[tname]
            dist = float(np.linalg.norm(pedge["mid"] - tedge["mid"]))
            if dist < best_dist:
                best_dist = dist
                best_match = {
                    "plug_edge": pname, "port_edge": tname,
                    "plug_mid": pedge["mid"], "port_mid": tedge["mid"],
                    "plug_pts": pedge["pts"], "port_pts": tedge["pts"],
                    "err_uv": tedge["mid"] - pedge["mid"], "err_px": dist, "reason": "initial"
                }
                
        if best_match and self.edge_pair_lock_enable:
            self._edge_pair_anchor[cam] = {
                "plug_edge": best_match["plug_edge"],
                "port_edge": best_match["port_edge"],
                "bad_frames": 0
            }
            self.get_logger().info(f"HYBRID_EDGE_PAIR_LOCK camera={cam} plug_edge={best_match['plug_edge']} port_edge={best_match['port_edge']} err_px={best_match['err_px']:.1f}")
        return best_match

    def _measure_errors(self):
        with self._yolo_lock:
            left_dets = self.latest_dets.get("left", [])
            right_dets = self.latest_dets.get("right", [])
            
        errs = []
        for cam, dets in [("left", left_dets), ("right", right_dets)]:
            if self._target_port_anchor[cam]:
                self._target_port_anchor[cam] = self._update_anchor(cam, self._target_port_anchor[cam], dets, "port")
                if self._target_port_anchor[cam] and self._target_port_anchor[cam]["missed_count"] > 10:
                    self._target_port_anchor[cam] = None
            if self._plug_anchor[cam]:
                self._plug_anchor[cam] = self._update_anchor(cam, self._plug_anchor[cam], dets, "plug")
                if self._plug_anchor[cam] and self._plug_anchor[cam]["missed_count"] > 10:
                    self._plug_anchor[cam] = None
            
            if not self._target_port_anchor[cam]:
                self.get_logger().info(f"HYBRID_EDGE_ANCHOR_MISS camera={cam} kind=port missed=11")
                port, _ = self._select_best_target_strictly(dets, self._task)
                if port:
                    self._target_port_anchor[cam] = {
                        "camera": cam, "instance_name": str(port.get("instance_name", "")),
                        "class_name": str(port.get("class_name", "")),
                        "bbox_xyxy": [float(v) for v in port.get("bbox_xyxy", [0]*4)],
                        "confidence": float(port.get("confidence", 0.0)),
                        "last_seen_time": time.time(), "missed_count": 0, "iou": 1.0
                    }
            if not self._plug_anchor[cam]:
                self.get_logger().info(f"HYBRID_EDGE_ANCHOR_MISS camera={cam} kind=plug missed=11")
                _, plug = self._select_best_target_strictly(dets, self._task)
                if plug:
                    self._plug_anchor[cam] = {
                        "camera": cam, "instance_name": str(plug.get("instance_name", "")),
                        "class_name": str(plug.get("class_name", "")),
                        "bbox_xyxy": [float(v) for v in plug.get("bbox_xyxy", [0]*4)],
                        "confidence": float(plug.get("confidence", 0.0)),
                        "last_seen_time": time.time(), "missed_count": 0, "iou": 1.0
                    }
                    
            if self._target_port_anchor[cam] and self._plug_anchor[cam]:
                port_a = self._target_port_anchor[cam]
                plug_a = self._plug_anchor[cam]
                if port_a["missed_count"] == 0 and plug_a["missed_count"] == 0:
                    self.get_logger().info(f"HYBRID_EDGE_ANCHOR camera={cam} port={port_a['instance_name']} port_conf={port_a['confidence']:.2f} plug={plug_a['instance_name']} plug_conf={plug_a['confidence']:.2f} port_iou={port_a.get('iou', 1.0):.2f} plug_iou={plug_a.get('iou', 1.0):.2f}")
                match = self._match_nearest_edges(plug_a["bbox_xyxy"], port_a["bbox_xyxy"], cam)
                if match:
                    match["camera"] = cam
                    match["weight"] = float(port_a["confidence"] * plug_a["confidence"])
                    match["port_instance"] = port_a["instance_name"]
                    match["plug_instance"] = plug_a["instance_name"]
                    errs.append(match)
        return errs

    def _run_edge_align(self, get_observation, move_robot, send_feedback, start_time):
        self.get_logger().info("HYBRID_PHASE EDGE_ALIGN")
        self._initialize_edge_anchors()
        align_start = time.monotonic()
        stable_count = 0

        while True:
            edge_elapsed = time.monotonic() - align_start
            if edge_elapsed >= self.edge_align_timeout_s:
                break
            if time.monotonic() - start_time > self.hybrid_time_limit_s:
                self.get_logger().info("HYBRID_EDGE_ALIGN_TIMEOUT reason=global_timeout")
                return False

            obs = get_observation()
            if not obs:
                time.sleep(0.05)
                continue

            pos, quat, force_n = self._get_tcp_state(obs)
            if force_n > self.force_hard_stop_n:
                self.get_logger().info("HYBRID_EDGE_ALIGN_TIMEOUT reason=force_limit")
                return False

            base_errs = self._measure_errors()
            if not base_errs:
                self.get_logger().info("HYBRID_EDGE_STATUS valid=none best=none median_err=inf")
                time.sleep(0.1)
                continue

            if self.debug_edge_overlay_enable:
                self._publish_debug_overlays(obs, base_errs)

            cams = [m["camera"] for m in base_errs]
            med_err = float(np.median([m["err_px"] for m in base_errs]))
            self.get_logger().info(
                f"HYBRID_EDGE_STATUS valid={','.join(cams)} best=none median_err={med_err:.1f} elapsed={edge_elapsed:.1f}s"
            )

            for m in base_errs:
                self.get_logger().info(
                    f"HYBRID_EDGE camera={m['camera']} plug_edge={m['plug_edge']} port_edge={m['port_edge']} "
                    f"err_uv=({m['err_uv'][0]:.1f}, {m['err_uv'][1]:.1f}) err_px={m['err_px']:.1f}"
                )

            if med_err <= self.edge_align_tol_px:
                stable_count += 1
                if stable_count >= self.edge_align_stable_frames:
                    edge_elapsed = time.monotonic() - align_start
                    if edge_elapsed >= self.edge_align_min_duration_s:
                        self.get_logger().info("HYBRID_EDGE_ALIGN_SUCCESS")
                        return True

                    remaining = max(0.0, self.edge_align_min_duration_s - edge_elapsed)
                    self.get_logger().info(
                        f"HYBRID_EDGE_ALIGN_HOLD remaining={remaining:.1f}s median_err={med_err:.1f}"
                    )
                    target = Pose(
                        position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                        orientation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])),
                    )
                    move_robot(motion_update=self._make_motion(target))
                    time.sleep(min(0.5, max(0.05, remaining)))
                    continue
            else:
                stable_count = 0

            probe_step = self.edge_probe_step_m
            probes = [(probe_step, 0), (-probe_step, 0), (0, probe_step), (0, -probe_step)]

            J = np.zeros((len(base_errs) * 2, 2))
            E = np.zeros(len(base_errs) * 2)
            cam_idx = {m["camera"]: i for i, m in enumerate(base_errs)}

            for i, m in enumerate(base_errs):
                E[2 * i] = m["err_uv"][0]
                E[2 * i + 1] = m["err_uv"][1]

            valid_probes = 0
            for dx, dy in probes:
                tgt = self._delta_to_pose(pos, quat, np.array([dx, dy, 0, 0, 0, 0], dtype=np.float64))
                move_robot(motion_update=self._make_motion(tgt))
                time.sleep(0.3)

                new_errs = self._measure_errors()

                tgt = self._delta_to_pose(pos, quat, np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
                move_robot(motion_update=self._make_motion(tgt))
                time.sleep(0.3)

                probe_used = False
                for nm in new_errs:
                    if nm["camera"] not in cam_idx:
                        continue
                    cam = nm["camera"]
                    idx = cam_idx[cam]
                    old_m = base_errs[idx]

                    if old_m["plug_edge"] != nm["plug_edge"] or old_m["port_edge"] != nm["port_edge"] or old_m.get("port_instance") != nm.get("port_instance") or old_m.get("plug_instance") != nm.get("plug_instance"):
                        self.get_logger().info(f"HYBRID_EDGE_PROBE_REJECT camera={cam} reason=anchor_or_edge_changed")
                        continue

                    d_u = nm["err_uv"][0] - old_m["err_uv"][0]
                    d_v = nm["err_uv"][1] - old_m["err_uv"][1]

                    if dx != 0:
                        J[2 * idx, 0] = d_u / dx
                        J[2 * idx + 1, 0] = d_v / dx
                        probe_used = True
                    elif dy != 0:
                        J[2 * idx, 1] = d_u / dy
                        J[2 * idx + 1, 1] = d_v / dy
                        probe_used = True

                if probe_used:
                    valid_probes += 1

            self.get_logger().info(f"HYBRID_EDGE_JACOBIAN valid_probes={valid_probes} J={J.flatten().tolist()}")
            
            if valid_probes < 2:
                self.get_logger().info("HYBRID_EDGE_ALIGN_HOLD reason=fewer_than_2_valid_probes")
                time.sleep(0.2)
                continue

            lambda_damp = self.edge_servo_damping
            JtJ = J.T @ J
            inv = np.linalg.pinv(JtJ + lambda_damp * np.eye(2))
            delta_xy = -self.edge_servo_gain * (inv @ J.T @ E)

            step_norm = np.linalg.norm(delta_xy)
            if step_norm > self.edge_max_xy_step_m:
                delta_xy *= self.edge_max_xy_step_m / step_norm

            self.get_logger().info(f"HYBRID_EDGE_CMD delta_xy=({delta_xy[0]:.5f}, {delta_xy[1]:.5f})")

            tgt = self._delta_to_pose(pos, quat, np.array([delta_xy[0], delta_xy[1], 0, 0, 0, 0], dtype=np.float64))
            move_robot(motion_update=self._make_motion(tgt))
            time.sleep(0.4)

            post_errs = self._measure_errors()
            if not post_errs:
                continue

            post_med = float(np.median([m["err_px"] for m in post_errs]))
            if post_med > med_err + 2.0:
                self.get_logger().info(f"HYBRID_EDGE_STEP before={med_err:.1f} after={post_med:.1f} revert")
                tgt = self._delta_to_pose(pos, quat, np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
                move_robot(motion_update=self._make_motion(tgt))
                time.sleep(0.4)
            else:
                self.get_logger().info(f"HYBRID_EDGE_STEP before={med_err:.1f} after={post_med:.1f} keep")

        self.get_logger().info(f"HYBRID_EDGE_ALIGN_TIMEOUT elapsed={time.monotonic() - align_start:.1f}s")
        return False

    def _publish_debug_overlays(self, obs, base_errs):
        try:
            if cv2 is None:
                return

            for m in base_errs:
                cam = m["camera"]
                pub = self._debug_pub_left if cam == "left" else self._debug_pub_right if cam == "right" else None
                if not pub: continue
                
                with self._annotated_lock:
                    img_msg = self._annotated_images.get(cam)
                if not img_msg:
                    img_msg = getattr(obs, f"{cam}_image", None)
                if not img_msg:
                    continue

                arr = self._ros_image_to_rgb(img_msg)
                arr_bgr = arr[:, :, ::-1].copy()
                h, w, _ = arr_bgr.shape
                
                def clamp(pt):
                    if not math.isfinite(pt[0]) or not math.isfinite(pt[1]):
                        return (0, 0)
                    return (int(max(0, min(pt[0], w - 1))), int(max(0, min(pt[1], h - 1))))
                
                p_mid = clamp((m["plug_mid"][0], m["plug_mid"][1]))
                t_mid = clamp((m["port_mid"][0], m["port_mid"][1]))
                
                is_clamped = False
                if p_mid[0] <= 0 or p_mid[0] >= w-1 or p_mid[1] <= 0 or p_mid[1] >= h-1: is_clamped = True
                if t_mid[0] <= 0 or t_mid[0] >= w-1 or t_mid[1] <= 0 or t_mid[1] >= h-1: is_clamped = True

                cv2.circle(arr_bgr, p_mid, 6, (0, 0, 255), -1)
                cv2.circle(arr_bgr, t_mid, 6, (0, 0, 255), -1)
                cv2.arrowedLine(arr_bgr, p_mid, t_mid, (0, 255, 0), 3)
                
                pt1 = clamp((m["plug_pts"][0][0], m["plug_pts"][0][1]))
                pt2 = clamp((m["plug_pts"][1][0], m["plug_pts"][1][1]))
                cv2.line(arr_bgr, pt1, pt2, (0, 0, 255), 3)

                pt3 = clamp((m["port_pts"][0][0], m["port_pts"][0][1]))
                pt4 = clamp((m["port_pts"][1][0], m["port_pts"][1][1]))
                cv2.line(arr_bgr, pt3, pt4, (0, 0, 255), 3)

                port_a = self._target_port_anchor[cam]
                plug_a = self._plug_anchor[cam]
                if port_a:
                    b = port_a["bbox_xyxy"]
                    p1 = clamp((b[0], b[1]))
                    p2 = clamp((b[2], b[3]))
                    cv2.rectangle(arr_bgr, p1, p2, (255, 0, 0), 3)
                if plug_a:
                    b = plug_a["bbox_xyxy"]
                    p1 = clamp((b[0], b[1]))
                    p2 = clamp((b[2], b[3]))
                    cv2.rectangle(arr_bgr, p1, p2, (0, 165, 255), 3)

                port_name = port_a["instance_name"] if port_a else "None"
                plug_name = plug_a["instance_name"] if plug_a else "None"
                pmiss = port_a["missed_count"] if port_a else 0
                lmiss = plug_a["missed_count"] if plug_a else 0

                text = f"{cam} | port: {port_name} (m:{pmiss}) | plug: {plug_name} (m:{lmiss}) | edge: {m['plug_edge']}-{m['port_edge']} | err_uv: ({m['err_uv'][0]:.1f}, {m['err_uv'][1]:.1f}) | err_px: {m['err_px']:.1f}"
                if is_clamped:
                    text += " [CLAMPED]"
                cv2.putText(arr_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                out_msg = Image()
                out_msg.header = img_msg.header
                out_msg.height = img_msg.height
                out_msg.width = img_msg.width
                out_msg.encoding = "bgr8"
                out_msg.step = img_msg.width * 3
                out_msg.data = arr_bgr.tobytes()
                
                pub.publish(out_msg)
                self.get_logger().info(f"HYBRID_EDGE_DEBUG_PUB camera={cam} topic=/hybrid/debug/{cam}_edge_match_image")

                if self.debug_edge_overlay_save_dir and self.debug_edge_overlay_save_every > 0:
                    if self._debug_save_count % self.debug_edge_overlay_save_every == 0:
                        import os
                        os.makedirs(self.debug_edge_overlay_save_dir, exist_ok=True)
                        p = os.path.join(self.debug_edge_overlay_save_dir, f"{cam}_{self._debug_save_count:06d}.png")
                        cv2.imwrite(p, arr_bgr)
            if self.debug_edge_overlay_save_dir and self.debug_edge_overlay_save_every > 0:
                self._debug_save_count += 1
        except Exception as e:
            self.get_logger().warning(f"Failed to publish debug overlay: {e}")

run_hybrid = RunHybrid
