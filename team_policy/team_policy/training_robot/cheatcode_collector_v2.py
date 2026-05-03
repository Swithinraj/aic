"""
Autonomous data collection policy — v2 (Schema v9).

Differences from cheatcode_collector.py (v1):
  * Subscribes to per-camera YOLO topics instead of only /fused_yolo/detections_json:
        /left_camera/yolo/detections_json
        /center_camera/yolo/detections_json
        /right_camera/yolo/detections_json
    (still subscribes to /fused_yolo/detections_json for the fused port_xyz hint)
  * Builds a 7D per-camera feature vector per frame:
        [confidence, bbox_cx_norm, bbox_cy_norm, bbox_w_norm, bbox_h_norm,
         valid_float, age_seconds]
  * Records tared_wrist_force_torque (6D), raw port_delta_tcp (3D), plug_type_onehot (2D),
    target_module_onehot (7D), and fused YOLO freshness/staleness
  * Uses EpisodeRecorderV2 which writes observations/yolo_per_camera/{left,center,right}
  * Image dimensions are read from the first observation frame for bbox normalisation

Usage (after pixi shell):
    # Terminal 1 — sim with ground truth
    distrobox enter -r aic_eval -- /entrypoint.sh \\
        ground_truth:=true start_aic_engine:=true gazebo_gui:=false

    # Terminal 2 — collector (change run_NNN each session; episodes go to Seagate)
    export SEAGATE=/media/$USER/seagate/aic_episodes
    ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=team_policy.training_robot.cheatcode_collector_v2 \\
        -p output_dir:=$SEAGATE/run_001 \\
        -p num_episodes:=3
"""
from __future__ import annotations

import json
import pathlib
import threading
import time
from typing import Dict, Optional

import numpy as np
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException

from team_policy.training_robot.episode_recorder_v2 import (  # type: ignore[import-unresolved]
    AGE_VALID_S,
    DEFAULT_MAX_FINAL_ERROR_M,
    DEFAULT_MAX_SUSTAINED_FORCE_DURATION_S,
    MAX_AGE_S,
    EpisodeRecorderV2,
    build_yolo_feature,
)

# Image dimensions of the three cameras (H × W).
# Must match the actual sensor resolution stored in HDF5.
_CAM_H = 1024
_CAM_W = 1152

_CAMERAS = ("left", "center", "right")
_CAM_TOPICS = {
    "left":   "/left_camera/yolo/detections_json",
    "center": "/center_camera/yolo/detections_json",
    "right":  "/right_camera/yolo/detections_json",
}


class DataCollectionPolicyV2(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        self._output_dir    = str(parent_node.declare_parameter("output_dir",   "/tmp/aic_dataset_v2").value)
        self._num_episodes  = int(parent_node.declare_parameter("num_episodes", 3).value)
        self._success_only  = bool(parent_node.declare_parameter("success_only", True).value)
        self._start_yolo_planner = bool(parent_node.declare_parameter("start_yolo_planner", True).value)
        self._max_final_error_by_port: dict = DEFAULT_MAX_FINAL_ERROR_M
        self._max_sustained_force_s: float  = DEFAULT_MAX_SUSTAINED_FORCE_DURATION_S

        self._recorder  = EpisodeRecorderV2(self._output_dir)
        self._cheatcode = CheatCode(parent_node)

        _existing = sorted(pathlib.Path(self._output_dir).glob("episode_*.hdf5"))
        self._episode_idx  = len(_existing)
        self._saved_count  = len(_existing)

        # --- Fused YOLO (port_xyz hint, kept for 30D base state) ---
        self._fused_lock       = threading.Lock()
        self._yolo_port_xyz    = np.zeros(3, dtype=np.float32)
        self._yolo_seen_target = False
        self._yolo_last_det_time: Optional[float] = None

        # --- Per-camera YOLO state ---
        self._cam_lock: threading.Lock = threading.Lock()
        # last valid detection feature per camera
        self._cam_feat: Dict[str, np.ndarray] = {
            cam: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, MAX_AGE_S], dtype=np.float32)
            for cam in _CAMERAS
        }
        self._cam_last_det_time: Dict[str, Optional[float]] = {cam: None for cam in _CAMERAS}
        # last raw detection (conf, bbox_xyxy) for building features at record time
        self._cam_last_conf: Dict[str, float] = {cam: 0.0 for cam in _CAMERAS}
        self._cam_last_bbox: Dict[str, Optional[list]] = {cam: None for cam in _CAMERAS}

        # task context (set per-episode, guards port matching)
        self._current_port_name:  str = ""
        self._current_port_type:  str = ""
        self._current_module_name: str = ""

        # insertion event from /scoring/insertion_event
        self._insertion_lock = threading.Lock()
        self._insertion_event_received: bool = False
        self._insertion_event_data: str = ""

        # dynamic image dims (updated from first observation to avoid hardcoding)
        self._img_h = _CAM_H
        self._img_w = _CAM_W

        parent_node.create_subscription(
            String, "/fused_yolo/detections_json", self._cb_fused_yolo, 10
        )
        parent_node.create_subscription(
            String, "/scoring/insertion_event", self._cb_insertion_event, 10
        )
        for cam in _CAMERAS:
            topic = _CAM_TOPICS[cam]
            parent_node.create_subscription(
                String, topic,
                lambda msg, c=cam: self._cb_per_cam_yolo(msg, c),
                10,
            )

        self._embedded_yolo_node     = None
        self._embedded_yolo_executor = None
        self._embedded_yolo_thread   = None
        if self._start_yolo_planner:
            self._start_embedded_yolo_planner()

        self.get_logger().info(
            f"DataCollectionPolicyV2 ready — "
            f"target={self._num_episodes} eps, "
            f"output={self._output_dir}, "
            f"schema=v9 (per-camera YOLO + tared force/torque + plug type + target module + fresh fused YOLO validity)"
        )

    # ------------------------------------------------------------------
    # Embedded YOLO planner
    # ------------------------------------------------------------------

    def _start_embedded_yolo_planner(self) -> None:
        from team_policy.planner.combined_yolo_depth_pose_planner import (
            CombinedYoloDepthPosePlanner,
        )
        self._embedded_yolo_node     = CombinedYoloDepthPosePlanner()
        self._embedded_yolo_executor = MultiThreadedExecutor(num_threads=2)
        self._embedded_yolo_executor.add_node(self._embedded_yolo_node)

        def spin_yolo() -> None:
            try:
                self._embedded_yolo_executor.spin()
            except ExternalShutdownException:
                pass
            except Exception as exc:
                self.get_logger().error(f"Embedded YOLO planner stopped: {exc}")

        self._embedded_yolo_thread = threading.Thread(
            target=spin_yolo, name="embedded_yolo_planner", daemon=True
        )
        self._embedded_yolo_thread.start()
        self.get_logger().info("Embedded YOLO planner started")

    def shutdown(self) -> None:
        if self._embedded_yolo_executor is not None:
            self._embedded_yolo_executor.shutdown()
        if self._embedded_yolo_thread is not None:
            self._embedded_yolo_thread.join(timeout=2.0)
        if self._embedded_yolo_node is not None:
            self._embedded_yolo_node.destroy_node()
        self._embedded_yolo_node     = None
        self._embedded_yolo_executor = None
        self._embedded_yolo_thread   = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # YOLO callbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_name(value: object) -> str:
        return str(value).strip().lower()

    def _target_match_rank(
        self, det: dict, target_type: str, target_port: str, target_module: str
    ) -> Optional[int]:
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
        if target_type == "sc":
            if any(name == "sc_port" or name.startswith("sc_port_") for name in names):
                return 1
        if target_type == "sfp":
            if any(name == "sfp_port" or name.startswith("sfp_port_") for name in names):
                return 1
        if target_port and any(target_port in name or name in target_port for name in names):
            return 2
        return None

    def _cb_fused_yolo(self, msg: String) -> None:
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        with self._fused_lock:
            target_port   = self._current_port_name
            target_type   = self._current_port_type
            target_module = self._current_module_name

        best_rank = None
        best_conf = float("-inf")
        best_xyz  = None
        for det in dets:
            rank = self._target_match_rank(det, target_type, target_port, target_module)
            if rank is None:
                continue
            pos = det.get("pose_base_link", {}).get("position", {})
            if not pos:
                continue
            xyz = np.array([
                float(pos.get("x", 0.0)),
                float(pos.get("y", 0.0)),
                float(pos.get("z", 0.0)),
            ], dtype=np.float32)
            conf = float(det.get("confidence", 0.0))
            if best_rank is None or rank < best_rank or (rank == best_rank and conf > best_conf):
                best_rank = rank
                best_conf = conf
                best_xyz  = xyz
        if best_xyz is not None:
            with self._fused_lock:
                self._yolo_port_xyz   = best_xyz
                self._yolo_seen_target = True
                self._yolo_last_det_time = time.time()

    def _cb_per_cam_yolo(self, msg: String, cam: str) -> None:
        """Parse per-camera detections_json and cache the best matching bbox."""
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(dets, list):
            return

        with self._fused_lock:
            target_port   = self._current_port_name
            target_type   = self._current_port_type
            target_module = self._current_module_name

        best_rank = None
        best_conf = float("-inf")
        best_bbox: Optional[list] = None

        for det in dets:
            if not isinstance(det, dict):
                continue
            # per-camera JSON has class_name but no instance_name
            rank = self._target_match_rank(det, target_type, target_port, target_module)
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

    def _cb_insertion_event(self, msg: String) -> None:
        with self._insertion_lock:
            self._insertion_event_received = True
            self._insertion_event_data = msg.data
        self.get_logger().info(f"Insertion event received: {msg.data}")

    # ------------------------------------------------------------------
    # Per-camera feature snapshot (called at record time)
    # ------------------------------------------------------------------

    def _snapshot_per_camera_features(self) -> Dict[str, np.ndarray]:
        now = time.time()
        feats: Dict[str, np.ndarray] = {}
        with self._cam_lock:
            for cam in _CAMERAS:
                feats[cam] = build_yolo_feature(
                    confidence   = self._cam_last_conf[cam],
                    bbox_xyxy    = self._cam_last_bbox[cam],
                    img_h        = self._img_h,
                    img_w        = self._img_w,
                    last_det_time= self._cam_last_det_time[cam],
                    now          = now,
                )
        return feats

    # ------------------------------------------------------------------
    # TF helpers (unchanged from v1)
    # ------------------------------------------------------------------

    @staticmethod
    def _task_frames(task: Task) -> tuple[str, str, str]:
        port_frame       = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame       = f"{task.cable_name}/{task.plug_name}_link"
        task_board_frame = "task_board/task_board_base_link"
        return port_frame, plug_frame, task_board_frame

    @staticmethod
    def _format_tf_pair(target_frame: str, source_frame: str) -> str:
        return f"{target_frame}<-{source_frame}"

    def _lookup_pose_array(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        try:
            tf_msg = self._parent_node._tf_buffer.lookup_transform(
                target_frame, source_frame, Time()
            ).transform
        except TransformException:
            return None
        return np.array([
            tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z,
            tf_msg.rotation.x, tf_msg.rotation.y, tf_msg.rotation.z, tf_msg.rotation.w,
        ], dtype=np.float32)

    def _relative_pose_plug_to_target(self, task: Task) -> Optional[np.ndarray]:
        port_frame, plug_frame, _ = self._task_frames(task)
        return self._lookup_pose_array(plug_frame, port_frame)

    def _privileged_tf_pairs(self, task: Task) -> list[tuple[str, str]]:
        port_frame, plug_frame, task_board_frame = self._task_frames(task)
        return [
            ("base_link", task_board_frame),
            ("base_link", port_frame),
            ("base_link", plug_frame),
            ("base_link", "gripper/tcp"),
            (plug_frame, port_frame),
        ]

    def _privileged_tf_snapshot(
        self, frame_pairs: list[tuple[str, str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        transforms = np.zeros((len(frame_pairs), 7), dtype=np.float32)
        valid      = np.zeros(len(frame_pairs), dtype=np.bool_)
        for idx, (target, source) in enumerate(frame_pairs):
            pose = self._lookup_pose_array(target, source)
            if pose is not None:
                transforms[idx] = pose
                valid[idx] = True
        return transforms, valid

    # ------------------------------------------------------------------
    # Policy entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:

        if self._episode_idx >= self._num_episodes:
            self.get_logger().info(
                f"Reached target of {self._num_episodes} episodes. Exiting gracefully."
            )
            return True

        episode_id = self._episode_idx
        self._episode_idx += 1

        port_type          = str(getattr(task, "port_type",          "unknown"))
        port_name          = str(getattr(task, "port_name",          "unknown"))
        cable_name         = str(getattr(task, "cable_name",         "unknown"))
        plug_name          = str(getattr(task, "plug_name",          "unknown"))
        cable_type         = str(getattr(task, "cable_type",         "unknown"))
        plug_type          = str(getattr(task, "plug_type",          "unknown"))
        target_module_name = str(getattr(task, "target_module_name", "unknown"))
        time_limit_s       = int(getattr(task, "time_limit",         0))
        task_id            = str(getattr(task, "id", str(episode_id)))

        send_feedback(
            f"collector_v2/episode={episode_id} port={port_type}/{port_name}"
        )

        # Reset per-episode state
        with self._fused_lock:
            self._current_port_name   = port_name.strip().lower()
            self._current_port_type   = port_type.strip().lower()
            self._current_module_name = target_module_name.strip().lower()
            self._yolo_port_xyz       = np.zeros(3, dtype=np.float32)
            self._yolo_seen_target    = False
            self._yolo_last_det_time  = None
        with self._cam_lock:
            for cam in _CAMERAS:
                self._cam_last_det_time[cam] = None
                self._cam_last_conf[cam]     = 0.0
                self._cam_last_bbox[cam]     = None
        with self._insertion_lock:
            self._insertion_event_received = False
            self._insertion_event_data     = ""

        self._recorder.start_episode(
            episode_id,
            task_id,
            port_type,
            port_name,
            cable_type=cable_type,
            cable_name=cable_name,
            plug_type=plug_type,
            plug_name=plug_name,
            target_module_name=target_module_name,
            time_limit_s=time_limit_s,
        )
        tf_pairs = self._privileged_tf_pairs(task)
        self._recorder.set_privileged_tf_frame_pairs(
            [self._format_tf_pair(t, s) for t, s in tf_pairs]
        )

        # Wrap move_robot to capture commanded pose and inject impedance compliance
        _WRENCH_COMPLIANCE = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        def recording_move_robot(motion_update=None, joint_motion_update=None):
            if motion_update is not None:
                motion_update.wrench_feedback_gains_at_tip = _WRENCH_COMPLIANCE
                p = getattr(motion_update, "pose", None)
                if p is not None:
                    pos = p.position
                    ori = p.orientation
                    self._recorder.update_commanded_pose(np.array(
                        [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w],
                        dtype=np.float32,
                    ))
                move_robot(motion_update=motion_update)
            elif joint_motion_update is not None:
                move_robot(joint_motion_update=joint_motion_update)

        # Measure resting force baseline (10 attempts, first success wins)
        force_baseline_n = 0.0
        tare_wrench = np.zeros(6, dtype=np.float32)
        for _ in range(10):
            obs0 = get_observation()
            if obs0 is not None:
                wrench = obs0.wrist_wrench.wrench
                w = wrench.force
                tare_wrench = np.array([
                    wrench.force.x, wrench.force.y, wrench.force.z,
                    wrench.torque.x, wrench.torque.y, wrench.torque.z,
                ], dtype=np.float32)
                force_baseline_n = float(np.sqrt(w.x*w.x + w.y*w.y + w.z*w.z))
                self._recorder.set_wrist_force_tare(tare_wrench)
                # Update image dims from first observation
                self._img_h = obs0.left_image.height or _CAM_H
                self._img_w = obs0.left_image.width  or _CAM_W
                break
            time.sleep(0.05)
        self.get_logger().info(
            f"collector_v2/episode={episode_id} "
            f"force_baseline={force_baseline_n:.1f}N "
            f"tare={[round(float(v), 3) for v in tare_wrench.tolist()]} "
            f"plug={plug_type} "
            f"img={self._img_h}x{self._img_w}"
        )

        # Background recording thread at ~10 Hz
        recording_active = threading.Event()
        recording_active.set()

        def obs_loop():
            _step = 0.10
            while recording_active.is_set():
                t0  = time.time()
                obs = get_observation()
                if obs is not None:
                    relative_pose = self._relative_pose_plug_to_target(task)
                    priv_tf, priv_tf_valid = self._privileged_tf_snapshot(tf_pairs)
                    with self._fused_lock:
                        yolo_xyz = self._yolo_port_xyz.copy()
                        yolo_seen = self._yolo_seen_target
                        last_det_time = self._yolo_last_det_time
                    if yolo_seen and last_det_time is not None:
                        yolo_age = min(MAX_AGE_S, time.time() - last_det_time)
                        yolo_valid = yolo_age < AGE_VALID_S
                    else:
                        yolo_age = MAX_AGE_S
                        yolo_valid = False
                    yolo_per_camera = self._snapshot_per_camera_features()
                    with self._insertion_lock:
                        ins_success = float(self._insertion_event_received)
                    self._recorder.record_frame(
                        obs,
                        relative_pose=relative_pose,
                        privileged_tf=priv_tf,
                        privileged_tf_valid=priv_tf_valid,
                        yolo_port_xyz=yolo_xyz if yolo_seen else None,
                        yolo_port_valid=yolo_valid,
                        yolo_port_age=yolo_age,
                        yolo_per_camera=yolo_per_camera,
                        insertion_success=ins_success,
                    )
                time.sleep(max(0.0, _step - (time.time() - t0)))

        obs_thread = threading.Thread(target=obs_loop, daemon=True)
        obs_thread.start()

        try:
            success = self._cheatcode.insert_cable(
                task=task,
                get_observation=get_observation,
                move_robot=recording_move_robot,
                send_feedback=send_feedback,
            )
        except Exception as exc:
            self.get_logger().error(f"CheatCode raised: {exc}")
            success = False
        finally:
            recording_active.clear()
            obs_thread.join(timeout=2.0)

        if self._success_only and not success:
            self._recorder.end_episode(success=False, force_baseline_n=force_baseline_n)
            send_feedback(f"collector_v2/episode={episode_id} FAILED — discarded")
        else:
            max_err = self._max_final_error_by_port.get("default")
            for key, threshold in self._max_final_error_by_port.items():
                if key != "default" and port_name.startswith(key):
                    max_err = threshold
                    break

            with self._insertion_lock:
                insertion_event_data = self._insertion_event_data
            path = self._recorder.end_episode(
                success=success,
                max_final_error_m=max_err,
                max_sustained_force_duration_s=self._max_sustained_force_s,
                force_baseline_n=force_baseline_n,
                insertion_event_data=insertion_event_data,
            )
            if path:
                self._saved_count += 1
                self.get_logger().info(
                    f"[{self._saved_count}/{self._num_episodes}] Saved {path}"
                )
                send_feedback(
                    f"collector_v2/saved episode={episode_id} success={success} "
                    f"total_saved={self._saved_count} path={path}"
                )
            else:
                send_feedback(
                    f"collector_v2/episode={episode_id} rejected by quality gate"
                )

        return success


# aic_model expects the class name to match the last segment of the module path.
# Usage: -p policy:=team_policy.training_robot.cheatcode_collector_v2
cheatcode_collector_v2 = DataCollectionPolicyV2


def main():
    raise SystemExit(
        "DataCollectionPolicyV2 is not a standalone node.\n"
        "Run via aic_model:\n"
        "  pixi run ros2 run aic_model aic_model --ros-args \\\n"
        "    -p use_sim_time:=true \\\n"
        "    -p policy:=team_policy.training_robot.cheatcode_collector_v2 \\\n"
        "    -p output_dir:=/media/$USER/seagate/aic_episodes/run_001"
    )
