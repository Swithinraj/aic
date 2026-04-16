from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, TransformStamped
from std_msgs.msg import ColorRGBA, String
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo

from team_policy.perception.yolov12_detector import YoloV12MultiCameraDetector
from team_policy.perception.cad_depth_registration import CadDepthPoseEstimator


class CombinedYoloDepthPosePlanner(YoloV12MultiCameraDetector):
    """
    YOLO + metric depth + CAD-registration pose pipeline.

    - Uses the existing YOLO detections and naming/rail logic from YoloV12MultiCameraDetector.
    - Uses metric depth crops from /<camera>_camera/stereo_depth/image.
    - Estimates an initial pose from the depth crop point cloud.
    - Refines with ICP against the AIC CAD assets when a mesh is available.
    - Publishes full 6D TFs and pose_base_link JSON for mypolicy.
    """

    def __init__(self):
        super().__init__()

        self._base_frame = "base_link"
        self._tf_camera = self._env_text("YOLO_POSE_TF_CAMERA", "center")
        self._axis_length_m = self._env_float("YOLO_DEPTH_POSE_AXIS_LENGTH_M", 0.05)
        self._axis_width_m = self._env_float("YOLO_DEPTH_POSE_AXIS_WIDTH_M", 0.004)
        self._text_scale = self._env_float("YOLO_DEPTH_POSE_TEXT_SCALE", 0.03)
        self._board_width_m = self._env_float("YOLO_DEPTH_POSE_BOARD_WIDTH_M", 0.300)
        self._board_height_m = self._env_float("YOLO_DEPTH_POSE_BOARD_HEIGHT_M", 0.425)
        self._depth_fallback_m = self._env_float("YOLO_DEPTH_POSE_FALLBACK_M", 0.30)
        self._assets_models_dir = self._env_path("YOLO_DEPTH_POSE_AIC_ASSETS_MODELS_DIR", "")
        self._preferred_camera_for_fusion = self._tf_camera
        self._force_table_flat = self._env_bool("YOLO_POSE_FORCE_TABLE_FLAT", True)
        self._table_normal_base = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._table_constrained_families = {"taskboard", "nic", "sc", "sfp_port", "sfp_module"}

        self._tf_broadcaster = TransformBroadcaster(self)
        self._fused_pose_pub = self.create_publisher(PoseArray, "/fused_yolo/poses_base_link", 10)
        self._fused_json_pub = self.create_publisher(String, "/fused_yolo/detections_json", 10)
        self._fused_marker_pub = self.create_publisher(MarkerArray, "/fused_yolo/pose_markers", 10)

        self._latest_instance_obs = {}
        self._latest_depth_images: Dict[str, Optional[np.ndarray]] = {"left": None, "center": None, "right": None}
        self._latest_depth_stamps: Dict[str, float] = {"left": 0.0, "center": 0.0, "right": 0.0}

        self._depth_subs = {
            "left": self.create_subscription(Image, "/left_camera/stereo_depth/image", lambda msg: self._depth_cb("left", msg), 10),
            "center": self.create_subscription(Image, "/center_camera/stereo_depth/image", lambda msg: self._depth_cb("center", msg), 10),
            "right": self.create_subscription(Image, "/right_camera/stereo_depth/image", lambda msg: self._depth_cb("right", msg), 10),
        }
        self._tf_sub = self.create_subscription(TFMessage, "/tf", self._tf_callback, 100)

        self._cad_estimator = CadDepthPoseEstimator(
            assets_models_dir=self._assets_models_dir or None,
            logger=lambda msg: self.get_logger().info(msg),
            voxel_size_m=self._env_float("YOLO_DEPTH_POSE_VOXEL_M", 0.004),
            min_points=max(40, int(self._env_float("YOLO_DEPTH_POSE_MIN_POINTS", 60))),
        )

        self.get_logger().info(
            f"Combined planner node started: YOLO + metric depth + CAD registration, TF camera={self._tf_camera}, base_frame={self._base_frame}"
        )

    def _env_float(self, key: str, default: float) -> float:
        import os
        try:
            return float(os.environ.get(key, str(default)))
        except Exception:
            return float(default)

    def _env_text(self, key: str, default: str) -> str:
        import os
        value = str(os.environ.get(key, default)).strip().lower()
        return value if value in {"left", "center", "right"} else str(default)

    def _env_bool(self, key: str, default: bool) -> bool:
        import os
        value = str(os.environ.get(key, "1" if default else "0")).strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _env_path(self, key: str, default: str) -> str:
        import os
        return str(os.environ.get(key, default)).strip()

    def _depth_cb(self, camera_name: str, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            depth = np.asarray(depth, dtype=np.float32)
        except Exception as exc:
            self.get_logger().warn(f"Depth decode failed for {camera_name}: {exc}")
            return
        with self._lock:
            self._latest_depth_images[camera_name] = depth
            self._latest_depth_stamps[camera_name] = self._stamp_to_sec(msg.header.stamp)

    def _get_latest_depth_copy(self, camera_name: str) -> Optional[np.ndarray]:
        with self._lock:
            depth = self._latest_depth_images.get(camera_name)
            return None if depth is None else depth.copy()

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            frames = dict(self._latest_frames)

        camera_results = {}
        for camera_name, msg in frames.items():
            if msg is None:
                continue
            if now - self._last_infer_time[camera_name] < self._min_period:
                continue
            try:
                result = self._run_inference_combined(camera_name, msg)
            except Exception as exc:
                self.get_logger().error(f"{camera_name} inference failed: {exc}")
                continue
            self._last_infer_time[camera_name] = now
            camera_results[camera_name] = {"msg": msg, "result": result}

        if not camera_results:
            return

        instance_obs = {}
        for camera_name, res in camera_results.items():
            msg = res["msg"]
            annotated, detections, classes, mask_overlay, mask_binary, mask_status, _, _, _ = res["result"]

            annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self._annotated_pubs[camera_name].publish(annotated_msg)

            json_msg = String()
            json_msg.data = json.dumps(detections, separators=(",", ":"))
            self._json_pubs[camera_name].publish(json_msg)

            classes_msg = String()
            classes_msg.data = ",".join(classes)
            self._classes_pubs[camera_name].publish(classes_msg)

            overlay_msg = self._bridge.cv2_to_imgmsg(mask_overlay, encoding="bgr8")
            overlay_msg.header = msg.header
            self._mask_overlay_pubs[camera_name].publish(overlay_msg)

            mask_msg = self._bridge.cv2_to_imgmsg(mask_binary, encoding="mono8")
            mask_msg.header = msg.header
            self._mask_binary_pubs[camera_name].publish(mask_msg)

            status_msg = String()
            status_msg.data = mask_status
            self._mask_status_pubs[camera_name].publish(status_msg)

            info = self._latest_infos.get(camera_name)
            if info is None:
                continue
            camera_frame_id = str(info.header.frame_id)
            for det in detections:
                name = det.get("instance_name", det.get("class_name", ""))
                if not name:
                    continue
                pose_cam = det.get("pose_camera")
                if not (isinstance(pose_cam, dict) and "t" in pose_cam and "q" in pose_cam):
                    continue
                family = self._detection_family_name(det)
                anchor_key = "NIC_CARD" if family == "nic" else "SC_PORT" if family == "sc" else "GENERIC"
                anchor = self._detection_anchor_point(det, anchor_key)
                anchor_uv = [float(anchor[0]), float(anchor[1])]
                instance_obs.setdefault(name, []).append(
                    {
                        "det": det,
                        "camera_frame_id": camera_frame_id,
                        "anchor_uv": anchor_uv,
                        "camera_name": camera_name,
                    }
                )

        with self._lock:
            self._latest_instance_obs = instance_obs

    def _tf_callback(self, msg: TFMessage) -> None:
        with self._lock:
            frames = dict(self._latest_frames)
            cached_obs = dict(getattr(self, "_latest_instance_obs", {}))

        instance_obs = cached_obs if cached_obs else self._run_fresh_detections(frames)
        if not instance_obs:
            return

        stamp = self.get_clock().now().to_msg()
        fused_records = []
        used_tf_names = set()

        for name, obs_list in instance_obs.items():
            best_obj = self._select_best_observation(obs_list)
            if best_obj is None:
                continue
            best_det = best_obj["det"]
            camera_frame_id = best_obj["camera_frame_id"]
            pose_cam = best_det.get("pose_camera")
            if not pose_cam:
                continue

            pose_base = self._transform_pose_dict(pose_cam, camera_frame_id, self._base_frame)
            if pose_base is None:
                continue
            pose_base = self._maybe_constrain_pose_to_table_plane(best_det, pose_base)

            display_name = str(name)
            tf_base = "det_" + self._tf_safe_name(display_name)
            tf_name = tf_base
            index = 1
            while tf_name in used_tf_names:
                tf_name = f"{tf_base}_{index}"
                index += 1
            used_tf_names.add(tf_name)

            fused_records.append(
                {
                    "name": display_name,
                    "tf_name": tf_name,
                    "bbox_xyxy": [float(v) for v in best_det.get("bbox_xyxy", [])],
                    "base_pose": self._pose_dict_to_msg(pose_base),
                    "class_name": str(best_det.get("class_name", "")),
                    "confidence": float(best_det.get("confidence", 0.0)),
                    "camera": best_obj.get("camera_name", "center"),
                    "anchor_uv": best_obj.get("anchor_uv", []),
                    "camera_frame_id": camera_frame_id,
                    "camera_name": best_obj.get("camera_name", "center"),
                    "pose_source": str(best_det.get("pose_source", "depth_registration")),
                }
            )

        self._publish_fused_records(stamp, fused_records)

    def _select_best_observation(self, obs_list: List[Dict]) -> Optional[Dict]:
        if not obs_list:
            return None
        preferred = [o for o in obs_list if o.get("camera_name") == self._preferred_camera_for_fusion]
        candidates = preferred if preferred else obs_list
        return max(candidates, key=lambda o: float(o["det"].get("confidence", 0.0)))

    def _run_fresh_detections(self, frames: Dict) -> Dict:
        instance_obs = {}
        for camera_name, msg in frames.items():
            if msg is None:
                continue
            try:
                result = self._run_inference_combined(camera_name, msg)
            except Exception:
                continue
            _, detections, _, _, _, _, _, _, _ = result
            with self._lock:
                info = self._latest_infos.get(camera_name)
            if info is None:
                continue
            camera_frame_id = str(info.header.frame_id)
            for det in detections:
                name = det.get("instance_name", det.get("class_name", ""))
                if not name:
                    continue
                pose_cam = det.get("pose_camera")
                if not (isinstance(pose_cam, dict) and "t" in pose_cam and "q" in pose_cam):
                    continue
                family = self._detection_family_name(det)
                anchor_key = "NIC_CARD" if family == "nic" else "SC_PORT" if family == "sc" else "GENERIC"
                anchor = self._detection_anchor_point(det, anchor_key)
                instance_obs.setdefault(name, []).append(
                    {
                        "det": det,
                        "camera_frame_id": camera_frame_id,
                        "anchor_uv": [float(anchor[0]), float(anchor[1])],
                        "camera_name": camera_name,
                    }
                )
        return instance_obs

    def _publish_fused_records(self, stamp, fused_records: List[Dict]):
        fused_pose_array = PoseArray()
        fused_pose_array.header.stamp = stamp
        fused_pose_array.header.frame_id = self._base_frame

        fused_markers = MarkerArray()
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        fused_markers.markers.append(del_m)

        fused_tf_list = []
        fused_detections_json = []
        marker_id = 0

        for rec in fused_records:
            base_pose = rec["base_pose"]
            fused_pose_array.poses.append(base_pose)
            fused_tf_list.append(self._pose_to_transform(stamp, self._base_frame, rec["tf_name"], base_pose))

            axes = self._make_axes_marker(marker_id, stamp, rec["name"], base_pose)
            fused_markers.markers.append(axes)
            marker_id += 1

            text = Marker()
            text.header.frame_id = self._base_frame
            text.header.stamp = stamp
            text.ns = "combined_detection_pose_text"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = base_pose
            text.pose.position.z += 0.04
            text.scale.z = self._text_scale
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            bp = base_pose.position
            text.text = f"{rec['name']} x={bp.x:.3f} y={bp.y:.3f} z={bp.z:.3f}"
            fused_markers.markers.append(text)
            marker_id += 1

            fused_detections_json.append(
                {
                    "class_name": rec["class_name"],
                    "instance_name": rec["name"],
                    "confidence": rec["confidence"],
                    "bbox_xyxy": rec["bbox_xyxy"],
                    "tf_frame": rec["tf_name"],
                    "source": rec["camera"],
                    "anchor_uv": rec.get("anchor_uv", []),
                    "camera_frame_id": rec.get("camera_frame_id", ""),
                    "camera_name": rec.get("camera_name", ""),
                    "pose_source": rec.get("pose_source", "depth_registration"),
                    "pose_base_link": {
                        "position": {
                            "x": float(bp.x),
                            "y": float(bp.y),
                            "z": float(bp.z),
                        },
                        "orientation": {
                            "x": float(base_pose.orientation.x),
                            "y": float(base_pose.orientation.y),
                            "z": float(base_pose.orientation.z),
                            "w": float(base_pose.orientation.w),
                        },
                    },
                }
            )

        self._fused_pose_pub.publish(fused_pose_array)
        self._fused_marker_pub.publish(fused_markers)
        if fused_tf_list:
            self._tf_broadcaster.sendTransform(fused_tf_list)

        jd = String()
        jd.data = json.dumps(fused_detections_json, separators=(",", ":"))
        self._fused_json_pub.publish(jd)

    def _run_inference_combined(self, camera_name: str, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self._model.predict(
            source=frame,
            device=self._device,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )
        result = results[0]
        names = result.names if hasattr(result, "names") else self._model.names

        detections: List[Dict] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((0,), dtype=np.float32)
            clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((0,), dtype=int)
            for box, conf, cls_idx in zip(xyxy, confs, clss):
                cls_name = str(names[int(cls_idx)]) if names is not None else str(int(cls_idx))
                detections.append(
                    {
                        "class_id": int(cls_idx),
                        "class_name": cls_name,
                        "confidence": float(conf),
                        "bbox_xyxy": [float(v) for v in box.tolist()],
                    }
                )

        detections = self._filter_target_detections(detections)
        mask_overlay, mask_binary, mask_status, projected_rails, fused_targets = self._fit_rails(camera_name, frame, detections, msg)
        detections = self._assign_detection_instance_names(detections, projected_rails, fused_targets, camera_name)
        detections = self._sort_output_detections(detections)

        pose_records = self._estimate_detection_poses(camera_name, detections)
        detections = self._attach_pose_data_to_detections(detections, pose_records)

        annotated = self._draw_filtered_detections(frame, detections)
        annotated = self._draw_instance_labels(annotated, detections)

        classes = [str(det.get("instance_name", det.get("class_name", ""))) for det in detections]
        return annotated, detections, classes, mask_overlay, mask_binary, mask_status, None, None, None

    def _estimate_detection_poses(self, camera_name: str, detections: List[Dict]) -> List[Dict]:
        with self._lock:
            info = self._latest_infos.get(camera_name)
            ctx = self._assignment_context.get(camera_name)
        depth = self._get_latest_depth_copy(camera_name)
        if info is None:
            return []

        observed_board_quad = None
        board_pose_camera = None
        if ctx is not None:
            observed_board_quad = ctx.get("observed_board_quad")
            if observed_board_quad is not None:
                observed_board_quad = np.asarray(observed_board_quad, dtype=np.float32).reshape(4, 2)

        records: List[Dict] = []
        used_tf_names = set()

        taskboard_det = self._find_best_detection(detections, self._taskboard_classes)
        if taskboard_det is not None:
            board_pose_camera = self._estimate_board_pose_geometry(info, depth, taskboard_det, observed_board_quad)
            if board_pose_camera is None and observed_board_quad is not None:
                board_pose_camera = self._estimate_board_pose_from_quad(info, observed_board_quad)

            if board_pose_camera is not None:
                pose = dict(board_pose_camera)
                pose["R"] = self._orthonormalize_rotation(np.asarray(pose["R"], dtype=np.float32))
                pose["q"] = self._matrix_to_quaternion(pose["R"])
                pose["source"] = str(pose.get("source", "board_depth_plane"))
                records.append(self._make_pose_record(taskboard_det, pose, used_tf_names))

        for det in detections:
            if taskboard_det is det:
                continue
            family = self._detection_family_name(det)
            raw_pose = None
            if depth is not None:
                mesh_family = family if family in {"nic", "sc", "sfp_module"} else None
                raw_pose = self._cad_estimator.estimate_pose(info, depth, det, mesh_family, init_pose_camera=board_pose_camera)

            pose = None
            if board_pose_camera is not None:
                translation_hint = None if raw_pose is None else np.asarray(raw_pose.get("t", None), dtype=np.float32).reshape(3)
                pose = self._estimate_component_pose_on_board_plane(
                    info=info,
                    depth=depth,
                    det=det,
                    board_pose_camera=board_pose_camera,
                    raw_pose_camera=raw_pose,
                    translation_hint=translation_hint,
                )
            elif raw_pose is not None:
                pose = raw_pose
            if pose is None:
                pose = self._fallback_pose_from_depth_or_anchor(info, depth, det, board_pose_camera)
            if pose is None:
                continue
            records.append(self._make_pose_record(det, pose, used_tf_names))

        return records

    def _fallback_pose_from_depth_or_anchor(self, info: CameraInfo, depth: Optional[np.ndarray], det: Dict, init_pose_camera: Optional[Dict]) -> Optional[Dict]:
        family = self._detection_family_name(det)
        anchor_key = "NIC_CARD" if family == "nic" else "SC_PORT" if family == "sc" else "GENERIC"
        anchor = self._detection_anchor_point(det, anchor_key)
        t = self._backproject_anchor(info, depth, anchor)
        if t is None:
            return None
        R = np.eye(3, dtype=np.float32)
        if init_pose_camera is not None:
            R = np.asarray(init_pose_camera["R"], dtype=np.float32).reshape(3, 3)
        R = self._orthonormalize_rotation(R)
        return {"R": R.astype(np.float32), "t": t.astype(np.float32), "q": self._matrix_to_quaternion(R), "source": "depth_anchor_board_oriented" if init_pose_camera is not None else "depth_anchor"}

    def _backproject_anchor(self, info: CameraInfo, depth: Optional[np.ndarray], anchor_uv: np.ndarray) -> Optional[np.ndarray]:
        u = float(anchor_uv[0])
        v = float(anchor_uv[1])
        Z = self._sample_depth(depth, u, v)
        if Z is None:
            Z = float(self._depth_fallback_m)
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 1e-6 or fy <= 1e-6:
            return None
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z], dtype=np.float32)

    def _sample_depth(self, depth: Optional[np.ndarray], u: float, v: float) -> Optional[float]:
        if depth is None:
            return None
        h, w = depth.shape[:2]
        r = int(np.clip(round(v), 0, h - 1))
        c = int(np.clip(round(u), 0, w - 1))
        r0, r1 = max(0, r - 2), min(h, r + 3)
        c0, c1 = max(0, c - 2), min(w, c + 3)
        patch = np.asarray(depth[r0:r1, c0:c1], dtype=np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _make_pose_record(self, det: Dict, pose_camera: Dict, used_tf_names: set) -> Dict:
        display_name = str(det.get("instance_name", det.get("class_name", "")))
        tf_base = "det_" + self._tf_safe_name(display_name)
        tf_name = tf_base
        index = 1
        while tf_name in used_tf_names:
            tf_name = f"{tf_base}_{index}"
            index += 1
        used_tf_names.add(tf_name)
        return {
            "name": display_name,
            "tf_name": tf_name,
            "bbox_xyxy": [float(v) for v in det.get("bbox_xyxy", [])],
            "pose_camera": pose_camera,
            "pose_source": str(pose_camera.get("source", "depth_registration")),
        }

    def _attach_pose_data_to_detections(self, detections: List[Dict], pose_records: List[Dict]) -> List[Dict]:
        out = [dict(d) for d in detections]
        for det in out:
            for rec in pose_records:
                rb = rec.get("bbox_xyxy")
                db = det.get("bbox_xyxy")
                if rb is None or db is None or len(rb) != len(db):
                    continue
                if sum(abs(float(a) - float(b)) for a, b in zip(rb, db)) < 1e-3:
                    pose_cam = rec.get("pose_camera")
                    if pose_cam:
                        det["pose_camera"] = {
                            "R": [[float(x) for x in row] for row in np.asarray(pose_cam.get("R", self._quaternion_to_matrix(np.asarray(pose_cam["q"], dtype=np.float32))), dtype=np.float32).reshape(3, 3)],
                            "t": [float(v) for v in pose_cam["t"]],
                            "q": [float(v) for v in pose_cam["q"]],
                        }
                    det["tf_frame"] = rec.get("tf_name")
                    det["pose_source"] = rec.get("pose_source", "depth_registration")
                    break
        return out

    def _camera_matrix_from_info(self, info) -> np.ndarray:
        return np.array(
            [[float(info.k[0]), 0.0, float(info.k[2])], [0.0, float(info.k[4]), float(info.k[5])], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _dist_coeffs_from_info(self, info) -> np.ndarray:
        if getattr(info, "d", None) is None or len(info.d) == 0:
            return np.zeros((5, 1), dtype=np.float32)
        return np.asarray(info.d, dtype=np.float32).reshape(-1, 1)

    def _board_corner_points_xy(self) -> np.ndarray:
        half_w = 0.5 * float(self._board_width_m)
        half_h = 0.5 * float(self._board_height_m)
        return np.array([[-half_w, half_h], [half_w, half_h], [half_w, -half_h], [-half_w, -half_h]], dtype=np.float32)

    def _board_corner_points_xyz(self) -> np.ndarray:
        xy = self._board_corner_points_xy()
        return np.column_stack([xy, np.zeros((4, 1), dtype=np.float32)]).astype(np.float32)

    def _estimate_board_pose_from_quad(self, info, board_quad_image: np.ndarray) -> Optional[Dict]:
        image_points = np.asarray(board_quad_image, dtype=np.float32).reshape(4, 2)
        object_points = self._board_corner_points_xyz()
        K = self._camera_matrix_from_info(info)
        dist = self._dist_coeffs_from_info(info)
        solve_flags = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=solve_flags)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float32).reshape(3)
        R = self._orthonormalize_rotation(np.asarray(R, dtype=np.float32).reshape(3, 3))
        return {"R": R, "t": t, "q": self._matrix_to_quaternion(R), "source": "board_pnp"}

    def _estimate_board_pose_geometry(self, info: CameraInfo, depth: Optional[np.ndarray], taskboard_det: Dict, observed_board_quad: Optional[np.ndarray]) -> Optional[Dict]:
        if depth is None:
            return self._estimate_board_pose_from_quad(info, observed_board_quad) if observed_board_quad is not None else None
        pts = self._depth_points_from_region(info, depth, taskboard_det, observed_board_quad)
        if pts is None or len(pts) < 80:
            return self._estimate_board_pose_from_quad(info, observed_board_quad) if observed_board_quad is not None else None
        plane = self._fit_dominant_plane_ransac(pts)
        if plane is None:
            return self._estimate_board_pose_from_quad(info, observed_board_quad) if observed_board_quad is not None else None
        normal, plane_point, plane_pts = plane
        quad3d = None
        if observed_board_quad is not None:
            quad3d = self._lift_quad_to_plane(info, observed_board_quad, normal, plane_point)
        if quad3d is not None:
            centroid = quad3d.mean(axis=0).astype(np.float32)
            x_hint = (quad3d[1] - quad3d[0]) + (quad3d[2] - quad3d[3])
            y_hint = (quad3d[3] - quad3d[0]) + (quad3d[2] - quad3d[1])
            x_axis = self._normalize_vec(self._project_vec_to_plane(x_hint, normal))
            y_axis = self._normalize_vec(self._project_vec_to_plane(y_hint, normal))
            if x_axis is not None and y_axis is not None and float(np.dot(np.cross(x_axis, y_axis), normal)) < 0.0:
                x_axis = -x_axis
        else:
            centroid = np.median(plane_pts, axis=0).astype(np.float32)
            x_axis = self._board_x_axis_from_image(info, centroid, normal, observed_board_quad, taskboard_det)
            y_axis = None
        if x_axis is None:
            x_axis = self._board_x_axis_from_image(info, centroid, normal, observed_board_quad, taskboard_det)
        if x_axis is None:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x_axis = self._normalize_vec(self._project_vec_to_plane(x_axis, normal))
        if x_axis is None:
            return None
        y_axis = self._normalize_vec(np.cross(normal, x_axis))
        if y_axis is None:
            return None
        x_axis = self._normalize_vec(np.cross(y_axis, normal))
        if x_axis is None:
            return None
        R = self._orthonormalize_rotation(np.column_stack([x_axis, y_axis, normal]).astype(np.float32))
        return {"R": R, "t": centroid.astype(np.float32), "q": self._matrix_to_quaternion(R), "source": "board_depth_plane_ransac"}

    def _depth_points_from_region(self, info: CameraInfo, depth: np.ndarray, det: Dict, quad: Optional[np.ndarray]) -> Optional[np.ndarray]:
        h, w = depth.shape[:2]
        if quad is not None:
            quad_i = np.round(np.asarray(quad, dtype=np.float32)).astype(np.int32).reshape(-1, 2)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, quad_i, 255)
        else:
            box = det.get("bbox_xyxy", [0, 0, 0, 0])
            x1, y1, x2, y2 = [float(v) for v in box]
            bw = max(2.0, x2 - x1)
            bh = max(2.0, y2 - y1)
            ix1 = int(np.clip(np.floor(x1 + 0.05 * bw), 0, w - 1))
            iy1 = int(np.clip(np.floor(y1 + 0.05 * bh), 0, h - 1))
            ix2 = int(np.clip(np.ceil(x2 - 0.05 * bw), ix1 + 1, w))
            iy2 = int(np.clip(np.ceil(y2 - 0.05 * bh), iy1 + 1, h))
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[iy1:iy2, ix1:ix2] = 255
        z = np.asarray(depth, dtype=np.float32)
        valid = (mask > 0) & np.isfinite(z) & (z > 0.05) & (z < 3.0)
        if not np.any(valid):
            return None
        valid_z = z[valid]
        z_med = float(np.median(valid_z))
        z_mad = float(np.median(np.abs(valid_z - z_med)))
        z_band = max(0.015, 3.0 * z_mad)
        valid &= (z >= z_med - z_band) & (z <= z_med + z_band)
        ys, xs = np.where(valid)
        if xs.size < 30:
            return None
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 1e-6 or fy <= 1e-6:
            return None
        zs = z[ys, xs].astype(np.float32)
        X = (xs.astype(np.float32) - cx) * zs / fx
        Y = (ys.astype(np.float32) - cy) * zs / fy
        pts = np.column_stack([X, Y, zs]).astype(np.float32)
        centroid = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - centroid.reshape(1, 3), axis=1)
        pts = pts[dist <= np.percentile(dist, 94.0)]
        return pts.astype(np.float32) if len(pts) >= 30 else None

    def _fit_dominant_plane_ransac(self, pts: np.ndarray, iterations: int = 120, threshold_m: float = 0.008) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] < 40:
            return None
        rng = np.random.default_rng(12345)
        best_inliers = None
        best_normal = None
        best_point = None
        n_pts = pts.shape[0]
        for _ in range(iterations):
            idx = rng.choice(n_pts, size=3, replace=False)
            p0, p1, p2 = pts[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            normal = self._normalize_vec(normal)
            if normal is None:
                continue
            d = np.abs((pts - p0.reshape(1, 3)) @ normal.reshape(3, 1)).reshape(-1)
            inliers = d < threshold_m
            if best_inliers is None or int(np.count_nonzero(inliers)) > int(np.count_nonzero(best_inliers)):
                best_inliers = inliers
                best_normal = normal
                best_point = p0
        if best_inliers is None or int(np.count_nonzero(best_inliers)) < 40:
            return None
        inlier_pts = pts[best_inliers]
        center = np.median(inlier_pts, axis=0)
        X = inlier_pts - center.reshape(1, 3)
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)].astype(np.float32)
        if float(np.dot(normal, center)) > 0.0:
            normal = -normal
        normal = self._normalize_vec(normal)
        if normal is None:
            return None
        dist = np.abs((pts - center.reshape(1, 3)) @ normal.reshape(3, 1)).reshape(-1)
        inlier_pts = pts[dist < threshold_m]
        if len(inlier_pts) < 40:
            return None
        plane_point = np.median(inlier_pts, axis=0).astype(np.float32)
        return normal.astype(np.float32), plane_point, inlier_pts.astype(np.float32)

    def _lift_quad_to_plane(self, info: CameraInfo, quad_uv: np.ndarray, normal: np.ndarray, plane_point: np.ndarray) -> Optional[np.ndarray]:
        quad_uv = np.asarray(quad_uv, dtype=np.float32).reshape(4, 2)
        pts = []
        for uv in quad_uv:
            ray = self._ray_from_pixel(info, float(uv[0]), float(uv[1]))
            if ray is None:
                return None
            denom = float(np.dot(normal, ray))
            if abs(denom) < 1e-6:
                return None
            t = float(np.dot(normal, plane_point) / denom)
            if t <= 0.01:
                return None
            pts.append((ray * t).astype(np.float32))
        pts = np.asarray(pts, dtype=np.float32)
        return pts if pts.shape == (4, 3) else None

    def _board_x_axis_from_image(self, info: CameraInfo, center_xyz: np.ndarray, normal: np.ndarray, quad: Optional[np.ndarray], taskboard_det: Dict) -> Optional[np.ndarray]:
        if quad is not None:
            quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
            top_edge = quad[1] - quad[0]
            bottom_edge = quad[2] - quad[3]
            uv_dir = top_edge + bottom_edge
            if float(np.linalg.norm(uv_dir)) < 1e-6:
                uv_dir = quad[1] - quad[0]
            center_uv = quad.mean(axis=0)
            return self._image_dir_to_plane_axis(info, center_xyz, normal, center_uv, uv_dir)
        box = np.asarray(taskboard_det.get("bbox_xyxy", [0, 0, 0, 0]), dtype=np.float32)
        center_uv = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)
        uv_dir = np.array([1.0, 0.0], dtype=np.float32)
        return self._image_dir_to_plane_axis(info, center_xyz, normal, center_uv, uv_dir)

    def _image_dir_to_plane_axis(self, info: CameraInfo, center_xyz: np.ndarray, normal: np.ndarray, center_uv: np.ndarray, uv_dir: np.ndarray) -> Optional[np.ndarray]:
        uv_dir = np.asarray(uv_dir, dtype=np.float32).reshape(2)
        nrm = float(np.linalg.norm(uv_dir))
        if nrm < 1e-6:
            return None
        uv_dir = uv_dir / nrm
        p0 = self._ray_from_pixel(info, float(center_uv[0]), float(center_uv[1]))
        p1 = self._ray_from_pixel(info, float(center_uv[0] + 8.0 * uv_dir[0]), float(center_uv[1] + 8.0 * uv_dir[1]))
        if p0 is None or p1 is None:
            return None
        axis = p1 - p0
        axis = self._project_vec_to_plane(axis, normal)
        return self._normalize_vec(axis)

    def _ray_from_pixel(self, info: CameraInfo, u: float, v: float) -> Optional[np.ndarray]:
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 1e-6 or fy <= 1e-6:
            return None
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32)
        return self._normalize_vec(ray)

    def _project_vec_to_plane(self, v: np.ndarray, normal: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(3)
        n = np.asarray(normal, dtype=np.float32).reshape(3)
        return v - n * float(np.dot(v, n))

    def _normalize_vec(self, v: np.ndarray) -> Optional[np.ndarray]:
        v = np.asarray(v, dtype=np.float32).reshape(3)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return None
        return (v / n).astype(np.float32)

    def _orthonormalize_rotation(self, R: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float32).reshape(3, 3))
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0.0:
            U[:, -1] *= -1.0
            Rn = U @ Vt
        return np.asarray(Rn, dtype=np.float32)

    def _bbox_long_axis_on_plane(self, info: CameraInfo, center_xyz: np.ndarray, normal: np.ndarray, det: Dict) -> Optional[np.ndarray]:
        box = np.asarray(det.get("bbox_xyxy", [0, 0, 0, 0]), dtype=np.float32)
        if box.size != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in box]
        if (x2 - x1) >= (y2 - y1):
            center_uv = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            uv_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            center_uv = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            uv_dir = np.array([0.0, 1.0], dtype=np.float32)
        return self._image_dir_to_plane_axis(info, center_xyz, normal, center_uv, uv_dir)

    def _principal_axis_from_points_on_plane(self, pts: np.ndarray, normal: np.ndarray, fallback_axis: np.ndarray) -> Optional[np.ndarray]:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] < 12:
            return None
        center = np.median(pts, axis=0)
        X = pts - center.reshape(1, 3)
        X = X - (X @ normal.reshape(3, 1)) * normal.reshape(1, 3)
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
        axis = self._normalize_vec(self._project_vec_to_plane(axis, normal))
        if axis is None:
            return None
        fallback_axis = self._normalize_vec(self._project_vec_to_plane(fallback_axis, normal))
        if fallback_axis is not None and float(np.dot(axis, fallback_axis)) < 0.0:
            axis = -axis
        return axis.astype(np.float32)

    def _estimate_component_pose_on_board_plane(self, info: CameraInfo, depth: Optional[np.ndarray], det: Dict, board_pose_camera: Dict, raw_pose_camera: Optional[Dict] = None, translation_hint: Optional[np.ndarray] = None) -> Optional[Dict]:
        if board_pose_camera is None:
            return None
        Rb = np.asarray(board_pose_camera["R"], dtype=np.float32).reshape(3, 3)
        board_x = Rb[:, 0]
        board_y = Rb[:, 1]
        normal = Rb[:, 2]
        pts = None if depth is None else self._depth_points_from_region(info, depth, det, None)
        if translation_hint is not None:
            t = np.asarray(translation_hint, dtype=np.float32).reshape(3)
        elif pts is not None and len(pts) >= 12:
            t = np.median(pts, axis=0).astype(np.float32)
        else:
            family = self._detection_family_name(det)
            anchor_key = "NIC_CARD" if family == "nic" else "SC_PORT" if family == "sc" else "GENERIC"
            anchor = self._detection_anchor_point(det, anchor_key)
            t = self._backproject_anchor(info, depth, anchor)
        if t is None:
            return None

        raw_axis = None
        if raw_pose_camera is not None and raw_pose_camera.get("R", None) is not None:
            raw_axis = np.asarray(raw_pose_camera["R"], dtype=np.float32).reshape(3, 3)[:, 0]
        if raw_axis is None:
            raw_axis = board_x

        x_axis = None
        if pts is not None and len(pts) >= 12:
            x_axis = self._principal_axis_from_points_on_plane(pts, normal, raw_axis)
        if x_axis is None:
            x_axis = self._bbox_long_axis_on_plane(info, t, normal, det)
        if x_axis is None and raw_axis is not None:
            x_axis = self._normalize_vec(self._project_vec_to_plane(raw_axis, normal))
        if x_axis is None:
            x_axis = board_x
        if float(np.dot(x_axis, board_x)) < 0.0:
            x_axis = -x_axis
        y_axis = self._normalize_vec(np.cross(normal, x_axis))
        if y_axis is None:
            y_axis = board_y
        if float(np.dot(y_axis, board_y)) < 0.0:
            y_axis = -y_axis
        x_axis = self._normalize_vec(np.cross(y_axis, normal))
        if x_axis is None:
            x_axis = board_x
        R = self._orthonormalize_rotation(np.column_stack([x_axis, y_axis, normal]).astype(np.float32))
        return {"R": R, "t": t.astype(np.float32), "q": self._matrix_to_quaternion(R), "source": "board_plane_component"}

    def _transform_pose_dict(self, pose: Optional[Dict], source_frame: str, target_frame: str) -> Optional[Dict]:
        if pose is None:
            return None
        t_val = pose.get("t")
        q_val = pose.get("q")
        R_val = pose.get("R")
        if t_val is None:
            return None
        t_src = np.asarray(t_val, dtype=np.float32).reshape(3)
        if R_val is not None:
            R_src = np.asarray(R_val, dtype=np.float32).reshape(3, 3)
        elif q_val is not None:
            R_src = self._quaternion_to_matrix(np.asarray(q_val, dtype=np.float32).reshape(4))
        else:
            return None
        q_src = self._matrix_to_quaternion(R_src)
        if source_frame == target_frame:
            return {"R": R_src.astype(np.float32), "t": t_src.astype(np.float32), "q": q_src.astype(np.float32)}
        try:
            R_tf, t_tf = self._lookup_transform(target_frame, source_frame)
        except Exception:
            return None
        R_tf = np.asarray(R_tf, dtype=np.float32).reshape(3, 3)
        t_tf = np.asarray(t_tf, dtype=np.float32).reshape(3)
        R_tgt = R_tf @ R_src
        t_tgt = R_tf @ t_src + t_tf
        return {"R": R_tgt.astype(np.float32), "t": t_tgt.astype(np.float32), "q": self._matrix_to_quaternion(R_tgt)}

    def _maybe_constrain_pose_to_table_plane(self, det: Dict, pose_base: Dict) -> Dict:
        if not self._force_table_flat:
            return pose_base
        family = self._detection_family_name(det)
        if family not in self._table_constrained_families:
            return pose_base

        R = np.asarray(pose_base.get("R"), dtype=np.float32).reshape(3, 3)
        t = np.asarray(pose_base.get("t"), dtype=np.float32).reshape(3)

        world_up = self._table_normal_base.astype(np.float32)
        z_axis = R[:, 2].astype(np.float32)
        if float(np.dot(z_axis, world_up)) < 0.0:
            world_up = -world_up

        x_hint = self._project_vec_to_plane(R[:, 0], world_up)
        x_axis = self._normalize_vec(x_hint)
        if x_axis is None:
            y_hint = self._project_vec_to_plane(R[:, 1], world_up)
            y_axis = self._normalize_vec(y_hint)
            if y_axis is None:
                return pose_base
            x_axis = self._normalize_vec(np.cross(y_axis, world_up))
            if x_axis is None:
                return pose_base
        y_axis = self._normalize_vec(np.cross(world_up, x_axis))
        if y_axis is None:
            return pose_base
        x_axis = self._normalize_vec(np.cross(y_axis, world_up))
        if x_axis is None:
            return pose_base

        R_flat = self._orthonormalize_rotation(np.column_stack([x_axis, y_axis, world_up]).astype(np.float32))
        return {
            "R": R_flat.astype(np.float32),
            "t": t.astype(np.float32),
            "q": self._matrix_to_quaternion(R_flat),
            "source": str(pose_base.get("source", "")) + "+table_flat",
        }

    def _pose_dict_to_msg(self, pose: Dict) -> Pose:
        msg = Pose()
        msg.position.x = float(pose["t"][0])
        msg.position.y = float(pose["t"][1])
        msg.position.z = float(pose["t"][2])
        msg.orientation = self._quaternion_msg_from_array(pose["q"])
        return msg

    def _quaternion_msg_from_array(self, q: np.ndarray) -> Quaternion:
        q = np.asarray(q, dtype=np.float32).reshape(4)
        msg = Quaternion()
        msg.x = float(q[0])
        msg.y = float(q[1])
        msg.z = float(q[2])
        msg.w = float(q[3])
        return msg

    def _matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(R))
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float64)
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        q = q / qn
        if q[3] < 0.0:
            q = -q
        return q.astype(np.float32)

    def _pose_to_transform(self, stamp, parent: str, child: str, pose: Pose) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(pose.position.x)
        tf.transform.translation.y = float(pose.position.y)
        tf.transform.translation.z = float(pose.position.z)
        tf.transform.rotation = pose.orientation
        return tf

    def _tf_safe_name(self, text: str) -> str:
        out = str(text).strip().lower().replace(" ", "_").replace("-", "_")
        out = "".join(ch for ch in out if ch.isalnum() or ch == "_")
        while "__" in out:
            out = out.replace("__", "_")
        return out if out else "detection"

    def _make_axes_marker(self, marker_id: int, stamp, name: str, pose: Pose) -> Marker:
        R = self._quaternion_to_matrix(
            np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=np.float32)
        )
        origin = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
        x_end = origin + R[:, 0] * float(self._axis_length_m)
        y_end = origin + R[:, 1] * float(self._axis_length_m)
        z_end = origin + R[:, 2] * float(self._axis_length_m)

        marker = Marker()
        marker.header.frame_id = self._base_frame
        marker.header.stamp = stamp
        marker.ns = "combined_detection_axes"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self._axis_width_m
        marker.pose.orientation.w = 1.0
        marker.points = [
            self._point_msg(origin),
            self._point_msg(x_end),
            self._point_msg(origin),
            self._point_msg(y_end),
            self._point_msg(origin),
            self._point_msg(z_end),
        ]
        marker.colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.4, b=1.0, a=1.0),
            ColorRGBA(r=0.0, g=0.4, b=1.0, a=1.0),
        ]
        return marker

    def _point_msg(self, xyz: np.ndarray) -> Point:
        p = Point()
        p.x = float(xyz[0])
        p.y = float(xyz[1])
        p.z = float(xyz[2])
        return p

    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = [float(v) for v in np.asarray(q, dtype=np.float32).reshape(4)]
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CombinedYoloDepthPosePlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
