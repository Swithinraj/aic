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

from team_policy.perception.yolov12_detector import YoloV12MultiCameraDetector


class CombinedYoloDepthPosePlanner(YoloV12MultiCameraDetector):
    def __init__(self):
        super().__init__()

        self._base_frame = "base_link"
        self._tf_camera = self._env_text("YOLO_POSE_TF_CAMERA", "center")
        self._detection_plane_z_m = self._env_float("YOLO_DEPTH_POSE_DETECTION_Z_M", 0.0)
        self._board_width_m = self._env_float("YOLO_DEPTH_POSE_BOARD_WIDTH_M", 0.300)
        self._board_height_m = self._env_float("YOLO_DEPTH_POSE_BOARD_HEIGHT_M", 0.425)
        self._axis_length_m = self._env_float("YOLO_DEPTH_POSE_AXIS_LENGTH_M", 0.05)
        self._axis_width_m = self._env_float("YOLO_DEPTH_POSE_AXIS_WIDTH_M", 0.004)
        self._text_scale = self._env_float("YOLO_DEPTH_POSE_TEXT_SCALE", 0.03)

        self._tf_broadcaster = TransformBroadcaster(self)

        self._fused_pose_pub = self.create_publisher(PoseArray, "/fused_yolo/poses_base_link", 10)
        self._fused_json_pub = self.create_publisher(String, "/fused_yolo/detections_json", 10)
        self._fused_marker_pub = self.create_publisher(MarkerArray, "/fused_yolo/pose_markers", 10)

        self._latest_instance_obs = {}
        self._tf_sub = self.create_subscription(TFMessage, "/tf", self._tf_callback, 100)

        self.get_logger().info(f"Combined planner node started: only true detection poses and TFs are published. TF camera={self._tf_camera}, base_frame={self._base_frame}")

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
        stamp = self.get_clock().now().to_msg()

        for camera_name, res in camera_results.items():
            msg = res["msg"]
            stamp = msg.header.stamp
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
            if info:
                camera_frame_id = str(info.header.frame_id)
                for det in detections:
                    name = det.get("instance_name", det.get("class_name", ""))
                    if not name:
                        continue
                    pose_cam = det.get("pose_camera")
                    if not (isinstance(pose_cam, dict) and "t" in pose_cam and "q" in pose_cam):
                        continue
                    # Compute anchor UV for visual servoing
                    family = self._detection_family_name(det)
                    anchor_key = "NIC_CARD" if family == "nic" else "SC_PORT" if family == "sc" else "GENERIC"
                    anchor = self._detection_anchor_point(det, anchor_key)
                    anchor_uv = [float(anchor[0]), float(anchor[1])]
                    if name not in instance_obs:
                        instance_obs[name] = []
                    instance_obs[name].append({
                        "det": det,
                        "camera_frame_id": camera_frame_id,
                        "anchor_uv": anchor_uv,
                        "camera_name": camera_name
                    })

        with self._lock:
            self._latest_instance_obs = instance_obs

    def _tf_callback(self, msg: TFMessage) -> None:
        with self._lock:
            instance_obs = dict(getattr(self, "_latest_instance_obs", {}))
            
        if not instance_obs:
            return
            
        stamp = self.get_clock().now().to_msg()
        fused_records = []
        used_tf_names = set()
        
        for name, obs_list in instance_obs.items():
            best_obj = max(obs_list, key=lambda o: float(o["det"].get("confidence", 0.0)))
            best_det = best_obj["det"]
            camera_frame_id = best_obj["camera_frame_id"]
            pose_cam = best_det.get("pose_camera")
            
            try:
                R_tf, t_tf = self._lookup_transform(self._base_frame, camera_frame_id)
            except Exception:
                continue

            R_cam = self._quaternion_to_matrix(np.array(pose_cam["q"], dtype=np.float32))
            t_cam = np.array(pose_cam["t"], dtype=np.float32)

            t_base = R_tf @ t_cam + t_tf
            R_base = R_tf @ R_cam
            q_base = self._matrix_to_quaternion(R_base)

            vx, vy, vz, vw = float(q_base[0]), float(q_base[1]), float(q_base[2]), float(q_base[3])
            yaw = np.arctan2(2.0 * (vw * vz + vx * vy), 1.0 - 2.0 * (vy * vy + vz * vz))

            x, y = float(t_base[0]), float(t_base[1])
            qz = float(np.sin(yaw / 2.0))
            qw = float(np.cos(yaw / 2.0))

            z = 0.17927 if "sfp_port" in str(name).lower() else 0.25

            pose_base = {
                "t": np.array([x, y, z], dtype=np.float32),
                "q": np.array([0.0, 0.0, qz, qw], dtype=np.float32)
            }

            display_name = str(name)
            tf_base = "det_" + self._tf_safe_name(display_name)
            tf_name = tf_base
            index = 1
            while tf_name in used_tf_names:
                tf_name = f"{tf_base}_{index}"
                index += 1
            used_tf_names.add(tf_name)

            fused_records.append({
                "name": display_name,
                "tf_name": tf_name,
                "bbox_xyxy": [float(v) for v in best_det.get("bbox_xyxy", [])],
                "base_pose": self._pose_dict_to_msg(pose_base),
                "class_name": str(best_det.get("class_name", "")),
                "confidence": float(best_det.get("confidence", 0.0)),
                "camera": "fused" if len(obs_list) > 1 else "single",
                "anchor_uv": best_obj.get("anchor_uv", []),
                "camera_frame_id": camera_frame_id,
                "camera_name": best_obj.get("camera_name", "center")
            })

        self._publish_fused_records(stamp, fused_records)

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

            fused_detections_json.append({
                "class_name": rec["class_name"],
                "instance_name": rec["name"],
                "confidence": rec["confidence"],
                "bbox_xyxy": rec["bbox_xyxy"],
                "tf_frame": rec["tf_name"],
                "source": rec["camera"],
                "anchor_uv": rec.get("anchor_uv", []),
                "camera_frame_id": rec.get("camera_frame_id", ""),
                "camera_name": rec.get("camera_name", ""),
                "pose_base_link": {
                    "position": {"x": float(bp.x), "y": float(bp.y), "z": float(bp.z)},
                    "orientation": {"x": 0.0, "y": 0.0, "z": base_pose.orientation.z, "w": base_pose.orientation.w}
                }
            })

        self._fused_pose_pub.publish(fused_pose_array)
        self._fused_marker_pub.publish(fused_markers)
        if len(fused_tf_list) > 0:
            self._tf_broadcaster.sendTransform(fused_tf_list)

        jd = String()
        jd.data = json.dumps(fused_detections_json, separators=(",", ":"))
        self._fused_json_pub.publish(jd)

    def _triangulate_instance(self, obs_list: List[Dict]) -> Optional[np.ndarray]:
        if len(obs_list) < self._min_triangulation_views:
            return None
        
        # Triangulate local to the first camera to maintain matrix stability against long baseline scaling
        anchor_frame = str(obs_list[0]["frame_id"])
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        used = 0

        for obs in obs_list:
            info = obs["info"]
            uv = obs["uv"]
            ray_cam = self._camera_ray_from_pixel(info, uv)
            try:
                R, t = self._lookup_transform(anchor_frame, str(obs["frame_id"]))
            except Exception:
                continue
            o = np.asarray(t, dtype=np.float64).reshape(3)
            d = (np.asarray(R, dtype=np.float64) @ np.asarray(ray_cam, dtype=np.float64).reshape(3))
            dn = np.linalg.norm(d)
            if dn < 1e-9:
                continue
            d = d / dn
            M = np.eye(3, dtype=np.float64) - np.outer(d, d)
            A += M
            b += M @ o
            used += 1

        if used < self._min_triangulation_views:
            return None
        try:
            X_anchor = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        # Transform the stabilized point back to base_link
        try:
            R_base, t_base = self._lookup_transform(self._base_frame, anchor_frame)
            X_base = (np.asarray(R_base, dtype=np.float64) @ X_anchor) + np.asarray(t_base, dtype=np.float64)
            return X_base.astype(np.float32)
        except Exception:
            return None

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

        if info is None or ctx is None:
            return []

        board_quad = ctx.get("projected_board_quad")
        if board_quad is None:
            return []

        board_quad = np.asarray(board_quad, dtype=np.float32).reshape(4, 2)
        board_pose_camera = self._estimate_board_pose_from_quad(info, board_quad)
        if board_pose_camera is None:
            return []

        image_to_board_h = self._compute_image_to_board_homography(board_quad)
        if image_to_board_h is None:
            return []

        records: List[Dict] = []
        used_tf_names = set()

        taskboard_det = self._find_best_detection(detections, self._taskboard_classes)
        if taskboard_det is not None:
            pose_camera = self._compose_local_pose(
                board_pose_camera,
                np.array([0.0, 0.0, self._detection_plane_z_m], dtype=np.float32),
                np.eye(3, dtype=np.float32),
            )
            if pose_camera is not None:
                records.append(
                    self._make_pose_record(
                        det=taskboard_det,
                        pose_camera=pose_camera,
                        used_tf_names=used_tf_names,
                    )
                )

        fix_det = self._find_best_detection(detections, self._fix_classes)
        if fix_det is not None:
            pose_camera = self._pose_from_detection_anchor(image_to_board_h, board_pose_camera, fix_det, "FIX")
            if pose_camera is not None:
                records.append(
                    self._make_pose_record(
                        det=fix_det,
                        pose_camera=pose_camera,
                        used_tf_names=used_tf_names,
                    )
                )

        for det in detections:
            family = self._detection_family_name(det)
            if family not in {"nic", "sc", "sfp_port"}:
                continue
            anchor_key = "NIC_CARD" if family == "nic" else "SC_PORT" if family == "sc" else "GENERIC"
            pose_camera = self._pose_from_detection_anchor(image_to_board_h, board_pose_camera, det, anchor_key)
            if pose_camera is None:
                continue
            records.append(
                self._make_pose_record(
                    det=det,
                    pose_camera=pose_camera,
                    used_tf_names=used_tf_names,
                )
            )

        return records

    def _pose_from_detection_anchor(self, image_to_board_h: np.ndarray, board_pose_base: Dict, det: Dict, anchor_key: str) -> Optional[Dict]:
        anchor_img = self._detection_anchor_point(det, anchor_key)
        board_xy = self._image_point_to_board_xy(image_to_board_h, anchor_img)
        if board_xy is None:
            return None
        return self._compose_local_pose(
            board_pose_base,
            np.array([float(board_xy[0]), float(board_xy[1]), self._detection_plane_z_m], dtype=np.float32),
            np.eye(3, dtype=np.float32),
        )

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
        }

    def _camera_matrix_from_info(self, info) -> np.ndarray:
        return np.array([[float(info.k[0]), 0.0, float(info.k[2])], [0.0, float(info.k[4]), float(info.k[5])], [0.0, 0.0, 1.0]], dtype=np.float32)

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
        R = np.asarray(R, dtype=np.float32).reshape(3, 3)
        return {"R": R, "t": t, "q": self._matrix_to_quaternion(R)}

    def _compute_image_to_board_homography(self, board_quad_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            return cv2.getPerspectiveTransform(np.asarray(board_quad_image, dtype=np.float32).reshape(4, 2), self._board_corner_points_xy()).astype(np.float32)
        except Exception:
            return None

    def _image_point_to_board_xy(self, H: np.ndarray, point_xy: np.ndarray) -> Optional[np.ndarray]:
        pts = np.asarray(point_xy, dtype=np.float32).reshape(1, 1, 2)
        out = cv2.perspectiveTransform(pts, H)
        if out is None:
            return None
        return np.asarray(out, dtype=np.float32).reshape(2)

    def _compose_local_pose(self, board_pose: Dict, local_point: np.ndarray, R_offset: np.ndarray) -> Optional[Dict]:
        if board_pose is None:
            return None
        R_board = np.asarray(board_pose["R"], dtype=np.float32).reshape(3, 3)
        t_board = np.asarray(board_pose["t"], dtype=np.float32).reshape(3)
        local_point = np.asarray(local_point, dtype=np.float32).reshape(3)
        R_offset = np.asarray(R_offset, dtype=np.float32).reshape(3, 3)
        R = R_board @ R_offset
        t = R_board @ local_point + t_board
        return {"R": R.astype(np.float32), "t": t.astype(np.float32), "q": self._matrix_to_quaternion(R)}

    def _transform_pose_dict(self, pose: Optional[Dict], source_frame: str, target_frame: str) -> Optional[Dict]:
        if pose is None:
            return None
        if source_frame == target_frame:
            return {"R": np.asarray(pose["R"], dtype=np.float32).copy(), "t": np.asarray(pose["t"], dtype=np.float32).copy(), "q": np.asarray(pose["q"], dtype=np.float32).copy()}
        try:
            R_tf, t_tf = self._lookup_transform(target_frame, source_frame)
        except Exception:
            return None
        R_src = np.asarray(pose["R"], dtype=np.float32).reshape(3, 3)
        t_src = np.asarray(pose["t"], dtype=np.float32).reshape(3)
        R_tgt = np.asarray(R_tf, dtype=np.float32) @ R_src
        t_tgt = np.asarray(R_tf, dtype=np.float32) @ t_src + np.asarray(t_tf, dtype=np.float32)
        return {"R": R_tgt.astype(np.float32), "t": t_tgt.astype(np.float32), "q": self._matrix_to_quaternion(R_tgt)}

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

    def _attach_pose_data_to_detections(self, detections: List[Dict], pose_records: List[Dict]) -> List[Dict]:
        out = [dict(d) for d in detections]
        for det in out:
            for rec in pose_records:
                rb = rec.get("bbox_xyxy")
                db = det.get("bbox_xyxy")
                if rb is None or db is None:
                    continue
                if len(rb) != len(db):
                    continue
                if sum(abs(float(a) - float(b)) for a, b in zip(rb, db)) < 1e-3:
                    pose_cam = rec.get("pose_camera")
                    if pose_cam:
                        det["pose_camera"] = {
                            "t": [float(v) for v in pose_cam["t"]],
                            "q": [float(v) for v in pose_cam["q"]]
                        }
                    det["tf_frame"] = rec.get("tf_name")
                    break
        return out

    def _pose_to_dict(self, pose: Optional[Pose]):
        if pose is None:
            return None
        return {
            "position": {"x": float(pose.position.x), "y": float(pose.position.y), "z": float(pose.position.z)},
            "orientation": {"x": float(pose.orientation.x), "y": float(pose.orientation.y), "z": float(pose.orientation.z), "w": float(pose.orientation.w)},
        }

    def _draw_pose_labels(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        out = image.copy()
        for det in detections:
            label = det.get("instance_name", det.get("class_name"))
            if not label:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in det["bbox_xyxy"]]
            pose = det.get("pose_base_link")
            text = str(label)
            if pose is not None:
                p = pose["position"]
                text = f"{label} ({p['x']:.3f},{p['y']:.3f},{p['z']:.3f})"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            tx = max(0, x1)
            ty = max(th + 6, y1 - 6)
            cv2.rectangle(out, (tx, ty - th - 6), (tx + tw + 8, ty + baseline - 2), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(out, text, (tx + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        return out

    def _build_pose_outputs(self, msg, pose_records: List[Dict]):
        base_pose_array = PoseArray()
        base_pose_array.header = msg.header
        base_pose_array.header.frame_id = self._base_frame

        markers = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        tf_list: List[TransformStamped] = []
        marker_id = 0
        now_msg = self.get_clock().now().to_msg()

        for rec in pose_records:
            base_pose = rec.get("base_pose")
            if base_pose is None:
                continue

            base_pose_array.poses.append(base_pose)
            tf_list.append(self._pose_to_transform(msg.header.stamp, self._base_frame, str(rec.get("tf_name", "")), base_pose))

            axis_marker = self._make_axes_marker(marker_id, now_msg, rec["name"], base_pose)
            markers.markers.append(axis_marker)
            marker_id += 1

            text = Marker()
            text.header.frame_id = self._base_frame
            text.header.stamp = now_msg
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
            markers.markers.append(text)
            marker_id += 1

        return base_pose_array, markers, tf_list

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
        R = self._quaternion_to_matrix(np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=np.float32))
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
            self._point_msg(origin), self._point_msg(x_end),
            self._point_msg(origin), self._point_msg(y_end),
            self._point_msg(origin), self._point_msg(z_end),
        ]
        marker.colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.4, b=1.0, a=1.0), ColorRGBA(r=0.0, g=0.4, b=1.0, a=1.0),
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