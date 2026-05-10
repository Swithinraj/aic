from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, TransformStamped
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from team_policy.perception.yolov12_detector import (
    _ALLOWED_TF_NAMES,
    YoloV12MultiCameraDetector,
)


class CombinedYoloDepthPosePlanner(YoloV12MultiCameraDetector):
    """YOLO canonical detections + metric depth anchor TFs.

    This planner deliberately avoids ICP, task-board masks,
    and duplicate TF aliases. Each canonical instance gets at
    most one fused record, selected from the best camera observation.
    """

    def __init__(self):
        super().__init__()

        self._base_frame = "base_link"
        self._tf_camera = self._env_text("YOLO_POSE_TF_CAMERA", "center")
        self._preferred_camera_for_fusion = self._tf_camera
        self._axis_length_m = self._env_float("YOLO_DEPTH_POSE_AXIS_LENGTH_M", 0.05)
        self._axis_width_m = self._env_float("YOLO_DEPTH_POSE_AXIS_WIDTH_M", 0.004)
        self._text_scale = self._env_float("YOLO_DEPTH_POSE_TEXT_SCALE", 0.03)
        self._depth_fallback_m = self._env_float("YOLO_DEPTH_POSE_FALLBACK_M", 0.30)
        self._fusion_max_age_s = self._env_float("YOLO_DEPTH_POSE_FUSION_MAX_AGE", 0.50)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._fused_pose_pub = self.create_publisher(PoseArray, "/fused_yolo/poses_base_link", 10)
        self._fused_json_pub = self.create_publisher(String, "/fused_yolo/detections_json", 10)
        self._fused_marker_pub = self.create_publisher(MarkerArray, "/fused_yolo/pose_markers", 10)

        self._latest_instance_obs: Dict[str, List[Dict]] = {}
        self._latest_depth_images: Dict[str, Optional[np.ndarray]] = {"left": None, "center": None, "right": None}
        self._latest_depth_stamps: Dict[str, float] = {"left": 0.0, "center": 0.0, "right": 0.0}

        self._depth_subs = {
            "left": self.create_subscription(Image, "/left_camera/stereo_depth/image", lambda msg: self._depth_cb("left", msg), 10),
            "center": self.create_subscription(Image, "/center_camera/stereo_depth/image", lambda msg: self._depth_cb("center", msg), 10),
            "right": self.create_subscription(Image, "/right_camera/stereo_depth/image", lambda msg: self._depth_cb("right", msg), 10),
        }

        self.get_logger().info(
            f"Combined planner node started: YOLO + metric depth anchor TF, TF camera={self._tf_camera}, base_frame={self._base_frame}"
        )

    def _env_float(self, key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, str(default)))
        except Exception:
            return float(default)

    def _env_text(self, key: str, default: str) -> str:
        value = str(os.environ.get(key, default)).strip().lower()
        return value if value in {"left", "center", "right"} else str(default)

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
                result = self._process_camera_frame(camera_name, msg)
            except Exception as exc:
                self.get_logger().error(f"{camera_name} inference failed: {exc}")
                continue
            self._last_infer_time[camera_name] = now
            camera_results[camera_name] = {"msg": msg, "result": result}

        if not camera_results:
            return

        instance_obs: Dict[str, List[Dict]] = {}
        for camera_name, res in camera_results.items():
            msg = res["msg"]
            annotated, detections, classes, mask_overlay, mask_binary, mask_status = res["result"][:6]
            detections = self._attach_depth_anchor_poses(camera_name, detections)
            self._publish_camera_outputs(
                camera_name,
                msg,
                annotated,
                detections,
                classes,
                mask_overlay,
                mask_binary,
                mask_status,
            )

            with self._lock:
                info = self._latest_infos.get(camera_name)
            if info is None:
                continue
            camera_frame_id = str(info.header.frame_id)
            for det in detections:
                name = str(det.get("instance_name", det.get("class_name", "")))
                if name not in _ALLOWED_TF_NAMES:
                    continue
                pose_cam = det.get("pose_camera")
                if not (isinstance(pose_cam, dict) and "t" in pose_cam and "q" in pose_cam):
                    continue
                instance_obs.setdefault(name, []).append(
                    {
                        "det": det,
                        "camera_frame_id": camera_frame_id,
                        "camera_name": camera_name,
                    }
                )

        with self._lock:
            self._latest_instance_obs = instance_obs

        self._publish_fused_from_instance_obs(instance_obs)

    def _attach_depth_anchor_poses(self, camera_name: str, detections: List[Dict]) -> List[Dict]:
        with self._lock:
            info = self._latest_infos.get(camera_name)
        if info is None:
            return detections
        depth = self._get_latest_depth_copy(camera_name)
        out: List[Dict] = []
        for det in detections:
            name = str(det.get("instance_name", det.get("class_name", "")))
            if name not in _ALLOWED_TF_NAMES:
                continue
            det2 = dict(det)
            anchor = self._anchor_uv_for_detection(det2)
            t_cam = self._backproject_anchor(info, depth, anchor)
            if t_cam is None:
                out.append(det2)
                continue
            q = [0.0, 0.0, 0.0, 1.0]
            det2["anchor_uv"] = [float(anchor[0]), float(anchor[1])]
            det2["pose_camera"] = {
                "t": [float(v) for v in t_cam],
                "q": q,
            }
            det2["pose_source"] = "depth_anchor"
            out.append(det2)
        return out

    def _anchor_uv_for_detection(self, det: Dict) -> np.ndarray:
        if isinstance(det.get("center_uv"), list) and len(det["center_uv"]) >= 2:
            return np.asarray(det["center_uv"][:2], dtype=np.float32)
        return self._detection_anchor_point(det)

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
        if fx <= 1e-6 or fy <= 1e-6 or Z <= 0.0:
            return None
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z], dtype=np.float32)

    def _sample_depth(self, depth: Optional[np.ndarray], u: float, v: float) -> Optional[float]:
        if depth is None:
            return None
        h, w = depth.shape[:2]
        if h <= 0 or w <= 0:
            return None
        r = int(np.clip(round(v), 0, h - 1))
        c = int(np.clip(round(u), 0, w - 1))
        r0, r1 = max(0, r - 3), min(h, r + 4)
        c0, c1 = max(0, c - 3), min(w, c + 4)
        patch = np.asarray(depth[r0:r1, c0:c1], dtype=np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _publish_fused_from_instance_obs(self, instance_obs: Dict[str, List[Dict]]) -> None:
        stamp = self.get_clock().now().to_msg()
        fused_records = []
        for name in sorted(instance_obs.keys(), key=self._tf_sort_key):
            if name not in _ALLOWED_TF_NAMES:
                continue
            best_obj = self._select_best_observation(instance_obs[name])
            if best_obj is None:
                continue
            det = best_obj["det"]
            camera_frame_id = best_obj["camera_frame_id"]
            pose_base = self._transform_pose_dict(det.get("pose_camera"), camera_frame_id, self._base_frame)
            if pose_base is None:
                continue
            tf_name = "det_" + self._tf_safe_name(name)
            anchor_keys = (
                "servo_anchor_valid",
                "servo_anchor_source",
                "servo_anchor_quality",
                "mouth_center_uv",
                "mouth_left_uv",
                "mouth_right_uv",
                "mouth_angle_rad",
                "sc_port_center_uv",
                "sc_port_axis_left_uv",
                "sc_port_axis_right_uv",
                "sc_port_axis_angle_rad",
                "tip_uv",
                "front_center_uv",
                "front_left_uv",
                "front_right_uv",
                "front_angle_rad",
                "sc_plug_tip_uv",
                "sc_plug_axis_left_uv",
                "sc_plug_axis_right_uv",
                "sc_plug_axis_angle_rad",
            )
            anchors = {key: det[key] for key in anchor_keys if key in det}
            fused_records.append(
                {
                    "name": name,
                    "tf_name": tf_name,
                    "bbox_xyxy": [float(v) for v in det.get("bbox_xyxy", [])],
                    "center_uv": det.get("center_uv", []),
                    "base_pose": self._pose_dict_to_msg(pose_base),
                    "pose_base": pose_base,
                    "class_name": str(det.get("class_name", "")),
                    "confidence": float(det.get("confidence", 0.0)),
                    "raw_confidence": float(det.get("raw_confidence", 0.0)),
                    "track_id": int(det.get("track_id", -1)),
                    "feature_track_id": int(det.get("feature_track_id", -1)),
                    "feature_quality_score": float(det.get("feature_quality_score", 0.0)),
                    "feature_tracked_count": int(det.get("feature_tracked_count", 0)),
                    "feature_inlier_count": int(det.get("feature_inlier_count", 0)),
                    "feature_mode": str(det.get("feature_mode", "")),
                    "camera": best_obj.get("camera_name", ""),
                    "camera_name": best_obj.get("camera_name", ""),
                    "camera_frame_id": camera_frame_id,
                    "pose_source": "depth_anchor",
                    **anchors,
                }
            )
        self._publish_fused_records(stamp, fused_records)

    def _select_best_observation(self, obs_list: List[Dict]) -> Optional[Dict]:
        if not obs_list:
            return None
        ranked = sorted(obs_list, key=lambda obs: float(obs["det"].get("confidence", 0.0)), reverse=True)
        best = ranked[0]
        best_conf = float(best["det"].get("confidence", 0.0))
        preferred = [
            obs for obs in ranked
            if obs.get("camera_name") == self._preferred_camera_for_fusion
            and best_conf - float(obs["det"].get("confidence", 0.0)) <= 0.05
        ]
        return preferred[0] if preferred else best

    def _publish_fused_records(self, stamp, fused_records: List[Dict]) -> None:
        fused_pose_array = PoseArray()
        fused_pose_array.header.stamp = stamp
        fused_pose_array.header.frame_id = self._base_frame

        fused_markers = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        fused_markers.markers.append(delete_marker)

        transforms = []
        fused_json = []
        marker_id = 0
        published_items = []

        for rec in fused_records:
            pose = rec["base_pose"]
            fused_pose_array.poses.append(pose)
            transforms.append(self._pose_to_transform(stamp, self._base_frame, rec["tf_name"], pose))
            published_items.append(f"{rec['name']}:{float(rec['confidence']):.2f}")

            axes = self._make_axes_marker(marker_id, stamp, rec["name"], pose)
            fused_markers.markers.append(axes)
            marker_id += 1

            text = Marker()
            text.header.frame_id = self._base_frame
            text.header.stamp = stamp
            text.ns = "simple_yolo_depth_anchor_text"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = pose
            text.pose.position.z += 0.04
            text.scale.z = self._text_scale
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            text.text = rec["name"]
            fused_markers.markers.append(text)
            marker_id += 1

            bp = pose.position
            fused_det = {
                    "class_name": rec["class_name"],
                    "instance_name": rec["name"],
                    "confidence": rec["confidence"],
                    "raw_confidence": rec.get("raw_confidence", 0.0),
                    "bbox_xyxy": rec["bbox_xyxy"],
                    "tf_frame": rec["tf_name"],
                    "source": rec["camera"],
                    "camera_name": rec["camera_name"],
                    "center_uv": rec.get("center_uv", []),
                    "camera_frame_id": rec.get("camera_frame_id", ""),
                    "pose_source": "depth_anchor",
                    "track_id": rec.get("track_id", -1),
                    "feature_track_id": rec.get("feature_track_id", -1),
                    "feature_quality_score": rec.get("feature_quality_score", 0.0),
                    "feature_tracked_count": rec.get("feature_tracked_count", 0),
                    "feature_inlier_count": rec.get("feature_inlier_count", 0),
                    "feature_mode": rec.get("feature_mode", ""),
                    "pose_base_link": {
                        "position": {
                            "x": float(bp.x),
                            "y": float(bp.y),
                            "z": float(bp.z),
                        },
                        "orientation": {
                            "x": float(pose.orientation.x),
                            "y": float(pose.orientation.y),
                            "z": float(pose.orientation.z),
                            "w": float(pose.orientation.w),
                        },
                    },
                }
            for key in (
                "servo_anchor_valid",
                "servo_anchor_source",
                "servo_anchor_quality",
                "mouth_center_uv",
                "mouth_left_uv",
                "mouth_right_uv",
                "mouth_angle_rad",
                "sc_port_center_uv",
                "sc_port_axis_left_uv",
                "sc_port_axis_right_uv",
                "sc_port_axis_angle_rad",
                "tip_uv",
                "front_center_uv",
                "front_left_uv",
                "front_right_uv",
                "front_angle_rad",
                "sc_plug_tip_uv",
                "sc_plug_axis_left_uv",
                "sc_plug_axis_right_uv",
                "sc_plug_axis_angle_rad",
            ):
                if key in rec:
                    fused_det[key] = rec[key]
            fused_json.append(fused_det)

        self._fused_pose_pub.publish(fused_pose_array)
        self._fused_marker_pub.publish(fused_markers)
        if transforms:
            self._tf_broadcaster.sendTransform(transforms)

        msg = String()
        msg.data = json.dumps(fused_json, separators=(",", ":"))
        self._fused_json_pub.publish(msg)
        self.get_logger().info(f"TRACK_YOLO_TF published={','.join(published_items)}")

    def _transform_pose_dict(self, pose: Optional[Dict], source_frame: str, target_frame: str) -> Optional[Dict]:
        if pose is None:
            return None
        t = np.asarray(pose.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        q = np.asarray(pose.get("q", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(4)
        if source_frame == target_frame:
            return {"t": t, "q": self._normalize_quat(q)}
        try:
            tf = self._tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except TransformException:
            try:
                tf = self._tf_buffer.lookup_transform(target_frame, source_frame, self.get_clock().now())
            except Exception:
                return None
        tr = tf.transform.translation
        rot = tf.transform.rotation
        tf_t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        tf_q = np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float64)
        R = self._quat_to_matrix(tf_q)
        out_t = R @ t + tf_t
        out_q = self._quat_multiply(tf_q, q)
        return {"t": out_t, "q": self._normalize_quat(out_q)}

    def _pose_dict_to_msg(self, pose: Dict) -> Pose:
        t = np.asarray(pose["t"], dtype=np.float64).reshape(3)
        q = self._normalize_quat(np.asarray(pose["q"], dtype=np.float64).reshape(4))
        return Pose(
            position=Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
            orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
        )

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

    def _make_axes_marker(self, marker_id: int, stamp, name: str, pose: Pose) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._base_frame
        marker.header.stamp = stamp
        marker.ns = "simple_yolo_depth_anchor_axes"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = self._axis_length_m
        marker.scale.y = self._axis_width_m
        marker.scale.z = self._axis_width_m
        marker.color = ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.9)
        marker.text = name
        return marker

    def _tf_safe_name(self, text: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text).strip().lower())
        return safe or "unknown"

    def _tf_sort_key(self, text: str):
        order = {
            "task_board": 0,
            "nic_card": 1,
            "sc_port": 2,
            "sfp_port": 3,
            "sfp_port_0": 3,
            "sfp_port_1": 4,
            "sfp_module": 5,
            "sc_plug": 6,
        }
        return (order.get(text, 99), text)

    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = CombinedYoloDepthPosePlanner._normalize_quat(q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = CombinedYoloDepthPosePlanner._normalize_quat(a)
        bx, by, bz, bw = CombinedYoloDepthPosePlanner._normalize_quat(b)
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
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
