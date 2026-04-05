"""
pose_estimator.py
-----------------
Standalone ROS2 test node that converts YOLO 2D bounding-box detections
into 3D poses expressed in the gripper (gripper/tcp) frame.

Pipeline per detection
  1.  Read bbox center (u, v) from /center_camera/yolo/detections_json
  2.  Read camera intrinsics from /center_camera/camera_info
  3.  Read metric depth at (u, v) from /center_camera/stereo_depth/image
      (published by the Depth-Anything-V2 node).
      Falls back to a configurable fixed depth if the depth node is not running.
  4.  Back-project (u, v, Z) → (X, Y, Z) in the camera optical frame
  5.  Transform the point into gripper/tcp via TF2
  6.  Log the result and publish a PointStamped on
      /yolo_pose/<class_name>/in_gripper for each detected class

Running
-------
  # make sure the depth node AND the YOLO node are running first, then:
  pixi shell
  ros2 run team_policy test_pose_estimator

  # Without depth-anything (fixed depth fallback):
  POSE_ESTIMATOR_FALLBACK_DEPTH_M=0.18 ros2 run team_policy test_pose_estimator

Environment variables
---------------------
  POSE_ESTIMATOR_CAMERA          center | left | right   (default: center)
  POSE_ESTIMATOR_GRIPPER_FRAME   gripper/tcp              (default: gripper/tcp)
  POSE_ESTIMATOR_FALLBACK_DEPTH_M  float in metres        (default: 0.20)
  POSE_ESTIMATOR_MAX_HZ          float                    (default: 5.0)
  POSE_ESTIMATOR_USE_DEPTH_NODE  1|0                      (default: 1)
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
import rclpy.duration
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

try:
    import tf2_geometry_msgs  # noqa: F401  registers transform support for PointStamped
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Detection:
    """One YOLO detection de-serialised from JSON."""

    def __init__(self, d: dict):
        self.class_id: int = int(d.get("class_id", 0))
        self.class_name: str = str(d.get("class_name", "unknown"))
        self.confidence: float = float(d.get("confidence", 0.0))
        x0, y0, x1, y1 = d.get("bbox_xyxy", [0, 0, 0, 0])
        self.u: float = (x0 + x1) / 2.0   # pixel column of bbox centre
        self.v: float = (y0 + y1) / 2.0   # pixel row   of bbox centre
        self.bbox: Tuple[float, float, float, float] = (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class PoseEstimatorNode(Node):
    """
    Converts YOLO detections to 3D poses in the gripper frame.
    Designed as a standalone test node — runs independently from mypolicy.
    """

    def __init__(self):
        super().__init__("yolo_pose_estimator")

        # ---- config from env ----
        self._camera: str = os.environ.get("POSE_ESTIMATOR_CAMERA", "center").strip().lower()
        self._gripper_frame: str = os.environ.get("POSE_ESTIMATOR_GRIPPER_FRAME", "gripper/tcp").strip()
        self._fallback_depth_m: float = float(os.environ.get("POSE_ESTIMATOR_FALLBACK_DEPTH_M", "0.20"))
        self._max_hz: float = max(0.1, float(os.environ.get("POSE_ESTIMATOR_MAX_HZ", "5.0")))
        self._use_depth_node: bool = os.environ.get("POSE_ESTIMATOR_USE_DEPTH_NODE", "1") != "0"
        self._min_period: float = 1.0 / self._max_hz

        # ---- state ----
        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._last_tick: float = 0.0

        self._latest_detections_json: Optional[str] = None
        self._latest_camera_info: Optional[CameraInfo] = None
        self._latest_depth: Optional[np.ndarray] = None   # shape (H, W), metric float32

        # ---- TF ----
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        # ---- topic names derived from chosen camera ----
        cam = self._camera
        det_topic   = f"/{cam}_camera/yolo/detections_json"
        info_topic  = f"/{cam}_camera/camera_info"
        depth_topic = f"/{cam}_camera/stereo_depth/image"   # from depth-anything node

        # ---- subscriptions ----
        self.create_subscription(String,     det_topic,   self._detections_cb, 10)
        self.create_subscription(CameraInfo, info_topic,  self._camera_info_cb, 10)
        if self._use_depth_node:
            self.create_subscription(Image,  depth_topic, self._depth_cb, 10)

        # ---- publishers (one per detection, created on demand) ----
        self._pose_pubs: Dict[str, object] = {}

        # ---- timer ----
        self.create_timer(1.0 / self._max_hz, self._tick)

        # ---- log startup info ----
        self.get_logger().info("=" * 60)
        self.get_logger().info("YOLO Pose Estimator started")
        self.get_logger().info(f"  Camera      : {cam}")
        self.get_logger().info(f"  Gripper TF  : {self._gripper_frame}")
        self.get_logger().info(f"  Depth node  : {'enabled — ' + depth_topic if self._use_depth_node else 'disabled (fallback used)'}")
        self.get_logger().info(f"  Fallback Z  : {self._fallback_depth_m:.3f} m")
        self.get_logger().info(f"  Rate        : {self._max_hz:.1f} Hz")
        self.get_logger().info("Subscribing to:")
        self.get_logger().info(f"  {det_topic}")
        self.get_logger().info(f"  {info_topic}")
        if self._use_depth_node:
            self.get_logger().info(f"  {depth_topic}")
        self.get_logger().info("=" * 60)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _detections_cb(self, msg: String) -> None:
        with self._lock:
            self._latest_detections_json = msg.data

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = msg

    def _depth_cb(self, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            with self._lock:
                self._latest_depth = np.array(depth, dtype=np.float32)
        except Exception as exc:
            self.get_logger().warn(f"Depth decode failed: {exc}")

    # ------------------------------------------------------------------
    # Processing tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = time.monotonic()
        if now - self._last_tick < self._min_period:
            return
        self._last_tick = now

        with self._lock:
            det_json = self._latest_detections_json
            info     = self._latest_camera_info
            depth    = self._latest_depth.copy() if self._latest_depth is not None else None

        if det_json is None:
            self.get_logger().warn("Waiting for YOLO detections …", throttle_duration_sec=3.0)
            return
        if info is None:
            self.get_logger().warn("Waiting for CameraInfo …", throttle_duration_sec=3.0)
            return

        # Parse detections
        try:
            raw = json.loads(det_json)
            detections: List[Detection] = [Detection(d) for d in raw]
        except Exception as exc:
            self.get_logger().error(f"JSON parse error: {exc}")
            return

        if not detections:
            return

        # Camera intrinsics
        fx, fy, cx, cy = self._parse_intrinsics(info)
        if fx < 1e-6 or fy < 1e-6:
            self.get_logger().warn("CameraInfo K matrix is zero — not calibrated?")
            return

        camera_frame = info.header.frame_id
        if not camera_frame:
            self.get_logger().warn("CameraInfo has empty frame_id — cannot transform.")
            return

        # Process each detection
        for det in detections:
            self._process_detection(det, fx, fy, cx, cy, depth, camera_frame)

    # ------------------------------------------------------------------
    # Back-projection + TF transform
    # ------------------------------------------------------------------

    def _process_detection(
        self,
        det: Detection,
        fx: float, fy: float, cx: float, cy: float,
        depth: Optional[np.ndarray],
        camera_frame: str,
    ) -> None:
        u, v = det.u, det.v

        # ---- Step 1: get depth Z at (u,v) ----
        Z = self._get_depth_at(u, v, depth)
        depth_source = "depth_node" if (depth is not None and self._use_depth_node) else "fallback"

        # ---- Step 2: back-project to camera 3D ----
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # ---- Step 3: build PointStamped in camera frame ----
        pt_cam = PointStamped()
        pt_cam.header.stamp    = self.get_clock().now().to_msg()
        pt_cam.header.frame_id = camera_frame
        pt_cam.point.x = float(X)
        pt_cam.point.y = float(Y)
        pt_cam.point.z = float(Z)

        # ---- Step 4: transform to gripper frame ----
        pt_gripper = self._transform_to_gripper(pt_cam)

        if pt_gripper is None:
            self.get_logger().warn(
                f"TF lookup failed for {det.class_name}: camera={camera_frame} → {self._gripper_frame}",
                throttle_duration_sec=2.0,
            )
            # Still log the camera-frame result
            self.get_logger().info(
                f"[{det.class_name}] conf={det.confidence:.2f} "
                f"pixel=({u:.1f},{v:.1f}) "
                f"camera_xyz=({X:.4f}, {Y:.4f}, {Z:.4f}) m [{depth_source}] "
                f"| TF to gripper: FAILED"
            )
            return

        gx = pt_gripper.point.x
        gy = pt_gripper.point.y
        gz = pt_gripper.point.z

        # ---- Step 5: publish + log ----
        self._publish_pose(det.class_name, pt_gripper)

        self.get_logger().info(
            f"[{det.class_name}] conf={det.confidence:.2f} "
            f"pixel=({u:.1f},{v:.1f}) "
            f"camera_xyz=({X:.4f}, {Y:.4f}, {Z:.4f}) m [{depth_source}] "
            f"→ gripper_xyz=({gx:.4f}, {gy:.4f}, {gz:.4f}) m"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_depth_at(self, u: float, v: float, depth: Optional[np.ndarray]) -> float:
        """Return metric depth (Z) at pixel (u,v). Falls back to fixed value."""
        if depth is not None and self._use_depth_node:
            h, w = depth.shape[:2]
            r = int(np.clip(round(v), 0, h - 1))
            c = int(np.clip(round(u), 0, w - 1))

            # Sample a small window around the center for robustness
            r0, r1 = max(0, r - 2), min(h, r + 3)
            c0, c1 = max(0, c - 2), min(w, c + 3)
            patch = depth[r0:r1, c0:c1]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            if valid.size > 0:
                return float(np.median(valid))

        return self._fallback_depth_m

    def _parse_intrinsics(self, info: CameraInfo) -> Tuple[float, float, float, float]:
        """Extract fx, fy, cx, cy from CameraInfo.K."""
        k = info.k
        if len(k) < 9:
            return 0.0, 0.0, 0.0, 0.0
        return float(k[0]), float(k[4]), float(k[2]), float(k[5])

    def _transform_to_gripper(self, pt: PointStamped) -> Optional[PointStamped]:
        """Transform a PointStamped from camera frame to gripper frame."""
        gripper_candidates = [self._gripper_frame, "tcp", "tool0", "ee_link"]
        for frame in gripper_candidates:
            try:
                transformed = self._tf_buffer.transform(
                    pt,
                    frame,
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
                if frame != self._gripper_frame:
                    self.get_logger().info(
                        f"Using fallback gripper frame '{frame}' ('{self._gripper_frame}' not found)",
                        throttle_duration_sec=5.0,
                    )
                return transformed
            except Exception:
                continue
        return None

    def _publish_pose(self, class_name: str, pt: PointStamped) -> None:
        """Publish on /yolo_pose/<class_name>/in_gripper (created lazily)."""
        safe_name = class_name.replace(" ", "_").replace("/", "_")
        topic = f"/yolo_pose/{safe_name}/in_gripper"
        if topic not in self._pose_pubs:
            self._pose_pubs[topic] = self.create_publisher(PointStamped, topic, 10)
            self.get_logger().info(f"Publishing pose on: {topic}")
        self._pose_pubs[topic].publish(pt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
