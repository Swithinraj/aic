"""
Corrected mypolicy with robust pixel-servo alignment and compliant insertion.

PIPELINE (6 stages)
-------------------
    Stage 1: YOLO sampling (15s) — average port pose + SFP module red axis
    Stage 2: Cartesian move to sampled pose
    Stage 3: Pitch orientation alignment (red-axis → down)
    Stage 4: Multi-camera pixel-servo fine alignment (FIXED)
    Stage 5: Pre-insert verification + local search
    Stage 6: Compliant force-guided insertion with spiral recovery

KEY FIXES over original:
    - Pixel servo now uses per-camera Jacobian with proper finite-difference probing
    - Port center tracked with exponential moving average + outlier rejection
    - Plug tip locked once from OBB geometry, never re-estimated
    - Convergence requires ALL active cameras below threshold for N stable frames
    - Insertion only fires after visual verification passes
    - Force-guided insertion uses delta-force (not absolute) for jam detection
    - Spiral search preserves visual alignment score
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge
from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3, Wrench
from nav_msgs.msg import Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from team_policy.planner.cartesian_planner import CartesianPlanner
from team_policy.planner.combined_yolo_depth_pose_planner import CombinedYoloDepthPosePlanner


# ═══════════════════════════════════════════════════════════════════════════════
# Quaternion / geometry helpers (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def _copy_pose(pose: Pose) -> Pose:
    return deepcopy(pose)


def _quat_to_np(q: Quaternion) -> List[float]:
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def _quat_normalize(q) -> List[float]:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / n, y / n, z / n, w / n]


def _quat_inverse(q) -> List[float]:
    x, y, z, w = _quat_normalize(q)
    return [-x, -y, -z, w]


def _quat_multiply(a, b) -> List[float]:
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_from_axis_angle(axis, angle: float) -> List[float]:
    axis_np = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis_np))
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    axis_np /= n
    s = math.sin(0.5 * float(angle))
    c = math.cos(0.5 * float(angle))
    return [float(axis_np[0] * s), float(axis_np[1] * s), float(axis_np[2] * s), float(c)]


def _quat_to_rotvec(q) -> tuple[List[float], float]:
    qq = _quat_normalize(q)
    if qq[3] < 0.0:
        qq = [-qq[0], -qq[1], -qq[2], -qq[3]]
    vx, vy, vz, vw = qq
    sin_half = math.sqrt(vx * vx + vy * vy + vz * vz)
    if sin_half < 1e-9:
        return [0.0, 0.0, 0.0], 0.0
    axis = [vx / sin_half, vy / sin_half, vz / sin_half]
    angle = 2.0 * math.atan2(sin_half, max(1e-12, vw))
    return [axis[0] * angle, axis[1] * angle, axis[2] * angle], abs(angle)


def _quat_error_rotvec(current: Quaternion, target: Quaternion) -> tuple[List[float], float]:
    qc = _quat_normalize(_quat_to_np(current))
    qt = _quat_normalize(_quat_to_np(target))
    return _quat_to_rotvec(_quat_multiply(qt, _quat_inverse(qc)))


def _rotate_vector_by_quaternion(q, v) -> np.ndarray:
    qn = _quat_normalize(q)
    vq = [float(v[0]), float(v[1]), float(v[2]), 0.0]
    qr = _quat_multiply(_quat_multiply(qn, vq), _quat_inverse(qn))
    return np.asarray(qr[:3], dtype=np.float64)


def _normalize_vec(v) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(vv))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vv / n


def _project_to_plane(v, plane_normal) -> np.ndarray:
    nn = _normalize_vec(plane_normal)
    if float(np.linalg.norm(nn)) < 1e-12:
        return np.asarray(v, dtype=np.float64).reshape(3)
    vv = np.asarray(v, dtype=np.float64).reshape(3)
    return vv - float(np.dot(vv, nn)) * nn


def _signed_angle_about_axis_rad(v_from, v_to, axis) -> float:
    aa = _normalize_vec(v_from)
    bb = _normalize_vec(v_to)
    kk = _normalize_vec(axis)
    if float(np.linalg.norm(aa)) < 1e-9 or float(np.linalg.norm(bb)) < 1e-9 or float(np.linalg.norm(kk)) < 1e-9:
        return 0.0
    sin_term = float(np.dot(kk, np.cross(aa, bb)))
    cos_term = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(math.atan2(sin_term, cos_term))


# ═══════════════════════════════════════════════════════════════════════════════
# DetectionListener (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionListener(Node):
    def __init__(self):
        super().__init__("mypolicy_detection_listener")
        self._lock = threading.Lock()
        self._latest_fused = {"time": 0.0, "detections": []}
        self._latest_per_camera = {
            "left": {"time": 0.0, "detections": []},
            "center": {"time": 0.0, "detections": []},
            "right": {"time": 0.0, "detections": []},
        }
        self.create_subscription(String, "/fused_yolo/detections_json", self._cb_fused, 10)
        self.create_subscription(String, "/left_camera/yolo/detections_json", lambda msg: self._cb_camera("left", msg), 10)
        self.create_subscription(String, "/center_camera/yolo/detections_json", lambda msg: self._cb_camera("center", msg), 10)
        self.create_subscription(String, "/right_camera/yolo/detections_json", lambda msg: self._cb_camera("right", msg), 10)

    def _parse_detection_list(self, raw: str) -> List[Dict]:
        try:
            parsed = json.loads(raw)
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []
        return [dict(item) for item in parsed if isinstance(item, dict)]

    def _cb_fused(self, msg: String) -> None:
        with self._lock:
            self._latest_fused = {"time": time.monotonic(), "detections": self._parse_detection_list(msg.data)}

    def _cb_camera(self, camera_name: str, msg: String) -> None:
        detections = self._parse_detection_list(msg.data)
        for det in detections:
            det["camera_name"] = camera_name
        with self._lock:
            self._latest_per_camera[camera_name] = {"time": time.monotonic(), "detections": detections}

    def get_all_detections(self, freshness_sec: float = 2.0) -> List[Dict]:
        now = time.monotonic()
        with self._lock:
            fused_time = float(self._latest_fused["time"])
            if fused_time > 0.0 and now - fused_time <= float(freshness_sec):
                return [dict(det) for det in self._latest_fused["detections"]]
            merged: List[Dict] = []
            for camera_name, snapshot in self._latest_per_camera.items():
                cam_time = float(snapshot["time"])
                if cam_time <= 0.0 or now - cam_time > float(freshness_sec):
                    continue
                for det in snapshot["detections"]:
                    item = dict(det)
                    item["camera_name"] = camera_name
                    merged.append(item)
            return merged

    def get_camera_detections(self, camera_name: str, freshness_sec: float = 1.0) -> List[Dict]:
        now = time.monotonic()
        with self._lock:
            snapshot = self._latest_per_camera.get(camera_name, {"time": 0.0, "detections": []})
            update_time = float(snapshot["time"])
            if update_time <= 0.0 or now - update_time > float(freshness_sec):
                return []
            return [dict(det) for det in snapshot["detections"]]


# ═══════════════════════════════════════════════════════════════════════════════
# MotionServoNode (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class MotionServoNode(Node):
    def __init__(self):
        super().__init__("mypolicy_motion_servo")
        self._lock = threading.Lock()
        self._current_state: Optional[ControllerState] = None
        self._current_tcp_pose: Optional[Pose] = None
        self._mode_request_sent = False

        self.command_frame = "base_link"
        self.position_tolerance = float(os.environ.get("MYPOLICY_POSITION_TOLERANCE_M", "0.008"))
        self.linear_kp = float(os.environ.get("MYPOLICY_LINEAR_KP", "1.0"))
        self.angular_kp = float(os.environ.get("MYPOLICY_ANGULAR_KP", "1.2"))
        self.max_linear_speed = float(os.environ.get("MYPOLICY_MAX_LINEAR_SPEED", "0.045"))
        self.min_linear_speed = float(os.environ.get("MYPOLICY_MIN_LINEAR_SPEED", "0.006"))
        self.max_angular_speed = float(os.environ.get("MYPOLICY_MAX_ANGULAR_SPEED", "0.30"))
        self.min_angular_speed = float(os.environ.get("MYPOLICY_MIN_ANGULAR_SPEED", "0.02"))
        self.trans_stiffness = float(os.environ.get("MYPOLICY_TRANS_STIFFNESS", "90.0"))
        self.rot_stiffness = float(os.environ.get("MYPOLICY_ROT_STIFFNESS", "55.0"))
        self.trans_damping = float(os.environ.get("MYPOLICY_TRANS_DAMPING", "55.0"))
        self.rot_damping = float(os.environ.get("MYPOLICY_ROT_DAMPING", "25.0"))

        self.create_subscription(ControllerState, "/aic_controller/controller_state", self._on_controller_state, 10)
        self.pose_command_pub = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)
        self.target_marker_pub = self.create_publisher(Marker, "/planner/target_marker", 10)
        self.waypoint_markers_pub = self.create_publisher(MarkerArray, "/planner/waypoint_markers", 10)
        self.path_pub = self.create_publisher(Path, "/planner/waypoint_path", 10)
        self.change_mode_client = self.create_client(ChangeTargetMode, "/aic_controller/change_target_mode")

    def _on_controller_state(self, msg: ControllerState) -> None:
        with self._lock:
            self._current_state = msg
            self._current_tcp_pose = _copy_pose(msg.tcp_pose)
        if not self._mode_request_sent:
            self.ensure_cartesian_mode()

    def ensure_cartesian_mode(self) -> None:
        if self._mode_request_sent:
            return
        if not self.change_mode_client.wait_for_service(timeout_sec=0.1):
            return
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN
        self.change_mode_client.call_async(request)
        self._mode_request_sent = True
        self.get_logger().info("Requested Cartesian target mode.")

    def get_current_pose(self) -> Optional[Pose]:
        with self._lock:
            return None if self._current_tcp_pose is None else _copy_pose(self._current_tcp_pose)

    def compute_twist_to_waypoint(self, current_pose: Pose, waypoint: Pose) -> Twist:
        dx = float(waypoint.position.x - current_pose.position.x)
        dy = float(waypoint.position.y - current_pose.position.y)
        dz = float(waypoint.position.z - current_pose.position.z)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        rotvec, angle = _quat_error_rotvec(current_pose.orientation, waypoint.orientation)
        twist = Twist()
        if distance >= 1e-6:
            commanded_speed = self.linear_kp * distance
            commanded_speed = min(self.max_linear_speed, max(self.min_linear_speed, commanded_speed))
            scale = commanded_speed / distance
            twist.linear.x = dx * scale
            twist.linear.y = dy * scale
            twist.linear.z = dz * scale
        if angle >= 1e-6:
            commanded_ang = self.angular_kp * angle
            commanded_ang = min(self.max_angular_speed, max(self.min_angular_speed, commanded_ang))
            rv_norm = math.sqrt(rotvec[0] ** 2 + rotvec[1] ** 2 + rotvec[2] ** 2)
            if rv_norm > 1e-9:
                a_scale = commanded_ang / rv_norm
                twist.angular.x = rotvec[0] * a_scale
                twist.angular.y = rotvec[1] * a_scale
                twist.angular.z = rotvec[2] * a_scale
        return twist

    def publish_twist_command(
        self,
        twist: Twist,
        frame_id: Optional[str] = None,
        trans_stiffness: Optional[float] = None,
        rot_stiffness: Optional[float] = None,
        trans_damping: Optional[float] = None,
        rot_damping: Optional[float] = None,
    ) -> None:
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.command_frame if frame_id is None else str(frame_id)
        msg.velocity = twist
        ts = self.trans_stiffness if trans_stiffness is None else float(trans_stiffness)
        rs = self.rot_stiffness if rot_stiffness is None else float(rot_stiffness)
        td = self.trans_damping if trans_damping is None else float(trans_damping)
        rd = self.rot_damping if rot_damping is None else float(rot_damping)
        msg.target_stiffness = [
            ts, 0, 0, 0, 0, 0,  0, ts, 0, 0, 0, 0,  0, 0, ts, 0, 0, 0,
            0, 0, 0, rs, 0, 0,  0, 0, 0, 0, rs, 0,   0, 0, 0, 0, 0, rs,
        ]
        msg.target_damping = [
            td, 0, 0, 0, 0, 0,  0, td, 0, 0, 0, 0,  0, 0, td, 0, 0, 0,
            0, 0, 0, rd, 0, 0,  0, 0, 0, 0, rd, 0,   0, 0, 0, 0, 0, rd,
        ]
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        self.pose_command_pub.publish(msg)

    def stop(self) -> None:
        self.publish_twist_command(Twist())

    def publish_target_marker(self, pose: Pose) -> None:
        marker = Marker()
        marker.header.frame_id = self.command_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "planner_target"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.08
        marker.scale.y = 0.012
        marker.scale.z = 0.012
        marker.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
        self.target_marker_pub.publish(marker)

    def publish_waypoint_visuals(self, waypoints: List[Pose]) -> None:
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        for index, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = self.command_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "planner_waypoints"
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = waypoint
            marker.scale.x = 0.05
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            marker_array.markers.append(marker)
        self.waypoint_markers_pub.publish(marker_array)
        path = Path()
        path.header.frame_id = self.command_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for waypoint in waypoints:
            ps = PoseStamped()
            ps.header.frame_id = self.command_frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose = waypoint
            path.poses.append(ps)
        self.path_pub.publish(path)


# ═══════════════════════════════════════════════════════════════════════════════
# PixelServoState — clean per-camera tracking state
# ═══════════════════════════════════════════════════════════════════════════════

class PixelServoState:
    """Tracks per-camera pixel alignment state with proper filtering."""

    def __init__(self):
        self.port_uv_ema: Dict[str, np.ndarray] = {}       # EMA-filtered port UV
        self.port_uv_history: Dict[str, deque] = {}         # raw history for outlier detection
        self.locked_tip_uv: Dict[str, np.ndarray] = {}      # one-shot locked plug tip UV
        self.jacobian: Dict[str, Optional[np.ndarray]] = {}  # per-camera 2x2 Jacobian (px/m)
        self.jacobian_valid: Dict[str, bool] = {}
        self.total_xy_correction_m = 0.0
        self.correction_history: List[np.ndarray] = []

    def reset(self):
        self.port_uv_ema.clear()
        self.port_uv_history.clear()
        self.locked_tip_uv.clear()
        self.jacobian.clear()
        self.jacobian_valid.clear()
        self.total_xy_correction_m = 0.0
        self.correction_history.clear()

    def update_port_uv(self, camera: str, raw_uv: np.ndarray, ema_alpha: float = 0.4) -> np.ndarray:
        """Update port UV with EMA + outlier rejection. Returns filtered UV."""
        if camera not in self.port_uv_history:
            self.port_uv_history[camera] = deque(maxlen=15)

        history = self.port_uv_history[camera]
        history.append(raw_uv.copy())

        # Outlier rejection: if we have enough history, reject jumps > 20px
        if camera in self.port_uv_ema and len(history) >= 3:
            prev = self.port_uv_ema[camera]
            jump = float(np.linalg.norm(raw_uv - prev))
            if jump > 20.0:
                # Reject this measurement, return previous EMA
                return self.port_uv_ema[camera].copy()

        # EMA update
        if camera not in self.port_uv_ema:
            self.port_uv_ema[camera] = raw_uv.copy()
        else:
            self.port_uv_ema[camera] = (
                ema_alpha * raw_uv + (1.0 - ema_alpha) * self.port_uv_ema[camera]
            )

        return self.port_uv_ema[camera].copy()

    def get_or_lock_tip(self, camera: str, module_det: Dict, port_uv: np.ndarray) -> np.ndarray:
        """Lock plug tip UV on first call per camera, never re-estimate."""
        if camera in self.locked_tip_uv:
            return self.locked_tip_uv[camera]

        tip_uv = self._estimate_plug_tip(module_det, port_uv)
        self.locked_tip_uv[camera] = tip_uv
        return tip_uv

    @staticmethod
    def _estimate_plug_tip(module_det: Dict, port_uv: np.ndarray) -> np.ndarray:
        """Estimate the plug tip as the midpoint of the OBB short edge closest to port."""
        corners = module_det.get("obb_corners_uv")
        if isinstance(corners, list) and len(corners) == 4:
            q = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            edge_mids = []
            edge_lens = []
            for i in range(4):
                a, b = q[i], q[(i + 1) % 4]
                edge_mids.append((a + b) * 0.5)
                edge_lens.append(float(np.linalg.norm(b - a)))

            # Pick the short-edge midpoint closest to port
            max_len = max(edge_lens) if edge_lens else 1.0
            best_mid = None
            best_score = float("inf")
            for mid, length in zip(edge_mids, edge_lens):
                # Prefer short edges (low length/max_len) that are close to port
                dist = float(np.linalg.norm(mid - port_uv))
                aspect_penalty = length / max(1.0, max_len)
                score = dist * (0.3 + 0.7 * aspect_penalty)
                if score < best_score:
                    best_score = score
                    best_mid = mid

            if best_mid is not None:
                return best_mid.astype(np.float64)

        # Fallback: bbox side closest to port
        bbox = module_det.get("bbox_xyxy", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            sides = [
                np.array([cx, y1]),
                np.array([cx, y2]),
                np.array([x1, cy]),
                np.array([x2, cy]),
            ]
            return min(sides, key=lambda s: float(np.linalg.norm(s - port_uv))).astype(np.float64)

        return port_uv.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Policy
# ═══════════════════════════════════════════════════════════════════════════════

class mypolicy(Policy):
    # ── timing ───────────────────────────────────────────────────────────────
    SAMPLE_SEC = float(os.environ.get("MYPOLICY_SAMPLE_SEC", "15.0"))
    SAMPLE_MIN_PORT_SAMPLES = int(os.environ.get("MYPOLICY_SAMPLE_MIN_PORT_SAMPLES", "5"))
    SAMPLE_FRESHNESS_SEC = float(os.environ.get("MYPOLICY_SAMPLE_FRESHNESS_SEC", "1.2"))
    MOVE_STEP_SEC = float(os.environ.get("MYPOLICY_MOVE_STEP_SEC", "0.05"))

    # ── module red axis ──────────────────────────────────────────────────────
    MODULE_RED_AXIS_LOCAL = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_LOCAL", "x")).strip().lower()
    MODULE_RED_AXIS_TARGET_BASE = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_TARGET_BASE", "down")).strip().lower()
    MODULE_RED_AXIS_PITCH_AXIS_BASE = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_PITCH_AXIS_BASE", "x")).strip().lower()
    MODULE_RED_AXIS_ALIGN_SIGN = float(os.environ.get("MYPOLICY_MODULE_RED_AXIS_ALIGN_SIGN", "1.0"))
    MODULE_RED_AXIS_MAX_ROT_STEP_DEG = float(os.environ.get("MYPOLICY_MODULE_RED_AXIS_MAX_ROT_STEP_DEG", "95.0"))
    MODULE_RED_AXIS_TOL_DEG = float(os.environ.get("MYPOLICY_MODULE_RED_AXIS_TOL_DEG", "2.5"))
    MODULE_RED_AXIS_MAX_ANGULAR_SPEED = float(os.environ.get("MYPOLICY_MODULE_RED_AXIS_MAX_ANGULAR_SPEED", "0.28"))
    MODULE_RED_AXIS_USE_LATEST_IF_NO_SAMPLE = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_USE_LATEST_IF_NO_SAMPLE", "1")).strip().lower() not in ("0", "false", "no", "off")
    MODULE_RED_AXIS_ALLOW_BIDIRECTIONAL_VERTICAL = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_ALLOW_BIDIRECTIONAL_VERTICAL", "0")).strip().lower() in ("1", "true", "yes", "on")
    SERVO_MIN_CONFIDENCE = float(os.environ.get("MYPOLICY_SERVO_MIN_CONFIDENCE", "0.20"))

    # ── pixel-servo alignment (CORRECTED tuning) ────────────────────────────
    PIXEL_SERVO_MAX_ITERS     = int(os.environ.get("MYPOLICY_PIXEL_SERVO_MAX_ITERS", "80"))
    PIXEL_SERVO_PX_TOL        = float(os.environ.get("MYPOLICY_PIXEL_SERVO_PX_TOL", "5.0"))
    PIXEL_SERVO_STABLE_FRAMES = int(os.environ.get("MYPOLICY_PIXEL_SERVO_STABLE_FRAMES", "5"))
    PIXEL_SERVO_LAMBDA        = float(os.environ.get("MYPOLICY_PIXEL_SERVO_LAMBDA", "0.50"))
    PIXEL_SERVO_MAX_STEP_M    = float(os.environ.get("MYPOLICY_PIXEL_SERVO_MAX_STEP_M", "0.0015"))
    PIXEL_SERVO_PROBE_M       = float(os.environ.get("MYPOLICY_PIXEL_SERVO_PROBE_M", "0.0010"))
    PIXEL_SERVO_SETTLE_SEC    = float(os.environ.get("MYPOLICY_PIXEL_SERVO_SETTLE_SEC", "0.25"))
    PIXEL_SERVO_MAX_TOTAL_M   = float(os.environ.get("MYPOLICY_PIXEL_SERVO_MAX_TOTAL_M", "0.030"))
    PIXEL_SERVO_CAMERA_ORDER  = ["center", "left", "right"]
    PIXEL_SERVO_REFIT_AFTER_DIVERGE = True
    PIXEL_SERVO_MEDIAN_FRAMES = int(os.environ.get("MYPOLICY_PIXEL_SERVO_MEDIAN_FRAMES", "5"))

    # ── compliant insertion ──────────────────────────────────────────────────
    INSERT_SPEED_MPS           = float(os.environ.get("MYPOLICY_INSERT_SPEED_MPS", "0.003"))
    INSERT_MAX_DEPTH_M         = float(os.environ.get("MYPOLICY_INSERT_MAX_DEPTH_M", "0.025"))
    INSERT_FORCE_DELTA_THRESH_N = float(os.environ.get("MYPOLICY_INSERT_FORCE_DELTA_THRESH_N", "6.0"))
    INSERT_FORCE_HARD_ABS_N    = float(os.environ.get("MYPOLICY_INSERT_FORCE_HARD_ABS_THRESH_N", "35.0"))
    INSERT_FORCE_JAM_COUNT     = int(os.environ.get("MYPOLICY_INSERT_FORCE_JAM_COUNT", "3"))
    INSERT_MAX_RETRIES         = int(os.environ.get("MYPOLICY_INSERT_MAX_RETRIES", "3"))
    INSERT_BASELINE_DURATION_S = float(os.environ.get("MYPOLICY_INSERT_BASELINE_DURATION_S", "0.5"))
    INSERT_BACKOFF_M           = float(os.environ.get("MYPOLICY_INSERT_BACKOFF_M", "0.003"))
    INSERT_CTRL_PERIOD_S       = 0.04

    INSERT_SPIRAL_ENABLE       = str(os.environ.get("MYPOLICY_INSERT_SPIRAL_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")
    INSERT_SPIRAL_RADII_M      = [0.00015, 0.00030, 0.00045, 0.00060, 0.00080, 0.00100]
    INSERT_SPIRAL_POINTS_PER_RING = 8
    INSERT_SPIRAL_PROBE_DEPTH_M = float(os.environ.get("MYPOLICY_INSERT_SPIRAL_PROBE_DEPTH_M", "0.00050"))
    INSERT_SPIRAL_BACKOFF_M    = float(os.environ.get("MYPOLICY_INSERT_SPIRAL_BACKOFF_M", "0.00100"))

    INSERT_FUNNEL_ENABLE       = str(os.environ.get("MYPOLICY_INSERT_FUNNEL_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")
    INSERT_FUNNEL_DELTA_XY_THRESH_N = float(os.environ.get("MYPOLICY_INSERT_FUNNEL_DELTA_XY_THRESH_N", "0.8"))
    INSERT_FUNNEL_GAIN_M_PER_N = float(os.environ.get("MYPOLICY_INSERT_FUNNEL_GAIN_M_PER_N", "0.00010"))
    INSERT_FUNNEL_MAX_XY_STEP_M = float(os.environ.get("MYPOLICY_INSERT_FUNNEL_MAX_XY_STEP_M", "0.00025"))
    INSERT_FUNNEL_MAX_TOTAL_XY_M = float(os.environ.get("MYPOLICY_INSERT_FUNNEL_MAX_TOTAL_XY_M", "0.0020"))

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._planner = CartesianPlanner()
        self._helper_executor = MultiThreadedExecutor(num_threads=3)
        self._detection_node = CombinedYoloDepthPosePlanner()
        self._detection_listener = DetectionListener()
        self._motion_servo = MotionServoNode()
        self._helper_executor.add_node(self._detection_node)
        self._helper_executor.add_node(self._detection_listener)
        self._helper_executor.add_node(self._motion_servo)
        self._helper_thread = threading.Thread(target=self._helper_executor.spin, daemon=True)
        self._helper_thread.start()

        self._taskboard_classes = self._parse_name_set(os.environ.get("YOLOV12_TASKBOARD_CLASSES", "taskboard,task_board,task board,board"))
        self._nic_classes = self._parse_name_set(os.environ.get("YOLOV12_NIC_CLASSES", "nic_card,nic card,nic,nic_card_0,nic_card_1,nic_card_2,nic_card_3,nic_card_4"))
        self._sfp_port_classes = self._parse_name_set(os.environ.get("YOLOV12_SFP_PORT_CLASSES", "sfp_port,sfp port,sfp_port_0,sfp_port_1,sfp_port_2,sfp_port_3"))
        self._sc_port_classes = self._parse_name_set(os.environ.get("YOLOV12_SC_PORT_CLASSES", "sc_port,sc port,sc_port_0,sc_port_1,sc_port_2,sc_port_3"))
        self._sfp_module_classes = self._parse_name_set(os.environ.get("YOLOV12_SFP_MODULE_CLASSES", "sfp_module,sfp module,transceiver,sfp_port_module,sfp port module,sfp-module,sfpmodule,sfp_transceiver"))

        self._hover_sampled_sfp_module_axis = None
        self._hover_sampled_sfp_module_pose = None
        self._hover_sampled_sfp_module_axis_count = 0
        self._hover_sampled_sfp_module_axis_cameras = []
        self._hover_sampled_sfp_module_axis_conf = 0.0
        self._cv_bridge = CvBridge()
        self._pixel_state = PixelServoState()
        self.get_logger().info("mypolicy_corrected: sample → move → pitch → pixel-servo → verify → insert")

    # ═══════════════════════════════════════════════════════════════════════════
    # Entry point
    # ═══════════════════════════════════════════════════════════════════════════

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        del move_robot
        send_feedback(f"mypolicy/start task={task.id}")
        observation = self._wait_for_observation(get_observation)
        if observation is None:
            send_feedback("mypolicy/fail no_observation")
            return False

        self._motion_servo.ensure_cartesian_mode()
        current_pose = self._motion_servo.get_current_pose() or observation.controller_state.tcp_pose
        gripper_orientation = Quaternion(
            x=float(current_pose.orientation.x),
            y=float(current_pose.orientation.y),
            z=float(current_pose.orientation.z),
            w=float(current_pose.orientation.w),
        )

        port_type = str(task.port_type).strip().lower()
        target_port_name = str(task.port_name).strip().lower()
        if port_type == "sfp":
            port_matcher = lambda det: self._matches_specific_port(det, target_port_name, self._sfp_port_classes)
        elif port_type == "sc":
            port_matcher = self._is_sc_port_detection
        else:
            port_matcher = self._is_nic_detection

        # ── Stage 1: Sample ──────────────────────────────────────────────────
        send_feedback("mypolicy/stage1_sampling_start")
        target_pose = self._sample_target_pose(
            label="hover_above_port",
            matcher=port_matcher,
            gripper_orientation=gripper_orientation,
            send_feedback=send_feedback,
        )
        if target_pose is None and port_type == "sfp":
            target_pose = self._sample_target_pose(
                label="hover_above_port_fallback",
                matcher=self._is_sfp_port_detection,
                gripper_orientation=gripper_orientation,
                send_feedback=send_feedback,
            )
        if target_pose is None:
            send_feedback("mypolicy/fail sampled_target_missing")
            self._motion_servo.stop()
            return False

        # ── Stage 2: Move ────────────────────────────────────────────────────
        send_feedback("mypolicy/stage2_move_start")
        move_ok = self._move_to_sampled_pose(target_pose, send_feedback=send_feedback)
        if not move_ok:
            send_feedback("mypolicy/fail move_failed")
            self._motion_servo.stop()
            return False

        # ── Stage 3: Pitch orientation ───────────────────────────────────────
        send_feedback("mypolicy/stage3_pitch_start")
        orient_ok = self._align_sfp_module_red_axis_after_hover(send_feedback=send_feedback)
        self._motion_servo.stop()
        if not orient_ok:
            send_feedback("mypolicy/fail pitch_failed")
            return False

        # ── Stage 4: Pixel-servo fine alignment (CORRECTED) ──────────────────
        send_feedback("mypolicy/stage4_pixel_servo_start")
        self._pixel_state.reset()
        servo_ok = self._pixel_servo_align(
            target_port_name=target_port_name,
            port_matcher=port_matcher,
            get_observation=get_observation,
            send_feedback=send_feedback,
        )
        self._motion_servo.stop()
        if not servo_ok:
            send_feedback("mypolicy/pixel_servo_failed falling_through_to_insert")
            # Don't abort — try insertion anyway if we're close

        # ── Stage 5: Pre-insert verification ─────────────────────────────────
        send_feedback("mypolicy/stage5_verify_start")
        verified = self._pre_insert_verify(
            target_port_name=target_port_name,
            port_matcher=port_matcher,
            get_observation=get_observation,
            send_feedback=send_feedback,
        )
        if not verified:
            send_feedback("mypolicy/verify_marginal proceeding_with_insertion")

        # ── Stage 6: Compliant insertion ─────────────────────────────────────
        send_feedback("mypolicy/stage6_insert_start")
        insert_ok = self._compliant_insert(
            target_port_name=target_port_name,
            port_matcher=port_matcher,
            get_observation=get_observation,
            send_feedback=send_feedback,
        )
        self._motion_servo.stop()
        send_feedback(f"mypolicy/done insert_ok={str(insert_ok).lower()}")
        return insert_ok

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 1: Sampling (unchanged from original)
    # ═══════════════════════════════════════════════════════════════════════════

    def _wait_for_observation(self, get_observation):
        while True:
            obs = get_observation()
            if obs is not None:
                return obs
            self.sleep_for(0.05)

    def _sample_target_pose(self, label, matcher, gripper_orientation, send_feedback) -> Optional[Pose]:
        send_feedback(f"mypolicy/{label}_sampling sec={self.SAMPLE_SEC:.1f}")
        deadline = time.monotonic() + max(0.5, float(self.SAMPLE_SEC))
        port_items: List[tuple] = []
        module_axis_items: List[tuple] = []
        module_pose_items: List[tuple] = []
        module_cameras = set()
        module_best_conf = 0.0

        self._hover_sampled_sfp_module_axis = None
        self._hover_sampled_sfp_module_pose = None
        self._hover_sampled_sfp_module_axis_count = 0
        self._hover_sampled_sfp_module_axis_cameras = []
        self._hover_sampled_sfp_module_axis_conf = 0.0

        while time.monotonic() < deadline:
            all_dets = self._detection_listener.get_all_detections(freshness_sec=self.SAMPLE_FRESHNESS_SEC)
            port_dets = [d for d in all_dets if matcher(d)]
            center_port_dets = [d for d in port_dets if str(d.get("camera_name", d.get("source", ""))) == "center"]
            if center_port_dets:
                port_dets = center_port_dets
            for det in port_dets:
                pose = self._pose_from_detection(det)
                if pose is None:
                    continue
                conf = max(0.05, float(det.get("confidence", 0.0)))
                cam = str(det.get("camera_name", det.get("source", "unknown")))
                weight = conf * (1.50 if cam == "center" else 1.0)
                port_items.append((pose, weight, cam))

            module_dets = [d for d in all_dets if self._is_sfp_module_detection(d)]
            center_module_dets = [d for d in module_dets if str(d.get("camera_name", d.get("source", ""))) == "center"]
            if center_module_dets:
                module_dets = center_module_dets
            for det in module_dets:
                pose = self._pose_from_detection(det)
                if pose is None:
                    continue
                red_axis = self._pose_local_axis_base(pose, self.MODULE_RED_AXIS_LOCAL)
                if float(np.linalg.norm(red_axis)) < 1e-9:
                    continue
                conf = max(0.05, float(det.get("confidence", 0.0)))
                cam = str(det.get("camera_name", det.get("source", "unknown")))
                weight = conf * (1.50 if cam == "center" else 1.0)
                module_axis_items.append((red_axis, weight))
                module_pose_items.append((pose, weight))
                module_cameras.add(cam)
                module_best_conf = max(module_best_conf, conf)
            self.sleep_for(0.05)

        sampled_axis = self._average_signed_axes(module_axis_items)
        if sampled_axis is not None:
            self._hover_sampled_sfp_module_axis = sampled_axis.copy()
            self._hover_sampled_sfp_module_pose = self._weighted_average_pose(module_pose_items)
            self._hover_sampled_sfp_module_axis_count = len(module_axis_items)
            self._hover_sampled_sfp_module_axis_cameras = sorted(module_cameras)
            self._hover_sampled_sfp_module_axis_conf = float(module_best_conf)
            send_feedback(f"mypolicy/{label}_module_axis samples={len(module_axis_items)}")

        if len(port_items) < max(1, int(self.SAMPLE_MIN_PORT_SAMPLES)):
            send_feedback(f"mypolicy/{label}_fail port_samples={len(port_items)}")
            return None

        target_pose = self._weighted_average_pose([(p, w) for p, w, _ in port_items])
        if target_pose is None:
            return None
        target_pose.orientation = Quaternion(
            x=float(gripper_orientation.x), y=float(gripper_orientation.y),
            z=float(gripper_orientation.z), w=float(gripper_orientation.w),
        )
        send_feedback(f"mypolicy/{label}_sampled samples={len(port_items)} xyz=({target_pose.position.x:.3f},{target_pose.position.y:.3f},{target_pose.position.z:.3f})")
        return target_pose

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 2: Move to sampled pose (unchanged)
    # ═══════════════════════════════════════════════════════════════════════════

    def _move_to_sampled_pose(self, target_pose: Pose, send_feedback) -> bool:
        self._motion_servo.ensure_cartesian_mode()
        current_pose = self._motion_servo.get_current_pose()
        while current_pose is None:
            self.sleep_for(0.05)
            current_pose = self._motion_servo.get_current_pose()

        waypoints = self._planner.plan_from_current_pose(current_pose, target_pose)
        if not waypoints:
            waypoints = [target_pose]

        self._motion_servo.publish_target_marker(target_pose)
        self._motion_servo.publish_waypoint_visuals(waypoints)
        send_feedback(f"mypolicy/move_start waypoints={len(waypoints)}")

        waypoint_idx = 0
        last_feedback_t = 0.0
        iter_count = 0

        while True:
            iter_count += 1
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(0.05)
                continue

            error = self._position_distance(current_pose, target_pose)
            if error <= self._motion_servo.position_tolerance:
                self._motion_servo.stop()
                send_feedback(f"mypolicy/move_complete err={error:.4f} iter={iter_count}")
                return True

            current_wp = waypoints[min(waypoint_idx, len(waypoints) - 1)]
            if self._position_distance(current_pose, current_wp) <= self._motion_servo.position_tolerance and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                current_wp = waypoints[waypoint_idx]

            twist = self._motion_servo.compute_twist_to_waypoint(current_pose, current_wp)
            twist.angular.x = twist.angular.y = twist.angular.z = 0.0
            self._motion_servo.publish_twist_command(twist, frame_id="base_link")

            now = time.monotonic()
            if now - last_feedback_t > 1.0:
                last_feedback_t = now
                send_feedback(f"mypolicy/move wp={waypoint_idx+1}/{len(waypoints)} err={error:.4f}")
            self.sleep_for(self.MOVE_STEP_SEC)

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 3: Pitch orientation (unchanged logic, condensed)
    # ═══════════════════════════════════════════════════════════════════════════

    def _align_sfp_module_red_axis_after_hover(self, send_feedback) -> bool:
        current_pose = self._motion_servo.get_current_pose()
        if current_pose is None:
            send_feedback("mypolicy/pitch_fail no_pose")
            return False

        axis_est = None
        if self._hover_sampled_sfp_module_axis is not None:
            axis_est = {"axis": np.asarray(self._hover_sampled_sfp_module_axis, dtype=np.float64)}
        elif self.MODULE_RED_AXIS_USE_LATEST_IF_NO_SAMPLE:
            axis_est = self._latest_sfp_module_red_axis_estimate()

        if axis_est is None:
            send_feedback("mypolicy/pitch_fail no_axis")
            return False

        red_axis = _normalize_vec(axis_est["axis"])
        target_pose, info = self._module_red_axis_alignment_target_pose_pitch_only(current_pose, red_axis)
        if not info.get("ok", False):
            send_feedback(f"mypolicy/pitch_fail reason={info.get('reason', 'unknown')}")
            return False

        send_feedback(f"mypolicy/pitch_apply err_deg={info['pitch_error_deg']:.2f} applied_deg={info['applied_deg']:.2f}")
        ok = self._execute_orientation_goal_pitch_only(
            target_pose=target_pose,
            pitch_axis_base=info["pitch_axis_base"],
            send_feedback=send_feedback,
            rotation_tolerance_rad=math.radians(max(0.2, float(self.MODULE_RED_AXIS_TOL_DEG))),
        )
        return ok

    def _module_red_axis_alignment_target_pose_pitch_only(self, current_pose, red_axis_base):
        red_axis = _normalize_vec(red_axis_base)
        desired = _normalize_vec(self._target_ground_direction_from_name(self.MODULE_RED_AXIS_TARGET_BASE))
        pitch_axis_base = _normalize_vec(self._target_ground_direction_from_name(self.MODULE_RED_AXIS_PITCH_AXIS_BASE))
        if float(np.linalg.norm(red_axis)) < 1e-9 or float(np.linalg.norm(desired)) < 1e-9:
            return _copy_pose(current_pose), {"ok": False, "reason": "bad_axis", "pitch_error_deg": float("inf"), "applied_deg": 0.0}

        if self.MODULE_RED_AXIS_ALLOW_BIDIRECTIONAL_VERTICAL and float(np.dot(red_axis, desired)) < 0.0:
            desired = -desired

        red_proj = _project_to_plane(red_axis, pitch_axis_base)
        desired_proj = _project_to_plane(desired, pitch_axis_base)
        if float(np.linalg.norm(red_proj)) < 1e-9 or float(np.linalg.norm(desired_proj)) < 1e-9:
            return _copy_pose(current_pose), {"ok": True, "reason": "parallel", "pitch_error_deg": 0.0, "applied_deg": 0.0, "pitch_axis_base": pitch_axis_base, "desired": desired}

        pitch_error = _signed_angle_about_axis_rad(red_proj, desired_proj, pitch_axis_base)
        max_step = math.radians(max(0.1, float(self.MODULE_RED_AXIS_MAX_ROT_STEP_DEG)))
        applied = float(np.clip(pitch_error * float(self.MODULE_RED_AXIS_ALIGN_SIGN), -max_step, max_step))
        q_cur = _quat_normalize(_quat_to_np(current_pose.orientation))
        q_delta = _quat_from_axis_angle(pitch_axis_base, applied)
        q_target = _quat_normalize(_quat_multiply(q_delta, q_cur))
        target = _copy_pose(current_pose)
        target.orientation = Quaternion(x=float(q_target[0]), y=float(q_target[1]), z=float(q_target[2]), w=float(q_target[3]))
        return target, {
            "ok": True, "reason": "computed",
            "pitch_error_deg": math.degrees(pitch_error), "applied_deg": math.degrees(applied),
            "pitch_axis_base": pitch_axis_base, "desired": desired,
        }

    def _execute_orientation_goal_pitch_only(self, target_pose, pitch_axis_base, send_feedback, rotation_tolerance_rad) -> bool:
        self._motion_servo.ensure_cartesian_mode()
        pitch_axis = _normalize_vec(pitch_axis_base)
        if float(np.linalg.norm(pitch_axis)) < 1e-9:
            return False

        iter_idx = 0
        last_feedback_t = 0.0
        while True:
            iter_idx += 1
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(0.03)
                continue

            rotvec, _ = _quat_error_rotvec(current_pose.orientation, target_pose.orientation)
            rotvec_np = np.asarray(rotvec, dtype=np.float64).reshape(3)
            pitch_err = float(np.dot(rotvec_np, pitch_axis))

            if abs(pitch_err) <= float(rotation_tolerance_rad):
                self._motion_servo.stop()
                send_feedback(f"mypolicy/pitch_complete iter={iter_idx} err_deg={math.degrees(abs(pitch_err)):.2f}")
                return True

            max_speed = max(0.02, float(self.MODULE_RED_AXIS_MAX_ANGULAR_SPEED))
            kp = max(0.05, float(self._motion_servo.angular_kp))
            pitch_cmd = float(np.clip(kp * pitch_err, -max_speed, max_speed))

            twist = Twist()
            twist.angular.x = float(pitch_axis[0] * pitch_cmd)
            twist.angular.y = float(pitch_axis[1] * pitch_cmd)
            twist.angular.z = float(pitch_axis[2] * pitch_cmd)
            self._motion_servo.publish_twist_command(twist, frame_id="base_link", trans_stiffness=90.0, rot_stiffness=45.0, trans_damping=55.0, rot_damping=20.0)

            now = time.monotonic()
            if now - last_feedback_t > 1.0:
                last_feedback_t = now
                send_feedback(f"mypolicy/pitch iter={iter_idx} err_deg={math.degrees(abs(pitch_err)):.2f}")
            self.sleep_for(0.04)

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 4: CORRECTED Pixel-Servo Fine Alignment
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_camera_image(self, camera_name: str, get_observation) -> Optional[np.ndarray]:
        """Decode camera image to BGR numpy array."""
        try:
            obs = get_observation()
            if obs is None:
                return None
            img_msg = getattr(obs, f"{camera_name}_image", None)
            if img_msg is None or img_msg.width == 0:
                return None
            return np.asarray(self._cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8"), dtype=np.uint8)
        except Exception:
            return None

    def _extract_port_center_uv(self, image: np.ndarray, port_det: Dict) -> Tuple[np.ndarray, str]:
        """Extract the port center from detection, trying dark-rectangle first."""
        # Try OBB center as primary (usually very accurate for SFP ports)
        obb = port_det.get("obb_cxcywh_deg")
        bbox = port_det.get("bbox_xyxy", [])

        if isinstance(obb, list) and len(obb) >= 2:
            fallback_uv = np.array([float(obb[0]), float(obb[1])], dtype=np.float64)
            source = "obb"
        elif len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            fallback_uv = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
            source = "bbox"
        else:
            h, w = image.shape[:2]
            return np.array([w * 0.5, h * 0.5], dtype=np.float64), "image_center"

        if len(bbox) != 4:
            return fallback_uv, source

        # Try dark-rectangle detection inside enlarged ROI
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        ih, iw = image.shape[:2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        # 1.5x enlarged ROI
        rx1 = max(0, int(cx - bw * 0.75))
        ry1 = max(0, int(cy - bh * 0.75))
        rx2 = min(iw, int(cx + bw * 0.75))
        ry2 = min(ih, int(cy + bh * 0.75))

        if rx2 - rx1 < 10 or ry2 - ry1 < 10:
            return fallback_uv, source

        try:
            crop = image[ry1:ry2, rx1:rx2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray)
            cl1 = cv2.medianBlur(cl1, 3)
            _, thresh = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            k = max(2, min(5, bw // 10))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_cnt = None
            best_score = -1.0
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < 20.0:
                    continue
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                short_side = min(rw, rh)
                long_side = max(rw, rh)
                if short_side < 2.0 or long_side / short_side > 5.0:
                    continue
                score = area / max(1.0, long_side / short_side)
                if score > best_score:
                    best_score = score
                    best_cnt = cnt

            if best_cnt is not None:
                M = cv2.moments(best_cnt)
                if abs(M["m00"]) > 1e-6:
                    cx_crop = M["m10"] / M["m00"]
                    cy_crop = M["m01"] / M["m00"]
                    return np.array([float(rx1) + cx_crop, float(ry1) + cy_crop], dtype=np.float64), "dark_rect"
        except Exception:
            pass

        return fallback_uv, source

    def _measure_pixel_error_single(
        self,
        camera: str,
        target_port_name: str,
        port_matcher,
        get_observation,
    ) -> Optional[Dict[str, Any]]:
        """Measure pixel error for a single camera: port_center - plug_tip."""
        image = self._get_camera_image(camera, get_observation)
        if image is None:
            return None

        dets = self._detection_listener.get_camera_detections(camera, freshness_sec=1.0)
        port_det = self._best_detection(dets, port_matcher)
        module_det = self._best_detection(dets, self._is_sfp_module_detection)

        if port_det is None or module_det is None:
            return None

        # Extract raw port center
        raw_port_uv, port_source = self._extract_port_center_uv(image, port_det)

        # EMA-filtered port center (rejects jitter and outliers)
        port_uv = self._pixel_state.update_port_uv(camera, raw_port_uv)

        # Lock plug tip once
        tip_uv = self._pixel_state.get_or_lock_tip(camera, module_det, port_uv)

        e = port_uv - tip_uv  # error: where port is relative to tip
        err_px = float(np.linalg.norm(e))

        return {
            "camera": camera,
            "err_px": err_px,
            "e": e.copy(),
            "port_uv": port_uv.copy(),
            "tip_uv": tip_uv.copy(),
            "port_source": port_source,
            "port_conf": float(port_det.get("confidence", 0.0)),
            "module_conf": float(module_det.get("confidence", 0.0)),
            "image": image,
        }

    def _measure_pixel_error_median(
        self,
        camera: str,
        target_port_name: str,
        port_matcher,
        get_observation,
        frames: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Median of N frames for robustness."""
        results = []
        for _ in range(frames):
            res = self._measure_pixel_error_single(camera, target_port_name, port_matcher, get_observation)
            if res is not None:
                results.append(res)
            self.sleep_for(0.04)

        if not results:
            return None

        errors = np.array([r["e"] for r in results])
        med_e = np.median(errors, axis=0)
        med_err = float(np.linalg.norm(med_e))

        # Pick the result closest to median
        best = min(results, key=lambda r: float(np.linalg.norm(r["e"] - med_e)))
        best["median_e"] = med_e
        best["median_err_px"] = med_err
        best["jitter_px"] = float(np.median([np.linalg.norm(r["e"] - med_e) for r in results]))
        return best

    def _select_alignment_camera(self, target_port_name: str, port_matcher) -> Optional[str]:
        """Find the best camera that sees both port and module."""
        for cam in self.PIXEL_SERVO_CAMERA_ORDER:
            dets = self._detection_listener.get_camera_detections(cam, freshness_sec=1.5)
            has_port = any(port_matcher(d) and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE for d in dets)
            has_module = any(self._is_sfp_module_detection(d) and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE for d in dets)
            if has_port and has_module:
                return cam
        # Fallback: any camera seeing the port
        for cam in self.PIXEL_SERVO_CAMERA_ORDER:
            dets = self._detection_listener.get_camera_detections(cam, freshness_sec=1.5)
            if any(port_matcher(d) and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE for d in dets):
                return cam
        return None

    def _estimate_pixel_jacobian(
        self,
        camera: str,
        target_port_name: str,
        port_matcher,
        get_observation,
        send_feedback,
        base_error: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Estimate 2x2 pixel Jacobian by finite-difference probing in X and Y.
        Moves +probe, measures, moves -probe (returns to start) for each axis.
        Returns J (2x2): de_px = J @ d_xy_meters.
        """
        probe_m = float(self.PIXEL_SERVO_PROBE_M)
        settle = float(self.PIXEL_SERVO_SETTLE_SEC)
        cols = []

        for axis_idx, (dx, dy) in enumerate([(probe_m, 0.0), (0.0, probe_m)]):
            # Move +probe
            self._command_small_xy(np.array([dx, dy]), max_step=probe_m * 1.5)
            self.sleep_for(settle)

            # Measure after probe
            meas = self._measure_pixel_error_median(
                camera, target_port_name, port_matcher, get_observation,
                frames=3,
            )

            # Return to start
            self._command_small_xy(np.array([-dx, -dy]), max_step=probe_m * 1.5)
            self.sleep_for(settle * 0.5)

            if meas is None:
                send_feedback(f"mypolicy/pixel_jacobian_fail axis={axis_idx} no_measurement")
                return None

            # Jacobian column: (error_after - error_before) / probe_distance
            col = (meas["median_e"] - base_error) / probe_m
            col_norm = float(np.linalg.norm(col))
            if col_norm < 1e-6:
                send_feedback(f"mypolicy/pixel_jacobian_fail axis={axis_idx} zero_sensitivity")
                return None

            cols.append(col)
            send_feedback(f"mypolicy/pixel_jacobian axis={axis_idx} de=({col[0]:+.1f},{col[1]:+.1f})px/m")

        J = np.column_stack(cols)
        if not np.all(np.isfinite(J)):
            return None

        # Validate: condition number shouldn't be too high
        cond = float(np.linalg.cond(J))
        if cond > 100.0:
            send_feedback(f"mypolicy/pixel_jacobian_warn cond={cond:.1f} (high)")

        return J

    def _pixel_servo_align(
        self,
        target_port_name: str,
        port_matcher,
        get_observation,
        send_feedback,
    ) -> bool:
        """
        CORRECTED pixel-servo loop:
          1. Select camera with both port + module visible
          2. Estimate 2x2 pixel Jacobian via finite-difference probing
          3. Iteratively correct XY using J^-1 @ error
          4. Refit Jacobian if correction diverges
          5. Declare success when error < tol for N stable frames
        """
        camera = self._select_alignment_camera(target_port_name, port_matcher)
        if camera is None:
            send_feedback("mypolicy/pixel_servo_fail no_camera")
            return False

        send_feedback(f"mypolicy/pixel_servo camera={camera}")

        # Initial measurement
        meas0 = self._measure_pixel_error_median(
            camera, target_port_name, port_matcher, get_observation,
            frames=self.PIXEL_SERVO_MEDIAN_FRAMES,
        )
        if meas0 is None:
            send_feedback("mypolicy/pixel_servo_fail no_initial_measurement")
            return False

        send_feedback(f"mypolicy/pixel_servo_initial err={meas0['median_err_px']:.1f}px port_src={meas0['port_source']}")

        if meas0["median_err_px"] <= self.PIXEL_SERVO_PX_TOL:
            send_feedback("mypolicy/pixel_servo_already_aligned")
            return True

        # Estimate Jacobian
        jacobian = self._estimate_pixel_jacobian(
            camera, target_port_name, port_matcher, get_observation,
            send_feedback, meas0["median_e"],
        )
        if jacobian is None:
            send_feedback("mypolicy/pixel_servo_fail jacobian_estimation_failed")
            return False

        self._pixel_state.jacobian[camera] = jacobian
        self._pixel_state.jacobian_valid[camera] = True

        # Servo loop
        stable_count = 0
        total_xy = 0.0
        best_err = meas0["median_err_px"]
        diverge_count = 0

        for iteration in range(self.PIXEL_SERVO_MAX_ITERS):
            # Measure current error
            meas = self._measure_pixel_error_median(
                camera, target_port_name, port_matcher, get_observation,
                frames=3,
            )
            if meas is None:
                # Camera lost — try to find another
                alt_cam = self._select_alignment_camera(target_port_name, port_matcher)
                if alt_cam is not None and alt_cam != camera:
                    camera = alt_cam
                    jacobian = None  # need new Jacobian for new camera
                    send_feedback(f"mypolicy/pixel_servo_camera_switch new={camera}")
                else:
                    send_feedback(f"mypolicy/pixel_servo_lost iter={iteration}")
                    break
                continue

            err_px = meas["median_err_px"]

            # Check convergence
            if err_px <= self.PIXEL_SERVO_PX_TOL:
                stable_count += 1
                if stable_count >= self.PIXEL_SERVO_STABLE_FRAMES:
                    send_feedback(f"mypolicy/pixel_servo_aligned iter={iteration} err={err_px:.1f}px stable={stable_count}")
                    return True
            else:
                stable_count = 0

            # Check for divergence (error getting worse)
            if err_px > best_err + 3.0:
                diverge_count += 1
                if diverge_count >= 3 and self.PIXEL_SERVO_REFIT_AFTER_DIVERGE:
                    send_feedback(f"mypolicy/pixel_servo_refit err={err_px:.1f}px best={best_err:.1f}px")
                    jacobian = self._estimate_pixel_jacobian(
                        camera, target_port_name, port_matcher, get_observation,
                        send_feedback, meas["median_e"],
                    )
                    if jacobian is None:
                        send_feedback("mypolicy/pixel_servo_fail refit_failed")
                        break
                    self._pixel_state.jacobian[camera] = jacobian
                    diverge_count = 0
                    continue
            else:
                diverge_count = 0

            if err_px < best_err:
                best_err = err_px

            # Total XY cap
            if total_xy >= self.PIXEL_SERVO_MAX_TOTAL_M:
                send_feedback(f"mypolicy/pixel_servo_cap total={total_xy*1000:.1f}mm err={err_px:.1f}px")
                break

            # Compute correction: delta_xy = -lambda * J^{-1} @ error
            if jacobian is None:
                jacobian = self._estimate_pixel_jacobian(
                    camera, target_port_name, port_matcher, get_observation,
                    send_feedback, meas["median_e"],
                )
                if jacobian is None:
                    break
                continue

            try:
                J_inv = np.linalg.pinv(jacobian, rcond=1e-3)
                delta_xy = -self.PIXEL_SERVO_LAMBDA * J_inv @ meas["median_e"]
            except np.linalg.LinAlgError:
                send_feedback("mypolicy/pixel_servo_fail singular_jacobian")
                break

            # Clamp step size
            step_norm = float(np.linalg.norm(delta_xy))
            if step_norm > self.PIXEL_SERVO_MAX_STEP_M:
                delta_xy = delta_xy / step_norm * self.PIXEL_SERVO_MAX_STEP_M
                step_norm = self.PIXEL_SERVO_MAX_STEP_M

            if step_norm < 1e-7:
                send_feedback(f"mypolicy/pixel_servo_tiny_step err={err_px:.1f}px")
                break

            # Execute correction
            self._command_small_xy(delta_xy, max_step=self.PIXEL_SERVO_MAX_STEP_M)
            total_xy += step_norm
            self.sleep_for(self.PIXEL_SERVO_SETTLE_SEC)

            if iteration < 3 or iteration % 5 == 0:
                send_feedback(
                    f"mypolicy/pixel_servo iter={iteration} err={err_px:.1f}px "
                    f"dxy=({delta_xy[0]*1000:+.2f},{delta_xy[1]*1000:+.2f})mm "
                    f"total={total_xy*1000:.1f}mm best={best_err:.1f}px"
                )

        # Final check
        final_meas = self._measure_pixel_error_median(
            camera, target_port_name, port_matcher, get_observation, frames=5,
        )
        if final_meas is not None:
            final_err = final_meas["median_err_px"]
            send_feedback(f"mypolicy/pixel_servo_final err={final_err:.1f}px")
            # Accept if reasonably close even if not perfect
            return final_err <= self.PIXEL_SERVO_PX_TOL * 2.0

        return False

    def _command_small_xy(self, delta_xy: np.ndarray, max_step: float = 0.002) -> None:
        """Move in XY only, clamping total step, preserving Z and orientation."""
        d = np.asarray(delta_xy, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(d))
        if norm < 1e-7:
            return
        if norm > max_step:
            d = d / norm * max_step

        dx, dy = float(d[0]), float(d[1])
        dist = math.sqrt(dx * dx + dy * dy)
        speed = float(self._motion_servo.max_linear_speed) * 0.20
        vx, vy = dx / dist * speed, dy / dist * speed

        cur = self._motion_servo.get_current_pose()
        if cur is None:
            return
        start_x, start_y = float(cur.position.x), float(cur.position.y)

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy

        t0 = time.monotonic()
        timeout = dist / speed + 0.6
        while time.monotonic() - t0 < timeout:
            p = self._motion_servo.get_current_pose()
            if p is None:
                break
            moved = math.sqrt((p.position.x - start_x) ** 2 + (p.position.y - start_y) ** 2)
            if moved >= dist * 0.85:
                break
            self._motion_servo.publish_twist_command(twist, frame_id="base_link")
            self.sleep_for(0.02)
        self._motion_servo.stop()

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 5: Pre-insert verification
    # ═══════════════════════════════════════════════════════════════════════════

    def _pre_insert_verify(
        self,
        target_port_name: str,
        port_matcher,
        get_observation,
        send_feedback,
    ) -> bool:
        """Quick multi-camera check that alignment is good enough for insertion."""
        errors = {}
        for cam in self.PIXEL_SERVO_CAMERA_ORDER:
            meas = self._measure_pixel_error_median(
                cam, target_port_name, port_matcher, get_observation, frames=3,
            )
            if meas is not None:
                errors[cam] = meas["median_err_px"]

        if not errors:
            send_feedback("mypolicy/verify_fail no_cameras")
            return False

        center_err = errors.get("center", 999.0)
        avg_err = sum(errors.values()) / len(errors)
        send_feedback(f"mypolicy/verify center={center_err:.1f}px avg={avg_err:.1f}px cams={list(errors.keys())}")

        # Pass if center camera is well aligned, or average is acceptable
        return center_err <= 12.0 or avg_err <= 15.0

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 6: Compliant Force-Guided Insertion
    # ═══════════════════════════════════════════════════════════════════════════

    def _read_wrist_force(self, obs) -> np.ndarray:
        if obs is None:
            return np.zeros(3, dtype=np.float64)
        try:
            w = obs.wrist_wrench.wrench.force
            return np.array([float(w.x), float(w.y), float(w.z)], dtype=np.float64)
        except Exception:
            return np.zeros(3, dtype=np.float64)

    def _sample_force_baseline(self, get_observation, duration_s=0.5) -> np.ndarray:
        samples = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            obs = get_observation()
            samples.append(self._read_wrist_force(obs))
            self.sleep_for(0.02)
        if not samples:
            return np.zeros(3, dtype=np.float64)
        return np.median(np.stack(samples), axis=0)

    def _move_z_relative(self, dz: float, speed: float = 0.003):
        if speed <= 0 or abs(dz) < 1e-6:
            return
        duration = abs(dz) / speed
        tw = Twist()
        tw.linear.z = speed if dz > 0 else -speed
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            self._motion_servo.publish_twist_command(tw, frame_id="base_link")
            self.sleep_for(0.025)
        self._motion_servo.stop()

    def _move_xy_relative(self, dx: float, dy: float, duration: float = 0.1):
        if duration <= 0:
            return
        tw = Twist()
        tw.linear.x = dx / duration
        tw.linear.y = dy / duration
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            self._motion_servo.publish_twist_command(tw, frame_id="base_link")
            self.sleep_for(0.025)
        self._motion_servo.stop()

    def _insert_backoff(self, backoff_m: float = 0.003):
        self._move_z_relative(backoff_m, speed=0.004)
        self.sleep_for(0.10)

    def _insert_spiral_search(self, get_observation, send_feedback, current_depth, reason="jam") -> bool:
        """Spiral search to find the hole when jammed."""
        send_feedback(f"mypolicy/spiral_start reason={reason} depth={current_depth*1000:.1f}mm")
        self._motion_servo.stop()

        backoff = float(self.INSERT_SPIRAL_BACKOFF_M)
        self._move_z_relative(backoff, speed=0.003)

        force_baseline = self._sample_force_baseline(get_observation, duration_s=0.2)

        best_offset = None
        min_force_delta = float("inf")

        candidates = [(0.0, 0.0)]
        for r in self.INSERT_SPIRAL_RADII_M:
            for i in range(self.INSERT_SPIRAL_POINTS_PER_RING):
                theta = i * (2 * math.pi / self.INSERT_SPIRAL_POINTS_PER_RING)
                candidates.append((r * math.cos(theta), r * math.sin(theta)))

        for dx, dy in candidates:
            self._move_xy_relative(dx, dy, duration=0.08)

            # Probe down
            probe_speed = 0.002
            probe_dur = float(self.INSERT_SPIRAL_PROBE_DEPTH_M) / probe_speed
            tw = Twist()
            tw.linear.z = -probe_speed

            max_delta = 0.0
            t0 = time.monotonic()
            while time.monotonic() - t0 < probe_dur:
                obs = get_observation()
                fvec = self._read_wrist_force(obs)
                delta = float(np.linalg.norm(fvec - force_baseline))
                max_delta = max(max_delta, delta)
                if delta > float(self.INSERT_FORCE_DELTA_THRESH_N):
                    break
                self._motion_servo.publish_twist_command(tw, frame_id="base_link")
                self.sleep_for(0.04)
            self._motion_servo.stop()

            # Back up from probe
            self._move_z_relative(float(self.INSERT_SPIRAL_PROBE_DEPTH_M), speed=probe_speed)
            # Return XY
            self._move_xy_relative(-dx, -dy, duration=0.08)

            if max_delta < min_force_delta:
                min_force_delta = max_delta
                best_offset = (dx, dy)

            send_feedback(f"mypolicy/spiral r={math.sqrt(dx*dx+dy*dy)*1000:.2f}mm delta={max_delta:.2f}N")

        if best_offset is not None and min_force_delta < float(self.INSERT_FORCE_DELTA_THRESH_N):
            dx, dy = best_offset
            send_feedback(f"mypolicy/spiral_best dxy=({dx*1000:.2f},{dy*1000:.2f})mm delta={min_force_delta:.2f}N")
            self._move_xy_relative(dx, dy, duration=0.08)
            self._move_z_relative(-backoff, speed=0.003)
            return True

        send_feedback("mypolicy/spiral_fail no_good_candidate")
        return False

    def _compliant_insert(
        self,
        target_port_name: str,
        port_matcher,
        get_observation,
        send_feedback,
    ) -> bool:
        """
        Compliant insertion using delta-force jam detection.
        Descends in -Z with small steps, monitors force changes from baseline.
        """
        send_feedback("mypolicy/insert_begin")

        start_pose = self._motion_servo.get_current_pose()
        if start_pose is None:
            send_feedback("mypolicy/insert_fail no_pose")
            return False

        # Sample force baseline before any descent
        force_baseline = self._sample_force_baseline(
            get_observation, duration_s=float(self.INSERT_BASELINE_DURATION_S)
        )
        send_feedback(f"mypolicy/insert_baseline mag={float(np.linalg.norm(force_baseline)):.2f}N")

        start_z = float(start_pose.position.z)
        inserted_depth = 0.0
        jam_count = 0
        retries = 0
        last_log_t = 0.0
        funnel_xy_total = [0.0, 0.0]
        stalled_cycles = 0
        insert_speed = float(self.INSERT_SPEED_MPS)
        insert_step = 0.0005  # 0.5mm per control tick

        while True:
            obs = get_observation()
            fvec = self._read_wrist_force(obs)
            abs_force = float(np.linalg.norm(fvec))
            delta_vec = fvec - force_baseline
            delta_mag = float(np.linalg.norm(delta_vec))

            # Hard safety
            if abs_force > float(self.INSERT_FORCE_HARD_ABS_N):
                self._motion_servo.stop()
                send_feedback(f"mypolicy/insert_abort hard_force={abs_force:.2f}N")
                return False

            # Log
            now = time.monotonic()
            if now - last_log_t >= 0.4:
                last_log_t = now
                send_feedback(f"mypolicy/insert depth={inserted_depth*1000:.1f}mm abs={abs_force:.2f}N delta={delta_mag:.2f}N")

            # Jam detection
            if delta_mag > float(self.INSERT_FORCE_DELTA_THRESH_N):
                jam_count += 1
                send_feedback(f"mypolicy/insert_contact delta={delta_mag:.2f}N count={jam_count}")
            else:
                jam_count = 0

            if jam_count >= int(self.INSERT_FORCE_JAM_COUNT):
                self._motion_servo.stop()
                retries += 1
                send_feedback(f"mypolicy/insert_jam retry={retries}")

                if retries > int(self.INSERT_MAX_RETRIES):
                    send_feedback("mypolicy/insert_fail max_retries")
                    return False

                # Try spiral recovery
                if self.INSERT_SPIRAL_ENABLE:
                    spiral_ok = self._insert_spiral_search(
                        get_observation, send_feedback, inserted_depth, reason="jam"
                    )
                    if spiral_ok:
                        force_baseline = self._sample_force_baseline(get_observation, 0.3)
                        cur = self._motion_servo.get_current_pose()
                        if cur is not None:
                            start_z = float(cur.position.z) + inserted_depth
                        jam_count = 0
                        continue

                # Fallback: backoff + re-align + retry
                self._insert_backoff(float(self.INSERT_BACKOFF_M))
                send_feedback("mypolicy/insert_realign_after_jam")
                self._pixel_state.reset()
                align_ok = self._pixel_servo_align(
                    target_port_name, port_matcher, get_observation, send_feedback,
                )
                self._motion_servo.stop()

                force_baseline = self._sample_force_baseline(get_observation, 0.3)
                cur = self._motion_servo.get_current_pose()
                if cur is not None:
                    start_z = float(cur.position.z)
                inserted_depth = 0.0
                jam_count = 0
                funnel_xy_total = [0.0, 0.0]
                continue

            # Contact-guided funneling (lateral XY from side forces)
            if self.INSERT_FUNNEL_ENABLE:
                df_xy = np.array([delta_vec[0], delta_vec[1]])
                delta_xy = float(np.linalg.norm(df_xy))
                if delta_xy > float(self.INSERT_FUNNEL_DELTA_XY_THRESH_N) and delta_mag < float(self.INSERT_FORCE_DELTA_THRESH_N):
                    gain = float(self.INSERT_FUNNEL_GAIN_M_PER_N)
                    max_step = float(self.INSERT_FUNNEL_MAX_XY_STEP_M)
                    max_total = float(self.INSERT_FUNNEL_MAX_TOTAL_XY_M)

                    step_xy = -gain * df_xy
                    sn = float(np.linalg.norm(step_xy))
                    if sn > max_step:
                        step_xy = step_xy * (max_step / sn)

                    new_tx = funnel_xy_total[0] + step_xy[0]
                    new_ty = funnel_xy_total[1] + step_xy[1]
                    if math.sqrt(new_tx ** 2 + new_ty ** 2) <= max_total:
                        self._move_xy_relative(float(step_xy[0]), float(step_xy[1]), duration=0.05)
                        funnel_xy_total = [new_tx, new_ty]

            # Descend one step
            duration = insert_step / insert_speed
            down_twist = Twist()
            down_twist.linear.z = -float(np.clip(insert_speed, 0.001, 0.010))
            self._motion_servo.publish_twist_command(
                down_twist, frame_id="base_link",
                trans_stiffness=60.0, rot_stiffness=50.0,
                trans_damping=50.0, rot_damping=20.0,
            )
            self.sleep_for(duration)
            self._motion_servo.stop()

            # Track depth
            cur = self._motion_servo.get_current_pose()
            if cur is not None:
                new_depth = max(0.0, start_z - float(cur.position.z))
                if abs(new_depth - inserted_depth) < 1e-5:
                    stalled_cycles += 1
                else:
                    stalled_cycles = 0
                inserted_depth = new_depth

            # Success: reached target depth
            if inserted_depth >= float(self.INSERT_MAX_DEPTH_M):
                self._motion_servo.stop()
                send_feedback(f"mypolicy/insert_success depth={inserted_depth*1000:.1f}mm")
                return True

            # Stall recovery
            if stalled_cycles >= 10 and inserted_depth < 0.001:
                send_feedback("mypolicy/insert_stall_recovery")
                self._move_z_relative(0.001, speed=0.005)
                stalled_cycles = 0

        self._motion_servo.stop()
        send_feedback("mypolicy/insert_fail unknown")
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Detection helpers (unchanged from original)
    # ═══════════════════════════════════════════════════════════════════════════

    def _best_detection(self, detections: List[Dict], matcher) -> Optional[Dict]:
        candidates = [d for d in detections if matcher(d) and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE]
        return max(candidates, key=lambda d: float(d.get("confidence", 0.0))) if candidates else None

    def _pose_from_detection(self, det: Dict) -> Optional[Pose]:
        pose_dict = det.get("pose_base_link")
        if not isinstance(pose_dict, dict):
            return None
        position = pose_dict.get("position")
        orientation = pose_dict.get("orientation")
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            return None
        pose = Pose()
        pose.position = Point(x=float(position.get("x", 0.0)), y=float(position.get("y", 0.0)), z=float(position.get("z", 0.0)))
        pose.orientation = Quaternion(x=float(orientation.get("x", 0.0)), y=float(orientation.get("y", 0.0)), z=float(orientation.get("z", 0.0)), w=float(orientation.get("w", 1.0)))
        return pose

    def _position_distance(self, a: Pose, b: Pose) -> float:
        dx = float(a.position.x - b.position.x)
        dy = float(a.position.y - b.position.y)
        dz = float(a.position.z - b.position.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _is_sfp_port_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._sfp_port_classes)

    def _is_sc_port_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._sc_port_classes)

    def _is_nic_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._nic_classes)

    def _is_sfp_module_detection(self, det: Dict) -> bool:
        if self._matches_any_name(det, self._sfp_module_classes):
            return True
        for key in ("class_name", "raw_class_name", "base_class_name", "instance_name"):
            norm = self._norm_name(det.get(key, ""))
            if ("sfp" in norm and "module" in norm) or "transceiver" in norm:
                return True
        return False

    def _matches_specific_port(self, det, target_name, allowed_classes) -> bool:
        if not self._matches_any_name(det, allowed_classes):
            return False
        if not target_name:
            return True
        normalized_target = self._norm_name(target_name)
        names = [self._norm_name(det.get(k, "")) for k in ("instance_name", "class_name", "raw_class_name", "base_class_name")]
        return normalized_target in names

    def _matches_any_name(self, det, allowed_names) -> bool:
        for key in ("class_name", "raw_class_name", "base_class_name", "instance_name"):
            norm = self._norm_name(det.get(key, ""))
            base = self._strip_numeric_suffix(norm)
            for allowed in allowed_names:
                if norm == allowed or base == allowed or norm.startswith(f"{allowed}_"):
                    return True
        return False

    def _parse_name_set(self, text: str) -> set:
        return {self._norm_name(x) for x in str(text).split(",") if str(x).strip()}

    def _strip_numeric_suffix(self, name: str) -> str:
        parts = self._norm_name(name).split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return self._norm_name(name)

    def _norm_name(self, name: str) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    # ── Axis / pose helpers ──────────────────────────────────────────────────

    def _weighted_average_pose(self, items) -> Optional[Pose]:
        if not items:
            return None
        weights = np.asarray([max(1e-6, float(w)) for _, w in items], dtype=np.float64)
        weights /= max(1e-9, float(np.sum(weights)))
        positions = np.asarray([[float(p.position.x), float(p.position.y), float(p.position.z)] for p, _ in items], dtype=np.float64)
        mean_p = np.sum(positions * weights.reshape(-1, 1), axis=0)
        q0 = np.asarray(_quat_to_np(items[0][0].orientation), dtype=np.float64)
        q_sum = np.zeros(4, dtype=np.float64)
        for (pose, _), weight in zip(items, weights):
            q = np.asarray(_quat_normalize(_quat_to_np(pose.orientation)), dtype=np.float64)
            if float(np.dot(q, q0)) < 0.0:
                q = -q
            q_sum += float(weight) * q
        q = _quat_normalize(q_sum.tolist())
        pose = Pose()
        pose.position = Point(x=float(mean_p[0]), y=float(mean_p[1]), z=float(mean_p[2]))
        pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        return pose

    def _pose_local_axis_base(self, pose, axis_local_name) -> np.ndarray:
        axis_local = self._axis_from_local_name(axis_local_name)
        return _normalize_vec(_rotate_vector_by_quaternion(_quat_to_np(pose.orientation), axis_local))

    def _average_signed_axes(self, axis_items) -> Optional[np.ndarray]:
        if not axis_items:
            return None
        ref = _normalize_vec(axis_items[0][0])
        if float(np.linalg.norm(ref)) < 1e-9:
            return None
        total = np.zeros(3, dtype=np.float64)
        weight_sum = 0.0
        for axis, weight in axis_items:
            a = _normalize_vec(axis)
            if float(np.linalg.norm(a)) < 1e-9:
                continue
            if float(np.dot(a, ref)) < 0.0:
                a = -a
            w = max(0.01, float(weight))
            total += w * a
            weight_sum += w
        if weight_sum <= 1e-9 or float(np.linalg.norm(total)) < 1e-9:
            return None
        return _normalize_vec(total / weight_sum)

    def _target_ground_direction_from_name(self, name: str) -> np.ndarray:
        value = str(name).strip().lower()
        directions = {
            "down": [0, 0, -1], "-z": [0, 0, -1], "ground": [0, 0, -1],
            "up": [0, 0, 1], "+z": [0, 0, 1],
            "x": [1, 0, 0], "+x": [1, 0, 0], "-x": [-1, 0, 0],
            "y": [0, 1, 0], "+y": [0, 1, 0], "-y": [0, -1, 0],
        }
        return np.asarray(directions.get(value, [0, 0, -1]), dtype=np.float64)

    def _axis_from_local_name(self, name: str) -> np.ndarray:
        value = str(name).strip().lower()
        axes = {
            "x": [1, 0, 0], "+x": [1, 0, 0], "roll": [1, 0, 0], "-x": [-1, 0, 0],
            "y": [0, 1, 0], "+y": [0, 1, 0], "pitch": [0, 1, 0], "-y": [0, -1, 0],
            "z": [0, 0, 1], "+z": [0, 0, 1], "yaw": [0, 0, 1], "-z": [0, 0, -1],
        }
        return np.asarray(axes.get(value, [1, 0, 0]), dtype=np.float64)

    def _latest_sfp_module_red_axis_estimate(self) -> Optional[Dict]:
        axis_items = []
        cameras = set()
        best_conf = 0.0
        for cam in ("center", "left", "right"):
            cam_dets = self._detection_listener.get_camera_detections(cam, freshness_sec=1.0)
            det = self._best_detection(cam_dets, self._is_sfp_module_detection)
            if det is None:
                continue
            pose = self._pose_from_detection(det)
            if pose is None:
                continue
            axis = self._pose_local_axis_base(pose, self.MODULE_RED_AXIS_LOCAL)
            if float(np.linalg.norm(axis)) < 1e-9:
                continue
            conf = max(0.05, float(det.get("confidence", 0.0)))
            axis_items.append((axis, conf * (1.50 if cam == "center" else 1.0)))
            cameras.add(cam)
            best_conf = max(best_conf, conf)
        axis = self._average_signed_axes(axis_items)
        if axis is None:
            return None
        return {"axis": axis, "cameras": sorted(cameras), "sample_count": len(axis_items), "confidence": best_conf}
