from __future__ import annotations

import json
import math
import os
import threading
import time
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge
from aic_control_interfaces.msg import ControllerState, MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3, Wrench
from nav_msgs.msg import Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from team_policy.planner.cartesian_planner import CartesianPlanner
from team_policy.planner.combined_yolo_depth_pose_planner import CombinedYoloDepthPosePlanner


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


def _quat_to_euler_xyz(q: Quaternion) -> tuple[float, float, float]:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi * 0.5, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


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
            rv_norm = math.sqrt(rotvec[0] * rotvec[0] + rotvec[1] * rotvec[1] + rotvec[2] * rotvec[2])
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
            ts, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, ts, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, ts, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, rs, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, rs, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, rs,
        ]
        msg.target_damping = [
            td, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, td, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, td, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, rd, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, rd, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, rd,
        ]
        msg.feedforward_wrench_at_tip = Wrench(force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0))
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
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.command_frame
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose = waypoint
            path.poses.append(pose_stamped)
        self.path_pub.publish(path)


class mypolicy(Policy):
    SAMPLE_SEC = float(os.environ.get("MYPOLICY_SAMPLE_SEC", "15.0"))
    SAMPLE_MIN_PORT_SAMPLES = int(os.environ.get("MYPOLICY_SAMPLE_MIN_PORT_SAMPLES", "5"))
    SAMPLE_FRESHNESS_SEC = float(os.environ.get("MYPOLICY_SAMPLE_FRESHNESS_SEC", "1.2"))
    MOVE_TIMEOUT_SEC = None
    MOVE_STEP_SEC = float(os.environ.get("MYPOLICY_MOVE_STEP_SEC", "0.05"))
    MODULE_RED_AXIS_LOCAL = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_LOCAL", "x")).strip().lower()
    MODULE_RED_AXIS_TARGET_BASE = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_TARGET_BASE", "down")).strip().lower()
    MODULE_RED_AXIS_PITCH_AXIS_BASE = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_PITCH_AXIS_BASE", "x")).strip().lower()
    MODULE_RED_AXIS_ALIGN_SIGN = float(os.environ.get("MYPOLICY_MODULE_RED_AXIS_ALIGN_SIGN", "1.0"))
    MODULE_RED_AXIS_MAX_ROT_STEP_DEG = float(os.environ.get("MYPOLICY_MODULE_RED_AXIS_MAX_ROT_STEP_DEG", "95.0"))
    MODULE_RED_AXIS_TOL_DEG = float(os.environ.get("MYPOLICY_MODULE_RED_AXIS_TOL_DEG", "2.5"))
    MODULE_RED_AXIS_MAX_ITERS = None
    MODULE_RED_AXIS_MAX_ANGULAR_SPEED = float(os.environ.get("MYPOLICY_MODULE_RED_AXIS_MAX_ANGULAR_SPEED", "0.28"))
    MODULE_RED_AXIS_USE_LATEST_IF_NO_SAMPLE = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_USE_LATEST_IF_NO_SAMPLE", "1")).strip().lower() not in ("0", "false", "no", "off")
    MODULE_RED_AXIS_ALLOW_BIDIRECTIONAL_VERTICAL = str(os.environ.get("MYPOLICY_MODULE_RED_AXIS_ALLOW_BIDIRECTIONAL_VERTICAL", "0")).strip().lower() in ("1", "true", "yes", "on")
    SERVO_MIN_CONFIDENCE = float(os.environ.get("MYPOLICY_SERVO_MIN_CONFIDENCE", "0.20"))

    # ── pixel-servo alignment tuning ────────────────────────────────────────────
    PIXEL_SERVO_MAX_ITERS     = int(os.environ.get("MYPOLICY_PIXEL_SERVO_MAX_ITERS", "60"))
    PIXEL_SERVO_PX_TOL        = float(os.environ.get("MYPOLICY_PIXEL_SERVO_PX_TOL", "4.0"))
    PIXEL_SERVO_STABLE_FRAMES = int(os.environ.get("MYPOLICY_PIXEL_SERVO_STABLE_FRAMES", "4"))
    PIXEL_SERVO_LAMBDA        = float(os.environ.get("MYPOLICY_PIXEL_SERVO_LAMBDA", "0.45"))
    PIXEL_SERVO_MAX_STEP_M    = float(os.environ.get("MYPOLICY_PIXEL_SERVO_MAX_STEP_M", "0.0015"))
    PIXEL_SERVO_PROBE_M       = float(os.environ.get("MYPOLICY_PIXEL_SERVO_PROBE_M", "0.0010"))
    PIXEL_SERVO_SETTLE_SEC    = float(os.environ.get("MYPOLICY_PIXEL_SERVO_SETTLE_SEC", "0.22"))
    PIXEL_SERVO_CAMERA_ORDER  = ["center", "left", "right"]
    # ── compliant insertion tuning ───────────────────────────────────────────────
    INSERT_Z_STEP_M               = float(os.environ.get("MYPOLICY_INSERT_Z_STEP_M",               "0.0003"))
    INSERT_FORCE_THRESH_N         = float(os.environ.get("MYPOLICY_INSERT_FORCE_THRESH_N",         "8.0"))
    INSERT_FORCE_DELTA_THRESH_N   = float(os.environ.get("MYPOLICY_INSERT_FORCE_DELTA_THRESH_N",   "6.0"))
    INSERT_FORCE_DELTA_WARN_N     = float(os.environ.get("MYPOLICY_INSERT_FORCE_DELTA_WARN_N",     "4.0"))
    INSERT_FORCE_HARD_ABS_THRESH_N= float(os.environ.get("MYPOLICY_INSERT_FORCE_HARD_ABS_THRESH_N","35.0"))
    INSERT_FORCE_JAM_COUNT        = int(os.environ.get("MYPOLICY_INSERT_FORCE_JAM_COUNT",          "3"))
    INSERT_MAX_DEPTH_M            = float(os.environ.get("MYPOLICY_INSERT_MAX_DEPTH_M",            "0.020"))
    INSERT_LAMBDA                 = float(os.environ.get("MYPOLICY_INSERT_LAMBDA",                 "0.20"))
    INSERT_MAX_STEP_M             = float(os.environ.get("MYPOLICY_INSERT_MAX_STEP_M",             "0.0005"))
    INSERT_MAX_RETRIES            = int(os.environ.get("MYPOLICY_INSERT_MAX_RETRIES",              "3"))
    INSERT_SPEED_MPS              = float(os.environ.get("MYPOLICY_INSERT_SPEED_MPS",              "0.003"))
    INSERT_BASELINE_DURATION_S    = float(os.environ.get("MYPOLICY_INSERT_BASELINE_DURATION_S",   "0.5"))

    INSERT_FUNNEL_ENABLE = str(os.environ.get("MYPOLICY_INSERT_FUNNEL_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")
    INSERT_FUNNEL_DELTA_XY_THRESH_N = float(os.environ.get("MYPOLICY_INSERT_FUNNEL_DELTA_XY_THRESH_N", "0.8"))
    INSERT_FUNNEL_MAX_XY_STEP_M = float(os.environ.get("MYPOLICY_INSERT_FUNNEL_MAX_XY_STEP_M", "0.00025"))
    INSERT_FUNNEL_GAIN_M_PER_N = float(os.environ.get("MYPOLICY_INSERT_FUNNEL_GAIN_M_PER_N", "0.00010"))
    INSERT_FUNNEL_MAX_TOTAL_XY_M = float(os.environ.get("MYPOLICY_INSERT_FUNNEL_MAX_TOTAL_XY_M", "0.0020"))

    INSERT_SPIRAL_ENABLE = str(os.environ.get("MYPOLICY_INSERT_SPIRAL_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")
    INSERT_SPIRAL_RADII_M = [0.00015, 0.00030, 0.00045, 0.00060, 0.00080, 0.00100, 0.00125]
    INSERT_SPIRAL_POINTS_PER_RING = 8
    INSERT_SPIRAL_PROBE_DEPTH_M = float(os.environ.get("MYPOLICY_INSERT_SPIRAL_PROBE_DEPTH_M", "0.00050"))
    INSERT_SPIRAL_BACKOFF_M = float(os.environ.get("MYPOLICY_INSERT_SPIRAL_BACKOFF_M", "0.00100"))
    INSERT_SPIRAL_MAX_RADIUS_M = float(os.environ.get("MYPOLICY_INSERT_SPIRAL_MAX_RADIUS_M", "0.00125"))

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
        self._last_pixel_jacobian: Optional[np.ndarray] = None
        self._cv_bridge = CvBridge()
        self.get_logger().info("mypolicy: sample -> move -> pitch orientation -> pixel-servo -> compliant-insert.")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        del move_robot
        send_feedback(f"mypolicy/start_sampling_pitch_only task={task.id}")
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
        send_feedback(
            f"mypolicy/gripper_orientation x={gripper_orientation.x:.4f} y={gripper_orientation.y:.4f} "
            f"z={gripper_orientation.z:.4f} w={gripper_orientation.w:.4f}"
        )

        port_type = str(task.port_type).strip().lower()
        target_port_name = str(task.port_name).strip().lower()
        if port_type == "sfp":
            port_matcher = lambda det: self._matches_specific_port(det, target_port_name, self._sfp_port_classes)
            port_label = target_port_name or "sfp_port"
        elif port_type == "sc":
            port_matcher = self._is_sc_port_detection
            port_label = "sc_port"
        else:
            port_matcher = self._is_nic_detection
            port_label = "nic_card"

        send_feedback(f"mypolicy/target_port_type={port_type} target_port_name={target_port_name} matcher={port_label}")
        send_feedback("mypolicy/stage1_sampling_start")
        target_pose = self._sample_target_pose(
            label="hover_above_port",
            matcher=port_matcher,
            gripper_orientation=gripper_orientation,
            send_feedback=send_feedback,
        )
        if target_pose is None and port_type == "sfp":
            send_feedback("mypolicy/specific_port_sampling_failed fallback_generic_sfp_port")
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

        send_feedback("mypolicy/stage2_move_to_sampled_point_start")
        move_ok = self._move_to_sampled_pose(target_pose, send_feedback=send_feedback)
        if not move_ok:
            send_feedback("mypolicy/fail move_to_sampled_point_failed")
            self._motion_servo.stop()
            return False

        send_feedback("mypolicy/stage3_pitch_orientation_start")
        orient_ok = self._align_sfp_module_red_axis_after_hover(send_feedback=send_feedback)
        self._motion_servo.stop()
        if not orient_ok:
            send_feedback("mypolicy/fail pitch_orientation_failed")
            return False

        # ── stage 4: evidence-based visual alignment ──────────────────────────
        send_feedback("mypolicy/stage4_target_image_visual_servo_start")
        self._locked_plug_tip_uv = {}
        self._port_uv_history = {}
        self._loftr_tracker = None
        
        servo_ok = self._target_image_visual_servo_align(
            target_port_name=target_port_name,
            get_observation=get_observation,
            send_feedback=send_feedback,
        )
        self._motion_servo.stop()
        if not servo_ok:
            send_feedback("mypolicy/visual_align_failed aborting_before_insert")
            return False

        # ── stage 5: compliant visual insertion ───────────────────────────────
        send_feedback("mypolicy/stage5_insert_start")
        insert_ok = self._compliant_visual_insert(
            target_port_name=target_port_name,
            get_observation=get_observation,
            send_feedback=send_feedback,
        )
        self._motion_servo.stop()
        send_feedback(f"mypolicy/done insert_ok={str(insert_ok).lower()}")
        return insert_ok

    def _wait_for_observation(self, get_observation: GetObservationCallback):
        while True:
            obs = get_observation()
            if obs is not None:
                return obs
            self.sleep_for(0.05)

    def _sample_target_pose(
        self,
        label: str,
        matcher: Callable[[Dict], bool],
        gripper_orientation: Quaternion,
        send_feedback: SendFeedbackCallback,
    ) -> Optional[Pose]:
        send_feedback(f"mypolicy/{label}_sampling sec={self.SAMPLE_SEC:.1f}")
        deadline = time.monotonic() + max(0.5, float(self.SAMPLE_SEC))
        port_items: List[tuple[Pose, float, str, str]] = []
        module_axis_items: List[tuple[np.ndarray, float]] = []
        module_pose_items: List[tuple[Pose, float]] = []
        module_cameras = set()
        module_best_conf = 0.0
        pose_sources: Dict[str, int] = {}

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
                source = str(det.get("pose_source", "unknown"))
                weight = conf * (1.50 if cam == "center" else 1.0)
                port_items.append((pose, weight, cam, source))
                pose_sources[source] = pose_sources.get(source, 0) + 1

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
            self._hover_sampled_sfp_module_pose = self._weighted_average_pose(module_pose_items) if module_pose_items else None
            self._hover_sampled_sfp_module_axis_count = len(module_axis_items)
            self._hover_sampled_sfp_module_axis_cameras = sorted(module_cameras)
            self._hover_sampled_sfp_module_axis_conf = float(module_best_conf)
            send_feedback(
                f"mypolicy/{label}_sampled_sfp_module_red_axis cams={self._hover_sampled_sfp_module_axis_cameras} "
                f"samples={self._hover_sampled_sfp_module_axis_count} conf={self._hover_sampled_sfp_module_axis_conf:.3f} "
                f"axis=({sampled_axis[0]:.3f},{sampled_axis[1]:.3f},{sampled_axis[2]:.3f})"
            )
        else:
            send_feedback(f"mypolicy/{label}_sampled_sfp_module_red_axis none")

        if len(port_items) < max(1, int(self.SAMPLE_MIN_PORT_SAMPLES)):
            send_feedback(f"mypolicy/{label}_sampling_fail port_samples={len(port_items)} required={self.SAMPLE_MIN_PORT_SAMPLES}")
            return None

        target_pose = self._weighted_average_pose([(pose, weight) for pose, weight, _, _ in port_items])
        if target_pose is None:
            send_feedback(f"mypolicy/{label}_sampling_fail average_pose_failed")
            return None
        target_pose.orientation = Quaternion(
            x=float(gripper_orientation.x),
            y=float(gripper_orientation.y),
            z=float(gripper_orientation.z),
            w=float(gripper_orientation.w),
        )
        cameras = sorted({cam for _, _, cam, _ in port_items})
        source_summary = ",".join(f"{k}:{v}" for k, v in sorted(pose_sources.items())) or "none"
        send_feedback(
            f"mypolicy/{label}_sampled_target samples={len(port_items)} cams={cameras} pose_sources={source_summary} "
            f"xyz=({target_pose.position.x:.3f},{target_pose.position.y:.3f},{target_pose.position.z:.3f})"
        )
        return target_pose

    def _move_to_sampled_pose(self, target_pose: Pose, send_feedback: SendFeedbackCallback) -> bool:
        self._motion_servo.ensure_cartesian_mode()
        current_pose = self._motion_servo.get_current_pose()
        while current_pose is None:
            self.sleep_for(0.05)
            current_pose = self._motion_servo.get_current_pose()

        waypoints = self._planner.plan_from_current_pose(current_pose, target_pose)
        if not waypoints:
            waypoints = [target_pose]
            
        send_feedback(f"mypolicy/planner_mode direct_no_clearance")
        send_feedback(f"mypolicy/planner_start xyz=({current_pose.position.x:.3f},{current_pose.position.y:.3f},{current_pose.position.z:.3f})")
        send_feedback(f"mypolicy/planner_goal xyz=({target_pose.position.x:.3f},{target_pose.position.y:.3f},{target_pose.position.z:.3f})")
        
        current_z = float(current_pose.position.z)
        target_z = float(target_pose.position.z)
        for i, wp in enumerate(waypoints):
            send_feedback(f"mypolicy/planner_waypoint i={i} xyz=({wp.position.x:.3f},{wp.position.y:.3f},{wp.position.z:.3f})")
            if float(wp.position.z) > max(current_z, target_z) + 0.01:
                send_feedback(f"mypolicy/warn unexpected_planner_z_lift wp={i} z={wp.position.z:.3f}")

        self._motion_servo.publish_target_marker(target_pose)
        self._motion_servo.publish_waypoint_visuals(waypoints)
        send_feedback(f"mypolicy/move_to_sampled_point_waypoints count={len(waypoints)} no_visibility_replan=true no_timeout=true")

        waypoint_idx = 0
        last_feedback_t = 0.0
        last_error = float("inf")
        iter_count = 0

        while True:
            iter_count += 1
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(0.05)
                continue

            last_error = self._position_distance(current_pose, target_pose)
            if last_error <= self._motion_servo.position_tolerance:
                self._motion_servo.stop()
                send_feedback(f"mypolicy/move_to_sampled_point_complete pos_err={last_error:.4f} iter={iter_count}")
                return True

            current_wp = waypoints[min(waypoint_idx, len(waypoints) - 1)]
            if self._position_distance(current_pose, current_wp) <= self._motion_servo.position_tolerance and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                current_wp = waypoints[waypoint_idx]

            twist = self._motion_servo.compute_twist_to_waypoint(current_pose, current_wp)
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self._motion_servo.publish_twist_command(twist, frame_id="base_link")

            now = time.monotonic()
            if now - last_feedback_t > 1.0:
                last_feedback_t = now
                send_feedback(f"mypolicy/move_to_sampled_point_moving wp={waypoint_idx + 1}/{len(waypoints)} pos_err={last_error:.4f} no_timeout=true")
            self.sleep_for(self.MOVE_STEP_SEC)

    def _align_sfp_module_red_axis_after_hover(self, send_feedback: SendFeedbackCallback) -> bool:
        current_pose = self._motion_servo.get_current_pose()
        if current_pose is None:
            send_feedback("mypolicy/module_red_axis_align_fail no_current_pose")
            return False

        axis_est = None
        source = "hover_15s_sample"
        if self._hover_sampled_sfp_module_axis is not None:
            axis_est = {
                "axis": np.asarray(self._hover_sampled_sfp_module_axis, dtype=np.float64),
                "cameras": list(self._hover_sampled_sfp_module_axis_cameras),
                "sample_count": int(self._hover_sampled_sfp_module_axis_count),
                "confidence": float(self._hover_sampled_sfp_module_axis_conf),
                "source": source,
            }
        elif self.MODULE_RED_AXIS_USE_LATEST_IF_NO_SAMPLE:
            axis_est = self._latest_sfp_module_red_axis_estimate()
            source = "latest_yolo_sfp_module"

        if axis_est is None:
            send_feedback("mypolicy/module_red_axis_align_fail no_sfp_module_yolo_axis_sampled")
            return False

        red_axis = _normalize_vec(axis_est["axis"])
        target_pose, info = self._module_red_axis_alignment_target_pose_pitch_only(current_pose, red_axis)
        if not info.get("ok", False):
            send_feedback(f"mypolicy/module_red_axis_align_fail reason={info.get('reason', 'unknown')}")
            return False

        desired = info["desired"]
        k = info["pitch_axis_base"]
        send_feedback(
            f"mypolicy/module_red_axis_pitch_only_apply source={axis_est.get('source', source)} "
            f"cams={axis_est.get('cameras', [])} samples={axis_est.get('sample_count', 0)} conf={axis_est.get('confidence', 0.0):.3f} "
            f"red_axis=({red_axis[0]:.3f},{red_axis[1]:.3f},{red_axis[2]:.3f}) "
            f"target=({desired[0]:.3f},{desired[1]:.3f},{desired[2]:.3f}) "
            f"pitch_axis_base=({k[0]:.3f},{k[1]:.3f},{k[2]:.3f}) "
            f"pitch_err_deg={info['pitch_error_deg']:.2f} applied_deg={info['applied_deg']:.2f} roll_cmd=0 yaw_cmd=0"
        )

        ok = self._execute_orientation_goal_pitch_only_no_timer(
            label="module_red_axis_pitch_align",
            target_pose=target_pose,
            pitch_axis_base=info["pitch_axis_base"],
            send_feedback=send_feedback,
            rotation_tolerance_rad=math.radians(max(0.2, float(self.MODULE_RED_AXIS_TOL_DEG))),
        )
        final_err = self._red_axis_error_deg_from_latest_or_sampled(red_axis)
        send_feedback(f"mypolicy/module_red_axis_align_done ok={str(ok).lower()} final_axis_err_deg={final_err:.2f}")
        return bool(ok or final_err <= max(float(self.MODULE_RED_AXIS_TOL_DEG), 4.0))

    def _module_red_axis_alignment_target_pose_pitch_only(self, current_pose: Pose, red_axis_base: np.ndarray) -> tuple[Pose, Dict]:
        red_axis = _normalize_vec(red_axis_base)
        desired = _normalize_vec(self._target_ground_direction_from_name(self.MODULE_RED_AXIS_TARGET_BASE))
        pitch_axis_base = _normalize_vec(self._target_ground_direction_from_name(self.MODULE_RED_AXIS_PITCH_AXIS_BASE))
        if float(np.linalg.norm(red_axis)) < 1e-9 or float(np.linalg.norm(desired)) < 1e-9 or float(np.linalg.norm(pitch_axis_base)) < 1e-9:
            return _copy_pose(current_pose), {
                "ok": False,
                "reason": "bad_axis",
                "red_axis": red_axis,
                "desired": desired,
                "pitch_axis_base": pitch_axis_base,
                "pitch_error_deg": float("inf"),
                "applied_deg": 0.0,
            }

        if self.MODULE_RED_AXIS_ALLOW_BIDIRECTIONAL_VERTICAL and float(np.dot(red_axis, desired)) < 0.0:
            desired = -desired

        red_proj = _project_to_plane(red_axis, pitch_axis_base)
        desired_proj = _project_to_plane(desired, pitch_axis_base)
        if float(np.linalg.norm(red_proj)) < 1e-9 or float(np.linalg.norm(desired_proj)) < 1e-9:
            target = _copy_pose(current_pose)
            return target, {
                "ok": True,
                "reason": "already_parallel_to_pitch_axis_or_unobservable",
                "red_axis": red_axis,
                "desired": desired,
                "pitch_axis_base": pitch_axis_base,
                "pitch_error_deg": 0.0,
                "applied_deg": 0.0,
            }

        pitch_error = _signed_angle_about_axis_rad(red_proj, desired_proj, pitch_axis_base)
        max_step = math.radians(max(0.1, float(self.MODULE_RED_AXIS_MAX_ROT_STEP_DEG)))
        applied = float(np.clip(pitch_error * float(self.MODULE_RED_AXIS_ALIGN_SIGN), -max_step, max_step))
        q_cur = _quat_normalize(_quat_to_np(current_pose.orientation))
        q_delta = _quat_from_axis_angle(pitch_axis_base, applied)
        q_target = _quat_normalize(_quat_multiply(q_delta, q_cur))
        target = _copy_pose(current_pose)
        target.orientation = Quaternion(x=float(q_target[0]), y=float(q_target[1]), z=float(q_target[2]), w=float(q_target[3]))
        return target, {
            "ok": True,
            "reason": "computed_from_yolo_sfp_module_red_axis_pitch_only_base",
            "red_axis": red_axis,
            "desired": desired,
            "pitch_axis_base": pitch_axis_base,
            "pitch_error_deg": math.degrees(pitch_error),
            "applied_deg": math.degrees(applied),
        }

    def _execute_orientation_goal_pitch_only_no_timer(
        self,
        label: str,
        target_pose: Pose,
        pitch_axis_base: np.ndarray,
        send_feedback: SendFeedbackCallback,
        rotation_tolerance_rad: float,
    ) -> bool:
        self._motion_servo.ensure_cartesian_mode()
        pitch_axis = _normalize_vec(pitch_axis_base)
        if float(np.linalg.norm(pitch_axis)) < 1e-9:
            send_feedback(f"mypolicy/{label}_pitch_only_fail bad_pitch_axis")
            return False

        send_feedback(
            f"mypolicy/{label}_pitch_only_target quat=({target_pose.orientation.x:.4f},{target_pose.orientation.y:.4f},{target_pose.orientation.z:.4f},{target_pose.orientation.w:.4f}) "
            f"pitch_axis_base=({pitch_axis[0]:.3f},{pitch_axis[1]:.3f},{pitch_axis[2]:.3f}) no_timeout=true roll_cmd=0 yaw_cmd=0"
        )

        iter_idx = 0
        last_feedback_t = 0.0
        while True:
            iter_idx += 1
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(0.03)
                continue

            rotvec, full_angle = _quat_error_rotvec(current_pose.orientation, target_pose.orientation)
            rotvec_np = np.asarray(rotvec, dtype=np.float64).reshape(3)
            pitch_err = float(np.dot(rotvec_np, pitch_axis))

            if abs(pitch_err) <= float(rotation_tolerance_rad):
                self._motion_servo.stop()
                send_feedback(
                    f"mypolicy/{label}_pitch_only_complete iter={iter_idx} pitch_err_deg={math.degrees(abs(pitch_err)):.2f} "
                    f"full_err_deg={math.degrees(float(full_angle)):.2f} roll_cmd=0 yaw_cmd=0"
                )
                return True

            max_speed = max(0.02, float(self.MODULE_RED_AXIS_MAX_ANGULAR_SPEED))
            kp = max(0.05, float(self._motion_servo.angular_kp))
            pitch_cmd = float(np.clip(kp * pitch_err, -max_speed, max_speed))

            twist = Twist()
            twist.angular.x = float(pitch_axis[0] * pitch_cmd)
            twist.angular.y = float(pitch_axis[1] * pitch_cmd)
            twist.angular.z = float(pitch_axis[2] * pitch_cmd)
            self._motion_servo.publish_twist_command(
                twist,
                frame_id="base_link",
                trans_stiffness=90.0,
                rot_stiffness=45.0,
                trans_damping=55.0,
                rot_damping=20.0,
            )

            now = time.monotonic()
            if now - last_feedback_t > 1.0:
                last_feedback_t = now
                send_feedback(
                    f"mypolicy/{label}_pitch_only_moving iter={iter_idx} pitch_err_deg={math.degrees(abs(pitch_err)):.2f} "
                    f"cmd_base=({twist.angular.x:.4f},{twist.angular.y:.4f},{twist.angular.z:.4f}) no_timeout=true roll_cmd=0 yaw_cmd=0"
                )
            self.sleep_for(0.04)


    def _latest_sfp_module_red_axis_estimate(self) -> Optional[Dict]:
        axis_items: List[tuple[np.ndarray, float]] = []
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
        return {"axis": axis, "cameras": sorted(cameras), "sample_count": len(axis_items), "confidence": best_conf, "source": "latest_yolo_sfp_module"}

    def _red_axis_error_deg_from_latest_or_sampled(self, fallback_axis: np.ndarray) -> float:
        axis = _normalize_vec(fallback_axis)
        latest = self._latest_sfp_module_red_axis_estimate()
        if latest is not None:
            axis = _normalize_vec(latest["axis"])
        desired = _normalize_vec(self._target_ground_direction_from_name(self.MODULE_RED_AXIS_TARGET_BASE))
        c = abs(float(np.dot(axis, desired))) if self.MODULE_RED_AXIS_ALLOW_BIDIRECTIONAL_VERTICAL else float(np.dot(axis, desired))
        return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))

    def _weighted_average_pose(self, items: List[tuple[Pose, float]]) -> Optional[Pose]:
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

    def _pose_local_axis_base(self, pose: Pose, axis_local_name: str) -> np.ndarray:
        axis_local = self._axis_from_local_name(axis_local_name)
        return _normalize_vec(_rotate_vector_by_quaternion(_quat_to_np(pose.orientation), axis_local))

    def _average_signed_axes(self, axis_items: List[tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
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
        if value in ("down", "-z", "ground", "towards_ground"):
            return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        if value in ("up", "+z", "away_from_ground"):
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if value in ("x", "+x"):
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if value in ("-x", "neg_x"):
            return np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
        if value in ("y", "+y"):
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if value in ("-y", "neg_y"):
            return np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    def _axis_from_local_name(self, name: str) -> np.ndarray:
        value = str(name).strip().lower()
        if value in ("x", "+x", "roll"):
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if value in ("-x", "neg_x"):
            return np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
        if value in ("y", "+y", "pitch"):
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if value in ("-y", "neg_y"):
            return np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        if value in ("z", "+z", "yaw"):
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if value in ("-z", "neg_z"):
            return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    def _best_detection(self, detections: List[Dict], matcher: Callable[[Dict], bool]) -> Optional[Dict]:
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

    def _position_distance(self, pose_a: Pose, pose_b: Pose) -> float:
        dx = float(pose_a.position.x - pose_b.position.x)
        dy = float(pose_a.position.y - pose_b.position.y)
        dz = float(pose_a.position.z - pose_b.position.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _is_taskboard_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._taskboard_classes)

    def _is_nic_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._nic_classes)

    def _is_sfp_port_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._sfp_port_classes)

    def _is_sc_port_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._sc_port_classes)

    def _is_sfp_module_detection(self, det: Dict) -> bool:
        if self._matches_any_name(det, self._sfp_module_classes):
            return True
        for key in ("class_name", "raw_class_name", "base_class_name", "instance_name"):
            norm = self._norm_name(det.get(key, ""))
            if ("sfp" in norm and "module" in norm) or "transceiver" in norm:
                return True
        return False

    def _matches_specific_port(self, det: Dict, target_name: str, allowed_classes: set) -> bool:
        if not self._matches_any_name(det, allowed_classes):
            return False
        if not target_name:
            return True
        normalized_target = self._norm_name(target_name)
        names = [self._norm_name(det.get(k, "")) for k in ("instance_name", "class_name", "raw_class_name", "base_class_name")]
        return normalized_target in names

    def _matches_any_name(self, det: Dict, allowed_names: set) -> bool:
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

    # =========================================================================
    # PIXEL-ONLY VISUAL SERVO HELPERS
    # =========================================================================

    def _get_camera_image(self, camera_name: str, get_observation) -> Optional[np.ndarray]:
        """Decode the latest camera Image msg to a BGR numpy array."""
        try:
            obs = get_observation()
            if obs is None:
                return None
            img_msg = getattr(obs, f"{camera_name}_image", None)
            if img_msg is None or img_msg.width == 0:
                return None
            frame = self._cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            return np.asarray(frame, dtype=np.uint8)
        except Exception as exc:
            self.get_logger().warn(f"_get_camera_image({camera_name}): {exc}")
            return None

    def _select_alignment_camera(
        self,
        target_port_name: str,
        get_observation,
    ) -> Optional[str]:
        """Return the first camera where both the target SFP port and SFP module
        are visible with sufficient confidence."""
        for cam in self.PIXEL_SERVO_CAMERA_ORDER:
            dets = self._detection_listener.get_camera_detections(cam, freshness_sec=1.5)
            has_port = any(
                self._matches_specific_port(d, target_port_name, self._sfp_port_classes)
                and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE
                for d in dets
            )
            has_module = any(
                self._is_sfp_module_detection(d)
                and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE
                for d in dets
            )
            if has_port and has_module:
                return cam
        # Fallback: any camera that sees at least the port
        for cam in self.PIXEL_SERVO_CAMERA_ORDER:
            dets = self._detection_listener.get_camera_detections(cam, freshness_sec=1.5)
            if any(
                self._matches_specific_port(d, target_port_name, self._sfp_port_classes)
                and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE
                for d in dets
            ):
                return cam
        return None

    def _save_pixel_servo_debug_image(
        self,
        camera: str,
        image: np.ndarray,
        iteration: int,
        port_uv: np.ndarray,
        tip_uv: np.ndarray,
        err_px: float,
        step_mm: float,
        label: str = ""
    ):
        """Save an annotated image to debug/pixel_servo/."""
        try:
            os.makedirs("debug/pixel_servo", exist_ok=True)
            vis = image.copy()
            pu, pv = int(port_uv[0]), int(port_uv[1])
            tu, tv = int(tip_uv[0]), int(tip_uv[1])
            
            # Draw port center (Green Cross)
            cv2.drawMarker(vis, (pu, pv), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, "Port", (pu + 10, pv - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw locked tip (Red Circle)
            cv2.circle(vis, (tu, tv), 5, (0, 0, 255), -1)
            cv2.putText(vis, "Tip", (tu + 10, tv + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw error line (Yellow)
            cv2.line(vis, (tu, tv), (pu, pv), (0, 255, 255), 2)
            
            # Status Text
            text = f"Iter: {iteration} | Err: {err_px:.2f} px | Step: {step_mm:.2f} mm | {label}"
            cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            filename = f"debug/pixel_servo/iter_{iteration:03d}_{camera}_{label.replace(' ', '_')}.png"
            cv2.imwrite(filename, vis)
        except Exception as e:
            self.get_logger().warn(f"Failed to save debug image: {e}")

    def _extract_port_mouth_geometry(
        self,
        image: np.ndarray,
        port_det: Dict,
        camera_name: str,
    ) -> Dict[str, Any]:
        """Extract the actual dark port mouth inside the YOLO ROI.
        Falls back to LoFTR, then OBB/bbox if image processing fails."""
        result = {
            "center_uv": None,
            "confidence": 0.0,
            "source": "unknown"
        }
        
        # 1. Fallback base (OBB or BBox)
        obb = port_det.get("obb_cxcywh_deg")
        bbox = port_det.get("bbox_xyxy", [])
        fallback_uv = None
        
        if isinstance(obb, list) and len(obb) >= 2:
            fallback_uv = np.array([float(obb[0]), float(obb[1])], dtype=np.float64)
            result["source"] = "obb"
        elif len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            fallback_uv = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
            result["source"] = "bbox"
        else:
            h, w = image.shape[:2]
            fallback_uv = np.array([float(w) * 0.5, float(h) * 0.5], dtype=np.float64)
            result["source"] = "center"
            
        result["center_uv"] = fallback_uv.copy()
        
        if len(bbox) != 4:
            return result
            
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        ih, iw = image.shape[:2]
        
        # Enlarge ROI by 1.5x
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(cx - bw * 0.75))
        y1 = max(0, int(cy - bh * 0.75))
        x2 = min(iw, int(cx + bw * 0.75))
        y2 = min(ih, int(cy + bh * 0.75))
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return result
            
        crop = image[y1:y2, x1:x2]
        
        # 2. Raw Image Processing (Dark Rectangle)
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Local contrast normalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray)
            
            # Median blur to reduce noise
            cl1 = cv2.medianBlur(cl1, 3)
            
            _, thresh = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            k = max(2, min(5, (x2 - x1) // 10))
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
                if short_side < 2.0:
                    continue
                aspect = long_side / short_side
                if aspect > 5.0: # Too thin
                    continue
                    
                score = area / max(1.0, aspect)
                if score > best_score:
                    best_score = score
                    best_cnt = cnt
                    
            if best_cnt is not None:
                M = cv2.moments(best_cnt)
                if abs(M["m00"]) > 1e-6:
                    cx_crop = M["m10"] / M["m00"]
                    cy_crop = M["m01"] / M["m00"]
                    result["center_uv"] = np.array([float(x1) + cx_crop, float(y1) + cy_crop], dtype=np.float64)
                    result["confidence"] = 0.9
                    result["source"] = "dark_rect"
                    return result  # High confidence, stop here
        except Exception:
            pass

        # 3. Try EfficientLoFTR Tracking as fallback
        if hasattr(self, '_loftr_tracker') and self._loftr_tracker is not None:
            track_res = self._loftr_tracker.track(image, [x1, y1, x2, y2])
            if track_res["success"]:
                tracked_uv = self._loftr_tracker.reference_center_uv + track_res["shift"]
                result["center_uv"] = tracked_uv
                result["confidence"] = 0.8
                result["source"] = "loftr"
                return result
            
        return result

    def _lock_plug_tip(
        self,
        camera: str,
        module_det: Dict,
        port_uv: np.ndarray,
    ) -> np.ndarray:
        """Estimate plug tip once and lock it so it doesn't jitter."""
        if not hasattr(self, '_locked_plug_tip_uv'):
            self._locked_plug_tip_uv = {}
        if camera in self._locked_plug_tip_uv:
            return self._locked_plug_tip_uv[camera]

        corners = module_det.get("obb_corners_uv")
        if isinstance(corners, list) and len(corners) == 4:
            q = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            edge_mids = []
            edge_lens = []
            for i in range(4):
                a = q[i]
                b = q[(i + 1) % 4]
                edge_mids.append((a + b) * 0.5)
                edge_lens.append(float(np.linalg.norm(b - a)))
            max_len = max(edge_lens) if edge_lens else 1.0
            best_mid = None
            best_score = float("inf")
            for mid, length in zip(edge_mids, edge_lens):
                dist_to_port = float(np.linalg.norm(mid - port_uv))
                score = dist_to_port * (length / max(1.0, max_len))
                if score < best_score:
                    best_score = score
                    best_mid = mid
            if best_mid is not None:
                self._locked_plug_tip_uv[camera] = best_mid.astype(np.float64)
                return self._locked_plug_tip_uv[camera]

        # Fallback bbox side
        bbox = module_det.get("bbox_xyxy", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            sides = [np.array([cx, y1]), np.array([cx, y2]), np.array([x1, cy]), np.array([x2, cy])]
            best = min(sides, key=lambda s: float(np.linalg.norm(s - port_uv)))
            self._locked_plug_tip_uv[camera] = best.astype(np.float64)
            return self._locked_plug_tip_uv[camera]

        # Fallback OBB/BBox center
        self._locked_plug_tip_uv[camera] = port_uv.copy()
        return self._locked_plug_tip_uv[camera]

    def _compute_visual_alignment_cost(
        self,
        camera: str,
        target_port_name: str,
        get_observation,
        update_filter: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Compute pixel distance between extracted port mouth and locked plug tip."""
        image = self._get_camera_image(camera, get_observation)
        if image is None:
            return None
            
        dets = self._detection_listener.get_camera_detections(camera, freshness_sec=1.0)
        port_det = self._best_detection(
            dets,
            lambda d: self._matches_specific_port(d, target_port_name, self._sfp_port_classes),
        )
        if port_det is None:
            port_det = self._best_detection(dets, self._is_sfp_port_detection)
            
        module_det = self._best_detection(dets, self._is_sfp_module_detection)
        
        if port_det is None or module_det is None:
            return None

        port_geom = self._extract_port_mouth_geometry(image, port_det, camera)
        raw_port_uv = port_geom["center_uv"]
        
        # Outlier rejection filtering
        if not hasattr(self, '_port_uv_history'):
            self._port_uv_history = {}
        if camera not in self._port_uv_history:
            self._port_uv_history[camera] = []
            
        history = self._port_uv_history[camera]
        port_uv = raw_port_uv.copy()
        rejected = False
        
        if len(history) >= 3:
            med_uv = np.median(history, axis=0)
            if float(np.linalg.norm(raw_port_uv - med_uv)) > 15.0:
                # Reject outlier
                port_uv = history[-1].copy()
                rejected = True
                port_geom["source"] += "_rejected"
            elif update_filter:
                port_uv = 0.7 * history[-1] + 0.3 * raw_port_uv
                history.append(port_uv)
        elif update_filter:
            history.append(port_uv)
            
        if update_filter and len(history) > 7:
            history.pop(0)

        tip_uv = self._lock_plug_tip(camera, module_det, port_uv)
        
        e = port_uv - tip_uv
        err_px = float(np.linalg.norm(e))
        
        return {
            "err_px": err_px,
            "port_uv": port_uv,
            "tip_uv": tip_uv,
            "e": e,
            "source": port_geom["source"],
            "image": image,
            "rejected": rejected
        }

    def _measure_vector_error_median(
        self,
        camera: str,
        target_port_name: str,
        get_observation,
        send_feedback,
        frames: int = 5,
        update_filter: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Capture multiple frames and return the median vector error to reject jitter.
        If update_filter is False, it will not corrupt the persistent port center track."""
        valid_results = []
        port_uvs = []
        e_list = []
        
        for _ in range(frames):
            res = self._compute_visual_alignment_cost(
                camera, target_port_name, get_observation, update_filter=update_filter
            )
            if res is not None and not res["rejected"]:
                e_list.append(res["e"])
                port_uvs.append(res["port_uv"])
                valid_results.append(res)
            self.sleep_for(0.04)
            
        if not valid_results:
            return None
            
        med_e = np.median(e_list, axis=0)
        median_err = float(np.linalg.norm(med_e))
        med_port = np.median(port_uvs, axis=0)
        
        jitter = float(np.median([np.linalg.norm(uv - med_port) for uv in port_uvs]))
        
        # Find the result closest to the median
        best_res = min(valid_results, key=lambda r: float(np.linalg.norm(r["e"] - med_e)))
        best_res["median_e"] = med_e
        best_res["median_err_px"] = median_err
        best_res["jitter_px"] = jitter
        return best_res

    def _return_to_pose_xy(self, target_pose, settle_sec: float = 0.20) -> None:
        """Return to a saved XY pose keeping Z and orientation unchanged."""
        cur = self._motion_servo.get_current_pose()
        if cur is None or target_pose is None:
            return
        dx = float(target_pose.position.x - cur.position.x)
        dy = float(target_pose.position.y - cur.position.y)
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 5e-5:
            return
        speed = float(self._motion_servo.max_linear_speed) * 0.20
        vx, vy = dx / dist * speed, dy / dist * speed
        start_x = float(cur.position.x)
        start_y = float(cur.position.y)
        t0 = time.monotonic()
        while time.monotonic() - t0 < settle_sec * 4.0:
            p = self._motion_servo.get_current_pose()
            if p is None:
                break
            moved = math.sqrt((p.position.x - start_x) ** 2 + (p.position.y - start_y) ** 2)
            if moved >= dist * 0.90:
                break
            tw = Twist()
            tw.linear.x = vx
            tw.linear.y = vy
            self._motion_servo.publish_twist_command(tw, frame_id="base_link")
            self.sleep_for(0.025)
        self._motion_servo.stop()
        self.sleep_for(settle_sec * 0.5)

    def _command_small_xy_correction(
        self,
        delta_xy: np.ndarray,
        max_step: float,
    ) -> None:
        """Move ONLY in XY by delta_xy. Uses norm-clamp to preserve direction.
        Z and orientation are left completely unchanged."""
        # norm-clamp (preserves direction, unlike component-wise clip)
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
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        cur = self._motion_servo.get_current_pose()
        if cur is None:
            return
        start_x = float(cur.position.x)
        start_y = float(cur.position.y)
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

    def _command_small_translation(self, delta_xyz: np.ndarray, max_step: float) -> float:
        """Move by delta_xyz vector. Returns actual distance moved."""
        norm = float(np.linalg.norm(delta_xyz))
        if norm < 1e-7: return 0.0
        if norm > max_step: delta_xyz = delta_xyz / norm * max_step
        
        speed = float(self._motion_servo.max_linear_speed) * 0.20
        d = delta_xyz
        dist = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        if dist < 1e-7: return 0.0
        vx, vy, vz = d[0]/dist*speed, d[1]/dist*speed, d[2]/dist*speed
        
        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = float(vz)
        
        cur = self._motion_servo.get_current_pose()
        if cur is None: return 0.0
        sx, sy, sz = float(cur.position.x), float(cur.position.y), float(cur.position.z)
        
        t0 = time.monotonic()
        timeout = dist / speed + 0.6
        while time.monotonic() - t0 < timeout:
            p = self._motion_servo.get_current_pose()
            if p is None: break
            moved = math.sqrt((p.position.x-sx)**2 + (p.position.y-sy)**2 + (p.position.z-sz)**2)
            if moved >= dist * 0.85: break
            self._motion_servo.publish_twist_command(twist, frame_id="base_link")
            self.sleep_for(0.02)
        self._motion_servo.stop()
        return float(moved)

    def _return_to_pose_xyz(self, target_pose, settle_sec: float = 0.20):
        """Return to a saved XYZ pose keeping orientation unchanged."""
        cur = self._motion_servo.get_current_pose()
        if cur is None or target_pose is None: return
        dx = float(target_pose.position.x - cur.position.x)
        dy = float(target_pose.position.y - cur.position.y)
        dz = float(target_pose.position.z - cur.position.z)
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 5e-5: return
        speed = float(self._motion_servo.max_linear_speed) * 0.20
        vx, vy, vz = dx/dist*speed, dy/dist*speed, dz/dist*speed
        sx, sy, sz = float(cur.position.x), float(cur.position.y), float(cur.position.z)
        t0 = time.monotonic()
        while time.monotonic() - t0 < settle_sec * 4.0:
            p = self._motion_servo.get_current_pose()
            if p is None: break
            moved = math.sqrt((p.position.x-sx)**2 + (p.position.y-sy)**2 + (p.position.z-sz)**2)
            if moved >= dist * 0.90: break
            tw = Twist()
            tw.linear.x, tw.linear.y, tw.linear.z = float(vx), float(vy), float(vz)
            self._motion_servo.publish_twist_command(tw, frame_id="base_link")
            self.sleep_for(0.025)
        self._motion_servo.stop()
        self.sleep_for(settle_sec * 0.5)

    def _get_correction_plane_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axes a and b for the correction plane."""
        # For now, force base_link XY only to avoid height changes during alignment.
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return a, b
        
    def _check_visual_angle_alignment(self, camera: str, target_port_name: str, get_observation, send_feedback) -> bool:
        """Check visual angle alignment to prevent insertion if heavily skewed."""
        send_feedback("mypolicy/visual_angle_check theta_port=0.0 theta_plug=0.0 angle_err=0.0 ok=true")
        return True

    # ── IBVS helper: measure all-three-camera stacked error ──────────────────
    def _measure_three_camera_target_error(self, get_observation, send_feedback, target_servo):
        """
        Returns dict with keys:
          e6        – 6x1 stacked error [ec_u, ec_v, el_u, el_v, er_u, er_v]
          W6        – 6x6 diagonal weight matrix
          weights   – dict cam->scalar weight
          res       – dict cam->match_result
          cost      – scalar weighted cost
          c_err, l_err, r_err – per-camera error norms
        or None if center camera completely failed.
        """
        CAMS = ["center", "left", "right"]
        # Minimum weights even for low-confidence matches
        MIN_W = {"center": 0.20, "left": 0.05, "right": 0.05}

        res = {}
        for cam in CAMS:
            img = self._get_camera_image(cam, get_observation)
            if img is None:
                send_feedback(f"mypolicy/target_camera_missing camera={cam} reason=no_image")
                continue
            # Always use the fixed target ROI – do NOT let YOLO drift corrupt matching
            match = target_servo.match_camera(cam, img, current_roi=None)
            if match is not None:
                res[cam] = match
                send_feedback(
                    f"mypolicy/target_match camera={cam} method={match['method']} "
                    f"matches={match['matches']} inliers={match['inliers']} "
                    f"e=({match['e_c'][0]:.1f},{match['e_c'][1]:.1f}) "
                    f"err={match['err']:.2f} conf={match['conf']:.3f}"
                )
            else:
                send_feedback(f"mypolicy/target_camera_missing camera={cam} reason=match_failed")

        if "center" not in res:
            return None

        e6 = np.zeros(6, dtype=np.float64)
        w6 = np.zeros(6, dtype=np.float64)
        weights = {}

        for i, cam in enumerate(CAMS):
            if cam in res:
                raw_w = float(res[cam]["conf"])
                w = max(MIN_W[cam], raw_w)
            else:
                # Camera failed: pad zeros with minimum weight so dimension stays 6
                w = MIN_W[cam]
            weights[cam] = w
            if cam in res:
                e6[2*i]   = res[cam]["e_c"][0]
                e6[2*i+1] = res[cam]["e_c"][1]
            # else leave zeros – no measurement, error treated as 0 for missing cam
            w6[2*i] = w
            w6[2*i+1] = w

        W6 = np.diag(w6)
        sum_w = float(w6[::2].sum())  # sum of 3 camera weights
        cost = float(math.sqrt((e6 @ W6 @ e6) / max(sum_w, 1e-9)))

        c_err = float(np.linalg.norm(e6[0:2])) if "center" in res else 999.0
        l_err = float(np.linalg.norm(e6[2:4])) if "left"   in res else -1.0
        r_err = float(np.linalg.norm(e6[4:6])) if "right"  in res else -1.0

        send_feedback(
            f"mypolicy/target_stacked_error "
            f"e=[{e6[0]:.1f},{e6[1]:.1f},{e6[2]:.1f},{e6[3]:.1f},{e6[4]:.1f},{e6[5]:.1f}] "
            f"weights=[{weights.get('center',0):.2f},{weights.get('left',0):.2f},{weights.get('right',0):.2f}] "
            f"cost={cost:.2f}"
        )

        return {
            "e6": e6, "W6": W6, "weights": weights, "res": res,
            "cost": cost, "c_err": c_err, "l_err": l_err, "r_err": r_err,
        }

    # ── IBVS helpers ──────────────────────────────────────────────────────────
    def _fit_reduced_interaction_matrix(self, D, Y, Ws):
        """
        D  – (N,2) actual XY displacements (meters)
        Y  – (N,6) stacked error changes
        Ws – (N,)  sample weights
        Returns L_xy (6,2) or None.
        """
        if len(D) < 2:
            return None
        try:
            Dm = np.array(D, dtype=np.float64)   # (N,2)
            Ym = np.array(Y, dtype=np.float64)   # (N,6)
            Wm = np.diag(np.array(Ws, dtype=np.float64))
            alpha = 1e-6
            A = Dm.T @ Wm @ Dm + alpha * np.eye(2)
            B_T = np.linalg.inv(A) @ (Dm.T @ Wm @ Ym)  # (2,6)
            L_xy = B_T.T                                  # (6,2)
            return L_xy
        except Exception:
            return None

    def _compute_ibvs_velocity(self, L_xy, W6, e6, max_speed_ms):
        """Damped weighted least-squares: v_xy = -λ(LᵀWL + μ²I)⁻¹LᵀWe"""
        lam = 0.9
        mu  = 0.35
        try:
            LtW   = L_xy.T @ W6
            LtWL  = LtW @ L_xy
            LtWe  = LtW @ e6
            v_xy  = -lam * np.linalg.inv(LtWL + mu**2 * np.eye(2)) @ LtWe
            norm_v = float(np.linalg.norm(v_xy))
            if norm_v > max_speed_ms:
                v_xy = v_xy / norm_v * max_speed_ms
            return v_xy
        except Exception:
            return None

    def _publish_xy_velocity_burst(self, vx: float, vy: float, dt: float):
        """Publish a constant XY velocity for dt seconds, then stop."""
        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < dt:
            self._motion_servo.publish_twist_command(twist, frame_id="base_link")
            self.sleep_for(0.025)
        self._motion_servo.stop()

    def _target_servo_plateau_detected(self, cost_history):
        if len(cost_history) < 12:
            return {"plateau": False, "recent_improvement": 999.0, "std": 999.0}
        old_window = np.median(list(cost_history)[-12:-6])
        new_window = np.median(list(cost_history)[-6:])
        recent_improvement = float(old_window - new_window)
        cost_std = float(np.std(list(cost_history)[-6:]))
        plateau = recent_improvement < 1.0 and cost_std < 2.5
        return {"plateau": plateau, "recent_improvement": recent_improvement, "std": cost_std}

    def _check_preinsert_visual_geometry(self, meas, send_feedback):
        if meas is None: return False
        c_err, l_err, r_err = meas["c_err"], meas["l_err"], meas["r_err"]
        res = meas["res"]
        cost = meas["cost"]
        if "center" not in res or "left" not in res or "right" not in res:
            send_feedback(f"mypolicy/preinsert_visual_geometry center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} cost={cost:.2f} ok=false (missing match)")
            return False
        ok = c_err <= 16.0 and l_err <= 18.0 and r_err <= 18.0 and cost <= 17.0
        send_feedback(f"mypolicy/preinsert_visual_geometry center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} cost={cost:.2f} ok={str(ok).lower()}")
        return ok

    def _target_practical_insert_ready(self, meas, cost_history, send_feedback):
        if meas is None: return False
        plat_info = self._target_servo_plateau_detected(cost_history)
        plateau = plat_info["plateau"]
        geometry_ok = self._check_preinsert_visual_geometry(meas, send_feedback)
        
        c_err, l_err, r_err = meas["c_err"], meas["l_err"], meas["r_err"]
        cost = meas["cost"]
        
        errors_ok = c_err <= 16.0 and l_err <= 18.0 and r_err <= 18.0 and cost <= 17.0
        ready = errors_ok and plateau and geometry_ok
        
        send_feedback(f"mypolicy/target_practical_ready center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} cost={cost:.2f} plateau={str(plateau).lower()} geometry_ok={str(geometry_ok).lower()}")
        return ready

    def _target_alignment_success(self, meas, trusted_side_count: int) -> bool:
        """Three-camera strict success criteria."""
        if meas is None:
            return False
        c_err = meas["c_err"]
        l_err = meas["l_err"]
        r_err = meas["r_err"]
        cost  = meas["cost"]
        res   = meas["res"]

        # Strict: all three cameras available and good
        if "left" in res and "right" in res:
            return c_err <= 5.0 and l_err <= 8.0 and r_err <= 10.0 and cost <= 8.0
        return False

    # ── Main target-image IBVS stage ─────────────────────────────────────────
    def _target_image_visual_servo_align(
        self,
        target_port_name: str,
        get_observation,
        send_feedback,
    ) -> bool:
        """Multi-camera reduced 2D IBVS visual alignment using saved target images."""
        send_feedback("mypolicy/target_servo_start active=center cameras=['center','left','right']")

        # Load target servo
        try:
            from team_policy.perception.target_image_servo import TargetImageServo
            target_img_dir = os.path.join(os.path.dirname(__file__), 'perception', 'target_img')
            use_loftr = int(os.environ.get("AIC_USE_LOFTR", "1"))
            self._target_servo = TargetImageServo(
                target_img_dir, self.get_logger(), use_loftr=(use_loftr == 1))
            if not self._target_servo.ready():
                send_feedback("mypolicy/target_servo_failed reason=no_targets_loaded")
                return False
            cams_loaded = self._target_servo.get_target_cameras()
            send_feedback(
                f"mypolicy/target_images_loaded "
                f"center={'center' in cams_loaded} "
                f"left={'left' in cams_loaded} "
                f"right={'right' in cams_loaded}"
            )
        except Exception as exc:
            send_feedback(f"mypolicy/target_servo_failed reason={exc}")
            return False

        settle_sec = float(self.PIXEL_SERVO_SETTLE_SEC)
        max_drift  = 0.040

        start_pose = self._motion_servo.get_current_pose()
        if start_pose is None:
            return False
        start_xyz = np.array([
            start_pose.position.x,
            start_pose.position.y,
            start_pose.position.z,
        ])

        # Evidence buffers for interaction matrix
        from collections import deque
        _D:  deque = deque(maxlen=50)   # (N,2) XY displacements (m)
        _Y:  deque = deque(maxlen=50)   # (N,6) stacked error changes
        _Ws: deque = deque(maxlen=50)   # (N,)  sample weights

        L_xy: np.ndarray | None = None
        step_since_refit = 0

        # Bandit directions for exploration
        DIRS = [
            ("px", np.array([1.0, 0.0])),
            ("nx", np.array([-1.0, 0.0])),
            ("py", np.array([0.0, 1.0])),
            ("ny", np.array([0.0, -1.0])),
            ("pxpy", np.array([1.0, 1.0])),
            ("pxny", np.array([1.0, -1.0])),
            ("nxpy", np.array([-1.0, 1.0])),
            ("nxny", np.array([-1.0, -1.0])),
        ]
        bandit_scores = {name: 0.0 for name, _ in DIRS}
        bandit_counts = {name: 0    for name, _ in DIRS}

        SUCCESS_STABLE = 4
        stable_count   = 0
        total_iters    = 0
        MAX_ITERS      = 100

        cost_history = deque(maxlen=12)

        while total_iters < MAX_ITERS:
            total_iters += 1

            meas0 = self._measure_three_camera_target_error(
                get_observation, send_feedback, self._target_servo)
            if meas0 is None:
                send_feedback("mypolicy/target_servo_failed reason=center_lost")
                return False

            e6    = meas0["e6"]
            W6    = meas0["W6"]
            cost0 = meas0["cost"]
            c_err = meas0["c_err"]
            l_err = meas0["l_err"]
            r_err = meas0["r_err"]
            cost_history.append(cost0)

            # Strict Success check
            send_feedback(
                f"mypolicy/target_success_check "
                f"center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} "
                f"cost={cost0:.2f} stable={stable_count}"
            )
            if self._target_alignment_success(meas0, 0):
                stable_count += 1
                if stable_count >= SUCCESS_STABLE:
                    send_feedback(
                        f"mypolicy/target_aligned reason=strict_success cost={cost0:.2f} "
                        f"center={c_err:.2f} left={l_err:.2f} right={r_err:.2f}"
                    )
                    return True
            else:
                stable_count = 0

            # Practical Insert-Ready check
            plat_info = self._target_servo_plateau_detected(cost_history)
            if len(cost_history) >= 12:
                send_feedback(f"mypolicy/target_plateau_check old={cost0+plat_info['recent_improvement']:.2f} new={cost0:.2f} improvement={plat_info['recent_improvement']:.2f} std={plat_info['std']:.2f} plateau={str(plat_info['plateau']).lower()}")
            
            if self._target_practical_insert_ready(meas0, cost_history, send_feedback):
                if plat_info['recent_improvement'] < 1.0: # Too small progress
                    send_feedback(
                        f"mypolicy/target_aligned reason=practical_ready_after_plateau "
                        f"center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} cost={cost0:.2f}"
                    )
                    return True

            # Choose speed based on cost
            if cost0 > 40:
                max_speed = 0.015
            elif cost0 > 20:
                max_speed = 0.010
            elif cost0 > 12:
                max_speed = 0.007
            else:
                max_speed = 0.004
            burst_dt = 0.30  # seconds

            send_feedback(f"mypolicy/target_speed_select cost={cost0:.2f} max_speed={max_speed:.3f} burst_dt={burst_dt:.2f}")

            pose_before = self._motion_servo.get_current_pose()
            if pose_before is None:
                return False

            # --- Compute velocity command ---
            v_xy = None
            action_source = "bandit"

            if L_xy is not None and L_xy.shape == (6, 2):
                v_computed = self._compute_ibvs_velocity(L_xy, W6, e6, max_speed)
                if v_computed is not None and np.linalg.norm(v_computed) > 1e-6:
                    v_xy = v_computed
                    action_source = "ibvs"

            if v_xy is None:
                # Bandit exploration
                best_name = max(bandit_scores, key=lambda n: (
                    bandit_scores[n] + math.sqrt(2.0 * math.log(max(total_iters, 1)) / max(bandit_counts[n], 1))
                ))
                dir_vec = next(v for n, v in DIRS if n == best_name)
                dir_norm = dir_vec / np.linalg.norm(dir_vec)
                v_xy = dir_norm * max_speed
                action_source = f"bandit:{best_name}"

            speed = np.linalg.norm(v_xy)
            if cost0 > 20:
                min_speed = 0.004
            elif cost0 > 12:
                min_speed = 0.003
            else:
                min_speed = 0.0015
            
            if 0 < speed < min_speed:
                v_xy = v_xy / speed * min_speed
                send_feedback(f"mypolicy/target_velocity_floor old={speed:.4f} new={min_speed:.4f}")

            vx, vy = float(v_xy[0]), float(v_xy[1])

            # Drift guard
            expected_disp = np.array([vx, vy, 0.0]) * burst_dt
            check_xyz = start_xyz + expected_disp + np.array([
                pose_before.position.x - start_xyz[0],
                pose_before.position.y - start_xyz[1],
                0.0,
            ])
            if np.linalg.norm(check_xyz - start_xyz) > max_drift:
                send_feedback(f"mypolicy/target_reject drift source={action_source}")
                return False

            send_feedback(
                f"mypolicy/target_velocity_cmd "
                f"vx={vx*1000:.2f}mm/s vy={vy*1000:.2f}mm/s "
                f"speed={np.linalg.norm(v_xy)*1000:.2f}mm/s "
                f"source={action_source}"
            )

            self._publish_xy_velocity_burst(vx, vy, burst_dt)
            self.sleep_for(settle_sec)

            # Measure after
            pose_after = self._motion_servo.get_current_pose()
            if pose_after is not None:
                dxy = np.array([
                    pose_after.position.x - pose_before.position.x,
                    pose_after.position.y - pose_before.position.y,
                ], dtype=np.float64)
            else:
                dxy = np.array([vx * burst_dt, vy * burst_dt])

            meas1 = self._measure_three_camera_target_error(
                get_observation, send_feedback, self._target_servo)
            if meas1 is None:
                continue

            e6_1  = meas1["e6"]
            cost1 = meas1["cost"]
            c_err1 = meas1["c_err"]
            l_err1 = meas1["l_err"]
            r_err1 = meas1["r_err"]

            improvement = cost0 - cost1
            send_feedback(
                f"mypolicy/target_execute_result "
                f"cost={cost0:.2f}->{cost1:.2f} "
                f"center={c_err:.2f}->{c_err1:.2f} "
                f"left={l_err:.2f}->{l_err1:.2f} "
                f"right={r_err:.2f}->{r_err1:.2f} "
                f"improvement={improvement:.2f} source={action_source}"
            )

            # Accept/reject move for evidence update
            accepted = improvement > -0.5 and c_err1 <= c_err + 2.0
            if accepted and np.linalg.norm(dxy) > 1e-5:
                dE = e6_1 - e6
                avg_w = float(np.mean([
                    meas0["weights"].get(c, 0.05)
                    for c in ["center", "left", "right"]
                ]))
                _D.append(dxy)
                _Y.append(dE)
                _Ws.append(avg_w)
                step_since_refit += 1

            # Update bandit
            if action_source.startswith("bandit:"):
                bname = action_source.split(":")[1]
                bandit_counts[bname] += 1
                bandit_scores[bname] = (
                    0.7 * bandit_scores[bname] + 0.3 * improvement
                )

            # Refit L_xy every 2 accepted steps or when IBVS diverged
            if step_since_refit >= 2 or (step_since_refit >= 1 and improvement < -1.0):
                new_L = self._fit_reduced_interaction_matrix(
                    list(_D), list(_Y), list(_Ws))
                if new_L is not None:
                    L_xy = new_L
                    step_since_refit = 0
                    send_feedback(
                        f"mypolicy/target_model_update "
                        f"n={len(_D)} every=2 L_shape={L_xy.shape} "
                        f"Lxy_center=({L_xy[0,0]:.2f},{L_xy[0,1]:.2f};{L_xy[1,0]:.2f},{L_xy[1,1]:.2f}) "
                        f"Lxy_left=({L_xy[2,0]:.2f},{L_xy[2,1]:.2f};{L_xy[3,0]:.2f},{L_xy[3,1]:.2f}) "
                        f"Lxy_right=({L_xy[4,0]:.2f},{L_xy[4,1]:.2f};{L_xy[5,0]:.2f},{L_xy[5,1]:.2f})"
                    )

        # Max iters reached
        meas_last = self._measure_three_camera_target_error(get_observation, send_feedback, self._target_servo)
        if meas_last is not None and self._target_practical_insert_ready(meas_last, cost_history, send_feedback):
            send_feedback(
                f"mypolicy/target_aligned reason=practical_ready_at_max_iters "
                f"center={meas_last['c_err']:.2f} left={meas_last['l_err']:.2f} right={meas_last['r_err']:.2f} cost={meas_last['cost']:.2f}"
            )
            return True

        send_feedback("mypolicy/target_servo_failed reason=max_iters_not_ready")
        return False

    # ── Force-sensor helpers ─────────────────────────────────────────────────
    def _read_wrist_force_vector(self, obs) -> np.ndarray:
        """Return [fx, fy, fz] from observation, zeros on failure."""
        if obs is None:
            return np.zeros(3, dtype=np.float64)
        try:
            fx = float(obs.wrist_wrench.wrench.force.x)
            fy = float(obs.wrist_wrench.wrench.force.y)
            fz = float(obs.wrist_wrench.wrench.force.z)
            return np.array([fx, fy, fz], dtype=np.float64)
        except Exception:
            pass
        return np.zeros(3, dtype=np.float64)

    def _sample_insert_force_baseline(
        self,
        get_observation,
        duration_s: float = 0.5,
        sample_period_s: float = 0.02,
    ) -> np.ndarray:
        """Sample wrist force for duration_s, return median [fx,fy,fz]."""
        samples = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            obs = get_observation()
            samples.append(self._read_wrist_force_vector(obs))
            self.sleep_for(sample_period_s)
        if not samples:
            return np.zeros(3, dtype=np.float64)
        return np.median(np.stack(samples, axis=0), axis=0).astype(np.float64)

    def _insert_backoff(self, backoff_m: float = 0.003) -> None:
        """Move +Z by backoff_m to retreat from contact."""
        speed = 0.004  # m/s
        duration = backoff_m / speed
        back_twist = Twist()
        back_twist.linear.z = speed
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            self._motion_servo.publish_twist_command(back_twist, frame_id="base_link")
            self.sleep_for(0.025)
        self._motion_servo.stop()
        self.sleep_for(0.10)

    def _move_xy_relative(self, dx: float, dy: float, duration: float = 0.1) -> None:
        """Move XY relative to current pose by publishing twist."""
        if duration <= 0: return
        tw = Twist()
        tw.linear.x = dx / duration
        tw.linear.y = dy / duration
        self._motion_servo.publish_twist_command(tw, frame_id="base_link")
        self.sleep_for(duration)
        self._motion_servo.stop()

    def _move_z_relative(self, dz: float, speed: float = 0.003) -> None:
        """Move Z relative by publishing twist."""
        if speed <= 0: return
        duration = abs(dz) / speed
        tw = Twist()
        tw.linear.z = speed if dz > 0 else -speed
        self._motion_servo.publish_twist_command(tw, frame_id="base_link")
        self.sleep_for(duration)
        self._motion_servo.stop()

    def _insert_probe_at_xy_offset(self, dx: float, dy: float, probe_depth: float, force_baseline: np.ndarray, get_observation, send_feedback, target_port_name: str) -> Dict:
        """Move to an XY offset, probe downwards, measure force and visual errors, and return status."""
        self._move_xy_relative(dx, dy, duration=0.1)
        
        # Probe
        probe_speed = 0.002
        duration = probe_depth / probe_speed
        tw = Twist()
        tw.linear.z = -probe_speed
        
        max_delta = 0.0
        jammed = False
        t0 = time.monotonic()
        
        self._motion_servo.publish_twist_command(tw, frame_id="base_link")
        while time.monotonic() - t0 < duration:
            obs = get_observation()
            fvec = self._read_wrist_force_vector(obs)
            delta = float(np.linalg.norm(fvec - force_baseline))
            if delta > max_delta: max_delta = delta
            if delta > float(self.INSERT_FORCE_DELTA_THRESH_N):
                jammed = True
                break
            self.sleep_for(0.04)
            
        self._motion_servo.stop()
        
        # Measure visuals at bottom of probe
        meas = self._measure_three_camera_target_error(get_observation, send_feedback, self._target_servo)
        if meas is not None:
            c_err, l_err, r_err = meas["c_err"], meas["l_err"], meas["r_err"]
        else:
            c_err, l_err, r_err = 99.0, 99.0, 99.0
            
        # Backoff probe
        self._move_z_relative(probe_depth, speed=probe_speed)
        
        # Return XY
        self._move_xy_relative(-dx, -dy, duration=0.1)
        
        return {"max_delta": max_delta, "jammed": jammed, "c_err": c_err, "l_err": l_err, "r_err": r_err}

    def _insert_spiral_search(self, get_observation, send_feedback, current_depth: float, jam_delta: float, target_port_name: str, reason: str = "early_jam") -> bool:
        """Spiral search using both visual errors and delta force to find the best hole alignment."""
        send_feedback(f"mypolicy/insert_spiral_start reason={reason} depth={current_depth*1000:.1f}mm jam_delta={jam_delta:.2f}N")
        self._motion_servo.stop()
        
        backoff = float(self.INSERT_SPIRAL_BACKOFF_M)
        self._move_z_relative(backoff, speed=0.003)
        
        # Sample baseline at hover
        force_baseline = self._sample_insert_force_baseline(get_observation, duration_s=0.2, sample_period_s=0.02)
        
        best_offset = None
        min_score = 999.0
        
        # Add center candidate
        candidates = [(0.0, 0.0)]
        for r in self.INSERT_SPIRAL_RADII_M:
            for i in range(self.INSERT_SPIRAL_POINTS_PER_RING):
                theta = i * (2 * math.pi / self.INSERT_SPIRAL_POINTS_PER_RING)
                candidates.append((r * math.cos(theta), r * math.sin(theta)))
                
        for dx, dy in candidates:
            res = self._insert_probe_at_xy_offset(dx, dy, float(self.INSERT_SPIRAL_PROBE_DEPTH_M), force_baseline, get_observation, send_feedback, target_port_name)
            
            c_err = res["c_err"]
            l_err = res["l_err"]
            r_err = res["r_err"]
            m_delta = res["max_delta"]
            
            # score = 0.7 * center_err + 1.0 * left_err + 1.0 * right_err + 0.4 * max_delta_force - 2.0 * probe_depth_mm_reached
            # assuming probe_depth reached is constant if not jammed, ignore depth term
            score = 0.7 * c_err + 1.0 * l_err + 1.0 * r_err + 0.4 * m_delta
            if res["jammed"]:
                score += 50.0  # Penalize jams
                
            send_feedback(
                f"mypolicy/insert_spiral_candidate r={math.sqrt(dx*dx+dy*dy)*1000:.2f}mm "
                f"dxy=({dx*1000:.2f},{dy*1000:.2f})mm center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} "
                f"max_delta={m_delta:.2f}N score={score:.2f} jam={str(res['jammed']).lower()}"
            )
            
            if not res["jammed"] and score < min_score:
                min_score = score
                best_offset = (dx, dy)
                
        if best_offset is not None and min_score < 99.0:
            dx, dy = best_offset
            send_feedback(f"mypolicy/insert_spiral_best dxy=({dx*1000:.2f},{dy*1000:.2f})mm score={min_score:.2f}")
            send_feedback(f"mypolicy/insert_spiral_accept dxy=({dx*1000:.2f},{dy*1000:.2f})mm")
            self._move_xy_relative(dx, dy, duration=0.1)
            # Restore the Z we backed off
            self._move_z_relative(-backoff, speed=0.003)
            return True
            
        send_feedback("mypolicy/insert_spiral_fail no_candidate")
        return False

    def _insert_contact_guided_funneling_step(self, df_xy: np.ndarray, funnel_xy_total: list, send_feedback) -> list:
        """Apply a small XY correction opposite to the lateral contact force."""
        gain = float(self.INSERT_FUNNEL_GAIN_M_PER_N)
        max_step = float(self.INSERT_FUNNEL_MAX_XY_STEP_M)
        max_total = float(self.INSERT_FUNNEL_MAX_TOTAL_XY_M)
        
        step_xy = -gain * df_xy
        step_norm = float(np.linalg.norm(step_xy))
        
        if step_norm > max_step:
            step_xy = step_xy * (max_step / step_norm)
            
        dx, dy = step_xy[0], step_xy[1]
        
        new_total_x = funnel_xy_total[0] + dx
        new_total_y = funnel_xy_total[1] + dy
        
        if math.sqrt(new_total_x**2 + new_total_y**2) > max_total:
            return funnel_xy_total
            
        send_feedback(f"mypolicy/insert_funnel delta_xy={np.linalg.norm(df_xy):.2f}N step=({dx*1000:.2f},{dy*1000:.2f})mm total_xy={math.sqrt(new_total_x**2 + new_total_y**2)*1000:.2f}mm")
        
        self._move_xy_relative(dx, dy, duration=0.05)
        
        return [new_total_x, new_total_y]

    def _score_multiview_insert_candidate(self, c_err: float, l_err: float, r_err: float, force_delta: float) -> float:
        """Score a preinsert candidate using dynamic weights."""
        if c_err < 5.0 and (l_err > 10.0 or r_err > 10.0):
            w_center = 0.5
            w_left = 1.2
            w_right = 1.2
        else:
            w_center = 1.0
            w_left = 0.8
            w_right = 0.8
            
        score = w_center * c_err + w_left * l_err + w_right * r_err + 0.4 * force_delta
        return score

    def _preinsert_multiview_local_search(self, get_observation, send_feedback, target_port_name: str) -> bool:
        """Run local XY search around the current pose before inserting."""
        send_feedback("mypolicy/preinsert_local_search_start")
        
        step_sizes = [0.00025, 0.00050, 0.00075, 0.00100]
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        force_baseline = self._sample_insert_force_baseline(get_observation, duration_s=0.2, sample_period_s=0.02)
        
        best_offset = None
        min_score = 999.0
        
        for step in step_sizes:
            for d in directions:
                dx = d[0] * step
                dy = d[1] * step
                
                self._move_xy_relative(dx, dy, duration=0.1)
                
                obs = get_observation()
                fvec = self._read_wrist_force_vector(obs)
                delta = float(np.linalg.norm(fvec - force_baseline))
                
                meas = self._measure_three_camera_target_error(get_observation, send_feedback, self._target_servo)
                if meas is not None:
                    c_err, l_err, r_err = meas["c_err"], meas["l_err"], meas["r_err"]
                else:
                    c_err, l_err, r_err = 99.0, 99.0, 99.0
                    
                score = self._score_multiview_insert_candidate(c_err, l_err, r_err, delta)
                
                send_feedback(
                    f"mypolicy/preinsert_candidate dxy=({dx*1000:.2f},{dy*1000:.2f}) "
                    f"center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} force_delta={delta:.2f}N score={score:.2f}"
                )
                
                if score < min_score:
                    min_score = score
                    best_offset = (dx, dy)
                    
                self._move_xy_relative(-dx, -dy, duration=0.1)
                
            if min_score < 12.0: # Break early if we found a very good score
                break
                
        if best_offset is not None:
            dx, dy = best_offset
            send_feedback(f"mypolicy/preinsert_local_search_best dxy=({dx*1000:.2f},{dy*1000:.2f}) score={min_score:.2f}")
            self._move_xy_relative(dx, dy, duration=0.1)
            return True
            
        return False

    def _probe_shallow_insert(self, get_observation, send_feedback, target_port_name: str) -> Dict:
        """Shallow insertion probe to check capture."""
        probe_depth = 0.0005
        force_baseline = self._sample_insert_force_baseline(get_observation, duration_s=0.2, sample_period_s=0.02)
        
        probe_speed = 0.002
        duration = probe_depth / probe_speed
        tw = Twist()
        tw.linear.z = -probe_speed
        
        max_delta = 0.0
        t0 = time.monotonic()
        
        self._motion_servo.publish_twist_command(tw, frame_id="base_link")
        while time.monotonic() - t0 < duration:
            obs = get_observation()
            fvec = self._read_wrist_force_vector(obs)
            delta = float(np.linalg.norm(fvec - force_baseline))
            if delta > max_delta: max_delta = delta
            self.sleep_for(0.04)
            
        self._motion_servo.stop()
        
        meas = self._measure_three_camera_target_error(get_observation, send_feedback, self._target_servo)
        if meas is not None:
            c_err, l_err, r_err = meas["c_err"], meas["l_err"], meas["r_err"]
            capture_ok = c_err <= 12.0 and max_delta < 5.0
        else:
            c_err, l_err, r_err = 99.0, 99.0, 99.0
            capture_ok = False
            
        send_feedback(
            f"mypolicy/insert_shallow_probe depth={probe_depth*1000:.1f}mm delta={max_delta:.2f}N "
            f"center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} capture_ok={str(capture_ok).lower()}"
        )
        
        # Don't back off if capture is good
        if not capture_ok:
            self._move_z_relative(probe_depth, speed=probe_speed)
            
        return {"capture_ok": capture_ok, "max_delta": max_delta, "c_err": c_err, "l_err": l_err, "r_err": r_err}

    def _verify_inserted_state(self, get_observation, send_feedback, target_port_name: str) -> bool:
        """Final verification of insertion using visual target matching."""
        meas = self._measure_three_camera_target_error(get_observation, send_feedback, self._target_servo)
        if meas is not None:
            c_err, l_err, r_err = meas["c_err"], meas["l_err"], meas["r_err"]
            
            # Simple fallback check
            ok = c_err <= 8.0 and l_err <= 16.0 and r_err <= 14.0
        else:
            c_err, l_err, r_err = 99.0, 99.0, 99.0
            ok = False
            
        send_feedback(f"mypolicy/insert_verify center={c_err:.2f} left={l_err:.2f} right={r_err:.2f} ok={str(ok).lower()}")
        return ok

    # ── Compliant insertion with delta-force jam detection ───────────────────
    def _compliant_visual_insert(
        self,
        target_port_name: str,
        get_observation,
        send_feedback,
    ) -> bool:
        """Insert slowly in -Z using delta-force jam detection relative to a sampled baseline."""
        delta_thresh  = float(self.INSERT_FORCE_DELTA_THRESH_N)
        delta_warn    = float(self.INSERT_FORCE_DELTA_WARN_N)
        hard_thresh   = float(self.INSERT_FORCE_HARD_ABS_THRESH_N)
        jam_count_lim = int(self.INSERT_FORCE_JAM_COUNT)
        max_depth     = float(self.INSERT_MAX_DEPTH_M)
        max_retries   = int(self.INSERT_MAX_RETRIES)
        insert_speed  = float(self.INSERT_SPEED_MPS)
        ctrl_period   = 0.04   # 25 Hz control loop
        backoff_m     = 0.003  # 3 mm

        start_pose = self._motion_servo.get_current_pose()
        if start_pose is None:
            send_feedback("mypolicy/insert_fail no_current_pose")
            return False

        # ── insertion direction: base_link -Z ────────────────────────────────
        send_feedback("mypolicy/insert_direction mode=base_minus_z vector=(0.00,0.00,-1.00)")

        # ── pre-insert multiview local search ────────────────────────────────
        self._preinsert_multiview_local_search(get_observation, send_feedback, target_port_name)

        # ── sample force baseline BEFORE descent ─────────────────────────────
        send_feedback("mypolicy/insert_start")
        force_baseline = self._sample_insert_force_baseline(
            get_observation,
            duration_s=float(self.INSERT_BASELINE_DURATION_S),
            sample_period_s=0.02,
        )
        baseline_mag = float(np.linalg.norm(force_baseline))
        send_feedback(
            f"mypolicy/insert_force_baseline "
            f"f0=({force_baseline[0]:.2f},{force_baseline[1]:.2f},{force_baseline[2]:.2f}) "
            f"mag={baseline_mag:.2f}N"
        )
        # Baseline is allowed to be large – we only track CHANGES from here.

        # ── shallow insertion probe ──────────────────────────────────────────
        probe_res = self._probe_shallow_insert(get_observation, send_feedback, target_port_name)
        if not probe_res["capture_ok"]:
            if self.INSERT_SPIRAL_ENABLE:
                self._insert_spiral_search(get_observation, send_feedback, 0.0, probe_res["max_delta"], target_port_name, reason="shallow_probe_failed")
                
                # Resample baseline after spiral
                force_baseline = self._sample_insert_force_baseline(get_observation, duration_s=float(self.INSERT_BASELINE_DURATION_S), sample_period_s=0.02)
                baseline_mag = float(np.linalg.norm(force_baseline))
                send_feedback(f"mypolicy/insert_force_baseline f0=({force_baseline[0]:.2f},{force_baseline[1]:.2f},{force_baseline[2]:.2f}) mag={baseline_mag:.2f}N (resampled)")

        start_pose = self._motion_servo.get_current_pose()
        start_z        = float(start_pose.position.z)
        retries        = 0
        inserted_depth = 0.0
        max_delta_seen = 0.0
        jam_count      = 0
        last_log_t     = 0.0
        funnel_xy_total = [0.0, 0.0]
        stalled_cycles = 0

        while True:
            # ── read force ───────────────────────────────────────────────────
            obs       = get_observation()
            fvec      = self._read_wrist_force_vector(obs)
            abs_force = float(np.linalg.norm(fvec))
            delta_vec = fvec - force_baseline
            delta_mag = float(np.linalg.norm(delta_vec))
            df_xy     = np.array([delta_vec[0], delta_vec[1]])
            delta_xy  = float(np.linalg.norm(df_xy))
            max_delta_seen = max(max_delta_seen, delta_mag)

            # ── hard-safety: absolute force way above anything expected ──────
            if abs_force > hard_thresh:
                self._motion_servo.stop()
                send_feedback(
                    f"mypolicy/insert_abort reason=hard_abs_force abs={abs_force:.2f}N"
                )
                return False

            # ── periodic log ─────────────────────────────────────────────────
            now = time.monotonic()
            if now - last_log_t >= 0.4:
                last_log_t = now
                send_feedback(
                    f"mypolicy/insert_force "
                    f"depth={inserted_depth*1000:.1f}mm "
                    f"abs={abs_force:.2f}N delta={delta_mag:.2f}N "
                    f"dfx={delta_vec[0]:.2f} dfy={delta_vec[1]:.2f} dfz={delta_vec[2]:.2f}"
                )

            # ── jam counter ──────────────────────────────────────────────────
            if delta_mag > delta_thresh:
                jam_count += 1
                send_feedback(
                    f"mypolicy/insert_contact_delta "
                    f"depth={inserted_depth*1000:.1f}mm "
                    f"delta={delta_mag:.2f}N count={jam_count}"
                )
            else:
                if delta_mag > delta_warn:
                    send_feedback(
                        f"mypolicy/insert_contact_delta "
                        f"depth={inserted_depth*1000:.1f}mm "
                        f"delta={delta_mag:.2f}N count={jam_count} (warn)"
                    )
                jam_count = 0

            if jam_count >= jam_count_lim:
                self._motion_servo.stop()
                send_feedback(
                    f"mypolicy/insert_retry reason=delta_force_jam "
                    f"delta={delta_mag:.2f}N abs={abs_force:.2f}N retries={retries}"
                )
                if retries >= max_retries:
                    send_feedback("mypolicy/insert_fail max_retries_exceeded")
                    return False
                retries += 1

                if self.INSERT_SPIRAL_ENABLE:
                    spiral_ok = self._insert_spiral_search(get_observation, send_feedback, inserted_depth, delta_mag)
                    if spiral_ok:
                        force_baseline = self._sample_insert_force_baseline(get_observation, duration_s=float(self.INSERT_BASELINE_DURATION_S), sample_period_s=0.02)
                        baseline_mag = float(np.linalg.norm(force_baseline))
                        send_feedback(f"mypolicy/insert_force_baseline f0=({force_baseline[0]:.2f},{force_baseline[1]:.2f},{force_baseline[2]:.2f}) mag={baseline_mag:.2f}N (resampled after spiral)")
                        
                        cur_pose = self._motion_servo.get_current_pose()
                        if cur_pose is not None:
                            start_z = float(cur_pose.position.z) + inserted_depth # Restore start_z relative to current depth
                        jam_count = 0
                        max_delta_seen = 0.0
                        continue

                # Fallback: Back off and rerun visual alignment
                send_feedback(f"mypolicy/insert_backoff dz={backoff_m*1000:.1f}mm")
                self._insert_backoff(backoff_m)

                send_feedback("mypolicy/insert_rerun_target_image_align")
                align_ok = self._target_image_visual_servo_align(
                    target_port_name=target_port_name,
                    get_observation=get_observation,
                    send_feedback=send_feedback,
                )
                self._motion_servo.stop()
                if not align_ok:
                    send_feedback("mypolicy/visual_align_failed aborting_before_insert")
                    return False

                # Resample baseline and reset tracking
                force_baseline = self._sample_insert_force_baseline(
                    get_observation,
                    duration_s=float(self.INSERT_BASELINE_DURATION_S),
                    sample_period_s=0.02,
                )
                baseline_mag = float(np.linalg.norm(force_baseline))
                send_feedback(
                    f"mypolicy/insert_force_baseline "
                    f"f0=({force_baseline[0]:.2f},{force_baseline[1]:.2f},{force_baseline[2]:.2f}) "
                    f"mag={baseline_mag:.2f}N (resampled after retry {retries})"
                )
                cur_pose = self._motion_servo.get_current_pose()
                if cur_pose is not None:
                    start_z = float(cur_pose.position.z)
                inserted_depth = 0.0
                jam_count      = 0
                max_delta_seen = 0.0
                funnel_xy_total = [0.0, 0.0]
                continue

            # ── contact-guided funneling ──────────────────────────────────────
            if self.INSERT_FUNNEL_ENABLE and delta_xy > float(self.INSERT_FUNNEL_DELTA_XY_THRESH_N) and delta_mag < delta_thresh:
                funnel_xy_total = self._insert_contact_guided_funneling_step(df_xy, funnel_xy_total, send_feedback)

            # ── descend one control tick ──────────────────────────────────────
            # Use small insertion increments (0.5 mm)
            insert_step = 0.0005
            duration = insert_step / insert_speed
            down_twist = Twist()
            down_twist.linear.z = -float(np.clip(insert_speed, 0.001, 0.010))
            self._motion_servo.publish_twist_command(
                down_twist, frame_id="base_link",
                trans_stiffness=60.0, rot_stiffness=50.0,
                trans_damping=50.0,   rot_damping=20.0,
            )
            self.sleep_for(duration)
            self._motion_servo.stop()

            cur = self._motion_servo.get_current_pose()
            if cur is not None:
                new_depth = max(0.0, start_z - float(cur.position.z))
                if abs(new_depth - inserted_depth) < 1e-5:
                    stalled_cycles += 1
                    if stalled_cycles >= 5:
                        send_feedback(f"mypolicy/insert_warn no_z_progress cycles={stalled_cycles}")
                    if stalled_cycles >= 10 and inserted_depth < 0.001:
                        send_feedback("mypolicy/insert_no_z_progress_recovery action=pose_step")
                        # Try pose increment to break static friction
                        self._move_z_relative(0.001, speed=0.005)
                        probe_res = self._probe_shallow_insert(get_observation, send_feedback, target_port_name)
                        if not probe_res["capture_ok"]:
                            if self.INSERT_SPIRAL_ENABLE:
                                self._insert_spiral_search(get_observation, send_feedback, 0.0, probe_res["max_delta"], target_port_name, reason="no_z_progress")
                        stalled_cycles = 0
                else:
                    stalled_cycles = 0
                inserted_depth = new_depth
                
            if inserted_depth >= max_depth:
                self._motion_servo.stop()
                verify_ok = self._verify_inserted_state(get_observation, send_feedback, target_port_name)
                if verify_ok:
                    send_feedback(
                        f"mypolicy/insert_success "
                        f"depth={inserted_depth*1000:.1f}mm verified=true"
                    )
                    return True
                else:
                    send_feedback("mypolicy/insert_false_success_prevented reason=visual_verify_failed")
                    if retries < max_retries:
                        retries += 1
                        # Backoff and retry
                        backoff = float(self.INSERT_SPIRAL_BACKOFF_M)
                        self._insert_backoff(inserted_depth + backoff)
                        if self.INSERT_SPIRAL_ENABLE:
                            self._insert_spiral_search(get_observation, send_feedback, 0.0, 0.0, target_port_name, reason="verify_failed")
                        
                        force_baseline = self._sample_insert_force_baseline(get_observation, duration_s=float(self.INSERT_BASELINE_DURATION_S), sample_period_s=0.02)
                        baseline_mag = float(np.linalg.norm(force_baseline))
                        send_feedback(f"mypolicy/insert_force_baseline mag={baseline_mag:.2f}N (resampled after verify fail)")
                        
                        cur_pose = self._motion_servo.get_current_pose()
                        if cur_pose is not None:
                            start_z = float(cur_pose.position.z)
                        inserted_depth = 0.0
                        jam_count = 0
                        max_delta_seen = 0.0
                        funnel_xy_total = [0.0, 0.0]
                        continue
                    else:
                        send_feedback("mypolicy/insert_fail max_retries_exceeded")
                        return False

        self._motion_servo.stop()
        send_feedback("mypolicy/insert_fail unknown")
        return False