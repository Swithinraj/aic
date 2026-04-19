from __future__ import annotations

import json
import math
import os
import threading
import time
from copy import deepcopy
from typing import Callable, Dict, List, Optional

import numpy as np
import rclpy
from aic_control_interfaces.msg import ControllerState, MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3, Wrench
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from team_policy.planner.cartesian_planner import CartesianPlanner
from team_policy.planner.combined_yolo_depth_pose_planner import CombinedYoloDepthPosePlanner
from team_policy.planner.rviz_click_to_move import (
    _quat_error_rotvec as rviz_quat_error_rotvec,
    _quat_from_axis_angle as rviz_quat_from_axis_angle,
    _quat_multiply as rviz_quat_multiply,
    _quat_to_msg as rviz_quat_to_msg,
)


def _copy_pose(pose: Pose) -> Pose:
    return deepcopy(pose)


def _quat_to_np(q: Quaternion):
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def _quat_normalize(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def _quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_inverse(q):
    x, y, z, w = _quat_normalize(q)
    return [-x, -y, -z, w]


def _quat_from_axis_angle(axis, angle):
    ax = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(ax))
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    ax = ax / n
    s = math.sin(0.5 * angle)
    c = math.cos(0.5 * angle)
    return [float(ax[0] * s), float(ax[1] * s), float(ax[2] * s), float(c)]


def _quat_to_rotvec(q):
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


def _quat_error_rotvec(current: Quaternion, target: Quaternion):
    qc = _quat_normalize(_quat_to_np(current))
    qt = _quat_normalize(_quat_to_np(target))
    q_err = _quat_multiply(qt, _quat_inverse(qc))
    return _quat_to_rotvec(q_err)


def _rotate_vector_by_quaternion(q, v):
    qn = _quat_normalize(q)
    vq = [float(v[0]), float(v[1]), float(v[2]), 0.0]
    qr = _quat_multiply(_quat_multiply(qn, vq), _quat_inverse(qn))
    return np.asarray(qr[:3], dtype=np.float64)


def _normalize_vec(v):
    vv = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(vv))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vv / n


def _project_to_plane(v, plane_normal):
    nn = _normalize_vec(plane_normal)
    return np.asarray(v, dtype=np.float64) - float(np.dot(v, nn)) * nn


def _signed_angle_about_axis_rad(v_from, v_to, axis):
    aa = _normalize_vec(v_from)
    bb = _normalize_vec(v_to)
    kk = _normalize_vec(axis)
    if np.linalg.norm(aa) < 1e-9 or np.linalg.norm(bb) < 1e-9 or np.linalg.norm(kk) < 1e-9:
        return 0.0
    cross_ab = np.cross(aa, bb)
    sin_term = float(np.dot(kk, cross_ab))
    cos_term = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(math.atan2(sin_term, cos_term))


def _local_pitch_error_to_base_direction_rad(
    current: Quaternion,
    local_forward_axis,
    local_pitch_axis,
    desired_base_direction,
) -> float:
    q_current = _quat_normalize(_quat_to_np(current))
    forward_base = _rotate_vector_by_quaternion(q_current, local_forward_axis)
    pitch_axis_base = _rotate_vector_by_quaternion(q_current, local_pitch_axis)
    desired_base = _normalize_vec(desired_base_direction)

    forward_proj = _project_to_plane(forward_base, pitch_axis_base)
    desired_proj = _project_to_plane(desired_base, pitch_axis_base)

    if float(np.linalg.norm(forward_proj)) < 1e-9 or float(np.linalg.norm(desired_proj)) < 1e-9:
        return 0.0

    return _signed_angle_about_axis_rad(forward_proj, desired_proj, pitch_axis_base)


def _yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    """Create a downward-facing quaternion with a given yaw rotation."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=float(math.sin(yaw_rad / 2.0)),
        w=float(math.cos(yaw_rad / 2.0)),
    )


def _quat_to_euler_xyz(q: Quaternion) -> tuple[float, float, float]:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi * 0.5, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return Quaternion(
        x=float(sr * cp * cy - cr * sp * sy),
        y=float(cr * sp * cy + sr * cp * sy),
        z=float(cr * cp * sy - sr * sp * cy),
        w=float(cr * cp * cy + sr * sp * sy),
    )


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
        detections = self._parse_detection_list(msg.data)
        with self._lock:
            self._latest_fused = {
                "time": time.monotonic(),
                "detections": detections,
            }

    def _cb_camera(self, camera_name: str, msg: String) -> None:
        detections = self._parse_detection_list(msg.data)
        for det in detections:
            det["camera_name"] = camera_name
        with self._lock:
            self._latest_per_camera[camera_name] = {
                "time": time.monotonic(),
                "detections": detections,
            }

    def find_best(
        self,
        matcher: Callable[[Dict], bool],
        preferred_camera: Optional[str] = None,
        min_update_time: float = 0.0,
        require_pose: bool = True,
        freshness_sec: float = 2.0,
    ) -> Optional[Dict]:
        now = time.monotonic()
        with self._lock:
            snapshot = {
                "time": float(self._latest_fused["time"]),
                "detections": [dict(det) for det in self._latest_fused["detections"]],
            }

        candidates: List[Dict] = []
        update_time = float(snapshot["time"])

        if update_time > 0.0 and update_time >= float(min_update_time) and now - update_time <= float(freshness_sec):
            for det in snapshot["detections"]:
                if require_pose and not self._has_pose(det):
                    continue
                if not matcher(det):
                    continue
                candidates.append(
                    {
                        "camera_name": det.get("camera_name", det.get("source", "fused")),
                        "update_time": update_time,
                        "detection": det,
                        "confidence": float(det.get("confidence", 0.0)),
                    }
                )

        if not candidates:
            return None

        if preferred_camera is not None:
            preferred = [c for c in candidates if c["camera_name"] == preferred_camera]
            if preferred:
                candidates = preferred

        candidates.sort(
            key=lambda item: (
                item["confidence"],
                item["update_time"],
            ),
            reverse=True,
        )
        return candidates[0]

    def get_all_detections(self, freshness_sec: float = 2.0) -> List[Dict]:
        now = time.monotonic()
        with self._lock:
            update_time = float(self._latest_fused["time"])
            if update_time > 0.0 and now - update_time <= freshness_sec:
                return [dict(det) for det in self._latest_fused["detections"]]

            merged: List[Dict] = []
            for camera_name, snapshot in self._latest_per_camera.items():
                cam_time = float(snapshot["time"])
                if cam_time <= 0.0 or now - cam_time > freshness_sec:
                    continue
                for det in snapshot["detections"]:
                    merged.append(dict(det))
            return merged

    def get_camera_detections(self, camera_name: str, freshness_sec: float = 0.5) -> List[Dict]:
        now = time.monotonic()
        with self._lock:
            snapshot = self._latest_per_camera.get(camera_name, {"time": 0.0, "detections": []})
            update_time = float(snapshot["time"])
            if update_time <= 0.0 or now - update_time > freshness_sec:
                return []
            return [dict(det) for det in snapshot["detections"]]

    def _has_pose(self, det: Dict) -> bool:
        pose = det.get("pose_base_link")
        return isinstance(pose, dict) and isinstance(pose.get("position"), dict) and isinstance(pose.get("orientation"), dict)

class MotionServoNode(Node):
    def __init__(self):
        super().__init__("mypolicy_motion_servo")
        self._lock = threading.Lock()
        self._current_state: Optional[ControllerState] = None
        self._current_tcp_pose: Optional[Pose] = None
        self._mode_request_sent = False

        self.command_frame = "base_link"
        self.position_tolerance = 0.020
        self.orientation_tolerance_rad = 0.08
        self.linear_kp = 1.0
        self.angular_kp = 1.2
        self.max_linear_speed = 0.05
        self.min_linear_speed = 0.015
        self.max_angular_speed = 0.8
        self.min_angular_speed = 0.08
        self.trans_stiffness = 90.0
        self.rot_stiffness = 5.0
        self.trans_damping = 50.0
        self.rot_damping = 5.0

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
            if self._current_tcp_pose is None:
                return None
            return _copy_pose(self._current_tcp_pose)

    def get_fts_wrench(self) -> Optional[Dict]:
        """Get the latest force/torque sensor readings from ControllerState."""
        with self._lock:
            if self._current_state is None:
                return None
            w = self._current_state.fts_tare_offset.wrench
            return {
                "fx": float(w.force.x),
                "fy": float(w.force.y),
                "fz": float(w.force.z),
                "tx": float(w.torque.x),
                "ty": float(w.torque.y),
                "tz": float(w.torque.z),
            }

    def compute_twist_to_waypoint(self, current_pose: Pose, waypoint: Pose) -> Twist:
        dx = waypoint.position.x - current_pose.position.x
        dy = waypoint.position.y - current_pose.position.y
        dz = waypoint.position.z - current_pose.position.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        rotvec, angle = _quat_error_rotvec(current_pose.orientation, waypoint.orientation)

        twist = Twist()

        if distance >= 1e-6:
            commanded_speed = self.linear_kp * distance
            if commanded_speed > self.max_linear_speed:
                commanded_speed = self.max_linear_speed
            elif commanded_speed < self.min_linear_speed:
                commanded_speed = self.min_linear_speed
            scale = commanded_speed / distance
            twist.linear.x = dx * scale
            twist.linear.y = dy * scale
            twist.linear.z = dz * scale

        if angle >= 1e-6:
            commanded_ang = self.angular_kp * angle
            if commanded_ang > self.max_angular_speed:
                commanded_ang = self.max_angular_speed
            elif commanded_ang < self.min_angular_speed:
                commanded_ang = self.min_angular_speed
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
        use_frame = self.command_frame if frame_id is None else str(frame_id)
        use_trans_stiffness = self.trans_stiffness if trans_stiffness is None else float(trans_stiffness)
        use_rot_stiffness = self.rot_stiffness if rot_stiffness is None else float(rot_stiffness)
        use_trans_damping = self.trans_damping if trans_damping is None else float(trans_damping)
        use_rot_damping = self.rot_damping if rot_damping is None else float(rot_damping)
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = use_frame
        msg.velocity = twist
        msg.target_stiffness = [
            use_trans_stiffness, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, use_trans_stiffness, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, use_trans_stiffness, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, use_rot_stiffness, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, use_rot_stiffness, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, use_rot_stiffness,
        ]
        msg.target_damping = [
            use_trans_damping, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, use_trans_damping, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, use_trans_damping, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, use_rot_damping, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, use_rot_damping, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, use_rot_damping,
        ]
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        self.pose_command_pub.publish(msg)

    def publish_compliant_insertion_command(
        self,
        twist: Twist,
        z_force: float = -3.0,
        z_stiffness: float = 20.0,
        xy_stiffness: float = 90.0,
    ) -> None:
        """Publish a motion command with reduced Z stiffness and a downward feedforward wrench for compliant insertion."""
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.command_frame
        msg.velocity = twist
        msg.target_stiffness = [
            xy_stiffness, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, xy_stiffness, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, z_stiffness, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, self.rot_stiffness, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, self.rot_stiffness, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, self.rot_stiffness,
        ]
        msg.target_damping = [
            self.trans_damping, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, self.trans_damping, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, self.trans_damping, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, self.rot_damping, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, self.rot_damping, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, self.rot_damping,
        ]
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=z_force),
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
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.command_frame
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose = waypoint
            path.poses.append(pose_stamped)
        self.path_pub.publish(path)


class mypolicy(Policy):
    PIXEL_CORRECTION_ENABLED = True
    SERVO_HOVER_Z = 0.0
    PATH_SETTLE_SEC = 1.0
    SERVO_WORLD_TOLERANCE = 0.003
    SERVO_MAX_ITERATIONS = 600
    SERVO_STEP_SEC = 0.10
    SERVO_TIMEOUT_SEC = 30.0

    SERVO_DLS_DAMPING = 1.2
    SERVO_GAIN_XY = 0.012
    SERVO_GAIN_Z = 0.008
    SERVO_MAX_LINEAR_SPEED_XY = 0.0014
    SERVO_MAX_LINEAR_SPEED_Z = 0.0003
    SERVO_MIN_LINEAR_SPEED_XY = 0.0008
    SERVO_CMD_SMOOTHING = 0.60
    SERVO_CONVERGED_PX = 160.0
    SERVO_CONVERGED_PX_STABLE_COUNT = 5
    SERVO_Z_ENABLE_PX = 80.0
    SERVO_ABORT_PX = 120.0

    SERVO_MAX_YAW_RATE = 0.10
    SERVO_MIN_CONFIDENCE = 0.20
    SERVO_LAST_VALID_PAIR_SEC = 0.60

    PITCH_GROUND_TIMEOUT_SEC = 20.0
    PITCH_GROUND_PITCH_TOL_RAD = 0.015
    PITCH_GROUND_STABLE_COUNT = 6
    PITCH_GROUND_MAX_RATE = 0.18
    PITCH_GROUND_MIN_RATE = 0.03
    PITCH_GROUND_KP = 1.6
    PITCH_GROUND_FORWARD_AXIS_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    PITCH_GROUND_AXIS_LOCAL = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    PITCH_GROUND_BASE_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    EDGE_ALIGN_TIMEOUT_SEC = 45.0
    EDGE_ALIGN_DLS_DAMPING = 3.5
    EDGE_ALIGN_GAIN_XY = 0.0010
    EDGE_ALIGN_MAX_LINEAR_SPEED_XY = 0.0005
    EDGE_ALIGN_CONVERGED_PX = 14.0
    EDGE_ALIGN_STABLE_COUNT = 8
    EDGE_ALIGN_MAX_PITCH_RATE = 0.030
    EDGE_ALIGN_PITCH_RATE_COARSE = 0.020
    EDGE_ALIGN_PITCH_RATE_FINE = 0.008
    EDGE_ALIGN_SIGN_FLIP_PATIENCE = 10
    EDGE_ALIGN_AXIAL_CONVERGED_PX = 8.0
    EDGE_ALIGN_SMOOTHING = 0.55

    SAMPLE_VERIFY_GOAL_LOST_MAX_COUNT = 4
    SAMPLE_VERIFY_GOAL_VISIBILITY_FRESHNESS_SEC = 1.0
    SAMPLE_VERIFY_IMAGE_MARGIN_PX = 2.0

    INSERT_Z_STEP = 0.001
    INSERT_MAX_LATERAL_FORCE = 15.0
    INSERT_MAX_Z_FORCE = 30.0
    INSERT_SETTLED_FORCE = 2.0
    INSERT_MAX_DEPTH = 0.04
    INSERT_STEP_SEC = 0.15
    INSERT_TIMEOUT_SEC = 20.0

    DEFAULT_FX = 600.0
    DEFAULT_FY = 600.0
    DEFAULT_CX = 320.0
    DEFAULT_CY = 240.0

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
        self._gripper_classes = self._parse_name_set(os.environ.get("YOLOV12_GRIPPER_CLASSES", "gripper,gripper_tip,gripper_tcp"))
        self.PIXEL_CORRECTION_ENABLED = str(os.environ.get("MYPOLICY_PIXEL_CORRECTION_ENABLED", "1")).strip().lower() not in ("0", "false", "no", "off")
        self._edge_pitch_sign = float(os.environ.get("MYPOLICY_EDGE_PITCH_SIGN", "1.0"))
        self._phase2_rotation_axis_local = str(os.environ.get("MYPOLICY_PHASE2_ROTATION_AXIS_LOCAL", "x")).strip().lower()
        self._phase2_pca_target_ground_direction = str(os.environ.get("MYPOLICY_PHASE2_PCA_TARGET_GROUND_DIRECTION", "down")).strip().lower()
        self._phase2_module_pose_freshness_sec = float(os.environ.get("MYPOLICY_PHASE2_MODULE_POSE_FRESHNESS_SEC", "1.0"))
        self._phase2_module_max_dist_to_tcp_m = float(os.environ.get("MYPOLICY_PHASE2_MODULE_MAX_DIST_TO_TCP_M", "0.35"))
        self._phase2_pca_max_abs_deg = float(os.environ.get("MYPOLICY_PHASE2_PCA_MAX_ABS_DEG", "95.0"))
        self._phase2_fixed_fallback_deg = float(os.environ.get("MYPOLICY_PHASE2_ROTATION_DEG", "-90.0"))
        self._phase2_allow_fixed_fallback = str(os.environ.get("MYPOLICY_PHASE2_ALLOW_FIXED_FALLBACK", "0")).strip().lower() in ("1", "true", "yes", "on")
        self._initial_prev_xy = np.zeros(2, dtype=np.float64)
        self._edge_prev_xy = np.zeros(2, dtype=np.float64)
        self._edge_prev_pitch = 0.0

        # Image parsing for visual servoing debugging
        self._cv_bridge = CvBridge()
        self._annotated_images = {}
        self._parent_node.create_subscription(Image, "/left_camera/yolo/annotated", lambda msg: self._annotated_img_cb("left", msg), 2)
        self._parent_node.create_subscription(Image, "/center_camera/yolo/annotated", lambda msg: self._annotated_img_cb("center", msg), 2)
        self._parent_node.create_subscription(Image, "/right_camera/yolo/annotated", lambda msg: self._annotated_img_cb("right", msg), 2)
        
        self._corner_match_pubs = {
            "left": self._parent_node.create_publisher(Image, "/left_camera/corner_match", 10),
            "center": self._parent_node.create_publisher(Image, "/center_camera/corner_match", 10),
            "right": self._parent_node.create_publisher(Image, "/right_camera/corner_match", 10),
        }

        self.get_logger().info("mypolicy.__init__()")
        self.get_logger().info("Started internal CombinedYoloDepthPosePlanner, detection listener, and motion servo.")

    def _annotated_img_cb(self, cam_name: str, msg: Image):
        self._annotated_images[cam_name] = msg

    def _camera_image_size(self, camera_name: str) -> Optional[tuple[int, int]]:
        msg = self._annotated_images.get(camera_name)
        if msg is not None:
            try:
                width = int(getattr(msg, "width", 0))
                height = int(getattr(msg, "height", 0))
                if width > 0 and height > 0:
                    return width, height
            except Exception:
                pass

        try:
            with self._detection_node._lock:
                info = self._detection_node._latest_infos.get(camera_name)
            if info is not None:
                width = int(getattr(info, "width", 0))
                height = int(getattr(info, "height", 0))
                if width > 0 and height > 0:
                    return width, height
        except Exception:
            pass
        return None

    def _uv_inside_image(self, camera_name: str, uv: Optional[np.ndarray], margin_px: Optional[float] = None) -> bool:
        if uv is None:
            return False
        size = self._camera_image_size(camera_name)
        if size is None:
            return True
        width, height = size
        margin = float(self.SAMPLE_VERIFY_IMAGE_MARGIN_PX if margin_px is None else margin_px)
        u = float(uv[0])
        v = float(uv[1])
        return (-margin <= u <= float(width) + margin) and (-margin <= v <= float(height) + margin)

    def _detection_intersects_image(self, camera_name: str, det: Optional[Dict], margin_px: Optional[float] = None) -> bool:
        if det is None:
            return False
        size = self._camera_image_size(camera_name)
        if size is None:
            return True
        width, height = size
        margin = float(self.SAMPLE_VERIFY_IMAGE_MARGIN_PX if margin_px is None else margin_px)

        bbox = det.get("bbox_xyxy", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                return not (x2 < -margin or x1 > float(width) + margin or y2 < -margin or y1 > float(height) + margin)
            except Exception:
                pass

        return self._uv_inside_image(camera_name, self._feature_uv(det), margin_px=margin)

    def _goal_feature_visible_cameras(self, matcher: Callable[[Dict], bool], freshness_sec: Optional[float] = None) -> List[str]:
        freshness = self.SAMPLE_VERIFY_GOAL_VISIBILITY_FRESHNESS_SEC if freshness_sec is None else float(freshness_sec)
        visible_cameras: List[str] = []

        for cam in ("left", "center", "right"):
            cam_dets = self._detection_listener.get_camera_detections(cam, freshness_sec=freshness)
            goal_dets = [
                d for d in cam_dets
                if matcher(d) and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE
            ]
            goal_dets.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
            if any(self._detection_intersects_image(cam, det) for det in goal_dets):
                visible_cameras.append(cam)

        return visible_cameras

    # ==================================================================
    # Main entry point — insert_cable
    # ==================================================================

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        send_feedback(f"mypolicy/start task={task.id}")

        observation = self._wait_for_observation(get_observation=get_observation, timeout_sec=5.0)
        if observation is None:
            send_feedback("mypolicy/fail no_observation")
            return False

        self._motion_servo.ensure_cartesian_mode()

        # Capture the current gripper orientation once — the robot is already
        # pointing downward in its home pose, so we reuse this for all targets.
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

        # Pick the right port detection class based on the task
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

        # ====== PHASE 1: Coarse approach via YOLO — go directly to port ======
        send_feedback("mypolicy/phase1_coarse_approach")

        # Detect specific target port
        send_feedback(f"mypolicy/search_{port_label}")
        port_result = self._wait_for_detection(
            matcher=port_matcher,
            timeout_sec=12.0,
            preferred_camera=None,
            min_update_time=0.0,
        )
        if port_result is None:
            # Fallback: try generic sfp_port matcher
            send_feedback(f"mypolicy/{port_label}_not_found, fallback to generic sfp_port")
            port_result = self._wait_for_detection(
                matcher=self._is_sfp_port_detection,
                timeout_sec=8.0,
                preferred_camera=None,
                min_update_time=0.0,
            )
            port_label = "sfp_port_fallback"

        if port_result is None:
            send_feedback("mypolicy/fail port_not_found")
            return False

        port_pose_raw = self._pose_from_detection(port_result["detection"])
        if port_pose_raw is None:
            send_feedback("mypolicy/fail port_pose_missing")
            return False

        send_feedback(
            f"mypolicy/port_detected class={port_result['detection'].get('class_name', '')} "
            f"conf={port_result['confidence']:.3f} xyz=({port_pose_raw.position.x:.3f},{port_pose_raw.position.y:.3f},{port_pose_raw.position.z:.3f})"
        )

        # Sample position accurately for 15s, plan static path, and monitor pixel alignment
        send_feedback("mypolicy/phase1_sample_and_verify")
        if not self._execute_sample_and_verify_goal(
            label="hover_above_port",
            matcher=port_matcher,
            gripper_orientation=gripper_orientation,
            z_offset=0.0,
            min_z=0.0,
            get_observation=get_observation,
            send_feedback=send_feedback,
            timeout_sec=60.0,
        ):
            send_feedback("mypolicy/fail hover_pose_failed")
            return False

        self._motion_servo.stop()
        send_feedback("mypolicy/phase1_path_complete")
        self.sleep_for(self.PATH_SETTLE_SEC)

        # ====== PHASE 2: Initial pixel correction ======
        send_feedback(f"mypolicy/pixel_correction_enabled={self.PIXEL_CORRECTION_ENABLED}")
        if self.PIXEL_CORRECTION_ENABLED:
            self._initial_prev_xy = np.zeros(2, dtype=np.float64)
            send_feedback("mypolicy/phase2_initial_pixel_correction_start")
            servo_ok = self._sfp_visual_servo_align(
                port_matcher=port_matcher,
                port_label=port_label,
                gripper_orientation=gripper_orientation,
                send_feedback=send_feedback,
            )
            if not servo_ok:
                send_feedback("mypolicy/phase2_initial_pixel_correction_failed")
                self._motion_servo.stop()
                return False

            self.sleep_for(0.5)

            send_feedback(
                f"mypolicy/phase2b_module_pca_pitch_start axis_local={self._phase2_rotation_axis_local} target_ground={self._phase2_pca_target_ground_direction}"
            )
            pitch_ok = self._align_sfp_module_pca_perpendicular_to_ground_with_rviz(
                send_feedback=send_feedback,
            )
            if not pitch_ok:
                send_feedback("mypolicy/phase2b_module_pca_pitch_failed")
                self._motion_servo.stop()
                return False

            self.sleep_for(0.5)
        else:
            send_feedback("mypolicy/phase2_pixel_correction_disabled")

        # ====== PHASE 3: Force-feedback insertion ======
        send_feedback("mypolicy/phase3_force_insertion_start")
        insert_ok = self._sfp_force_insert(
            gripper_orientation=gripper_orientation,
            get_observation=get_observation,
            move_robot=move_robot,
            send_feedback=send_feedback,
        )

        self._motion_servo.stop()

        if insert_ok:
            send_feedback("mypolicy/done success=true")
        else:
            send_feedback("mypolicy/done success=false (insertion uncertain)")
        return insert_ok

    # ==================================================================
    # Phase 2: Visual Servoing
    # ==================================================================

    def _sfp_visual_servo_align(
        self,
        port_matcher: Callable[[Dict], bool],
        port_label: str,
        gripper_orientation: Quaternion,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        del port_label, gripper_orientation
        deadline = time.monotonic() + 15.0
        best_metric_error = float("inf")
        best_center_error = float("inf")
        stable_count = 0
        last_valid_pairs: Dict[str, Dict] = {}
        goal_lost_all_count = 0
        self._initial_prev_xy[:] = 0.0

        for iteration in range(self.SERVO_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                success = best_center_error <= self.SERVO_CONVERGED_PX or best_metric_error <= (self.SERVO_CONVERGED_PX + 8.0)
                send_feedback(
                    f"mypolicy/initial_pixel_correction_timeout best_px_err={best_metric_error:.1f} best_center_px_err={best_center_error:.1f}"
                )
                self._motion_servo.stop()
                return success

            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            visible_goal_cameras = self._goal_feature_visible_cameras(port_matcher, freshness_sec=0.80)
            if visible_goal_cameras:
                goal_lost_all_count = 0
            else:
                goal_lost_all_count += 1
                self._motion_servo.stop()
                send_feedback(
                    f"mypolicy/initial_pixel_correction_goal_out_of_all_cameras lost_count={goal_lost_all_count}/{self.SAMPLE_VERIFY_GOAL_LOST_MAX_COUNT}"
                )
                if goal_lost_all_count >= self.SAMPLE_VERIFY_GOAL_LOST_MAX_COUNT:
                    return False
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            per_cam = {}
            now = time.monotonic()
            for cam in ("left", "center", "right"):
                cam_dets = self._detection_listener.get_camera_detections(cam, freshness_sec=0.80)
                port_det = self._best_cam_detection(cam_dets, port_matcher)
                module_det = self._best_cam_detection(cam_dets, self._is_sfp_module_detection)
                if port_det is not None and module_det is not None:
                    last_valid_pairs[cam] = {"time": now, "port": dict(port_det), "module": dict(module_det)}
                else:
                    cached = last_valid_pairs.get(cam)
                    if cached is not None and now - float(cached.get("time", 0.0)) <= self.SERVO_LAST_VALID_PAIR_SEC:
                        if port_det is None:
                            port_det = dict(cached["port"])
                        if module_det is None:
                            module_det = dict(cached["module"])
                self._publish_servo_debug_image(cam, cam_dets, port_det, module_det)
                if port_det is None or module_det is None:
                    continue
                J_cam, e_cam, px_err = self._compute_center_ibvs_camera_system(
                    camera_name=cam,
                    current_pose=current_pose,
                    port_det=port_det,
                    plug_det=module_det,
                )
                if J_cam is None or e_cam is None or J_cam.shape[0] < 2:
                    continue
                per_cam[cam] = {
                    "J": J_cam,
                    "e": e_cam.reshape(-1, 1),
                    "err": float(px_err),
                    "weight": max(0.35, min(1.0, math.sqrt(max(1e-6, float(port_det.get("confidence", 0.0)) * float(module_det.get("confidence", 0.0)))))),
                }

            if not per_cam:
                self._motion_servo.stop()
                send_feedback(f"mypolicy/initial_pixel_correction_iter_{iteration} waiting_for_valid_port_and_module")
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            # Prefer center camera alignment of module center to port center.
            use_cams = ["center"] if "center" in per_cam else list(per_cam.keys())
            J_rows = []
            e_rows = []
            per_cam_errors = []
            for cam in use_cams:
                item = per_cam[cam]
                J_rows.append(item["weight"] * item["J"])
                e_rows.append(item["weight"] * item["e"])
                per_cam_errors.append(item["err"])
            J = np.vstack(J_rows)
            e = np.vstack(e_rows)
            H = J.T @ J + (self.SERVO_DLS_DAMPING ** 2) * np.eye(3, dtype=np.float64)
            g = J.T @ e
            try:
                v_cmd = -np.linalg.solve(H, g).reshape(3)
            except np.linalg.LinAlgError:
                v_cmd = -(np.linalg.pinv(J) @ e).reshape(3)

            vx = float(self.SERVO_GAIN_XY * v_cmd[0])
            vy = float(self.SERVO_GAIN_XY * v_cmd[1])
            vz_raw = float(self.SERVO_GAIN_Z * v_cmd[2])
            v_xy = np.asarray([vx, vy], dtype=np.float64)
            metric_px_error = float(max(per_cam_errors))
            avg_px_error = float(sum(per_cam_errors) / len(per_cam_errors))
            best_metric_error = min(best_metric_error, metric_px_error)
            if "center" in per_cam:
                best_center_error = min(best_center_error, float(per_cam["center"]["err"]))

            speed_xy = float(np.linalg.norm(v_xy))
            if speed_xy > self.SERVO_MAX_LINEAR_SPEED_XY:
                v_xy *= self.SERVO_MAX_LINEAR_SPEED_XY / max(speed_xy, 1e-9)
                speed_xy = float(np.linalg.norm(v_xy))
            if metric_px_error > 40.0 and speed_xy > 1e-9 and speed_xy < self.SERVO_MIN_LINEAR_SPEED_XY:
                v_xy *= self.SERVO_MIN_LINEAR_SPEED_XY / max(speed_xy, 1e-9)

            v_xy = self.SERVO_CMD_SMOOTHING * self._initial_prev_xy + (1.0 - self.SERVO_CMD_SMOOTHING) * v_xy
            self._initial_prev_xy = v_xy.copy()
            vz = float(np.clip(vz_raw, -self.SERVO_MAX_LINEAR_SPEED_Z, self.SERVO_MAX_LINEAR_SPEED_Z)) if metric_px_error <= self.SERVO_Z_ENABLE_PX else 0.0

            conv_err = float(per_cam["center"]["err"]) if "center" in per_cam else metric_px_error
            if conv_err <= self.SERVO_CONVERGED_PX:
                stable_count += 1
            else:
                stable_count = 0

            send_feedback(
                f"mypolicy/initial_pixel_correction_iter_{iteration} cams={use_cams} px_err_max={metric_px_error:.1f} px_err_avg={avg_px_error:.1f} center_px_err={(float(per_cam['center']['err']) if 'center' in per_cam else -1.0):.1f} vx={v_xy[0]:.4f} vy={v_xy[1]:.4f} vz={vz:.4f}"
            )

            if stable_count >= self.SERVO_CONVERGED_PX_STABLE_COUNT:
                self._motion_servo.stop()
                send_feedback(f"mypolicy/initial_pixel_correction_converged px_err={conv_err:.1f}")
                return True

            twist = Twist()
            twist.linear.x = float(v_xy[0])
            twist.linear.y = float(v_xy[1])
            twist.linear.z = float(vz)
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self._motion_servo.publish_twist_command(twist)
            self.sleep_for(self.SERVO_STEP_SEC)

        self._motion_servo.stop()
        send_feedback(f"mypolicy/initial_pixel_correction_max_iterations best_px_err={best_metric_error:.1f} best_center_px_err={best_center_error:.1f}")
        return best_center_error <= self.SERVO_CONVERGED_PX or best_metric_error <= (self.SERVO_CONVERGED_PX + 8.0)

    def _is_sfp_module_detection(self, det: Dict) -> bool:
        if self._matches_any_name(det, self._sfp_module_classes):
            return True
        for key in ("class_name", "raw_class_name", "base_class_name", "instance_name"):
            norm = self._norm_name(det.get(key, ""))
            if not norm:
                continue
            if ("sfp" in norm and "module" in norm) or ("transceiver" in norm):
                return True
        return False

    def _parse_points_uv(self, raw) -> Optional[np.ndarray]:
        if raw is None:
            return None
        pts = None
        if isinstance(raw, list) and len(raw) >= 2:
            if isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 2:
                try:
                    pts = np.asarray([[float(p[0]), float(p[1])] for p in raw], dtype=np.float64)
                except Exception:
                    pts = None
            elif len(raw) % 2 == 0:
                try:
                    arr = np.asarray([float(v) for v in raw], dtype=np.float64).reshape(-1, 2)
                    pts = arr
                except Exception:
                    pts = None
        return pts if pts is not None and len(pts) >= 2 else None

    def _det_shape_points_uv(self, det: Dict) -> Optional[np.ndarray]:
        pts = self._parse_points_uv(det.get("mask_polygon_uv"))
        if pts is not None and len(pts) >= 4:
            return pts
        pts = self._parse_points_uv(det.get("obb_corners_uv"))
        if pts is not None and len(pts) >= 4:
            return pts
        bbox = det.get("bbox_xyxy", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        return None

    def _pca_axis_from_points(self, pts: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if pts is None or len(pts) < 2:
            return None
        center = pts.mean(axis=0)
        centered = pts - center
        cov = centered.T @ centered / max(len(pts), 1)
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        axis = vecs[:, int(np.argmax(vals))]
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        proj = centered @ axis
        end_a = center + axis * float(np.min(proj))
        end_b = center + axis * float(np.max(proj))
        return center, axis, end_a, end_b

    def _module_free_end_uv(self, module_det: Dict, gripper_det: Optional[Dict]) -> Optional[Dict]:
        pts = self._det_shape_points_uv(module_det)
        pca = self._pca_axis_from_points(pts)
        if pca is None:
            return None
        center, axis, end_a, end_b = pca
        grip_uv = self._feature_uv(gripper_det) if gripper_det is not None else None
        if grip_uv is None:
            grip_uv = self._bbox_center_uv(module_det)
        if grip_uv is None:
            return None
        da = float(np.linalg.norm(end_a - grip_uv))
        db = float(np.linalg.norm(end_b - grip_uv))
        free_end = end_a if da >= db else end_b
        hold_end = end_b if da >= db else end_a
        return {
            "center": center,
            "axis": axis,
            "free_end": free_end,
            "hold_end": hold_end,
        }

    def _best_cam_detection(self, cam_dets: List[Dict], matcher: Callable[[Dict], bool]) -> Optional[Dict]:
        cands = [d for d in cam_dets if matcher(d) and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE]
        return max(cands, key=lambda d: float(d.get("confidence", 0.0))) if cands else None

    def _port_target_end_uv(self, port_det: Dict, module_free_end_uv: Optional[np.ndarray]) -> Optional[Dict]:
        pts = self._det_shape_points_uv(port_det)
        pca = self._pca_axis_from_points(pts)
        if pca is None:
            return None
        center, axis, end_a, end_b = pca
        ref_uv = module_free_end_uv if module_free_end_uv is not None else center
        da = float(np.linalg.norm(end_a - ref_uv))
        db = float(np.linalg.norm(end_b - ref_uv))
        target_end = end_a if da <= db else end_b
        other_end = end_b if da <= db else end_a
        return {
            "center": center,
            "axis": axis,
            "target_end": target_end,
            "other_end": other_end,
        }

    def _publish_edge_align_debug_image(
        self,
        cam: str,
        cam_dets: List[Dict],
        port_det: Optional[Dict],
        module_det: Optional[Dict],
        gripper_det: Optional[Dict],
        free_end_uv: Optional[np.ndarray],
        port_end_uv: Optional[np.ndarray],
    ) -> None:
        if cam not in self._annotated_images:
            return
        try:
            cv_img = self._cv_bridge.imgmsg_to_cv2(self._annotated_images[cam], "bgr8")
            for det, color in ((port_det, (255, 0, 0)), (module_det, (0, 255, 255)), (gripper_det, (0, 0, 255))):
                if det is None:
                    continue
                bbox = det.get("bbox_xyxy", [])
                if isinstance(bbox, list) and len(bbox) == 4:
                    cv2.rectangle(cv_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            if module_det is not None:
                info = self._module_free_end_uv(module_det, gripper_det)
                if info is not None:
                    c = tuple(np.round(info["center"]).astype(int))
                    e1 = tuple(np.round(info["hold_end"]).astype(int))
                    e2 = tuple(np.round(info["free_end"]).astype(int))
                    cv2.line(cv_img, e1, e2, (0, 255, 0), 2)
                    cv2.circle(cv_img, c, 4, (255, 255, 255), -1)
                    cv2.circle(cv_img, e2, 6, (0, 255, 0), -1)
            if port_det is not None:
                pinfo = self._port_target_end_uv(port_det, free_end_uv)
                if pinfo is not None:
                    e1 = tuple(np.round(pinfo["other_end"]).astype(int))
                    e2 = tuple(np.round(pinfo["target_end"]).astype(int))
                    cv2.line(cv_img, e1, e2, (255, 0, 0), 2)
                    cv2.circle(cv_img, e2, 6, (255, 0, 0), -1)
            if free_end_uv is not None and port_end_uv is not None:
                cv2.line(cv_img, tuple(np.round(free_end_uv).astype(int)), tuple(np.round(port_end_uv).astype(int)), (0, 255, 255), 2)
            msg_out = self._cv_bridge.cv2_to_imgmsg(cv_img, "bgr8")
            msg_out.header = self._annotated_images[cam].header
            self._corner_match_pubs[cam].publish(msg_out)
        except Exception:
            return

    def _sfp_module_edge_pitch_align(
        self,
        port_matcher: Callable[[Dict], bool],
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        deadline = time.monotonic() + self.EDGE_ALIGN_TIMEOUT_SEC
        stable_count = 0
        best_px_error = float("inf")
        best_axial_error = float("inf")
        self._edge_prev_pitch = 0.0
        pitch_sign = float(self._edge_pitch_sign)
        no_improve_count = 0

        for iteration in range(self.SERVO_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                self._motion_servo.stop()
                send_feedback(
                    f"mypolicy/module_edge_pitch_timeout best_px_err={best_px_error:.1f} best_axial_err={best_axial_error:.1f}"
                )
                return best_px_error <= self.EDGE_ALIGN_CONVERGED_PX and best_axial_error <= self.EDGE_ALIGN_AXIAL_CONVERGED_PX

            used_cameras = []
            per_cam_errors = []
            axial_errors = []

            for cam in ("left", "right"):
                cam_dets = self._detection_listener.get_camera_detections(cam, freshness_sec=0.80)
                port_det = self._best_cam_detection(cam_dets, port_matcher)
                module_det = self._best_cam_detection(cam_dets, self._is_sfp_module_detection)
                gripper_det = self._best_cam_detection(cam_dets, self._is_gripper_detection)
                free_end_info = self._module_free_end_uv(module_det, gripper_det) if module_det is not None else None
                port_end_info = self._port_target_end_uv(port_det, None if free_end_info is None else free_end_info["free_end"]) if port_det is not None else None
                free_end_uv = None if free_end_info is None else free_end_info["free_end"]
                port_end_uv = None if port_end_info is None else port_end_info["target_end"]
                self._publish_edge_align_debug_image(cam, cam_dets, port_det, module_det, gripper_det, free_end_uv, port_end_uv)
                if free_end_info is None or port_end_info is None:
                    continue

                err_vec = np.asarray(port_end_info["target_end"] - free_end_info["free_end"], dtype=np.float64)
                err_norm = float(np.linalg.norm(err_vec))
                axis = np.asarray(free_end_info["axis"], dtype=np.float64)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm < 1e-9:
                    continue
                axis = axis / axis_norm
                axial_err = float(np.dot(err_vec, axis))

                used_cameras.append(cam)
                per_cam_errors.append(err_norm)
                axial_errors.append(axial_err)

            if len(used_cameras) < 2:
                self._motion_servo.stop()
                send_feedback(f"mypolicy/module_edge_pitch_iter_{iteration} waiting_for_module_and_port_left_right")
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            metric_px_error = float(max(per_cam_errors))
            avg_px_error = float(sum(per_cam_errors) / len(per_cam_errors))
            avg_axial_err = float(sum(axial_errors) / len(axial_errors))
            abs_axial = abs(avg_axial_err)

            improved = False
            if metric_px_error < best_px_error - 1.5:
                best_px_error = metric_px_error
                improved = True
            if abs_axial < best_axial_error - 1.5:
                best_axial_error = abs_axial
                improved = True

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.EDGE_ALIGN_SIGN_FLIP_PATIENCE:
                pitch_sign *= -1.0
                self._edge_prev_pitch = 0.0
                no_improve_count = 0
                send_feedback(f"mypolicy/module_edge_pitch_flip_sign new_sign={pitch_sign:+.1f}")

            if abs_axial > 25.0:
                target_pitch_rate = self.EDGE_ALIGN_PITCH_RATE_COARSE
            elif abs_axial > self.EDGE_ALIGN_AXIAL_CONVERGED_PX:
                target_pitch_rate = self.EDGE_ALIGN_PITCH_RATE_FINE
            else:
                target_pitch_rate = 0.0

            if avg_axial_err < 0.0:
                pitch_dir = 1.0
            elif avg_axial_err > 0.0:
                pitch_dir = -1.0
            else:
                pitch_dir = 0.0

            pitch_cmd = float(np.clip(pitch_sign * pitch_dir * target_pitch_rate, -self.EDGE_ALIGN_MAX_PITCH_RATE, self.EDGE_ALIGN_MAX_PITCH_RATE))
            self._edge_prev_pitch = self.EDGE_ALIGN_SMOOTHING * self._edge_prev_pitch + (1.0 - self.EDGE_ALIGN_SMOOTHING) * pitch_cmd

            if metric_px_error <= self.EDGE_ALIGN_CONVERGED_PX and abs_axial <= self.EDGE_ALIGN_AXIAL_CONVERGED_PX:
                stable_count += 1
            else:
                stable_count = 0

            send_feedback(
                f"mypolicy/module_edge_pitch_iter_{iteration} cams={used_cameras} px_err_max={metric_px_error:.1f} "
                f"px_err_avg={avg_px_error:.1f} axial_err={avg_axial_err:.1f} "
                f"pitch_y={self._edge_prev_pitch:.5f} sign={pitch_sign:+.1f} mode=velocity_local_pitch frame=gripper/tcp"
            )

            if stable_count >= self.EDGE_ALIGN_STABLE_COUNT:
                self._motion_servo.stop()
                send_feedback(f"mypolicy/module_edge_pitch_converged px_err={metric_px_error:.1f} axial_err={avg_axial_err:.1f}")
                return True

            self._publish_local_pitch_velocity(self._edge_prev_pitch)
            self.sleep_for(self.SERVO_STEP_SEC)

        self._motion_servo.stop()
        send_feedback(f"mypolicy/module_edge_pitch_max_iterations best_px_err={best_px_error:.1f} best_axial_err={best_axial_error:.1f}")
        return best_px_error <= self.EDGE_ALIGN_CONVERGED_PX and best_axial_error <= self.EDGE_ALIGN_AXIAL_CONVERGED_PX


    def _phase2_axis_vector_local(self) -> np.ndarray:
        axis_name = str(self._phase2_rotation_axis_local).strip().lower()
        if axis_name == "x":
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if axis_name == "y":
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if axis_name == "z":
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.get_logger().warn(f"Unknown MYPOLICY_PHASE2_ROTATION_AXIS_LOCAL={axis_name!r}; falling back to 'x'.")
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def _phase2_target_ground_direction_base(self) -> np.ndarray:
        name = str(self._phase2_pca_target_ground_direction).strip().lower()
        if name in ("up", "+z", "upward"):
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def _det_pose_position_np(self, det: Dict) -> Optional[np.ndarray]:
        pose = det.get("pose_base_link")
        if not isinstance(pose, dict):
            return None
        pos = pose.get("position")
        if not isinstance(pos, dict):
            return None
        try:
            return np.array([
                float(pos.get("x", 0.0)),
                float(pos.get("y", 0.0)),
                float(pos.get("z", 0.0)),
            ], dtype=np.float64)
        except Exception:
            return None

    def _det_pose_orientation_quaternion(self, det: Dict) -> Optional[Quaternion]:
        pose = det.get("pose_base_link")
        if not isinstance(pose, dict):
            return None
        ori = pose.get("orientation")
        if not isinstance(ori, dict):
            return None
        try:
            return Quaternion(
                x=float(ori.get("x", 0.0)),
                y=float(ori.get("y", 0.0)),
                z=float(ori.get("z", 0.0)),
                w=float(ori.get("w", 1.0)),
            )
        except Exception:
            return None

    def _module_pca_axis_from_detection_base(self, det: Dict) -> Optional[np.ndarray]:
        q = self._det_pose_orientation_quaternion(det)
        if q is None:
            return None
        axis = _rotate_vector_by_quaternion(_quat_to_np(q), np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1e-9:
            return None
        return _normalize_vec(axis)

    def _det_camera_name(self, det: Dict) -> Optional[str]:
        for key in ("camera_name", "source"):
            name = str(det.get(key, "")).strip().lower()
            if name in {"left", "center", "right"}:
                return name
        return None

    def _pose_base_link_dict_from_pose_dict(self, pose_base: Dict) -> Optional[Dict]:
        try:
            t = np.asarray(pose_base.get("t"), dtype=np.float64).reshape(3)
            q = np.asarray(pose_base.get("q"), dtype=np.float64).reshape(4)
        except Exception:
            return None
        return {
            "position": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "orientation": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
        }

    def _estimate_module_pose_from_camera_depth(self, det: Dict, camera_name: str) -> Optional[Dict]:
        try:
            with self._detection_node._lock:
                info = self._detection_node._latest_infos.get(camera_name)
            if info is None:
                return None
            depth = self._detection_node._get_latest_depth_copy(camera_name)
            if depth is None:
                return None
            raw_pose = self._detection_node._cad_estimator.estimate_pose(
                info,
                depth,
                det,
                "sfp_module",
                init_pose_camera=None,
            )
            if raw_pose is None:
                return None
            frame_id = str(info.header.frame_id)
            pose_base = self._detection_node._transform_pose_dict(raw_pose, frame_id, "base_link")
            if pose_base is None:
                return None
            pose_base_link = self._pose_base_link_dict_from_pose_dict(pose_base)
            if pose_base_link is None:
                return None
            out = dict(det)
            out["camera_name"] = camera_name
            out["pose_base_link"] = pose_base_link
            out["pose_source"] = str(raw_pose.get("source", "depth_pca"))
            return out
        except Exception:
            return None

    def _find_active_sfp_module_detection_from_direct_depth(self, current_pose: Pose) -> Optional[Dict]:
        tcp_pos = np.array([
            float(current_pose.position.x),
            float(current_pose.position.y),
            float(current_pose.position.z),
        ], dtype=np.float64)
        candidates = []
        for camera_name in ("center", "left", "right"):
            cam_dets = self._detection_listener.get_camera_detections(camera_name, freshness_sec=self._phase2_module_pose_freshness_sec)
            for det in cam_dets:
                if not self._is_sfp_module_detection(det):
                    continue
                est = self._estimate_module_pose_from_camera_depth(det, camera_name)
                if est is None:
                    continue
                pos = self._det_pose_position_np(est)
                axis = self._module_pca_axis_from_detection_base(est)
                if pos is None or axis is None:
                    continue
                dist = float(np.linalg.norm(pos - tcp_pos))
                conf = float(est.get("confidence", det.get("confidence", 0.0)))
                pose_source = str(est.get("pose_source", ""))
                source_rank = 2 if pose_source == "depth_icp" else (1 if pose_source == "depth_pca" else 0)
                candidates.append((dist, -source_rank, -conf, est))
        if not candidates:
            return None
        nearby = [item for item in candidates if item[0] <= self._phase2_module_max_dist_to_tcp_m]
        nearby.sort(key=lambda item: (item[0], item[1], item[2]))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return dict((nearby[0] if nearby else candidates[0])[3])

    def _find_active_sfp_module_detection(self, current_pose: Pose) -> Optional[Dict]:
        tcp_pos = np.array([
            float(current_pose.position.x),
            float(current_pose.position.y),
            float(current_pose.position.z),
        ], dtype=np.float64)
        dets = self._detection_listener.get_all_detections(freshness_sec=self._phase2_module_pose_freshness_sec)
        scored = []
        for det in dets:
            if not self._is_sfp_module_detection(det):
                continue
            pos = self._det_pose_position_np(det)
            axis = self._module_pca_axis_from_detection_base(det)
            if pos is None or axis is None:
                continue
            dist = float(np.linalg.norm(pos - tcp_pos))
            conf = float(det.get("confidence", 0.0))
            pose_source = str(det.get("pose_source", ""))
            source_rank = 2 if pose_source == "depth_icp" else (1 if pose_source == "depth_pca" else 0)
            scored.append((dist, -source_rank, -conf, det))
        if not scored:
            return None
        nearby = [item for item in scored if item[0] <= self._phase2_module_max_dist_to_tcp_m]
        nearby.sort(key=lambda item: (item[0], item[1], item[2]))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return dict((nearby[0] if nearby else scored[0])[3])

    def _compute_module_pca_pitch_target(self, current_pose: Pose, module_det: Dict) -> Optional[Dict]:
        axis_local = self._phase2_axis_vector_local()
        control_axis_base = _rotate_vector_by_quaternion(_quat_to_np(current_pose.orientation), axis_local)
        if float(np.linalg.norm(control_axis_base)) < 1e-9:
            return None
        control_axis_base = _normalize_vec(control_axis_base)

        raw_axis = self._module_pca_axis_from_detection_base(module_det)
        module_pos = self._det_pose_position_np(module_det)
        if raw_axis is None or module_pos is None:
            return None

        tcp_pos = np.array([
            float(current_pose.position.x),
            float(current_pose.position.y),
            float(current_pose.position.z),
        ], dtype=np.float64)
        tcp_to_module = module_pos - tcp_pos
        if float(np.linalg.norm(tcp_to_module)) > 1e-9:
            tcp_to_module_dir = _normalize_vec(tcp_to_module)
        else:
            tcp_to_module_dir = np.zeros(3, dtype=np.float64)

        desired_base = _normalize_vec(self._phase2_target_ground_direction_base())

        # Choose the PCA-axis sign using the TCP->module direction first, then refine by residual pitch error.
        if float(np.linalg.norm(tcp_to_module_dir)) > 1e-9:
            if float(np.dot(raw_axis, tcp_to_module_dir)) < 0.0:
                signed_axis = -raw_axis
            else:
                signed_axis = raw_axis
            axis_sign_source = "tcp_to_module"
        else:
            signed_axis = raw_axis
            axis_sign_source = "default_raw"

        raw_proj = _project_to_plane(signed_axis, control_axis_base)
        flip_proj = _project_to_plane(-signed_axis, control_axis_base)
        desired_proj = _project_to_plane(desired_base, control_axis_base)
        if float(np.linalg.norm(raw_proj)) < 1e-9 or float(np.linalg.norm(desired_proj)) < 1e-9:
            return None

        err_signed = _signed_angle_about_axis_rad(raw_proj, desired_proj, control_axis_base)
        err_flipped = _signed_angle_about_axis_rad(flip_proj, desired_proj, control_axis_base)
        if abs(err_flipped) + 1e-9 < abs(err_signed):
            chosen_axis = -signed_axis
            angle_cmd_rad = float(err_flipped)
            axis_sign_source = axis_sign_source + "+flip_min_error"
        else:
            chosen_axis = signed_axis
            angle_cmd_rad = float(err_signed)

        max_abs_rad = math.radians(max(1.0, float(self._phase2_pca_max_abs_deg)))
        angle_cmd_rad = float(np.clip(angle_cmd_rad, -max_abs_rad, max_abs_rad))
        vertical_alignment_deg = math.degrees(
            math.acos(float(np.clip(abs(np.dot(chosen_axis, desired_base)), 0.0, 1.0)))
        )
        ground_plane_angle_deg = 90.0 - vertical_alignment_deg

        return {
            "angle_cmd_rad": angle_cmd_rad,
            "angle_cmd_deg": math.degrees(angle_cmd_rad),
            "module_axis_base": chosen_axis,
            "control_axis_base": control_axis_base,
            "ground_plane_angle_deg": float(ground_plane_angle_deg),
            "vertical_error_deg": float(vertical_alignment_deg),
            "pose_source": str(module_det.get("pose_source", "")),
            "instance_name": str(module_det.get("instance_name", module_det.get("class_name", "sfp_module"))),
            "confidence": float(module_det.get("confidence", 0.0)),
            "distance_to_tcp_m": float(np.linalg.norm(module_pos - tcp_pos)),
            "axis_sign_source": axis_sign_source,
            "tcp_to_module_dir": tcp_to_module_dir,
        }

    def _build_local_axis_rotated_pose(self, source_pose: Pose, axis_local: np.ndarray, angle_rad: float) -> Pose:
        q_current = _quat_normalize(_quat_to_np(source_pose.orientation))
        q_delta_local = rviz_quat_from_axis_angle(axis_local.tolist(), float(angle_rad))
        q_target = rviz_quat_multiply(q_current, q_delta_local)
        target_pose = _copy_pose(source_pose)
        target_pose.orientation = rviz_quat_to_msg(q_target)
        return target_pose

    def _align_sfp_module_pca_perpendicular_to_ground_with_rviz(
        self,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        current_pose = self._motion_servo.get_current_pose()
        if current_pose is None:
            send_feedback("mypolicy/module_pca_pitch_no_current_pose")
            return False

        module_det = self._find_active_sfp_module_detection_from_direct_depth(current_pose)
        if module_det is None:
            module_det = self._find_active_sfp_module_detection(current_pose)
        if module_det is None:
            if self._phase2_allow_fixed_fallback:
                send_feedback(
                    f"mypolicy/module_pca_pitch_no_detection fallback_fixed_deg={self._phase2_fixed_fallback_deg:.1f}"
                )
                axis_local = self._phase2_axis_vector_local()
                angle_rad = math.radians(float(self._phase2_fixed_fallback_deg))
                target_info = {
                    "angle_cmd_rad": angle_rad,
                    "angle_cmd_deg": math.degrees(angle_rad),
                    "module_axis_base": np.zeros(3, dtype=np.float64),
                    "control_axis_base": _rotate_vector_by_quaternion(_quat_to_np(current_pose.orientation), axis_local),
                    "ground_plane_angle_deg": float("nan"),
                    "vertical_error_deg": float("nan"),
                    "pose_source": "fallback_fixed",
                    "instance_name": "fallback_fixed",
                    "confidence": 0.0,
                    "distance_to_tcp_m": 0.0,
                    "axis_sign_source": "fallback_fixed",
                    "tcp_to_module_dir": np.zeros(3, dtype=np.float64),
                }
            else:
                send_feedback("mypolicy/module_pca_pitch_no_detection")
                return False
        else:
            target_info = self._compute_module_pca_pitch_target(current_pose, module_det)
            if target_info is None:
                send_feedback("mypolicy/module_pca_pitch_target_failed")
                return False

        axis_local = self._phase2_axis_vector_local()
        target_pose = self._build_local_axis_rotated_pose(
            source_pose=current_pose,
            axis_local=axis_local,
            angle_rad=float(target_info["angle_cmd_rad"]),
        )
        self._motion_servo.publish_target_marker(target_pose)
        self._motion_servo.publish_waypoint_visuals([target_pose])

        sign_src = str(target_info.get("axis_sign_source", "unknown"))
        tcp_to_module_dir = np.asarray(target_info.get("tcp_to_module_dir", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
        send_feedback(
            f"mypolicy/module_pca_selected name={target_info['instance_name']} conf={target_info['confidence']:.3f} "
            f"pose_source={target_info['pose_source']} dist_to_tcp={target_info['distance_to_tcp_m']:.3f}m "
            f"ground_angle={target_info['ground_plane_angle_deg']:.2f}deg vertical_err={target_info['vertical_error_deg']:.2f}deg "
            f"rotate_deg={target_info['angle_cmd_deg']:.2f} axis_local={self._phase2_rotation_axis_local} sign_src={sign_src} "
            f"module_axis_base=({target_info['module_axis_base'][0]:.3f},{target_info['module_axis_base'][1]:.3f},{target_info['module_axis_base'][2]:.3f}) "
            f"tcp_to_module=({tcp_to_module_dir[0]:.3f},{tcp_to_module_dir[1]:.3f},{tcp_to_module_dir[2]:.3f})"
        )

        deadline = time.monotonic() + self.PITCH_GROUND_TIMEOUT_SEC
        while time.monotonic() < deadline:
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            rotvec, ang_error = rviz_quat_error_rotvec(current_pose.orientation, target_pose.orientation)
            if ang_error <= self.PITCH_GROUND_PITCH_TOL_RAD:
                self._motion_servo.stop()
                send_feedback(
                    f"mypolicy/module_pca_pitch_converged axis_local={self._phase2_rotation_axis_local} "
                    f"rotate_deg={target_info['angle_cmd_deg']:.2f} ang_err={math.degrees(ang_error):.2f}deg "
                    f"rotvec=({rotvec[0]:.4f},{rotvec[1]:.4f},{rotvec[2]:.4f})"
                )
                return True

            twist = self._motion_servo.compute_twist_to_waypoint(current_pose, target_pose)
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            send_feedback(
                f"mypolicy/module_pca_pitch_iter axis_local={self._phase2_rotation_axis_local} "
                f"rotate_deg={target_info['angle_cmd_deg']:.2f} ang_err={math.degrees(ang_error):.2f}deg "
                f"cmd_ang=({twist.angular.x:.4f},{twist.angular.y:.4f},{twist.angular.z:.4f}) "
                f"mode=rviz_orientation frame=base_link"
            )

            self._motion_servo.publish_twist_command(
                twist,
                frame_id="base_link",
                trans_stiffness=90.0,
                rot_stiffness=50.0,
                trans_damping=50.0,
                rot_damping=20.0,
            )
            self.sleep_for(self.SERVO_STEP_SEC)

        self._motion_servo.stop()
        send_feedback(
            f"mypolicy/module_pca_pitch_timeout axis_local={self._phase2_rotation_axis_local} rotate_deg={target_info['angle_cmd_deg']:.2f}"
        )
        return False


    def _compute_pixel_correction(
        self,
        du: float,
        dv: float,
        fx: float,
        fy: float,
        z_height: float,
    ) -> tuple:
        """Convert pixel offset (du, dv) into Cartesian correction (dx, dy) in base_link frame.
        
        Uses pinhole camera model: dx = du * Z / fx.
        The camera is pointing down, so camera-X → world-Y, camera-Y → world-X (with sign flip
        depending on camera mounting). We apply conservative gain to avoid overshoot.
        """
        # Raw metric offset in camera frame
        cam_dx = du * z_height / max(fx, 1.0)
        cam_dy = dv * z_height / max(fy, 1.0)

        # Camera is mounted pointing down: camera X-axis ~ base-link Y, camera Y-axis ~ base-link X
        # This mapping depends on the camera mount. For a downward-looking camera with
        # optical frame convention (Z forward = down, X right, Y down):
        world_dx = cam_dx * self.SERVO_GAIN
        world_dy = cam_dy * self.SERVO_GAIN

        # Clamp to prevent large jumps
        max_step = 0.015  # 15mm max correction per iteration
        mag = math.sqrt(world_dx * world_dx + world_dy * world_dy)
        return world_dx, world_dy

    def _feature_uv(self, det: Optional[Dict]) -> Optional[np.ndarray]:
        if det is None:
            return None
        anchor_uv = det.get("anchor_uv", [])
        if isinstance(anchor_uv, list) and len(anchor_uv) == 2:
            try:
                return np.asarray([float(anchor_uv[0]), float(anchor_uv[1])], dtype=np.float64)
            except Exception:
                pass
        return self._bbox_center_uv(det)

    def _bbox_center_uv(self, det: Dict) -> Optional[np.ndarray]:
        bbox = det.get("bbox_xyxy", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)

    def _bbox_size_uv(self, det: Dict) -> Optional[np.ndarray]:
        bbox = det.get("bbox_xyxy", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return np.asarray([abs(x2 - x1), abs(y2 - y1)], dtype=np.float64)

    def _angle_wrap(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _bbox_axis_frame(self, det: Dict) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
        bbox = det.get("bbox_xyxy", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w < 4.0 or h < 4.0:
            return None
        center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
        if w >= h:
            major = np.asarray([1.0, 0.0], dtype=np.float64)
            minor = np.asarray([0.0, 1.0], dtype=np.float64)
            major_half = 0.5 * w
            minor_half = 0.5 * h
        else:
            major = np.asarray([0.0, 1.0], dtype=np.float64)
            minor = np.asarray([1.0, 0.0], dtype=np.float64)
            major_half = 0.5 * h
            minor_half = 0.5 * w
        return center, major, minor, major_half, minor_half

    def _bbox_orientation_angle(self, det: Dict) -> Optional[float]:
        frame = self._bbox_axis_frame(det)
        if frame is None:
            return None
        _, major, _, _, _ = frame
        return math.atan2(float(major[1]), float(major[0]))

    def _axis_angle_error(self, target_angle: float, current_angle: float) -> float:
        err = float(target_angle) - float(current_angle)
        while err > (math.pi * 0.5):
            err -= math.pi
        while err < -(math.pi * 0.5):
            err += math.pi
        return err

    def _estimate_detection_yaw_error(self, port_det: Dict, plug_det: Dict) -> Optional[float]:
        port_angle = self._bbox_orientation_angle(port_det)
        plug_angle = self._bbox_orientation_angle(plug_det)
        if port_angle is None or plug_angle is None:
            return None
        return self._axis_angle_error(port_angle, plug_angle)

    def _bbox_ibvs_feature_points(self, det: Dict, ref_major: Optional[np.ndarray] = None) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, List[np.ndarray]]]:
        frame = self._bbox_axis_frame(det)
        if frame is None:
            return None
        center, major, minor, major_half, minor_half = frame
        if ref_major is not None and float(np.dot(major, ref_major)) < 0.0:
            major = -major
            minor = -minor
        pts = [
            center,
            center + major * major_half,
            center - major * major_half,
            center + minor * minor_half,
            center - minor * minor_half,
        ]
        return center, major, minor, major_half, minor_half, pts

    def _camera_rotation_image_gain(self, camera_name: str) -> float:
        try:
            with self._detection_node._lock:
                info = self._detection_node._latest_infos.get(camera_name)
            if info is None:
                return 1.0
            camera_frame = str(info.header.frame_id)
            R_cam_base, _ = self._detection_node._lookup_transform(camera_frame, "base_link")
            base_z_in_cam = np.asarray(R_cam_base, dtype=np.float64) @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            gain = float(base_z_in_cam[2])
            if abs(gain) < 0.2:
                return 0.2 if gain >= 0.0 else -0.2
            return gain
        except Exception:
            return 1.0


    def _compute_center_ibvs_camera_system(
        self,
        camera_name: str,
        current_pose: Pose,
        port_det: Dict,
        plug_det: Dict,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        J_xyz, _ = self._compute_ibvs_jacobian_xyz(camera_name, current_pose)
        if J_xyz is None:
            return None, None, float("inf")

        port_uv = self._bbox_center_uv(port_det)
        plug_uv = self._bbox_center_uv(plug_det)
        if port_uv is None or plug_uv is None:
            return None, None, float("inf")

        e = np.asarray(plug_uv - port_uv, dtype=np.float64).reshape(2, 1)
        px_err = float(np.linalg.norm(e.reshape(-1)))
        return J_xyz, e, px_err

    def _compute_ibvs_jacobian_xyz(
        self,
        camera_name: str,
        current_pose: Pose,
    ) -> tuple[Optional[np.ndarray], float]:
        try:
            with self._detection_node._lock:
                info = self._detection_node._latest_infos.get(camera_name)
            if info is None:
                return None, 0.0

            camera_frame = str(info.header.frame_id)
            R_cam_base, t_cam_base = self._detection_node._lookup_transform(camera_frame, "base_link")

            p_base = np.asarray([
                float(current_pose.position.x),
                float(current_pose.position.y),
                float(current_pose.position.z),
            ], dtype=np.float64)

            p_cam = np.asarray(R_cam_base, dtype=np.float64) @ p_base + np.asarray(t_cam_base, dtype=np.float64)

            X = float(p_cam[0])
            Y = float(p_cam[1])
            Z = float(p_cam[2])

            if Z <= 1e-4:
                return None, 0.0

            fx = float(info.k[0]) if len(info.k) >= 6 else self.DEFAULT_FX
            fy = float(info.k[4]) if len(info.k) >= 6 else self.DEFAULT_FY

            L_trans = np.asarray([
                [fx / Z, 0.0, -fx * X / (Z * Z)],
                [0.0, fy / Z, -fy * Y / (Z * Z)],
            ], dtype=np.float64)

            G_xyz = np.asarray(R_cam_base, dtype=np.float64)[:, :3]
            J_xyz = L_trans @ G_xyz
            return J_xyz, Z
        except Exception:
            return None, 0.0

    def _compute_pure_ibvs_camera_system(
        self,
        camera_name: str,
        current_pose: Pose,
        port_det: Dict,
        plug_det: Dict,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float, Optional[float]]:
        J_xy, _ = self._compute_ibvs_jacobian_xy(camera_name, current_pose)
        if J_xy is None:
            return None, None, float("inf"), None

        plug_frame = self._bbox_ibvs_feature_points(plug_det)
        port_frame = self._bbox_ibvs_feature_points(port_det)
        if plug_frame is None or port_frame is None:
            return None, None, float("inf"), None

        plug_center, plug_major, plug_minor, _, _, plug_pts = plug_frame
        port_center, port_major, port_minor, _, _, _ = port_frame

        if float(np.dot(port_major, plug_major)) < 0.0:
            port_major = -port_major
            port_minor = -port_minor

        port_pts = [
            port_center,
            port_center + port_major * np.linalg.norm(plug_pts[1] - plug_center),
            port_center - port_major * np.linalg.norm(plug_pts[1] - plug_center),
            port_center + port_minor * np.linalg.norm(plug_pts[3] - plug_center),
            port_center - port_minor * np.linalg.norm(plug_pts[3] - plug_center),
        ]

        rot_gain = self._camera_rotation_image_gain(camera_name)
        J_rows = []
        e_rows = []

        for plug_pt, port_pt in zip(plug_pts, port_pts):
            rel = np.asarray(plug_pt - plug_center, dtype=np.float64)
            j_wz = rot_gain * np.asarray([-rel[1], rel[0]], dtype=np.float64).reshape(2, 1)
            J_feat = np.hstack([np.asarray(J_xy, dtype=np.float64), j_wz])
            err = np.asarray(plug_pt - port_pt, dtype=np.float64).reshape(2, 1)
            J_rows.append(J_feat)
            e_rows.append(err)

        J = np.vstack(J_rows)
        e = np.vstack(e_rows)
        center_px_err = float(np.linalg.norm(plug_center - port_center))
        yaw_err = self._estimate_detection_yaw_error(port_det, plug_det)
        return J, e, center_px_err, yaw_err

    def _compute_ibvs_jacobian_xy(self, camera_name: str, current_pose: Pose) -> tuple[Optional[np.ndarray], float]:
        try:
            with self._detection_node._lock:
                info = self._detection_node._latest_infos.get(camera_name)
            if info is None:
                return None, 0.0

            camera_frame = str(info.header.frame_id)
            R_cam_base, t_cam_base = self._detection_node._lookup_transform(camera_frame, "base_link")

            p_base = np.asarray([
                float(current_pose.position.x),
                float(current_pose.position.y),
                float(current_pose.position.z),
            ], dtype=np.float64)

            p_cam = np.asarray(R_cam_base, dtype=np.float64) @ p_base + np.asarray(t_cam_base, dtype=np.float64)

            X = float(p_cam[0])
            Y = float(p_cam[1])
            Z = float(p_cam[2])

            if Z <= 1e-4:
                return None, 0.0

            fx = float(info.k[0]) if len(info.k) >= 6 else self.DEFAULT_FX
            fy = float(info.k[4]) if len(info.k) >= 6 else self.DEFAULT_FY

            L_trans = np.asarray([
                [fx / Z, 0.0, -fx * X / (Z * Z)],
                [0.0, fy / Z, -fy * Y / (Z * Z)],
            ], dtype=np.float64)

            G_xy = np.asarray(R_cam_base, dtype=np.float64)[:, :2]
            J_xy = L_trans @ G_xy
            return J_xy, Z
        except Exception:
            return None, 0.0

    def _publish_servo_debug_image(
        self,
        cam: str,
        cam_dets: List[Dict],
        port_det: Optional[Dict],
        plug_det: Optional[Dict],
    ) -> None:
        if cam not in self._annotated_images:
            return

        try:
            cv_img = self._cv_bridge.imgmsg_to_cv2(self._annotated_images[cam], "bgr8")

            pb = port_det.get("bbox_xyxy", []) if port_det else []
            gb = plug_det.get("bbox_xyxy", []) if plug_det else []

            if len(pb) == 4:
                cv2.rectangle(cv_img, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (255, 0, 0), 2)
            if len(gb) == 4:
                cv2.rectangle(cv_img, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 0, 255), 2)

            def draw_axis(det: Optional[Dict], color_major, color_minor):
                if det is None:
                    return
                frame = self._bbox_ibvs_feature_points(det)
                if frame is None:
                    return
                center, major, minor, major_half, minor_half, _ = frame
                c = (int(round(center[0])), int(round(center[1])))
                m1 = center + major * major_half
                m2 = center - major * major_half
                s1 = center + minor * minor_half
                s2 = center - minor * minor_half
                cv2.circle(cv_img, c, 5, (255, 255, 255), -1)
                cv2.line(cv_img, c, (int(round(m1[0])), int(round(m1[1]))), color_major, 2)
                cv2.line(cv_img, c, (int(round(m2[0])), int(round(m2[1]))), color_major, 2)
                cv2.line(cv_img, c, (int(round(s1[0])), int(round(s1[1]))), color_minor, 2)
                cv2.line(cv_img, c, (int(round(s2[0])), int(round(s2[1]))), color_minor, 2)

            draw_axis(port_det, (255, 255, 0), (255, 128, 0))
            draw_axis(plug_det, (0, 255, 255), (0, 128, 255))

            port_uv = self._feature_uv(port_det) if port_det else None
            plug_uv = self._feature_uv(plug_det) if plug_det else None

            if port_uv is not None and plug_uv is not None:
                cv2.line(
                    cv_img,
                    (int(plug_uv[0]), int(plug_uv[1])),
                    (int(port_uv[0]), int(port_uv[1])),
                    (0, 255, 0),
                    3,
                )
                err = float(np.linalg.norm(plug_uv - port_uv))
                cv2.putText(
                    cv_img,
                    f"err={err:.1f}px",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            y_off = 60
            for d in cam_dets[:8]:
                cn = str(d.get("instance_name", d.get("class_name", "")))
                conf = float(d.get("confidence", 0.0))
                cv2.putText(
                    cv_img,
                    f"{cn} {conf:.2f}",
                    (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y_off += 24

            msg_out = self._cv_bridge.cv2_to_imgmsg(cv_img, "bgr8")
            msg_out.header = self._annotated_images[cam].header
            self._corner_match_pubs[cam].publish(msg_out)
        except Exception:
            return

    def _is_gripper_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._gripper_classes)

    def _detect_slot_orientation(self, port_det: Dict) -> Optional[float]:
        """Estimate the SFP slot yaw from its bounding box aspect ratio and angle."""
        bbox = port_det.get("bbox_xyxy", [])
        if len(bbox) != 4:
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        if w < 5 or h < 5:
            return None

        # SFP ports are rectangular — the long axis is the insertion direction.
        # If width > height, the slot is horizontal (yaw ~0).
        # If height > width, the slot is vertical (yaw ~pi/2).
        if w > h * 1.3:
            return 0.0
        elif h > w * 1.3:
            return math.pi / 2.0
        else:
            # Nearly square — can't determine orientation reliably
            return None

    def _get_camera_intrinsics(self, camera_name: str) -> tuple:
        """Retrieve fx, fy, cx, cy from the detection node's cached camera info."""
        try:
            with self._detection_node._lock:
                info = self._detection_node._latest_infos.get(camera_name)
            if info is not None and len(info.k) >= 6:
                fx = float(info.k[0])
                fy = float(info.k[4])
                cx = float(info.k[2])
                cy = float(info.k[5])
                return fx, fy, cx, cy
        except Exception:
            pass
        return self.DEFAULT_FX, self.DEFAULT_FY, self.DEFAULT_CX, self.DEFAULT_CY

    def _pose_with_local_pitch_delta(self, current_pose: Pose, pitch_delta_rad: float) -> Pose:
        q_cur = _quat_normalize(_quat_to_np(current_pose.orientation))
        q_delta = _quat_from_axis_angle([0.0, 1.0, 0.0], float(pitch_delta_rad))
        q_tgt = _quat_normalize(_quat_multiply(q_cur, q_delta))
        pose = _copy_pose(current_pose)
        pose.orientation = Quaternion(
            x=float(q_tgt[0]),
            y=float(q_tgt[1]),
            z=float(q_tgt[2]),
            w=float(q_tgt[3]),
        )
        return pose

    def _publish_local_pitch_velocity(self, pitch_rate_cmd: float) -> None:
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = float(np.clip(pitch_rate_cmd, -self.EDGE_ALIGN_MAX_PITCH_RATE, self.EDGE_ALIGN_MAX_PITCH_RATE))
        twist.angular.z = 0.0
        self._motion_servo.publish_twist_command(
            twist,
            frame_id="gripper/tcp",
            trans_stiffness=90.0,
            rot_stiffness=50.0,
            trans_damping=50.0,
            rot_damping=20.0,
        )

    # ==================================================================
    # Phase 3: Force-Feedback Insertion
    # ==================================================================

    def _sfp_force_insert(
        self,
        gripper_orientation: Quaternion,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        del gripper_orientation, move_robot
        current_pose = self._motion_servo.get_current_pose()
        if current_pose is None:
            send_feedback("mypolicy/insert_fail no_pose")
            return False

        tare_fx, tare_fy, tare_fz = 0.0, 0.0, 0.0
        tare_obs = get_observation()
        if tare_obs is not None:
            w = tare_obs.wrist_wrench.wrench
            tare_fx = float(w.force.x)
            tare_fy = float(w.force.y)
            tare_fz = float(w.force.z)
            send_feedback(f"mypolicy/insert_tare fx={tare_fx:.2f} fy={tare_fy:.2f} fz={tare_fz:.2f}")

        start_z = float(current_pose.position.z)
        target_z = start_z - self.INSERT_MAX_DEPTH
        send_feedback(f"mypolicy/insert_start z={start_z:.4f} target_z={target_z:.4f}")

        insertion_started = False
        settled_count = 0
        step_z = start_z

        while True:
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(self.INSERT_STEP_SEC)
                continue

            current_z = float(current_pose.position.z)
            if step_z <= target_z:
                send_feedback(f"mypolicy/insert_max_depth_reached z={current_z:.4f}")
                break

            observation = get_observation()
            if observation is not None:
                w = observation.wrist_wrench.wrench
                raw_fz = float(w.force.z) - tare_fz
                raw_fx = float(w.force.x) - tare_fx
                raw_fy = float(w.force.y) - tare_fy
                fz = abs(raw_fz)
                lateral_force = math.sqrt(raw_fx * raw_fx + raw_fy * raw_fy)

                send_feedback(
                    f"mypolicy/insert z={current_z:.4f} cmd_z={step_z:.4f} fz={raw_fz:.2f} "
                    f"fx={raw_fx:.2f} fy={raw_fy:.2f} lat={lateral_force:.2f}"
                )

                if lateral_force > self.INSERT_MAX_LATERAL_FORCE:
                    send_feedback(f"mypolicy/insert_abort lateral_force={lateral_force:.2f}")
                    return False

                if fz > self.INSERT_MAX_Z_FORCE:
                    send_feedback(f"mypolicy/insert_abort z_force={fz:.2f}")
                    return False

                if insertion_started and fz < self.INSERT_SETTLED_FORCE:
                    settled_count += 1
                    if settled_count >= 5:
                        send_feedback(f"mypolicy/insert_seated z={current_z:.4f}")
                        return True
                else:
                    settled_count = 0

                if fz > 2.0:
                    insertion_started = True

            twist = Twist()
            twist.linear.z = -0.010
            self._motion_servo.publish_twist_command(twist)
            self.sleep_for(self.INSERT_STEP_SEC)
            step_z -= abs(twist.linear.z) * self.INSERT_STEP_SEC

        return insertion_started

    def _retract_z(self, current_pose: Pose, distance: float, orientation: Quaternion) -> None:
        """Retract upward by a small distance after an abort."""
        retract_pose = _copy_pose(current_pose)
        retract_pose.position.z += distance
        retract_pose.orientation = orientation
        for _ in range(10):
            cur = self._motion_servo.get_current_pose()
            if cur is None:
                break
            twist = self._motion_servo.compute_twist_to_waypoint(cur, retract_pose)
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self._motion_servo.publish_twist_command(twist)
            self.sleep_for(0.05)
        self._motion_servo.stop()

    # ==================================================================
    # Shared helpers
    # ==================================================================

    def _wait_for_observation(self, get_observation: GetObservationCallback, timeout_sec: float):
        deadline = self.time_now() + Duration(seconds=timeout_sec)
        while self.time_now() < deadline:
            observation = get_observation()
            if observation is not None:
                return observation
            self.sleep_for(0.05)
        return None

    def _wait_for_detection(
        self,
        matcher: Callable[[Dict], bool],
        timeout_sec: float,
        preferred_camera: Optional[str],
        min_update_time: float,
    ) -> Optional[Dict]:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            result = self._detection_listener.find_best(
                matcher=matcher,
                preferred_camera=preferred_camera,
                min_update_time=min_update_time,
                require_pose=True,
                freshness_sec=2.0,
            )
            if result is not None:
                return result
            self.sleep_for(0.05)
        return None
    def _execute_sample_and_verify_goal(
        self,
        label: str,
        matcher: Callable[[Dict], bool],
        gripper_orientation: Quaternion,
        z_offset: float,
        min_z: float,
        get_observation: GetObservationCallback,
        send_feedback: SendFeedbackCallback,
        timeout_sec: float,
    ) -> bool:
        """
        State 1: Sample for 15s to find average port location.
        State 2: Execute the planned path only.
        State 3: Replan only if the goal is completely out of frame in all cameras.

        No internal hover timer is used here.
        """
        del timeout_sec
        self._motion_servo.ensure_cartesian_mode()

        while True:
            send_feedback(f"mypolicy/{label}_sampling (15s)")
            sample_deadline = time.monotonic() + 15.0

            positions_x, positions_y, positions_z = [], [], []

            while time.monotonic() < sample_deadline:
                all_dets = self._detection_listener.get_all_detections(freshness_sec=1.0)
                port_dets = [d for d in all_dets if matcher(d)]
                center_dets = [d for d in port_dets if d.get("camera_name") == "center"]
                if center_dets:
                    port_dets = center_dets
                for det in port_dets:
                    pose = self._pose_from_detection(det)
                    if pose:
                        positions_x.append(float(pose.position.x))
                        positions_y.append(float(pose.position.y))
                        positions_z.append(float(pose.position.z))
                self.sleep_for(0.1)

            if not positions_x:
                send_feedback(f"mypolicy/{label}_sampling_fail: no detections. Retrying.")
                continue

            raw_target_pose = Pose()
            raw_target_pose.position.x = sum(positions_x) / len(positions_x)
            raw_target_pose.position.y = sum(positions_y) / len(positions_y)
            raw_target_pose.position.z = sum(positions_z) / len(positions_z)

            target_pose = self._make_target_pose(raw_target_pose, gripper_orientation, z_offset, min_z)
            send_feedback(
                f"mypolicy/{label}_sampled_target xyz=({raw_target_pose.position.x:.3f},{raw_target_pose.position.y:.3f},{raw_target_pose.position.z:.3f})"
            )

            current_pose = self._motion_servo.get_current_pose()
            while current_pose is None:
                self.sleep_for(0.05)
                current_pose = self._motion_servo.get_current_pose()

            waypoints = self._planner.plan_from_current_pose(current_pose, target_pose)
            if not waypoints:
                waypoints = [target_pose]

            self._motion_servo.publish_target_marker(target_pose)
            self._motion_servo.publish_waypoint_visuals(waypoints)
            send_feedback(f"mypolicy/{label}_moving")

            waypoint_idx = 0
            goal_feature_lost_count = 0

            while True:
                current_pose = self._motion_servo.get_current_pose()
                if current_pose is None:
                    self.sleep_for(0.05)
                    continue

                if self._position_distance(current_pose, target_pose) <= self._motion_servo.position_tolerance:
                    self._motion_servo.stop()
                    send_feedback(f"mypolicy/{label}_path_complete")
                    return True

                visible_goal_cameras = self._goal_feature_visible_cameras(matcher, freshness_sec=1.0)
                if visible_goal_cameras:
                    goal_feature_lost_count = 0
                else:
                    goal_feature_lost_count += 1
                    self._motion_servo.stop()
                    send_feedback(
                        f"mypolicy/{label}_goal_out_of_all_cameras lost_count={goal_feature_lost_count}/{self.SAMPLE_VERIFY_GOAL_LOST_MAX_COUNT}"
                    )
                    if goal_feature_lost_count >= self.SAMPLE_VERIFY_GOAL_LOST_MAX_COUNT:
                        send_feedback(f"mypolicy/{label}_replan_goal_out_of_all_cameras")
                        break
                    self.sleep_for(0.05)
                    continue

                current_wp = waypoints[min(waypoint_idx, len(waypoints) - 1)]
                if self._position_distance(current_pose, current_wp) <= self._motion_servo.position_tolerance and waypoint_idx < len(waypoints) - 1:
                    waypoint_idx += 1
                    current_wp = waypoints[waypoint_idx]

                twist = self._motion_servo.compute_twist_to_waypoint(current_pose, current_wp)
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = 0.0
                self._motion_servo.publish_twist_command(twist)
                self.sleep_for(0.05)

            continue

    def _execute_pose_goal(
        self,
        label: str,
        target_pose: Pose,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        timeout_sec: float,
    ) -> bool:
        self._motion_servo.ensure_cartesian_mode()
        current_pose = self._motion_servo.get_current_pose()
        if current_pose is None:
            observation = self._wait_for_observation(get_observation=get_observation, timeout_sec=3.0)
            if observation is None:
                send_feedback(f"mypolicy/fail {label}_no_observation")
                return False
            current_pose = observation.controller_state.tcp_pose

        waypoints = self._planner.plan_from_current_pose(current_pose=current_pose, target_pose=target_pose)
        if not waypoints:
            waypoints = [target_pose]

        self._motion_servo.publish_target_marker(target_pose)
        self._motion_servo.publish_waypoint_visuals(waypoints)
        send_feedback(f"mypolicy/{label}_waypoints count={len(waypoints)}")
        for index, waypoint in enumerate(waypoints, start=1):
            send_feedback(
                f"mypolicy/{label}_waypoint {index}/{len(waypoints)} xyz=({waypoint.position.x:.3f},{waypoint.position.y:.3f},{waypoint.position.z:.3f})"
            )
            if not self._servo_to_pose(waypoint=waypoint, timeout_sec=timeout_sec):
                self._motion_servo.stop()
                return False
        self._motion_servo.stop()
        return True

    def _servo_to_pose(self, waypoint: Pose, timeout_sec: float) -> bool:
        del timeout_sec
        while True:
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(0.05)
                continue

            pos_error = self._position_distance(current_pose, waypoint)
            if pos_error <= self._motion_servo.position_tolerance:
                self._motion_servo.stop()
                return True

            twist = self._motion_servo.compute_twist_to_waypoint(current_pose, waypoint)
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self._motion_servo.publish_twist_command(twist)
            self.sleep_for(0.05)

    def _make_target_pose(self, detected_pose: Pose, orientation: Quaternion, z_offset: float = 0.0, min_z: float = 0.0) -> Pose:
        """Build a target pose with no policy-side positional offsets."""
        target_pose = Pose()
        target_pose.position = Point(
            x=float(detected_pose.position.x),
            y=float(detected_pose.position.y),
            z=float(detected_pose.position.z),
        )
        target_pose.orientation = Quaternion(
            x=float(orientation.x),
            y=float(orientation.y),
            z=float(orientation.z),
            w=float(orientation.w),
        )
        return target_pose

    # ==================================================================
    # Matching helpers
    # ==================================================================

    def _is_taskboard_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._taskboard_classes)

    def _is_nic_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._nic_classes)

    def _is_sfp_port_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._sfp_port_classes)

    def _is_sc_port_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._sc_port_classes)

    def _matches_specific_port(self, det: Dict, target_name: str, allowed_classes: set) -> bool:
        """Match a detection to a specific port name (e.g., 'sfp_port_0')."""
        if not self._matches_any_name(det, allowed_classes):
            return False
        if not target_name:
            return True  # No specific port requested — accept any
        # Check if the instance name matches the target exactly
        instance_name = self._norm_name(det.get("instance_name", ""))
        normalized_target = self._norm_name(target_name)
        return instance_name == normalized_target

    def _matches_any_name(self, det: Dict, allowed_names: set) -> bool:
        for key in ("class_name", "raw_class_name", "base_class_name", "instance_name"):
            value = det.get(key, "")
            norm = self._norm_name(value)
            base = self._strip_numeric_suffix(norm)
            for allowed in allowed_names:
                if norm == allowed or base == allowed or norm.startswith(f"{allowed}_"):
                    return True
        return False

    def _pose_from_detection(self, det: Dict) -> Optional[Pose]:
        pose_dict = det.get("pose_base_link")
        if not isinstance(pose_dict, dict):
            return None
        position = pose_dict.get("position")
        orientation = pose_dict.get("orientation")
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            return None
        pose = Pose()
        pose.position = Point(
            x=float(position.get("x", 0.0)),
            y=float(position.get("y", 0.0)),
            z=float(position.get("z", 0.0)),
        )
        pose.orientation = Quaternion(
            x=float(orientation.get("x", 0.0)),
            y=float(orientation.get("y", 0.0)),
            z=float(orientation.get("z", 0.0)),
            w=float(orientation.get("w", 1.0)),
        )
        return pose

    def _position_distance(self, pose_a: Pose, pose_b: Pose) -> float:
        dx = float(pose_a.position.x - pose_b.position.x)
        dy = float(pose_a.position.y - pose_b.position.y)
        dz = float(pose_a.position.z - pose_b.position.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _parse_name_set(self, text: str) -> set:
        return {self._norm_name(x) for x in str(text).split(",") if str(x).strip()}

    def _strip_numeric_suffix(self, name: str) -> str:
        parts = self._norm_name(name).split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return self._norm_name(name)

    def _norm_name(self, name: str) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")