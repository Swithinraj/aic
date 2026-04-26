from __future__ import annotations

import json
import math
import os
import threading
import time
from copy import deepcopy
from typing import Callable, Dict, List, Optional

import numpy as np
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
        self.get_logger().info("mypolicy simplified: sample -> move -> pitch orientation only, no movement/orientation timeouts.")

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

        send_feedback("mypolicy/done sampling_move_pitch_complete")
        return True

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