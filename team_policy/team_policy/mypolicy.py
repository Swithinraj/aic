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
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from team_policy.planner.cartesian_planner import CartesianPlanner
from team_policy.planner.combined_yolo_depth_pose_planner import CombinedYoloDepthPosePlanner


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


def _quat_error_rotvec(current: Quaternion, target: Quaternion):
    qc = _quat_normalize(_quat_to_np(current))
    qt = _quat_normalize(_quat_to_np(target))
    q_err = _quat_multiply(qt, _quat_inverse(qc))
    if q_err[3] < 0.0:
        q_err = [-q_err[0], -q_err[1], -q_err[2], -q_err[3]]
    vx, vy, vz, vw = q_err
    sin_half = math.sqrt(vx * vx + vy * vy + vz * vz)
    if sin_half < 1e-9:
        return [0.0, 0.0, 0.0], 0.0
    axis = [vx / sin_half, vy / sin_half, vz / sin_half]
    angle = 2.0 * math.atan2(sin_half, max(1e-12, vw))
    return [axis[0] * angle, axis[1] * angle, axis[2] * angle], abs(angle)


class DetectionListener(Node):
    def __init__(self):
        super().__init__("mypolicy_detection_listener")
        self._lock = threading.Lock()
        self._latest: Dict[str, Dict] = {
            "left": {"time": 0.0, "detections": []},
            "center": {"time": 0.0, "detections": []},
            "right": {"time": 0.0, "detections": []},
        }
        self.create_subscription(String, "/left_camera/yolo/detections_json", lambda msg: self._cb("left", msg), 10)
        self.create_subscription(String, "/center_camera/yolo/detections_json", lambda msg: self._cb("center", msg), 10)
        self.create_subscription(String, "/right_camera/yolo/detections_json", lambda msg: self._cb("right", msg), 10)

    def _cb(self, camera_name: str, msg: String) -> None:
        try:
            parsed = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(parsed, list):
            return
        detections = [dict(item) for item in parsed if isinstance(item, dict)]
        with self._lock:
            self._latest[camera_name] = {"time": time.monotonic(), "detections": detections}

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
                camera: {
                    "time": float(entry["time"]),
                    "detections": [dict(det) for det in entry["detections"]],
                }
                for camera, entry in self._latest.items()
            }

        candidates: List[Dict] = []
        for camera_name, entry in snapshot.items():
            update_time = float(entry["time"])
            if update_time <= 0.0:
                continue
            if update_time < float(min_update_time):
                continue
            if now - update_time > float(freshness_sec):
                continue
            for det in entry["detections"]:
                if require_pose and not self._has_pose(det):
                    continue
                if not matcher(det):
                    continue
                candidates.append(
                    {
                        "camera_name": camera_name,
                        "update_time": update_time,
                        "detection": det,
                        "confidence": float(det.get("confidence", 0.0)),
                        "preferred": 1 if preferred_camera is not None and camera_name == preferred_camera else 0,
                    }
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item["preferred"],
                item["confidence"],
                item["update_time"],
            ),
            reverse=True,
        )
        return candidates[0]

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
        self.position_tolerance = 0.012
        self.orientation_tolerance_rad = 0.08
        self.linear_kp = 1.0
        self.angular_kp = 1.2
        self.max_linear_speed = 0.05
        self.min_linear_speed = 0.015
        self.max_angular_speed = 0.8
        self.min_angular_speed = 0.08
        self.trans_stiffness = 90.0
        self.rot_stiffness = 50.0
        self.trans_damping = 50.0
        self.rot_damping = 20.0

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

    def publish_twist_command(self, twist: Twist) -> None:
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.command_frame
        msg.velocity = twist
        msg.target_stiffness = [
            self.trans_stiffness, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, self.trans_stiffness, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, self.trans_stiffness, 0.0, 0.0, 0.0,
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
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.command_frame
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose = waypoint
            path.poses.append(pose_stamped)
        self.path_pub.publish(path)


class mypolicy(Policy):
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

        self.get_logger().info("mypolicy.__init__()")
        self.get_logger().info("Started internal CombinedYoloDepthPosePlanner, detection listener, and motion servo.")

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

        send_feedback("mypolicy/search_taskboard_all_cameras")
        board_result = self._wait_for_detection(
            matcher=self._is_taskboard_detection,
            timeout_sec=12.0,
            preferred_camera=None,
            min_update_time=0.0,
        )
        if board_result is None:
            send_feedback("mypolicy/fail taskboard_not_found")
            return False

        board_pose_raw = self._pose_from_detection(board_result["detection"])
        if board_pose_raw is None:
            send_feedback("mypolicy/fail taskboard_pose_missing")
            return False

        current_pose = self._motion_servo.get_current_pose() or observation.controller_state.tcp_pose
        board_pose = self._make_position_only_target_pose(board_pose_raw, current_pose)

        send_feedback(
            f"mypolicy/taskboard_found camera={board_result['camera_name']} class={board_result['detection'].get('class_name', '')} conf={board_result['confidence']:.3f}"
        )
        if not self._execute_pose_goal(
            label="taskboard_pose",
            target_pose=board_pose,
            get_observation=get_observation,
            move_robot=move_robot,
            send_feedback=send_feedback,
            timeout_sec=18.0,
        ):
            send_feedback("mypolicy/fail taskboard_pose_failed")
            return False

        detection_after_board_move_time = time.monotonic()
        self.sleep_for(1.0)

        observation = self._wait_for_observation(get_observation=get_observation, timeout_sec=3.0)
        if observation is None:
            send_feedback("mypolicy/fail no_observation_after_taskboard_move")
            return False

        send_feedback("mypolicy/search_nic_card")
        nic_result = self._wait_for_detection(
            matcher=self._is_nic_detection,
            timeout_sec=12.0,
            preferred_camera="center",
            min_update_time=detection_after_board_move_time,
        )
        if nic_result is None:
            send_feedback("mypolicy/fail nic_card_not_found")
            return False

        nic_pose_raw = self._pose_from_detection(nic_result["detection"])
        if nic_pose_raw is None:
            send_feedback("mypolicy/fail nic_pose_missing")
            return False

        current_pose = self._motion_servo.get_current_pose() or observation.controller_state.tcp_pose
        nic_pose = self._make_position_only_target_pose(nic_pose_raw, current_pose)

        send_feedback(
            f"mypolicy/nic_found camera={nic_result['camera_name']} class={nic_result['detection'].get('class_name', '')} conf={nic_result['confidence']:.3f}"
        )
        if not self._execute_pose_goal(
            label="nic_pose",
            target_pose=nic_pose,
            get_observation=get_observation,
            move_robot=move_robot,
            send_feedback=send_feedback,
            timeout_sec=18.0,
        ):
            send_feedback("mypolicy/fail nic_pose_failed")
            return False

        self._motion_servo.stop()
        send_feedback("mypolicy/done success=true")
        return True

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
            self.sleep_for(0.10)
        return None

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
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(0.05)
                continue

            pos_error = self._position_distance(current_pose, waypoint)
            _, ang_error = _quat_error_rotvec(current_pose.orientation, waypoint.orientation)
            if pos_error <= self._motion_servo.position_tolerance and ang_error <= self._motion_servo.orientation_tolerance_rad:
                self._motion_servo.stop()
                return True

            twist = self._motion_servo.compute_twist_to_waypoint(current_pose, waypoint)
            self._motion_servo.publish_twist_command(twist)
            self.sleep_for(0.05)

        self._motion_servo.stop()
        return False

    def _make_position_only_target_pose(self, detected_pose: Pose, orientation_source_pose: Pose) -> Pose:
        target_pose = Pose()
        target_pose.position = Point(
            x=float(detected_pose.position.x),
            y=float(detected_pose.position.y),
            z=float(detected_pose.position.z),
        )
        target_pose.orientation = Quaternion(
            x=float(orientation_source_pose.orientation.x),
            y=float(orientation_source_pose.orientation.y),
            z=float(orientation_source_pose.orientation.z),
            w=float(orientation_source_pose.orientation.w),
        )
        return target_pose

    def _is_taskboard_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._taskboard_classes)

    def _is_nic_detection(self, det: Dict) -> bool:
        return self._matches_any_name(det, self._nic_classes)

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