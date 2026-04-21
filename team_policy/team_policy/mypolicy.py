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


def _yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    """Create a downward-facing quaternion with a given yaw rotation."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=float(math.sin(yaw_rad / 2.0)),
        w=float(math.cos(yaw_rad / 2.0)),
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
    SERVO_HOVER_Z = 0.06
    # Bug 4 fix: relaxed from 0.003 — 10mm world-frame tolerance is achievable with noisy detections
    SERVO_WORLD_TOLERANCE = 0.010
    SERVO_MAX_ITERATIONS = 500
    SERVO_GAIN_XY = 1.20
    SERVO_GAIN_Z = 0.35
    SERVO_STEP_SEC = 0.10
    SERVO_TIMEOUT_SEC = 50.0

    SERVO_DLS_DAMPING = 1.5
    SERVO_MAX_LINEAR_SPEED_XY = 0.030
    SERVO_MAX_LINEAR_SPEED_Z = 0.008
    SERVO_MIN_LINEAR_SPEED_XY = 0.0
    # Bug 4 fix: relaxed from 22px — 22px = 2.2mm which is impossible with noisy detections
    # 60px ≈ 6mm at hover height, reliably achievable from the world-frame fallback path
    SERVO_CONVERGED_PX = 60.0
    SERVO_CONVERGED_PX_STABLE_COUNT = 4
    SERVO_Z_ENABLE_PX = 90.0
    SERVO_ABORT_PX = 120.0

    SERVO_MAX_YAW_RATE = 0.20
    SERVO_YAW_GAIN = 0.8
    # Bug 3 fix: SC port yaw tolerance loosened — SC slot has a mechanical guide
    # that corrects small angular errors on insertion; 0.10 rad was blocking servo convergence
    SERVO_YAW_CONVERGED_RAD = 0.10        # kept for SFP
    SERVO_YAW_CONVERGED_RAD_SC = 0.30     # looser for SC (≈17°)
    SERVO_MIN_CONFIDENCE = 0.20
    SERVO_LAST_VALID_PAIR_SEC = 0.60

    INSERT_Z_STEP = 0.001
    INSERT_MAX_LATERAL_FORCE = 15.0
    INSERT_MAX_Z_FORCE = 30.0
    INSERT_SETTLED_FORCE = 4.0
    INSERT_MIN_DEPTH_FOR_SETTLED_MM = 5.0
    INSERT_DEEP_SUCCESS_MM = 8.0
    INSERT_MAX_DEPTH = 0.160
    # SC insertion travels in Y — max travel from standoff to full engagement
    SC_INSERT_MAX_TRAVEL = 0.080          # 80mm max Y travel
    SC_INSERT_SPEED = 0.008               # 8mm/s horizontal insertion speed
    SC_HOVER_Y_STANDOFF = 0.06           # 6cm standoff in +Y before inserting
    SC_PORT_FACE_Z = 0.25                 # SC ports are higher on the board than SFP
    INSERT_STEP_SEC = 0.15
    INSERT_TIMEOUT_SEC = 45.0
    
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
        self._sfp_module_classes = self._parse_name_set(os.environ.get("YOLOV12_SFP_MODULE_CLASSES", "sfp_module,sfp module,transceiver"))
        self._gripper_classes = self._parse_name_set(os.environ.get("YOLOV12_GRIPPER_CLASSES", "gripper,gripper_tip,gripper_tcp"))

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

        current_pose = self._motion_servo.get_current_pose() or observation.controller_state.tcp_pose

        # --- Determine port type and set PER-TYPE config ---
        # This is the root cause of the orientation slant bug:
        # SC ports require a HORIZONTAL approach (-Y axis), not a downward -Z approach.
        # SFP ports face upward on the NIC card and need a vertical -Z approach.
        # Using the same downward orientation for both causes the SC plug to hit the
        # top rim of the port housing instead of entering from the side.
        port_type = str(task.port_type).strip().lower()
        target_port_name = str(task.port_name).strip().lower()

        if port_type == "sc":
            # SC port: sc_port_base_link has pose roll=π/2, pitch=π
            # → insertion axis is horizontal in world frame, approach in -Y
            # Rotate home orientation 90° around world X to point gripper sideways.
            # Home pose is roughly pointing down (x≈1,y≈0,z≈0,w≈0).
            # After 90° rotation around X: gripper Z-axis points in -Y world direction.
            home_q = [
                float(current_pose.orientation.x),
                float(current_pose.orientation.y),
                float(current_pose.orientation.z),
                float(current_pose.orientation.w),
            ]
            rot_90_x = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]  # 90° around X
            sc_q = _quat_multiply(home_q, rot_90_x)
            sc_q = _quat_normalize(sc_q)
            gripper_orientation = Quaternion(x=sc_q[0], y=sc_q[1], z=sc_q[2], w=sc_q[3])
            insertion_axis = "y"
            hover_z_offset = 0.0        # SC: hover at same Z as port, not above it
            hover_min_z = 0.15
            yaw_converged_rad = self.SERVO_YAW_CONVERGED_RAD_SC   # Bug 3 fix: loose for SC
            port_matcher = self._is_sc_port_detection
            port_label = "sc_port"
            # SC fallback stays within SC class — never fall back to SFP
            fallback_matcher = self._is_sc_port_detection
        elif port_type == "sfp":
            # SFP port: faces upward on NIC card, insert straight down (-Z)
            # Keep home orientation (already pointing down)
            gripper_orientation = Quaternion(
                x=float(current_pose.orientation.x),
                y=float(current_pose.orientation.y),
                z=float(current_pose.orientation.z),
                w=float(current_pose.orientation.w),
            )
            insertion_axis = "z"
            hover_z_offset = self.SERVO_HOVER_Z
            hover_min_z = 0.20
            yaw_converged_rad = self.SERVO_YAW_CONVERGED_RAD
            port_matcher = lambda det: self._matches_specific_port(det, target_port_name, self._sfp_port_classes)
            port_label = target_port_name or "sfp_port"
            fallback_matcher = self._is_sfp_port_detection
        else:
            gripper_orientation = Quaternion(
                x=float(current_pose.orientation.x),
                y=float(current_pose.orientation.y),
                z=float(current_pose.orientation.z),
                w=float(current_pose.orientation.w),
            )
            insertion_axis = "z"
            hover_z_offset = self.SERVO_HOVER_Z
            hover_min_z = 0.20
            yaw_converged_rad = self.SERVO_YAW_CONVERGED_RAD
            port_matcher = self._is_nic_detection
            port_label = "nic_card"
            fallback_matcher = self._is_nic_detection

        send_feedback(
            f"mypolicy/config port_type={port_type} insertion_axis={insertion_axis} "
            f"gripper_q=({gripper_orientation.x:.3f},{gripper_orientation.y:.3f},"
            f"{gripper_orientation.z:.3f},{gripper_orientation.w:.3f}) "
            f"yaw_tol_deg={math.degrees(yaw_converged_rad):.1f}"
        )

        # ====== PHASE 1: Detect port and coarse approach ======
        send_feedback(f"mypolicy/phase1_search_{port_label}")
        port_result = self._wait_for_detection(
            matcher=port_matcher,
            timeout_sec=12.0,
            preferred_camera=None,
            min_update_time=0.0,
        )
        if port_result is None:
            # Bug 1 fix: fallback stays within the correct cable family
            send_feedback(f"mypolicy/{port_label}_not_found, fallback within same type")
            port_result = self._wait_for_detection(
                matcher=fallback_matcher,
                timeout_sec=8.0,
                preferred_camera=None,
                min_update_time=0.0,
            )
            port_label = f"{port_label}_fallback"

        if port_result is None:
            send_feedback("mypolicy/fail port_not_found")
            return False

        port_pose_raw = self._pose_from_detection(port_result["detection"])
        if port_pose_raw is None:
            send_feedback("mypolicy/fail port_pose_missing")
            return False

        send_feedback(
            f"mypolicy/port_detected class={port_result['detection'].get('class_name', '')} "
            f"conf={port_result['confidence']:.3f} "
            f"xyz=({port_pose_raw.position.x:.3f},{port_pose_raw.position.y:.3f},{port_pose_raw.position.z:.3f})"
        )

        send_feedback("mypolicy/phase1_sample_and_verify")
        if not self._execute_sample_and_verify_goal(
            label="hover_above_port",
            matcher=port_matcher,
            gripper_orientation=gripper_orientation,
            z_offset=hover_z_offset,
            min_z=hover_min_z,
            get_observation=get_observation,
            send_feedback=send_feedback,
            timeout_sec=60.0,
        ):
            send_feedback("mypolicy/fail hover_pose_failed")
            return False

        self.sleep_for(0.5)

        # ====== PHASE 2: Visual servoing alignment ======
        send_feedback("mypolicy/phase2_visual_servo_start")
        servo_ok = self._sfp_visual_servo_align(
            port_matcher=port_matcher,
            port_label=port_label,
            gripper_orientation=gripper_orientation,
            send_feedback=send_feedback,
            yaw_converged_rad=yaw_converged_rad,   # Bug 3 fix: pass per-type tolerance
        )
        if not servo_ok:
            send_feedback("mypolicy/phase2_servo_failed (stopping before insertion)")
            self._motion_servo.stop()
            return False

        self.sleep_for(0.5)

        # ====== PHASE 3: Insertion ======
        send_feedback(f"mypolicy/phase3_insertion_start axis={insertion_axis}")

        if insertion_axis == "y":
            # SC cable: horizontal insertion in -Y direction
            insert_ok = self._sc_force_insert(
                port_matcher=port_matcher,
                gripper_orientation=gripper_orientation,
                get_observation=get_observation,
                send_feedback=send_feedback,
            )
        else:
            # SFP/NIC cable: vertical insertion in -Z direction
            port_face_hint_z = float(port_pose_raw.position.z) if port_pose_raw is not None else None
            insert_ok = self._sfp_force_insert(
                gripper_orientation=gripper_orientation,
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                port_face_hint_z=port_face_hint_z,
                port_matcher=port_matcher,    # Bug 5 fix: passed through for live XY hold
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
        yaw_converged_rad: float = None,    # Bug 3 fix: per-type yaw tolerance
    ) -> bool:
        # Use the passed tolerance, or fall back to the class default
        if yaw_converged_rad is None:
            yaw_converged_rad = self.SERVO_YAW_CONVERGED_RAD

        deadline = time.monotonic() + self.SERVO_TIMEOUT_SEC
        best_metric_error = float("inf")
        stable_count = 0
        last_valid_pairs: Dict[str, Dict] = {}

        # Cache the initial port pose for use as a stable world-frame fallback reference.
        # This prevents the noisy live fused detection from jumping the fallback target around.
        # We update this cache when a fresh high-confidence fused detection is available.
        cached_port_pose: Optional[Pose] = None

        # Log servo entry state
        initial_fused = self._detection_listener.find_best(
            matcher=port_matcher, require_pose=True, freshness_sec=2.0
        )
        if initial_fused is None:
            # Bug 1 fix: fall back within the same matcher, not always sfp_port
            initial_fused = self._detection_listener.find_best(
                matcher=port_matcher, require_pose=True, freshness_sec=5.0
            )
            
        if initial_fused is not None:
            ip = self._pose_from_detection(initial_fused["detection"])
            if ip is not None:
                # Make a copy, not a reference, to avoid mutations
                cached_port_pose = Pose()
                cached_port_pose.position.x = float(ip.position.x)
                cached_port_pose.position.y = float(ip.position.y)
                cached_port_pose.position.z = float(ip.position.z)
                cached_port_pose.orientation.x = float(ip.orientation.x)
                cached_port_pose.orientation.y = float(ip.orientation.y)
                cached_port_pose.orientation.z = float(ip.orientation.z)
                cached_port_pose.orientation.w = float(ip.orientation.w)
                port_yaw_init = math.degrees(2.0 * math.atan2(float(ip.orientation.z), float(ip.orientation.w)))
                send_feedback(
                    f"mypolicy/servo_entry"
                    f" port_xyz=({ip.position.x:.4f},{ip.position.y:.4f},{ip.position.z:.4f})"
                    f" port_yaw_deg={port_yaw_init:.2f}"
                    f" conf={initial_fused['confidence']:.3f}"
                )
        tcp0 = self._motion_servo.get_current_pose()
        if tcp0 is not None:
            tcp_yaw_init = math.degrees(self._quat_yaw_rad(tcp0.orientation))
            send_feedback(
                f"mypolicy/servo_entry"
                f" tcp_xyz=({tcp0.position.x:.4f},{tcp0.position.y:.4f},{tcp0.position.z:.4f})"
                f" tcp_yaw_deg={tcp_yaw_init:.2f}"
            )
        send_feedback(
            f"mypolicy/servo_params"
            f" converged_px={self.SERVO_CONVERGED_PX}"
            f" stable_count_req={self.SERVO_CONVERGED_PX_STABLE_COUNT}"
            f" yaw_converged_deg={math.degrees(self.SERVO_YAW_CONVERGED_RAD):.1f}"
            f" timeout_sec={self.SERVO_TIMEOUT_SEC}"
            f" gain_xy={self.SERVO_GAIN_XY} gain_z={self.SERVO_GAIN_Z}"
            f" yaw_gain={self.SERVO_YAW_GAIN} max_yaw_rate={self.SERVO_MAX_YAW_RATE}"
        )

        for iteration in range(self.SERVO_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                send_feedback(f"mypolicy/servo_timeout best_px_err={best_metric_error:.1f}")
                self._motion_servo.stop()
                # Even on timeout, allow insertion if world-frame XY is close enough.
                # Gripper detection is intermittent so pixel convergence may not be reached,
                # but the robot may still be aligned well enough for the force-insertion phase.
                if best_metric_error <= self.SERVO_CONVERGED_PX:
                    return True
                cur_at_timeout = self._motion_servo.get_current_pose()
                fused_at_timeout = self._detection_listener.find_best(
                    matcher=port_matcher, require_pose=True, freshness_sec=10.0
                )
                send_feedback(
                    f"mypolicy/servo_timeout_fused_found={fused_at_timeout is not None}"
                    f" cur_found={cur_at_timeout is not None}"
                )
                if cur_at_timeout is not None and fused_at_timeout is not None:
                    fp = self._pose_from_detection(fused_at_timeout["detection"])
                    if fp is not None:
                        dx = float(fp.position.x) - float(cur_at_timeout.position.x)
                        dy = float(fp.position.y) - float(cur_at_timeout.position.y)
                        xy_err_mm = math.sqrt(dx * dx + dy * dy) * 1000.0
                        # Check yaw convergence at timeout — same check as main IBVS path
                        pqz_to = float(fp.orientation.z)
                        pqw_to = float(fp.orientation.w)
                        port_yaw_to = 2.0 * math.atan2(pqz_to, pqw_to)
                        tcp_yaw_to = self._quat_yaw_rad(cur_at_timeout.orientation)
                        yaw_err_to = abs(self._angle_wrap(port_yaw_to - tcp_yaw_to))
                        yaw_ok_to = yaw_err_to <= yaw_converged_rad
                        send_feedback(
                            f"mypolicy/servo_timeout_world_xy_err={xy_err_mm:.1f}mm"
                            f" yaw_err_deg={math.degrees(yaw_err_to):.2f} yaw_ok={yaw_ok_to}"
                            f" tcp_xyz=({cur_at_timeout.position.x:.3f},{cur_at_timeout.position.y:.3f},{cur_at_timeout.position.z:.3f})"
                            f" port_xyz=({fp.position.x:.3f},{fp.position.y:.3f},{fp.position.z:.3f})"
                        )
                        # Strict gate: same threshold as max-iter path, yaw must also be valid
                        if xy_err_mm <= 15.0 and yaw_ok_to:
                            send_feedback("mypolicy/servo_timeout_close_enough proceeding to insertion")
                            return True
                        send_feedback(
                            f"mypolicy/servo_timeout_rejected xy_err={xy_err_mm:.1f}mm (need<=15) yaw_ok={yaw_ok_to}"
                        )
                return False

            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            J_rows = []
            e_rows = []
            per_cam_errors = []
            used_cameras = []
            best_fallback_port = None
            best_fallback_conf = -1.0
            now = time.monotonic()

            for cam in ("left", "center", "right"):
                cam_dets = self._detection_listener.get_camera_detections(cam, freshness_sec=0.80)

                port_cands = [
                    d for d in cam_dets
                    if port_matcher(d) and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE
                ]
                plug_cands = [
                    d for d in cam_dets
                    if self._is_gripper_detection(d) and float(d.get("confidence", 0.0)) >= self.SERVO_MIN_CONFIDENCE
                ]

                port_det = max(port_cands, key=lambda d: float(d.get("confidence", 0.0))) if port_cands else None
                plug_det = max(plug_cands, key=lambda d: float(d.get("confidence", 0.0))) if plug_cands else None

                if port_det is not None:
                    conf = float(port_det.get("confidence", 0.0))
                    if conf > best_fallback_conf:
                        best_fallback_conf = conf
                        best_fallback_port = port_det

                if port_det is not None and plug_det is not None:
                    last_valid_pairs[cam] = {
                        "time": now,
                        "port": dict(port_det),
                        "plug": dict(plug_det),
                    }
                else:
                    cached = last_valid_pairs.get(cam)
                    if cached is not None and now - float(cached.get("time", 0.0)) <= self.SERVO_LAST_VALID_PAIR_SEC:
                        if port_det is None:
                            port_det = dict(cached["port"])
                        if plug_det is None:
                            plug_det = dict(cached["plug"])

                self._publish_servo_debug_image(cam, cam_dets, port_det, plug_det)

                if port_det is None or plug_det is None:
                    continue

                J_cam, e_cam, px_err = self._compute_center_ibvs_camera_system(
                    camera_name=cam,
                    current_pose=current_pose,
                    port_det=port_det,
                    plug_det=plug_det,
                )
                if J_cam is None or e_cam is None or J_cam.shape[0] < 2:
                    continue

                port_conf = float(port_det.get("confidence", 0.0))
                plug_conf = float(plug_det.get("confidence", 0.0))
                weight = max(0.35, min(1.0, math.sqrt(max(1e-6, port_conf * plug_conf))))

                J_rows.append(weight * J_cam)
                e_rows.append(weight * e_cam.reshape(-1, 1))
                per_cam_errors.append(float(px_err))
                used_cameras.append(cam)

            if J_rows:
                # Pre-check pixel error before computing velocities.
                # If ALL cameras show error > SERVO_ABORT_PX, the IBVS detections are
                # likely misidentified or the robot is far from the port. Fall through
                # to the world-frame fallback which is more reliable at large distances.
                avg_px_error_pre = float(sum(per_cam_errors) / len(per_cam_errors))
                metric_px_error_pre = float(max(per_cam_errors))
                if metric_px_error_pre > self.SERVO_ABORT_PX:
                    # Log this skip and fall through to world-frame fallback below
                    send_feedback(
                        f"mypolicy/servo_iter_{iteration}"
                        f" ibvs_skipped_px_too_large px_err_max={metric_px_error_pre:.1f}"
                        f" (>{self.SERVO_ABORT_PX}) falling_back_to_world"
                    )
                    J_rows = []  # Force fall-through to world-frame path

            if J_rows:
                if len(used_cameras) >= 2:
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
                else:
                    J2 = np.asarray(J_rows[0][:, :2], dtype=np.float64)
                    e2 = np.asarray(e_rows[0], dtype=np.float64)
                    H2 = J2.T @ J2 + (self.SERVO_DLS_DAMPING ** 2) * np.eye(2, dtype=np.float64)
                    g2 = J2.T @ e2
                    try:
                        vxy_cmd = -np.linalg.solve(H2, g2).reshape(2)
                    except np.linalg.LinAlgError:
                        vxy_cmd = -(np.linalg.pinv(J2) @ e2).reshape(2)
                    vx = float(self.SERVO_GAIN_XY * vxy_cmd[0])
                    vy = float(self.SERVO_GAIN_XY * vxy_cmd[1])
                    vz_raw = 0.0

                v_xy = np.asarray([vx, vy], dtype=np.float64)
                speed_xy = float(np.linalg.norm(v_xy))
                if speed_xy > self.SERVO_MAX_LINEAR_SPEED_XY:
                    v_xy *= self.SERVO_MAX_LINEAR_SPEED_XY / speed_xy

                avg_px_error = float(sum(per_cam_errors) / len(per_cam_errors))
                metric_px_error = float(max(per_cam_errors))
                best_metric_error = min(best_metric_error, metric_px_error)

                if len(used_cameras) >= 2 and metric_px_error <= self.SERVO_Z_ENABLE_PX:
                    vz = float(np.clip(vz_raw, -self.SERVO_MAX_LINEAR_SPEED_Z, self.SERVO_MAX_LINEAR_SPEED_Z))
                else:
                    vz = 0.0

                if metric_px_error <= self.SERVO_CONVERGED_PX:
                    stable_count += 1
                else:
                    stable_count = 0

                # ---- Yaw correction from fused port orientation ----
                # The pose_base_link.orientation in each fused detection carries a real
                # board-frame yaw derived from the PnP homography solve — NOT from bbox
                # axis alignment. We read the latest fused port detection and compute the
                # yaw error between the port's orientation and the current TCP's yaw.
                wz = 0.0
                yaw_err_rad = 0.0
                port_yaw_deg = float("nan")
                tcp_yaw_deg = float("nan")

                # For yaw estimation, try the specific port first, then fall back to any sfp_port.
                # This handles the case where the named port (e.g. sfp_port_0) is not in the
                # fused topic but a nearby sfp port is — orientation is still valid.
                fused_for_yaw = self._detection_listener.find_best(
                    matcher=port_matcher,
                    require_pose=True,
                    freshness_sec=3.0,
                )
                # Bug 1 fix: fallback uses port_matcher, not hardcoded sfp_port
                if fused_for_yaw is None:
                    fused_for_yaw = self._detection_listener.find_best(
                        matcher=port_matcher,
                        require_pose=True,
                        freshness_sec=8.0,
                    )
                if fused_for_yaw is not None:
                    port_pose_yaw = self._pose_from_detection(fused_for_yaw["detection"])
                    if port_pose_yaw is not None:
                        # Extract port yaw from pose_base_link orientation (pure yaw quaternion: x=0, y=0)
                        pqz = float(port_pose_yaw.orientation.z)
                        pqw = float(port_pose_yaw.orientation.w)
                        port_yaw = 2.0 * math.atan2(pqz, pqw)

                        # Extract current TCP yaw using full Euler formula (stable for any orientation)
                        tcp_yaw = self._quat_yaw_rad(current_pose.orientation)

                        # Yaw error (target - current), wrapped to [-pi, pi]
                        yaw_err_rad = self._angle_wrap(port_yaw - tcp_yaw)
                        port_yaw_deg = math.degrees(port_yaw)
                        tcp_yaw_deg = math.degrees(tcp_yaw)

                        # Proportional angular correction, clamped
                        wz_raw = self.SERVO_YAW_GAIN * yaw_err_rad
                        wz = float(np.clip(wz_raw, -self.SERVO_MAX_YAW_RATE, self.SERVO_MAX_YAW_RATE))

                yaw_converged = abs(yaw_err_rad) <= yaw_converged_rad

                send_feedback(
                    f"mypolicy/servo_iter_{iteration}"
                    f" cams={used_cameras}"
                    f" px_err_max={metric_px_error:.1f} px_err_avg={avg_px_error:.1f}"
                    f" vx={v_xy[0]:.4f} vy={v_xy[1]:.4f} vz={vz:.4f} wz={wz:.4f}"
                    f" yaw_err_deg={math.degrees(yaw_err_rad):.2f}"
                    f" port_yaw={port_yaw_deg:.2f} tcp_yaw={tcp_yaw_deg:.2f}"
                    f" stable={stable_count}/{self.SERVO_CONVERGED_PX_STABLE_COUNT}"
                    f" yaw_conv={yaw_converged}"
                )

                if stable_count >= self.SERVO_CONVERGED_PX_STABLE_COUNT and yaw_converged:
                    self._motion_servo.stop()
                    send_feedback(
                        f"mypolicy/servo_converged px_err={metric_px_error:.1f}"
                        f" yaw_err_deg={math.degrees(yaw_err_rad):.2f}"
                    )
                    return True

                twist = Twist()
                twist.linear.x = float(v_xy[0])
                twist.linear.y = float(v_xy[1])
                twist.linear.z = float(vz)
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = float(wz)

                self._motion_servo.publish_twist_command(twist)
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            self._motion_servo.stop()

            # No valid port+plug pair available from per-camera detections.
            # Fall back to fused-topic detection (which carries pose_base_link) and
            # use world-frame XY correction to keep the TCP aligned with the port.
            # We update the cached port pose only when a fresh detection is available
            # and is within 50mm of the last known position (to avoid jumping to a wrong port).
            fused_port = self._detection_listener.find_best(
                matcher=port_matcher,
                require_pose=True,
                freshness_sec=2.0,
            )
            live_fallback_pose = self._pose_from_detection(fused_port["detection"]) if fused_port else None
            if live_fallback_pose is not None:
                if cached_port_pose is None:
                    # Make a copy, not a reference, to avoid mutations of the original
                    cached_port_pose = Pose()
                    cached_port_pose.position.x = float(live_fallback_pose.position.x)
                    cached_port_pose.position.y = float(live_fallback_pose.position.y)
                    cached_port_pose.position.z = float(live_fallback_pose.position.z)
                    cached_port_pose.orientation.x = float(live_fallback_pose.orientation.x)
                    cached_port_pose.orientation.y = float(live_fallback_pose.orientation.y)
                    cached_port_pose.orientation.z = float(live_fallback_pose.orientation.z)
                    cached_port_pose.orientation.w = float(live_fallback_pose.orientation.w)
                else:
                    # Only update cache if the new detection is within 50mm of the cached position
                    # (prevents large jumps from noisy or wrong detections)
                    cache_dx = float(live_fallback_pose.position.x) - float(cached_port_pose.position.x)
                    cache_dy = float(live_fallback_pose.position.y) - float(cached_port_pose.position.y)
                    cache_dist = math.sqrt(cache_dx * cache_dx + cache_dy * cache_dy)
                    if cache_dist <= 0.050:
                        # Smooth update: blend new detection toward cached (EMA with alpha=0.3) for all axes
                        alpha = 0.3
                        cached_port_pose.position.x = (1.0 - alpha) * float(cached_port_pose.position.x) + alpha * float(live_fallback_pose.position.x)
                        cached_port_pose.position.y = (1.0 - alpha) * float(cached_port_pose.position.y) + alpha * float(live_fallback_pose.position.y)
                        cached_port_pose.position.z = (1.0 - alpha) * float(cached_port_pose.position.z) + alpha * float(live_fallback_pose.position.z)

            fallback_pose = cached_port_pose
            if fallback_pose is not None:
                world_dx = float(fallback_pose.position.x) - float(current_pose.position.x)
                world_dy = float(fallback_pose.position.y) - float(current_pose.position.y)
                world_dist = math.sqrt(world_dx * world_dx + world_dy * world_dy)

                # Compute yaw error from cached port pose orientation for the fallback path
                fb_pqz = float(fallback_pose.orientation.z)
                fb_pqw = float(fallback_pose.orientation.w)
                fb_port_yaw = 2.0 * math.atan2(fb_pqz, fb_pqw)
                # Use full Euler yaw extraction (stable for downward-facing gripper with x≈1, w≈0)
                fb_tcp_yaw = self._quat_yaw_rad(current_pose.orientation)
                fb_yaw_err = self._angle_wrap(fb_port_yaw - fb_tcp_yaw)
                fb_yaw_converged = abs(fb_yaw_err) <= yaw_converged_rad
                # Proportional yaw correction even in fallback mode
                fb_wz_raw = self.SERVO_YAW_GAIN * fb_yaw_err
                fb_wz = float(np.clip(fb_wz_raw, -self.SERVO_MAX_YAW_RATE, self.SERVO_MAX_YAW_RATE))

                send_feedback(
                    f"mypolicy/servo_iter_{iteration} fallback_to_port err={world_dist*1000.0:.1f}mm"
                    f" yaw_err_deg={math.degrees(fb_yaw_err):.2f} yaw_conv={fb_yaw_converged}"
                    f" wz={fb_wz:.4f} cached={'live' if live_fallback_pose is not None else 'stale'}"
                )

                # Only allow fallback-based convergence if yaw is also validated
                if world_dist <= self.SERVO_WORLD_TOLERANCE and fb_yaw_converged:
                    self._motion_servo.stop()
                    send_feedback(f"mypolicy/servo_converged_fallback err={world_dist*1000.0:.1f}mm yaw_err_deg={math.degrees(fb_yaw_err):.2f}")
                    return True

                cmd = np.asarray([world_dx, world_dy], dtype=np.float64) * 0.45
                mag = float(np.linalg.norm(cmd))
                if mag > self.SERVO_MAX_LINEAR_SPEED_XY:
                    cmd *= self.SERVO_MAX_LINEAR_SPEED_XY / mag

                twist = Twist()
                twist.linear.x = float(cmd[0])
                twist.linear.y = float(cmd[1])
                twist.linear.z = 0.0
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = float(fb_wz)  # Apply yaw correction in fallback mode too
                self._motion_servo.publish_twist_command(twist)
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            send_feedback(f"mypolicy/servo_iter_{iteration} waiting_for_valid_port_and_plug")
            self.sleep_for(self.SERVO_STEP_SEC)

        self._motion_servo.stop()
        send_feedback(
            f"mypolicy/servo_max_iterations best_px_err={best_metric_error:.1f}"
            f" converged_px={self.SERVO_CONVERGED_PX}"
        )
        if best_metric_error <= self.SERVO_CONVERGED_PX:
            return True
        # Allow insertion if world-frame XY is close enough even after max iterations.
        cur_at_max = self._motion_servo.get_current_pose()
        fused_at_max = self._detection_listener.find_best(
            matcher=port_matcher, require_pose=True, freshness_sec=10.0
        )
        send_feedback(
            f"mypolicy/servo_max_iter_fused_found={fused_at_max is not None}"
            f" cur_found={cur_at_max is not None}"
        )
        if cur_at_max is not None and fused_at_max is not None:
            fp = self._pose_from_detection(fused_at_max["detection"])
            if fp is not None:
                dx = float(fp.position.x) - float(cur_at_max.position.x)
                dy = float(fp.position.y) - float(cur_at_max.position.y)
                xy_err_mm = math.sqrt(dx * dx + dy * dy) * 1000.0
                # Check yaw convergence at max-iter — match strict gate
                pqz_mi = float(fp.orientation.z)
                pqw_mi = float(fp.orientation.w)
                port_yaw_mi = 2.0 * math.atan2(pqz_mi, pqw_mi)
                tcp_yaw_mi = self._quat_yaw_rad(cur_at_max.orientation)
                yaw_err_mi = abs(self._angle_wrap(port_yaw_mi - tcp_yaw_mi))
                yaw_ok_mi = yaw_err_mi <= yaw_converged_rad
                send_feedback(
                    f"mypolicy/servo_max_iter_world_xy_err={xy_err_mm:.1f}mm"
                    f" yaw_err_deg={math.degrees(yaw_err_mi):.2f} yaw_ok={yaw_ok_mi}"
                    f" tcp_xyz=({cur_at_max.position.x:.3f},{cur_at_max.position.y:.3f},{cur_at_max.position.z:.3f})"
                    f" port_xyz=({fp.position.x:.3f},{fp.position.y:.3f},{fp.position.z:.3f})"
                )
                if xy_err_mm <= 15.0 and yaw_ok_mi:
                    send_feedback("mypolicy/servo_max_iter_close_enough proceeding to insertion")
                    return True
                send_feedback(
                    f"mypolicy/servo_max_iter_rejected xy_err={xy_err_mm:.1f}mm (need<=15) yaw_ok={yaw_ok_mi}"
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

    def _quat_yaw_rad(self, q: Quaternion) -> float:
        """Extract the Z-axis (yaw) component from a quaternion using the full Euler formula.

        This is numerically stable even for downward-facing orientations (x≈1, w≈0)
        unlike the simplified 2*atan2(qz, qw) which assumes a pure yaw rotation.
        """
        qx = float(q.x)
        qy = float(q.y)
        qz = float(q.z)
        qw = float(q.w)
        # Normalize
        n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n < 1e-12:
            return 0.0
        qx /= n; qy /= n; qz /= n; qw /= n
        # Full Euler yaw extraction (rotation around Z-axis of fixed frame)
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

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

    # ==================================================================
    # Phase 3: Force-Feedback Insertion
    # ==================================================================

    def _sfp_force_insert(
        self,
        gripper_orientation: Quaternion,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        port_face_hint_z: Optional[float] = None,
        port_matcher: Callable[[Dict], bool] = None,   # Bug 5 fix: for live XY hold
    ) -> bool:
        """Perform insertion by stepping down in velocity mode.

        Contact detection is DEPTH-BASED: the simulation FTS does not reliably register
        contact forces during plug insertion, so we use TCP Z position to determine when
        the plug has entered the port.

        Port face is estimated from:
        1. port_face_hint_z: board-surface Z from initial camera detection (most accurate)
        2. fused detection Z (also board-surface, from same pipeline)
        3. hardcoded fallback 0.17927

        'Contact' is declared when TCP descends to within PORT_FACE_APPROACH_MM of port face.
        Success is declared when 8mm of insertion depth past contact is achieved.
        """
        current_pose = self._motion_servo.get_current_pose()
        if current_pose is None:
            send_feedback("mypolicy/insert_fail no_pose")
            return False

        # ---- Tare the wrench ----
        tare_fx, tare_fy, tare_fz = 0.0, 0.0, 0.0
        tare_obs = get_observation()
        if tare_obs is not None:
            w = tare_obs.wrist_wrench.wrench
            tare_fx = float(w.force.x)
            tare_fy = float(w.force.y)
            tare_fz = float(w.force.z)
            send_feedback(f"mypolicy/insert_tare fx={tare_fx:.2f} fy={tare_fy:.2f} fz={tare_fz:.2f}")

        start_z = float(current_pose.position.z)
        # Max-depth guard: never descend more than INSERT_MAX_DEPTH from the start
        abs_min_z = start_z - self.INSERT_MAX_DEPTH
        deadline = time.monotonic() + self.INSERT_TIMEOUT_SEC

        # Determine PORT_FACE_Z: the Z of the port opening/entry face.
        # Priority: hint from initial detection > fused detection > hardcoded fallback.
        # Note: the fused detection hardcodes z=0.17927 for sfp_port (same as fallback),
        # so we prefer the hint from the board homography which gives the real board-surface Z.
        PORT_FACE_Z = 0.17927  # fallback hardcoded value
        port_face_source = "hardcoded"

        if port_face_hint_z is not None and port_face_hint_z > 0.05:
            PORT_FACE_Z = float(port_face_hint_z)
            port_face_source = f"initial_detection(z={port_face_hint_z:.4f})"
        else:
            fused_port_at_insert = self._detection_listener.find_best(
                matcher=self._is_sfp_port_detection,
                require_pose=True,
                freshness_sec=5.0,
            )
            if fused_port_at_insert is not None:
                fp_insert = self._pose_from_detection(fused_port_at_insert["detection"])
                # Only use if it looks like a real measurement (not the hardcoded 0.17927 from fused topic)
                if fp_insert is not None and float(fp_insert.position.z) > 0.19:
                    PORT_FACE_Z = float(fp_insert.position.z)
                    port_face_source = f"fused(conf={fused_port_at_insert['confidence']:.2f})"

        # Contact is declared when TCP Z drops below PORT_FACE_Z + approach_margin.
        # The approach margin of 5mm allows for detection noise while ensuring we
        # don't declare contact too early (robot still 5mm above port opening).
        PORT_FACE_APPROACH_MM = 5.0
        contact_trigger_z = PORT_FACE_Z + (PORT_FACE_APPROACH_MM / 1000.0)

        send_feedback(
            f"mypolicy/insert_start z={start_z:.4f} abs_min_z={abs_min_z:.4f}"
            f" port_face_z={PORT_FACE_Z:.4f} port_face_source={port_face_source}"
            f" contact_trigger_z={contact_trigger_z:.4f}"
            f" gap_to_port={start_z - PORT_FACE_Z:.4f}m"
        )

        insertion_started = False
        insertion_started_z = None  # Z at which depth-based contact was declared
        settled_count = 0

        while time.monotonic() < deadline:
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(self.INSERT_STEP_SEC)
                continue

            current_z = float(current_pose.position.z)

            # Hard max-depth guard using ACTUAL tcp Z (not a stale step_z)
            if current_z <= abs_min_z:
                send_feedback(f"mypolicy/insert_max_depth_reached current_z={current_z:.4f} abs_min_z={abs_min_z:.4f}")
                break

            # Monitor forces for abort conditions only (not for contact detection)
            observation = get_observation()
            if observation is not None:
                w = observation.wrist_wrench.wrench
                raw_fz = float(w.force.z) - tare_fz
                raw_fx = float(w.force.x) - tare_fx
                raw_fy = float(w.force.y) - tare_fy

                fz = abs(raw_fz)
                fx = abs(raw_fx)
                fy = abs(raw_fy)
                lateral_force = math.sqrt(fx * fx + fy * fy)

                send_feedback(
                    f"mypolicy/insert z={current_z:.4f}"
                    f" dist_to_port={current_z - PORT_FACE_Z:.4f}m"
                    f" fz={raw_fz:.2f} fx={raw_fx:.2f} fy={raw_fy:.2f} lat={lateral_force:.2f}"
                    f" ins_started={insertion_started}"
                )

                # Safety abort on extreme forces
                if lateral_force > self.INSERT_MAX_LATERAL_FORCE:
                    send_feedback(f"mypolicy/insert_abort lateral_force={lateral_force:.2f}")
                    return False

                if fz > self.INSERT_MAX_Z_FORCE:
                    send_feedback(f"mypolicy/insert_abort z_force={fz:.2f}")
                    return False

            # DEPTH-BASED contact detection: declare contact when TCP reaches contact_trigger_z.
            # This is the primary contact signal because the simulation FTS does not register
            # contact forces reliably during plug insertion.
            if not insertion_started and current_z <= contact_trigger_z:
                insertion_started = True
                insertion_started_z = current_z
                send_feedback(
                    f"mypolicy/insert_contact_depth z={current_z:.4f}"
                    f" port_face_z={PORT_FACE_Z:.4f} (depth-based contact)"
                )

            if insertion_started and insertion_started_z is not None:
                depth_past_contact = insertion_started_z - current_z
                deep_enough = depth_past_contact >= (self.INSERT_DEEP_SUCCESS_MM / 1000.0)
                if deep_enough:
                    send_feedback(
                        f"mypolicy/insert_seated_deep z={current_z:.4f}"
                        f" depth_past_contact={depth_past_contact*1000:.1f}mm"
                        f" (depth-only success, need>={self.INSERT_DEEP_SUCCESS_MM}mm)"
                    )
                    return True

            # Step down using pure velocity Twist
            # Step down with live XY correction to hold alignment during descent.
            # Bug 5 fix: pure -Z descent drifts off-center from cable drag / compliance.
            # We read the latest fused port detection and apply a small proportional
            # correction in X and Y each step to keep the plug centered on the port.
            twist = Twist()
            twist.linear.z = -0.010  # Descend at 10mm/sec

            if port_matcher is not None:
                fused_xy = self._detection_listener.find_best(
                    matcher=port_matcher, require_pose=True, freshness_sec=1.0
                )
                if fused_xy is not None:
                    fp_xy = self._pose_from_detection(fused_xy["detection"])
                    if fp_xy is not None:
                        # Proportional XY correction capped at ±5mm/s
                        twist.linear.x = float(np.clip(
                            (float(fp_xy.position.x) - float(current_pose.position.x)) * 2.0,
                            -0.005, 0.005
                        ))
                        twist.linear.y = float(np.clip(
                            (float(fp_xy.position.y) - float(current_pose.position.y)) * 2.0,
                            -0.005, 0.005
                        ))

            self._motion_servo.publish_twist_command(twist)

            self.sleep_for(self.INSERT_STEP_SEC)

        self._motion_servo.stop()
        send_feedback("mypolicy/insert_timeout")
        # Report success if we achieved >= INSERT_DEEP_SUCCESS_MM past contact depth
        if insertion_started and insertion_started_z is not None:
            current_pose = self._motion_servo.get_current_pose()
            final_z = float(current_pose.position.z) if current_pose is not None else start_z
            depth_achieved = insertion_started_z - final_z
            send_feedback(f"mypolicy/insert_timeout_depth_achieved={depth_achieved*1000:.1f}mm (need>={self.INSERT_DEEP_SUCCESS_MM}mm)")
            return depth_achieved >= (self.INSERT_DEEP_SUCCESS_MM / 1000.0)
        return False

    def _sc_force_insert(
        self,
        port_matcher: Callable[[Dict], bool],
        gripper_orientation: Quaternion,
        get_observation: GetObservationCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Horizontal insertion for SC cables.

        SC ports face outward from the taskboard rail with insertion axis along world -Y.
        The sc_port_base_link has pose (roll=π/2, pitch=π) meaning the port Z-axis
        (insertion direction) points horizontally. We move the gripper in -Y world direction.

        The gripper was already reoriented in insert_cable() so its tool-Z points in -Y.
        Here we just step in -Y and use depth-based contact detection on the Y axis.
        """
        current_pose = self._motion_servo.get_current_pose()
        if current_pose is None:
            send_feedback("mypolicy/sc_insert_fail no_pose")
            return False

        # Tare wrench
        tare_fx, tare_fy, tare_fz = 0.0, 0.0, 0.0
        tare_obs = get_observation()
        if tare_obs is not None:
            w = tare_obs.wrist_wrench.wrench
            tare_fx = float(w.force.x)
            tare_fy = float(w.force.y)
            tare_fz = float(w.force.z)
            send_feedback(f"mypolicy/sc_insert_tare fx={tare_fx:.2f} fy={tare_fy:.2f} fz={tare_fz:.2f}")

        start_y = float(current_pose.position.y)
        # SC port entrance is 15.64mm from base_link along port axis.
        # We approach from SC_HOVER_Y_STANDOFF cm away and travel at most SC_INSERT_MAX_TRAVEL.
        abs_min_y = start_y - self.SC_INSERT_MAX_TRAVEL
        deadline = time.monotonic() + self.INSERT_TIMEOUT_SEC

        # Determine port face Y from fused detection
        PORT_FACE_Y = start_y - 0.030   # default: 30mm ahead
        port_face_source = "default_offset"
        fused_port = self._detection_listener.find_best(
            matcher=port_matcher, require_pose=True, freshness_sec=5.0
        )
        if fused_port is not None:
            fp = self._pose_from_detection(fused_port["detection"])
            if fp is not None:
                PORT_FACE_Y = float(fp.position.y)
                port_face_source = f"fused(conf={fused_port['confidence']:.2f})"

        contact_trigger_y = PORT_FACE_Y + 0.005   # 5mm before port face
        insertion_started = False
        insertion_started_y = None

        send_feedback(
            f"mypolicy/sc_insert_start start_y={start_y:.4f} abs_min_y={abs_min_y:.4f} "
            f"port_face_y={PORT_FACE_Y:.4f} source={port_face_source} "
            f"contact_trigger_y={contact_trigger_y:.4f}"
        )

        while time.monotonic() < deadline:
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(self.INSERT_STEP_SEC)
                continue

            current_y = float(current_pose.position.y)

            if current_y <= abs_min_y:
                send_feedback(f"mypolicy/sc_insert_max_travel current_y={current_y:.4f}")
                break

            # Force monitoring for safety abort
            observation = get_observation()
            if observation is not None:
                w = observation.wrist_wrench.wrench
                raw_fx = float(w.force.x) - tare_fx
                raw_fy = float(w.force.y) - tare_fy
                raw_fz = float(w.force.z) - tare_fz
                lateral_force = math.sqrt(raw_fx**2 + raw_fz**2)   # X and Z are lateral for SC
                insert_force = abs(raw_fy)                           # Y is the insertion axis

                send_feedback(
                    f"mypolicy/sc_insert y={current_y:.4f} dist_to_port={current_y - PORT_FACE_Y:.4f}m "
                    f"fy={raw_fy:.2f} lat={lateral_force:.2f} ins_started={insertion_started}"
                )

                if lateral_force > self.INSERT_MAX_LATERAL_FORCE:
                    send_feedback(f"mypolicy/sc_insert_abort lateral_force={lateral_force:.2f}")
                    return False
                if insert_force > self.INSERT_MAX_Z_FORCE:
                    send_feedback(f"mypolicy/sc_insert_abort insert_force={insert_force:.2f}")
                    return False

            # Depth-based contact detection on Y axis
            if not insertion_started and current_y <= contact_trigger_y:
                insertion_started = True
                insertion_started_y = current_y
                send_feedback(
                    f"mypolicy/sc_insert_contact y={current_y:.4f} port_face_y={PORT_FACE_Y:.4f}"
                )

            if insertion_started and insertion_started_y is not None:
                depth = insertion_started_y - current_y
                if depth >= (self.INSERT_DEEP_SUCCESS_MM / 1000.0):
                    send_feedback(
                        f"mypolicy/sc_insert_success y={current_y:.4f} "
                        f"depth={depth*1000:.1f}mm"
                    )
                    return True

            # Move in -Y with live XZ hold (keep plug centered on port during insertion)
            twist = Twist()
            twist.linear.y = -self.SC_INSERT_SPEED

            fused_xy = self._detection_listener.find_best(
                matcher=port_matcher, require_pose=True, freshness_sec=1.0
            )
            if fused_xy is not None:
                fp_xy = self._pose_from_detection(fused_xy["detection"])
                if fp_xy is not None:
                    twist.linear.x = float(np.clip(
                        (float(fp_xy.position.x) - float(current_pose.position.x)) * 2.0,
                        -0.005, 0.005
                    ))
                    twist.linear.z = float(np.clip(
                        (float(fp_xy.position.z) - float(current_pose.position.z)) * 2.0,
                        -0.005, 0.005
                    ))

            self._motion_servo.publish_twist_command(twist)
            self.sleep_for(self.INSERT_STEP_SEC)

        self._motion_servo.stop()
        send_feedback("mypolicy/sc_insert_timeout")

        if insertion_started and insertion_started_y is not None:
            final_pose = self._motion_servo.get_current_pose()
            final_y = float(final_pose.position.y) if final_pose is not None else start_y
            depth_achieved = insertion_started_y - final_y
            send_feedback(
                f"mypolicy/sc_insert_timeout_depth={depth_achieved*1000:.1f}mm "
                f"(need>={self.INSERT_DEEP_SUCCESS_MM}mm)"
            )
            return depth_achieved >= (self.INSERT_DEEP_SUCCESS_MM / 1000.0)
        return False

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
        State 1: Sample for 15s to find Average port location
        State 2: Execute trajectory.
        State 3: Monitor Pixel error continually. If error diverges consistently, abort and repeat State 1.
        """
        self._motion_servo.ensure_cartesian_mode()
        total_deadline = time.monotonic() + timeout_sec
        
        while time.monotonic() < total_deadline:
            # ==============================
            # STATE 1: Sample (15s)
            # ==============================
            send_feedback(f"mypolicy/{label}_sampling (15s)")
            sample_deadline = time.monotonic() + 15.0
            
            positions_x, positions_y, positions_z = [], [], []
            
            while time.monotonic() < sample_deadline:
                all_dets = self._detection_listener.get_all_detections(freshness_sec=1.0)
                port_dets = [d for d in all_dets if matcher(d)]
                
                # The port is elevated off the board. Using angled cameras (left/right) introduces extreme
                # projection XY parallax since the homography assumes Z=0 (board surface).
                # The center camera avoids this completely because it natively points perfectly vertical.
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

            # Average out the coordinate noise
            avg_x = sum(positions_x) / len(positions_x)
            avg_y = sum(positions_y) / len(positions_y)
            avg_z = sum(positions_z) / len(positions_z)

            raw_target_pose = Pose()
            raw_target_pose.position.x = avg_x
            raw_target_pose.position.y = avg_y
            raw_target_pose.position.z = avg_z

            target_pose = self._make_target_pose(raw_target_pose, gripper_orientation, z_offset, min_z)
            send_feedback(f"mypolicy/{label}_sampled_target xyz=({avg_x:.3f},{avg_y:.3f},{avg_z:.3f})")

            # ==============================
            # STATE 2: Plan
            # ==============================
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(0.5)
                continue

            waypoints = self._planner.plan_from_current_pose(current_pose, target_pose)
            if not waypoints:
                waypoints = [target_pose]
                
            self._motion_servo.publish_target_marker(target_pose)
            self._motion_servo.publish_waypoint_visuals(waypoints)

            # ==============================
            # STATE 3: Move & Monitor
            # ==============================
            send_feedback(f"mypolicy/{label}_moving")
            
            waypoint_idx = 0
            path_aborted = False
            last_pixel_error = float('inf')
            error_increase_count = 0

            while waypoint_idx < len(waypoints) and not path_aborted and time.monotonic() < total_deadline:
                current_pose = self._motion_servo.get_current_pose()
                if current_pose is None:
                    self.sleep_for(0.05)
                    continue

                # ---- Validate Pixel Error ----
                live_dets = self._detection_listener.get_all_detections(freshness_sec=1.0)
                port_candidates = sorted([d for d in live_dets if matcher(d)], key=lambda x: -float(x.get("confidence", 0.0)))
                
                if port_candidates:
                    port_det = port_candidates[0]
                    best_cam = port_det.get("camera_name", "center")
                    
                    gripper_candidates = [d for d in live_dets if self._is_gripper_detection(d) and d.get("camera_name") == best_cam]
                    if gripper_candidates:
                        gripper_det = sorted(gripper_candidates, key=lambda x: -float(x.get("confidence", 0.0)))[0]
                        port_uv = self._feature_uv(port_det)
                        grip_uv = self._feature_uv(gripper_det)
                        if port_uv is None or grip_uv is None:
                            self.sleep_for(0.05)
                            continue

                        du = float(port_uv[0]) - float(grip_uv[0])
                        dv = float(port_uv[1]) - float(grip_uv[1])
                        pixel_error = math.sqrt(du*du + dv*dv)
                        
                        # Monitor if it is strictly reducing (with a 3px noise margin)
                        prev_pixel_error = last_pixel_error
                        if prev_pixel_error != float('inf') and pixel_error > prev_pixel_error + 20.0:
                            error_increase_count += 1
                        else:
                            error_increase_count = max(0, error_increase_count - 1)
                        last_pixel_error = pixel_error

                        if error_increase_count > 5:
                            send_feedback(f"mypolicy/abort_increase_err ({prev_pixel_error:.1f}px -> {pixel_error:.1f}px)")
                            self._motion_servo.stop()
                            path_aborted = True
                            break

                # ---- Servo Execution ----
                if self._position_distance(current_pose, target_pose) <= self._motion_servo.position_tolerance:
                    self._motion_servo.stop()
                    return True

                current_wp = waypoints[waypoint_idx]
                if self._position_distance(current_pose, current_wp) <= self._motion_servo.position_tolerance:
                    if waypoint_idx < len(waypoints) - 1:
                        waypoint_idx += 1
                        current_wp = waypoints[waypoint_idx]

                twist = self._motion_servo.compute_twist_to_waypoint(current_pose, current_wp)
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = 0.0
                self._motion_servo.publish_twist_command(twist)
                self.sleep_for(0.05)

            if path_aborted:
                # Do not return. Let the outer loop run the sampling phase again.
                continue

            # Waypoints exhausted (or total_deadline expired during traversal).
            # Only return True if the robot actually reached within position tolerance of the target.
            self._motion_servo.stop()
            current_pose_final = self._motion_servo.get_current_pose()
            if current_pose_final is not None:
                final_dist = self._position_distance(current_pose_final, target_pose)
                if final_dist <= self._motion_servo.position_tolerance:
                    return True
                send_feedback(
                    f"mypolicy/{label}_path_ended_far_from_target dist={final_dist*1000:.1f}mm (threshold={self._motion_servo.position_tolerance*1000:.0f}mm) re-sampling"
                )
                # Re-sample if there is time remaining
                continue
            return True

        self._motion_servo.stop()
        return False

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
            # Position-only convergence check.
            # Orientation is kept constant (== starting TCP orientation) so we
            # skip the angular tolerance which is unreliable at the 180° singularity.
            if pos_error <= self._motion_servo.position_tolerance:
                self._motion_servo.stop()
                return True

            twist = self._motion_servo.compute_twist_to_waypoint(current_pose, waypoint)
            # Zero out angular velocity to avoid feeding the orientation
            # singularity — we are not requesting any rotation change.
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self._motion_servo.publish_twist_command(twist)
            self.sleep_for(0.05)

        self._motion_servo.stop()
        return False

    def _make_target_pose(self, detected_pose: Pose, orientation: Quaternion, z_offset: float = 0.0, min_z: float = 0.0) -> Pose:
        """Build a target pose using the detected position and a known-safe orientation.

        Args:
            detected_pose: Pose from the perception system (position used, orientation ignored).
            orientation: The gripper orientation to command (typically the current TCP orientation).
            z_offset: Extra height above the detected z so the gripper approaches from above.
            min_z: Minimum allowed z value. Use to prevent collision with vertical components.
        """
        target_pose = Pose()
        z_raw = float(detected_pose.position.z) + float(z_offset)
        target_pose.position = Point(
            x=float(detected_pose.position.x),
            y=float(detected_pose.position.y),
            z=max(z_raw, float(min_z)),
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
