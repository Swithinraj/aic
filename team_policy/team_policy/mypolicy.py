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
        self._latest = {"time": 0.0, "detections": []}
        self.create_subscription(String, "/fused_yolo/detections_json", self._cb, 10)

    def _cb(self, msg: String) -> None:
        try:
            parsed = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(parsed, list):
            return
        detections = [dict(item) for item in parsed if isinstance(item, dict)]
        with self._lock:
            self._latest = {"time": time.monotonic(), "detections": detections}

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
                "time": float(self._latest["time"]),
                "detections": [dict(det) for det in self._latest["detections"]],
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
                        "camera_name": det.get("source", "fused"),
                        "update_time": update_time,
                        "detection": det,
                        "confidence": float(det.get("confidence", 0.0)),
                    }
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item["confidence"],
                item["update_time"],
            ),
            reverse=True,
        )
        return candidates[0]

    def get_all_detections(self, freshness_sec: float = 2.0) -> List[Dict]:
        """Return all current detections if fresh enough."""
        now = time.monotonic()
        with self._lock:
            update_time = float(self._latest["time"])
            if update_time <= 0.0 or now - update_time > freshness_sec:
                return []
            return [dict(det) for det in self._latest["detections"]]

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
    # ---- Visual servo tuning constants ----
    SERVO_HOVER_Z = 0.06              # Hover height above port for visual servo (m)
    SERVO_PIXEL_TOLERANCE = 5.0       # Pixel error threshold for alignment convergence
    SERVO_MAX_ITERATIONS = 80         # Max visual servo correction iterations
    SERVO_GAIN = 0.4                  # Proportional gain for pixel→Cartesian correction
    SERVO_STEP_SEC = 0.10             # Sleep between servo iterations
    SERVO_TIMEOUT_SEC = 15.0          # Total visual servo timeout

    # ---- Force insertion tuning constants ----
    INSERT_Z_STEP = 0.002             # Step down increment per iteration (m)
    INSERT_Z_FORCE = 3.0              # Downward feedforward force (N). Positive is DOWN in tip frame!
    INSERT_Z_STIFFNESS = 20.0         # Reduced Z stiffness during insertion
    INSERT_MAX_LATERAL_FORCE = 8.0    # Max FX/FY before declaring misalignment (N)
    INSERT_MAX_Z_FORCE = 15.0         # Max FZ before declaring jam (N)
    INSERT_SETTLED_FORCE = 1.5        # Z force threshold to detect plug seated (N)
    INSERT_MAX_DEPTH = 0.04           # Maximum insertion travel (m)
    INSERT_STEP_SEC = 0.08            # Sleep between insertion steps
    INSERT_TIMEOUT_SEC = 12.0         # Total insertion timeout
    
    _gripper_classes = {"gripper", "sfp_module", "plug"}

    # ---- Camera intrinsic defaults (overridden from camera_info) ----
    DEFAULT_FX = 450.0
    DEFAULT_FY = 450.0
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
        self._sfp_module_classes = self._parse_name_set(os.environ.get("YOLOV12_SFP_MODULE_CLASSES", "sfp_module,sfp module"))

        self.get_logger().info("mypolicy.__init__()")
        self.get_logger().info("Started internal CombinedYoloDepthPosePlanner, detection listener, and motion servo.")

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
            z_offset=self.SERVO_HOVER_Z,
            min_z=0.20,
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
        )
        if not servo_ok:
            send_feedback("mypolicy/phase2_servo_failed (continuing with best position)")

        self.sleep_for(0.5)

        # ====== PHASE 3: Force-feedback insertion ======
        send_feedback("mypolicy/phase3_force_insertion_start")
        insert_ok = self._sfp_force_insert(
            gripper_orientation=gripper_orientation,
            get_observation=get_observation,
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
        """Iteratively correct XY position using Pixel IBVS logic between the plug and the slot."""
        deadline = time.monotonic() + self.SERVO_TIMEOUT_SEC

        for iteration in range(self.SERVO_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                send_feedback("mypolicy/servo_timeout")
                return False

            all_dets = self._detection_listener.get_all_detections(freshness_sec=1.0)
            if not all_dets:
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            # Find port and gripper in the SAME camera view
            port_det = None
            gripper_det = None
            best_cam = None

            # First, find the port
            port_candidates = sorted([d for d in all_dets if port_matcher(d)], key=lambda x: -float(x.get("confidence", 0.0)))
            if not port_candidates:
                send_feedback(f"mypolicy/servo_iter_{iteration} port_not_visible")
                self.sleep_for(self.SERVO_STEP_SEC)
                continue
            
            port_det = port_candidates[0]
            best_cam = port_det.get("camera_name", "center")

            # Then find the gripper specifically in that SAME camera view
            gripper_candidates = [d for d in all_dets if self._is_gripper_detection(d) and d.get("camera_name") == best_cam]
            if gripper_candidates:
                gripper_det = sorted(gripper_candidates, key=lambda x: -float(x.get("confidence", 0.0)))[0]
            
            if gripper_det is None:
                # If we don't see the gripper, we assume image center, but this is a fallback
                fx, fy, cx, cy = self._get_camera_intrinsics(best_cam)
                target_u, target_v = cx, cy
                gripper_pose_x, gripper_pose_y = None, None
            else:
                target_u, target_v = gripper_det.get("anchor_uv", [0, 0])
                gripper_pose = self._pose_from_detection(gripper_det)
                gripper_pose_x = gripper_pose.position.x if gripper_pose else None
                gripper_pose_y = gripper_pose.position.y if gripper_pose else None

            anchor_uv = port_det.get("anchor_uv", [])
            if len(anchor_uv) != 2:
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            du = float(anchor_uv[0]) - float(target_u)
            dv = float(anchor_uv[1]) - float(target_v)
            pixel_error = math.sqrt(du * du + dv * dv)

            send_feedback(
                f"mypolicy/servo_iter_{iteration} cam={best_cam} "
                f"uv=({anchor_uv[0]:.1f},{anchor_uv[1]:.1f}) error={pixel_error:.1f}px"
            )

            if pixel_error <= self.SERVO_PIXEL_TOLERANCE:
                send_feedback(f"mypolicy/servo_converged pixel_error={pixel_error:.1f}")
                return True

            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            # To avoid manual Jacobian camera axis assumptions, we take advantage of the combined planner's robust true-metric pose projections.
            port_pose = self._pose_from_detection(port_det)
            if not port_pose:
                self.sleep_for(self.SERVO_STEP_SEC)
                continue

            # Use task-space error mapping calculated accurately from the homography pipeline projection
            if gripper_pose_x is not None and gripper_pose_y is not None:
                world_dx = float(port_pose.position.x) - gripper_pose_x
                world_dy = float(port_pose.position.y) - gripper_pose_y
            else:
                # Fallback purely on the port's target position mapped by the system
                world_dx = float(port_pose.position.x) - float(current_pose.position.x)
                world_dy = float(port_pose.position.y) - float(current_pose.position.y)

            # Apply gain and clamp
            world_dx *= self.SERVO_GAIN
            world_dy *= self.SERVO_GAIN

            max_step = 0.015
            mag = math.sqrt(world_dx * world_dx + world_dy * world_dy)
            if mag > max_step:
                world_dx = (world_dx / mag) * max_step
                world_dy = (world_dy / mag) * max_step

            corrected_pose = _copy_pose(current_pose)
            corrected_pose.position.x += world_dx
            corrected_pose.position.y += world_dy

            # Also detect and match slot orientation
            slot_yaw = self._detect_slot_orientation(port_det)
            if slot_yaw is not None:
                send_feedback(f"mypolicy/servo_slot_yaw={math.degrees(slot_yaw):.1f}deg")
                corrected_pose.orientation = Quaternion(
                    x=float(gripper_orientation.x),
                    y=float(gripper_orientation.y),
                    z=float(math.sin(slot_yaw / 2.0)),
                    w=float(math.cos(slot_yaw / 2.0)),
                )

            # Send correction twist
            twist = self._motion_servo.compute_twist_to_waypoint(current_pose, corrected_pose)
            # Zero angular velocity — we handle orientation separately
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self._motion_servo.publish_twist_command(twist)

            self.sleep_for(self.SERVO_STEP_SEC)

        send_feedback("mypolicy/servo_max_iterations_reached")
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
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Perform compliant insertion: slowly descend with force monitoring."""
        current_pose = self._motion_servo.get_current_pose()
        if current_pose is None:
            send_feedback("mypolicy/insert_fail no_pose")
            return False

        start_z = float(current_pose.position.z)
        target_z = start_z - self.INSERT_MAX_DEPTH
        deadline = time.monotonic() + self.INSERT_TIMEOUT_SEC

        send_feedback(f"mypolicy/insert_start z={start_z:.4f} target_z={target_z:.4f}")

        insertion_started = False
        settled_count = 0

        while time.monotonic() < deadline:
            current_pose = self._motion_servo.get_current_pose()
            if current_pose is None:
                self.sleep_for(self.INSERT_STEP_SEC)
                continue

            current_z = float(current_pose.position.z)

            if current_z <= target_z:
                send_feedback(f"mypolicy/insert_max_depth_reached z={current_z:.4f}")
                break

            observation = get_observation()
            if observation is not None:
                # Actual real-time FTS wrench readings!
                w = observation.wrist_wrench.wrench
                raw_fz = float(w.force.z)
                raw_fx = float(w.force.x)
                raw_fy = float(w.force.y)
                
                fz = abs(raw_fz)
                fx = abs(raw_fx)
                fy = abs(raw_fy)
                lateral_force = math.sqrt(fx * fx + fy * fy)

                send_feedback(
                    f"mypolicy/insert z={current_z:.4f} fz={raw_fz:.2f} "
                    f"fx={raw_fx:.2f} fy={raw_fy:.2f} lat={lateral_force:.2f}"
                )

                if lateral_force > self.INSERT_MAX_LATERAL_FORCE:
                    send_feedback(f"mypolicy/insert_abort lateral_force={lateral_force:.2f}")
                    self._motion_servo.stop()
                    self._retract_z(current_pose, 0.01, gripper_orientation)
                    return False

                if fz > self.INSERT_MAX_Z_FORCE:
                    send_feedback(f"mypolicy/insert_abort z_force={fz:.2f}")
                    self._motion_servo.stop()
                    self._retract_z(current_pose, 0.01, gripper_orientation)
                    return False

                if insertion_started and fz < self.INSERT_SETTLED_FORCE:
                    settled_count += 1
                    if settled_count >= 5:
                        send_feedback(f"mypolicy/insert_seated z={current_z:.4f}")
                        self._motion_servo.stop()
                        return True
                else:
                    settled_count = 0

                if fz > 1.0:
                    insertion_started = True

            descent_pose = _copy_pose(current_pose)
            descent_pose.position.z -= self.INSERT_Z_STEP
            descent_pose.orientation = gripper_orientation

            twist = self._motion_servo.compute_twist_to_waypoint(current_pose, descent_pose)
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0

            self._motion_servo.publish_compliant_insertion_command(
                twist=twist,
                z_force=self.INSERT_Z_FORCE,
                z_stiffness=self.INSERT_Z_STIFFNESS,
            )

            self.sleep_for(self.INSERT_STEP_SEC)

        self._motion_servo.stop()
        send_feedback("mypolicy/insert_timeout")
        return insertion_started  # Partial success if we at least made contact

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
                        port_uv = port_det.get("anchor_uv", [0, 0])
                        grip_uv = gripper_det.get("anchor_uv", [0, 0])
                        
                        du = float(port_uv[0]) - float(grip_uv[0])
                        dv = float(port_uv[1]) - float(grip_uv[1])
                        pixel_error = math.sqrt(du*du + dv*dv)
                        
                        # Monitor if it is strictly reducing (with a 3px noise margin)
                        if last_pixel_error != float('inf') and pixel_error > last_pixel_error + 3.0:
                            error_increase_count += 1
                        else:
                            error_increase_count = max(0, error_increase_count - 1)
                            last_pixel_error = min(last_pixel_error, pixel_error)
                            
                        if error_increase_count > 5:
                            send_feedback(f"mypolicy/abort_increase_err ({last_pixel_error:.1f}px -> {pixel_error:.1f}px)")
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
                
            # If we exhausted waypoints naturally safely
            self._motion_servo.stop()
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