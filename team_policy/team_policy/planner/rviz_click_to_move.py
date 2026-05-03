from __future__ import annotations

from typing import List, Optional

import math
import rclpy
from rclpy.node import Node

from aic_control_interfaces.msg import ControllerState, MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Twist, Vector3, Wrench
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
    MarkerArray,
)

from team_policy.planner.cartesian_planner import CartesianPlanner


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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


def _quat_to_msg(q) -> Quaternion:
    q = _quat_normalize(q)
    msg = Quaternion()
    msg.x = float(q[0])
    msg.y = float(q[1])
    msg.z = float(q[2])
    msg.w = float(q[3])
    return msg


def _quat_from_axis_angle(axis, angle: float):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    ax /= n
    ay /= n
    az /= n
    s = math.sin(0.5 * angle)
    c = math.cos(0.5 * angle)
    return [ax * s, ay * s, az * s, c]


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


class RvizClickToMove(Node):
    def __init__(self) -> None:
        super().__init__("rviz_click_to_move")

        self.declare_parameter("command_frame", "base_link")
        self.declare_parameter("position_tolerance", 0.012)
        self.declare_parameter("orientation_tolerance_rad", 0.08)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("target_marker_scale", 0.14)
        self.declare_parameter("linear_kp", 1.0)
        self.declare_parameter("angular_kp", 1.2)
        self.declare_parameter("max_linear_speed", 0.05)
        self.declare_parameter("min_linear_speed", 0.015)
        self.declare_parameter("max_angular_speed", 0.8)
        self.declare_parameter("min_angular_speed", 0.08)
        self.declare_parameter("trans_stiffness", 90.0)
        self.declare_parameter("rot_stiffness", 50.0)
        self.declare_parameter("trans_damping", 50.0)
        self.declare_parameter("rot_damping", 20.0)

        self.command_frame = str(self.get_parameter("command_frame").value)
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.orientation_tolerance_rad = float(self.get_parameter("orientation_tolerance_rad").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.target_marker_scale = float(self.get_parameter("target_marker_scale").value)
        self.linear_kp = float(self.get_parameter("linear_kp").value)
        self.angular_kp = float(self.get_parameter("angular_kp").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.min_linear_speed = float(self.get_parameter("min_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_angular_speed = float(self.get_parameter("min_angular_speed").value)
        self.trans_stiffness = float(self.get_parameter("trans_stiffness").value)
        self.rot_stiffness = float(self.get_parameter("rot_stiffness").value)
        self.trans_damping = float(self.get_parameter("trans_damping").value)
        self.rot_damping = float(self.get_parameter("rot_damping").value)

        self.planner = CartesianPlanner()
        self.current_state: Optional[ControllerState] = None
        self.current_tcp_pose: Optional[Pose] = None
        self.current_target_pose: Optional[Pose] = None
        self.last_planned_target_pose: Optional[Pose] = None
        self.active_waypoints: List[Pose] = []
        self.active_waypoint_index = 0
        self.execution_active = False
        self.mode_request_sent = False
        self.marker_initialized = False
        self.zero_sent = False

        self.controller_state_sub = self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._on_controller_state,
            10,
        )

        self.pose_command_pub = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)
        self.target_marker_pub = self.create_publisher(Marker, "/planner/target_marker", 10)
        self.waypoint_markers_pub = self.create_publisher(MarkerArray, "/planner/waypoint_markers", 10)
        self.path_pub = self.create_publisher(Path, "/planner/waypoint_path", 10)

        self.change_mode_client = self.create_client(ChangeTargetMode, "/aic_controller/change_target_mode")
        self.server = InteractiveMarkerServer(self, "planner_target")

        self.timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        self.get_logger().info("rviz_click_to_move started with 6-DoF interactive target marker.")
        self.get_logger().info("Position and end-effector orientation are both executed with Cartesian velocity servo.")
        self.get_logger().info("Add InteractiveMarkers display for '/planner_target/update' in RViz.")

    def _on_controller_state(self, msg: ControllerState) -> None:
        self.current_state = msg
        self.current_tcp_pose = msg.tcp_pose
        if not self.mode_request_sent:
            self._request_cartesian_mode()
        if not self.marker_initialized:
            self._initialize_interactive_target(msg.tcp_pose)

    def _request_cartesian_mode(self) -> None:
        if not self.change_mode_client.wait_for_service(timeout_sec=0.1):
            return
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN
        self.change_mode_client.call_async(request)
        self.mode_request_sent = True
        self.get_logger().info("Requested Cartesian target mode.")

    def _initialize_interactive_target(self, tcp_pose: Pose) -> None:
        target_pose = Pose()
        target_pose.position.x = tcp_pose.position.x
        target_pose.position.y = tcp_pose.position.y
        target_pose.position.z = min(tcp_pose.position.z + 0.05, 0.55)
        target_pose.orientation = tcp_pose.orientation

        self.current_target_pose = target_pose
        self._make_interactive_marker(target_pose)
        self._publish_target_marker(target_pose)
        self.marker_initialized = True
        self.get_logger().info(
            f"Interactive target initialized at ({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f})."
        )

    def _make_interactive_marker(self, pose: Pose) -> None:
        marker = InteractiveMarker()
        marker.header.frame_id = self.command_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.name = "planner_target"
        marker.description = "drag / rotate and release to execute"
        marker.scale = self.target_marker_scale
        marker.pose = pose

        center_control = InteractiveMarkerControl()
        center_control.always_visible = True
        center_control.interaction_mode = InteractiveMarkerControl.NONE
        center_control.name = "target_visual"
        center_control.markers.extend(self._make_center_markers())
        marker.controls.append(center_control)

        marker.controls.append(self._make_move_axis_control("move_x", 1.0, 1.0, 0.0, 0.0))
        marker.controls.append(self._make_move_axis_control("move_y", 1.0, 0.0, 1.0, 0.0))
        marker.controls.append(self._make_move_axis_control("move_z", 1.0, 0.0, 0.0, 1.0))

        marker.controls.append(self._make_rotate_axis_control("rotate_x", 1.0, 1.0, 0.0, 0.0))
        marker.controls.append(self._make_rotate_axis_control("rotate_y", 1.0, 0.0, 1.0, 0.0))
        marker.controls.append(self._make_rotate_axis_control("rotate_z", 1.0, 0.0, 0.0, 1.0))

        marker.controls.append(self._make_move_rotate_3d_control())

        self.server.insert(marker, feedback_callback=self._on_marker_feedback)
        self.server.applyChanges()

    def _make_center_markers(self) -> List[Marker]:
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = 0.03
        sphere.scale.y = 0.03
        sphere.scale.z = 0.03
        sphere.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9)

        x_axis = Marker()
        x_axis.type = Marker.ARROW
        x_axis.scale.x = 0.06
        x_axis.scale.y = 0.008
        x_axis.scale.z = 0.008
        x_axis.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
        x_axis.pose.orientation = _quat_to_msg(_quat_from_axis_angle([0.0, 0.0, 1.0], 0.0))

        y_axis = Marker()
        y_axis.type = Marker.ARROW
        y_axis.scale.x = 0.06
        y_axis.scale.y = 0.008
        y_axis.scale.z = 0.008
        y_axis.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9)
        y_axis.pose.orientation = _quat_to_msg(_quat_from_axis_angle([0.0, 0.0, 1.0], math.pi * 0.5))

        z_axis = Marker()
        z_axis.type = Marker.ARROW
        z_axis.scale.x = 0.06
        z_axis.scale.y = 0.008
        z_axis.scale.z = 0.008
        z_axis.color = ColorRGBA(r=0.0, g=0.4, b=1.0, a=0.9)
        z_axis.pose.orientation = _quat_to_msg(_quat_multiply(
            _quat_from_axis_angle([0.0, 1.0, 0.0], -math.pi * 0.5),
            [0.0, 0.0, 0.0, 1.0],
        ))

        return [sphere, x_axis, y_axis, z_axis]

    def _make_move_axis_control(self, name: str, w: float, x: float, y: float, z: float) -> InteractiveMarkerControl:
        control = InteractiveMarkerControl()
        control.name = name
        control.orientation = Quaternion(w=w, x=x, y=y, z=z)
        control.orientation_mode = InteractiveMarkerControl.INHERIT
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.always_visible = False
        return control

    def _make_rotate_axis_control(self, name: str, w: float, x: float, y: float, z: float) -> InteractiveMarkerControl:
        control = InteractiveMarkerControl()
        control.name = name
        control.orientation = Quaternion(w=w, x=x, y=y, z=z)
        control.orientation_mode = InteractiveMarkerControl.INHERIT
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control.always_visible = False
        return control

    def _make_move_rotate_3d_control(self) -> InteractiveMarkerControl:
        control = InteractiveMarkerControl()
        control.name = "move_rotate_3d"
        control.orientation_mode = InteractiveMarkerControl.INHERIT
        control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        control.always_visible = False
        return control

    def _on_marker_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        self.current_target_pose = feedback.pose
        self._publish_target_marker(feedback.pose)

        if feedback.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return

        if self.last_planned_target_pose is not None:
            pos_same = self._position_distance(feedback.pose, self.last_planned_target_pose) < 0.003
            _, ang_same = _quat_error_rotvec(feedback.pose.orientation, self.last_planned_target_pose.orientation)
            if pos_same and ang_same < 0.02:
                return

        self._plan_and_start(feedback.pose)

    def _plan_and_start(self, target_pose: Pose) -> None:
        if self.current_tcp_pose is None:
            self.get_logger().warn("No controller state received yet. Cannot plan.")
            return

        plan_target = Pose()
        plan_target.position.x = target_pose.position.x
        plan_target.position.y = target_pose.position.y
        plan_target.position.z = target_pose.position.z
        plan_target.orientation = target_pose.orientation

        self.active_waypoints = self.planner.plan_from_current_pose(
            current_pose=self.current_tcp_pose,
            target_pose=plan_target,
        )
        if len(self.active_waypoints) == 0:
            self.active_waypoints = [plan_target]

        self.active_waypoint_index = 0
        self.execution_active = len(self.active_waypoints) > 0
        self.zero_sent = False
        self.last_planned_target_pose = plan_target

        self._publish_waypoint_visuals(self.active_waypoints)
        self._publish_target_marker(plan_target)

        self.get_logger().info(
            f"Target set to ({plan_target.position.x:.3f}, {plan_target.position.y:.3f}, {plan_target.position.z:.3f}) "
            f"with orientation ({plan_target.orientation.x:.3f}, {plan_target.orientation.y:.3f}, "
            f"{plan_target.orientation.z:.3f}, {plan_target.orientation.w:.3f}). "
            f"Generated {len(self.active_waypoints)} waypoints."
        )

    def _on_timer(self) -> None:
        if self.current_tcp_pose is None:
            return

        if not self.execution_active:
            if not self.zero_sent:
                self._publish_twist_command(Twist())
                self.zero_sent = True
            return

        if self.active_waypoint_index >= len(self.active_waypoints):
            self.execution_active = False
            self.zero_sent = False
            self.get_logger().info("Waypoint execution complete.")
            return

        waypoint = self.active_waypoints[self.active_waypoint_index]
        pos_error = self._position_distance(self.current_tcp_pose, waypoint)
        _, ang_error = _quat_error_rotvec(self.current_tcp_pose.orientation, waypoint.orientation)

        if pos_error <= self.position_tolerance and ang_error <= self.orientation_tolerance_rad:
            self.active_waypoint_index += 1
            if self.active_waypoint_index < len(self.active_waypoints):
                self.get_logger().info(
                    f"Advancing to waypoint {self.active_waypoint_index + 1}/{len(self.active_waypoints)}"
                )
            else:
                self.execution_active = False
                self.zero_sent = False
                self.get_logger().info("Final waypoint reached.")
            return

        twist = self._compute_twist_to_waypoint(self.current_tcp_pose, waypoint)
        self._publish_twist_command(twist)
        self.zero_sent = False

    def _compute_twist_to_waypoint(self, current_pose: Pose, waypoint: Pose) -> Twist:
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

    def _publish_twist_command(self, twist: Twist) -> None:
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

    def _publish_target_marker(self, pose: Pose) -> None:
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

    def _publish_waypoint_visuals(self, waypoints: List[Pose]) -> None:
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

    def _position_distance(self, pose_a: Pose, pose_b: Pose) -> float:
        dx = pose_a.position.x - pose_b.position.x
        dy = pose_a.position.y - pose_b.position.y
        dz = pose_a.position.z - pose_b.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


def main() -> None:
    rclpy.init()
    node = RvizClickToMove()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()