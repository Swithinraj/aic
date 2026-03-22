from __future__ import annotations

from typing import List, Optional

import rclpy
from rclpy.node import Node

from aic_control_interfaces.msg import ControllerState, MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
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


class RvizClickToMove(Node):
    def __init__(self) -> None:
        super().__init__("rviz_click_to_move")

        self.declare_parameter("command_frame", "base_link")
        self.declare_parameter("position_tolerance", 0.01)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("target_marker_scale", 0.10)

        self.command_frame = str(self.get_parameter("command_frame").value)
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.target_marker_scale = float(self.get_parameter("target_marker_scale").value)

        self.planner = CartesianPlanner()
        self.current_tcp_pose: Optional[Pose] = None
        self.current_target_pose: Optional[Pose] = None
        self.active_waypoints: List[Pose] = []
        self.active_waypoint_index = 0
        self.execution_active = False
        self.mode_request_sent = False
        self.marker_initialized = False

        self.controller_state_sub = self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._on_controller_state,
            10,
        )

        self.pose_command_pub = self.create_publisher(
            MotionUpdate,
            "/aic_controller/pose_commands",
            10,
        )
        self.target_marker_pub = self.create_publisher(Marker, "/planner/target_marker", 10)
        self.waypoint_markers_pub = self.create_publisher(MarkerArray, "/planner/waypoint_markers", 10)
        self.path_pub = self.create_publisher(Path, "/planner/waypoint_path", 10)

        self.change_mode_client = self.create_client(ChangeTargetMode, "/aic_controller/change_target_mode")
        self.server = InteractiveMarkerServer(self, "planner_target")

        self.timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        self.get_logger().info("rviz_click_to_move started with a 3D interactive target marker.")
        self.get_logger().info("Set RViz Fixed Frame to base_link and add an InteractiveMarkers display for '/planner_target/update'.")
        self.get_logger().info("Drag the target marker in 3D space and release the mouse to plan and execute.")

    def _on_controller_state(self, msg: ControllerState) -> None:
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
        target_pose.position.z = tcp_pose.position.z + 0.05
        target_pose.orientation = tcp_pose.orientation

        self.current_target_pose = target_pose
        self._make_interactive_marker(target_pose)
        self._publish_target_marker(target_pose)
        self.marker_initialized = True
        self.get_logger().info(
            f"Interactive target initialized at ({target_pose.position.x:.3f}, "
            f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f})."
        )

    def _make_interactive_marker(self, pose: Pose) -> None:
        marker = InteractiveMarker()
        marker.header.frame_id = self.command_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.name = "planner_target"
        marker.description = "drag and release to execute"
        marker.scale = self.target_marker_scale
        marker.pose = pose

        center_control = InteractiveMarkerControl()
        center_control.always_visible = True
        center_control.interaction_mode = InteractiveMarkerControl.BUTTON
        center_control.name = "target_visual"
        center_control.markers.append(self._make_center_marker())
        marker.controls.append(center_control)

        marker.controls.append(self._make_move_axis_control("move_x", 1.0, 1.0, 0.0, 0.0))
        marker.controls.append(self._make_move_axis_control("move_y", 1.0, 0.0, 1.0, 0.0))
        marker.controls.append(self._make_move_axis_control("move_z", 1.0, 0.0, 0.0, 1.0))

        self.server.insert(marker, feedback_callback=self._on_marker_feedback)
        self.server.applyChanges()

    def _make_center_marker(self) -> Marker:
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = 0.035
        marker.scale.y = 0.035
        marker.scale.z = 0.035
        marker.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9)
        return marker

    def _make_move_axis_control(self, name: str, w: float, x: float, y: float, z: float) -> InteractiveMarkerControl:
        control = InteractiveMarkerControl()
        control.name = name
        control.orientation = Quaternion(w=w, x=x, y=y, z=z)
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.always_visible = False
        return control

    def _on_marker_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        self.current_target_pose = feedback.pose
        self._publish_target_marker(feedback.pose)

        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self._plan_and_start(feedback.pose)
        elif feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            self._plan_and_start(feedback.pose)

    def _plan_and_start(self, target_pose: Pose) -> None:
        if self.current_tcp_pose is None:
            self.get_logger().warn("No controller state received yet. Cannot plan.")
            return

        plan_target = Pose()
        plan_target.position.x = target_pose.position.x
        plan_target.position.y = target_pose.position.y
        plan_target.position.z = target_pose.position.z
        plan_target.orientation = self.current_tcp_pose.orientation

        self.active_waypoints = self.planner.plan_from_current_pose(
            current_pose=self.current_tcp_pose,
            target_pose=plan_target,
        )
        self.active_waypoint_index = 0
        self.execution_active = len(self.active_waypoints) > 0

        self._publish_waypoint_visuals(self.active_waypoints)
        self._publish_target_marker(plan_target)

        self.get_logger().info(
            f"Target set to ({plan_target.position.x:.3f}, {plan_target.position.y:.3f}, {plan_target.position.z:.3f}). "
            f"Generated {len(self.active_waypoints)} waypoints."
        )

    def _on_timer(self) -> None:
        if not self.execution_active:
            return
        if self.current_tcp_pose is None:
            return
        if self.active_waypoint_index >= len(self.active_waypoints):
            self.execution_active = False
            self.get_logger().info("Waypoint execution complete.")
            return

        waypoint = self.active_waypoints[self.active_waypoint_index]
        self._publish_motion_command(waypoint)

        if self._position_distance(self.current_tcp_pose, waypoint) <= self.position_tolerance:
            self.active_waypoint_index += 1
            if self.active_waypoint_index < len(self.active_waypoints):
                self.get_logger().info(
                    f"Advancing to waypoint {self.active_waypoint_index + 1}/{len(self.active_waypoints)}"
                )
            else:
                self.execution_active = False
                self.get_logger().info("Final waypoint reached.")

    def _publish_motion_command(self, pose: Pose) -> None:
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.command_frame
        msg.pose = pose
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        self.pose_command_pub.publish(msg)

    def _publish_target_marker(self, pose: Pose) -> None:
        marker = Marker()
        marker.header.frame_id = self.command_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "planner_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
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
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = waypoint
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
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
        return (dx * dx + dy * dy + dz * dz) ** 0.5


def main() -> None:
    rclpy.init()
    node = RvizClickToMove()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
