from __future__ import annotations

import os
from typing import List, Optional

import rclpy
import rclpy.duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from aic_control_interfaces.msg import ControllerState, MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Twist, Vector3, Wrench
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_geometry_msgs  # Registers transform support for PointStamped # noqa: F401

from team_policy.planner.cartesian_planner import CartesianPlanner


class PoseValidationMove(Node):
    def __init__(self) -> None:
        super().__init__("pose_validation_move")

        # Config
        self.command_frame = os.environ.get("POSE_VALIDATION_FRAME", "base_link")
        self.pose_topic = os.environ.get("POSE_VALIDATION_TOPIC", "/yolo_pose/fix/in_gripper")
        self.z_offset = float(os.environ.get("POSE_VALIDATION_Z_OFFSET", "0.05"))

        self.declare_parameter("position_tolerance", 0.012)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("linear_kp", 1.0)
        self.declare_parameter("max_linear_speed", 0.05)
        self.declare_parameter("min_linear_speed", 0.015)
        self.declare_parameter("trans_stiffness", 90.0)
        self.declare_parameter("rot_stiffness", 50.0)
        self.declare_parameter("trans_damping", 50.0)
        self.declare_parameter("rot_damping", 20.0)

        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.linear_kp = float(self.get_parameter("linear_kp").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.min_linear_speed = float(self.get_parameter("min_linear_speed").value)
        self.trans_stiffness = float(self.get_parameter("trans_stiffness").value)
        self.rot_stiffness = float(self.get_parameter("rot_stiffness").value)
        self.trans_damping = float(self.get_parameter("trans_damping").value)
        self.rot_damping = float(self.get_parameter("rot_damping").value)

        # State
        self.planner = CartesianPlanner()
        self.current_state: Optional[ControllerState] = None
        self.current_tcp_pose: Optional[Pose] = None
        self.current_target_pose: Optional[Pose] = None
        self.last_planned_target_pose: Optional[Pose] = None
        self.active_waypoints: List[Pose] = []
        self.active_waypoint_index = 0
        self.execution_active = False
        self.mode_request_sent = False
        self.zero_sent = False

        # TF
        self._tf_buffer = Buffer()

        # Create pubs/clients FIRST
        self.pose_command_pub = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)
        self.target_marker_pub = self.create_publisher(Marker, "/planner/validation_target_marker", 10)
        self.waypoint_markers_pub = self.create_publisher(MarkerArray, "/planner/validation_waypoint_markers", 10)
        self.path_pub = self.create_publisher(Path, "/planner/validation_waypoint_path", 10)
        self.change_mode_client = self.create_client(ChangeTargetMode, "/aic_controller/change_target_mode")

        # Then TF listener
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        # Then subscriptions
        self.controller_state_sub = self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._on_controller_state,
            10,
        )

        self.pose_sub = self.create_subscription(
            PointStamped,
            self.pose_topic,
            self._on_target_pose,
            10,
        )

        # Timer LAST
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        self.get_logger().info("=" * 60)
        self.get_logger().info("Pose Validation Move Node started")
        self.get_logger().info(f"Listening to PointStamped on: {self.pose_topic}")
        self.get_logger().info(f"Target command frame      : {self.command_frame}")
        self.get_logger().info(f"Z-offset                  : {self.z_offset:.3f} m")
        self.get_logger().info("Hovering above the point. Will act dynamically on receipt of poses.")
        self.get_logger().info("=" * 60)

    def _request_cartesian_mode(self) -> None:
        if not hasattr(self, "change_mode_client"):
            return
        if not self.change_mode_client.wait_for_service(timeout_sec=0.1):
            return
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN
        self.change_mode_client.call_async(request)
        self.mode_request_sent = True
        self.get_logger().info("Requested Cartesian target mode.")

    def _on_controller_state(self, msg: ControllerState) -> None:
        self.current_state = msg
        self.current_tcp_pose = msg.tcp_pose
        if not self.mode_request_sent:
            self._request_cartesian_mode()

    def _request_cartesian_mode(self) -> None:
        if not self.change_mode_client.wait_for_service(timeout_sec=0.1):
            return
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN
        self.change_mode_client.call_async(request)
        self.mode_request_sent = True
        self.get_logger().info("Requested Cartesian target mode.")

    def _on_target_pose(self, msg: PointStamped) -> None:
        if self.current_tcp_pose is None:
            self.get_logger().warn("No controller state received yet. Cannot plan.", throttle_duration_sec=2.0)
            return

        # Transform PointStamped into command frame
        try:
            # Zero out the timestamp so TF will just get the latest available transform.
            # This prevents extrapolation errors if `use_sim_time` isn't set uniformly across nodes.
            msg.header.stamp = rclpy.time.Time().to_msg()
            pt_in_cmd = self._tf_buffer.transform(
                msg,
                self.command_frame,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to transform target to '{self.command_frame}': {e}", throttle_duration_sec=2.0)
            return

        target_pose = Pose()
        target_pose.position.x = pt_in_cmd.point.x
        target_pose.position.y = pt_in_cmd.point.y
        target_pose.position.z = pt_in_cmd.point.z + self.z_offset
        # Keep the gripper orientation unchanged
        target_pose.orientation = self.current_tcp_pose.orientation

        # Do not replan if we are already close enough to the last plan and actively moving
        if self.last_planned_target_pose is not None and self.execution_active:
            if self._position_distance(target_pose, self.last_planned_target_pose) < 0.03:
                return  

        self._plan_and_start(target_pose)

    def _plan_and_start(self, target_pose: Pose) -> None:
        self.active_waypoints = self.planner.plan_from_current_pose(
            current_pose=self.current_tcp_pose,
            target_pose=target_pose,
        )
        self.active_waypoint_index = 0
        self.execution_active = len(self.active_waypoints) > 0
        self.zero_sent = False
        self.last_planned_target_pose = target_pose
        self.current_target_pose = target_pose

        self._publish_waypoint_visuals(self.active_waypoints)
        self._publish_target_marker(target_pose)

        self.get_logger().info(
            f"New target pose arrived! Planning translation to "
            f"({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f}). "
            f"Waypoints: {len(self.active_waypoints)}."
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
            self.get_logger().info("Waypoint execution complete. Robot arrived at object estimation.")
            return

        waypoint = self.active_waypoints[self.active_waypoint_index]
        distance = self._position_distance(self.current_tcp_pose, waypoint)
        if distance <= self.position_tolerance:
            self.active_waypoint_index += 1
            if self.active_waypoint_index >= len(self.active_waypoints):
                self.execution_active = False
                self.zero_sent = False
                self.get_logger().info("Final waypoint reached. Target acquired.")
            return

        twist = self._compute_twist_to_waypoint(self.current_tcp_pose, waypoint)
        self._publish_twist_command(twist)
        self.zero_sent = False

    def _compute_twist_to_waypoint(self, current_pose: Pose, waypoint: Pose) -> Twist:
        dx = waypoint.position.x - current_pose.position.x
        dy = waypoint.position.y - current_pose.position.y
        dz = waypoint.position.z - current_pose.position.z
        distance = (dx * dx + dy * dy + dz * dz) ** 0.5

        twist = Twist()
        if distance < 1e-6:
            return twist

        commanded_speed = self.linear_kp * distance
        if commanded_speed > self.max_linear_speed:
            commanded_speed = self.max_linear_speed
        elif commanded_speed < self.min_linear_speed:
            commanded_speed = self.min_linear_speed

        scale = commanded_speed / distance
        twist.linear.x = dx * scale
        twist.linear.y = dy * scale
        twist.linear.z = dz * scale
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
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
        marker.ns = "validation_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.025
        marker.scale.y = 0.025
        marker.scale.z = 0.025
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0) # magenta sphere
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
            marker.ns = "validation_waypoints"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = waypoint
            marker.scale.x = 0.015
            marker.scale.y = 0.015
            marker.scale.z = 0.015
            marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)
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
    node = PoseValidationMove()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
