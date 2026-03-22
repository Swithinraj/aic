from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration

from team_policy.planner.cartesian_planner import CartesianPlanner


class mypolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._planner = CartesianPlanner()
        self.get_logger().info("mypolicy.__init__()")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        send_feedback(f"mypolicy/start task={task.id} plug={task.plug_name} port={task.port_name}")

        observation = self._wait_for_observation(get_observation=get_observation, timeout_sec=3.0)
        if observation is None:
            self.get_logger().error("No observation received before planning.")
            send_feedback("mypolicy/fail no_observation")
            return False

        current_pose = observation.controller_state.tcp_pose
        target_pose = self._build_demo_target_from_current_pose(current_pose)
        waypoints = self._planner.plan(target_pose=target_pose, observation=observation)

        send_feedback(f"mypolicy/waypoints_generated count={len(waypoints)}")
        if not waypoints:
            send_feedback("mypolicy/fail no_waypoints")
            return False

        all_reached = True
        for index, waypoint in enumerate(waypoints, start=1):
            send_feedback(f"mypolicy/executing_waypoint {index}/{len(waypoints)}")
            self.set_pose_target(move_robot=move_robot, pose=waypoint, frame_id="base_link")
            reached = self._wait_until_position_close(
                get_observation=get_observation,
                target_pose=waypoint,
                position_tolerance=0.01,
                timeout_sec=4.0,
            )
            if not reached:
                all_reached = False
                self.get_logger().warn(f"Waypoint {index} not reached within timeout.")
                send_feedback(f"mypolicy/waypoint_timeout index={index}")
                break

        send_feedback(f"mypolicy/done success={all_reached}")
        return all_reached

    def _wait_for_observation(
        self,
        get_observation: GetObservationCallback,
        timeout_sec: float,
    ):
        deadline = self.time_now() + Duration(seconds=timeout_sec)
        while self.time_now() < deadline:
            observation = get_observation()
            if observation is not None:
                return observation
            self.sleep_for(0.05)
        return None

    def _build_demo_target_from_current_pose(self, current_pose: Pose) -> Pose:
        return Pose(
            position=Point(
                x=current_pose.position.x,
                y=current_pose.position.y,
                z=current_pose.position.z + 0.05,
            ),
            orientation=Quaternion(
                x=current_pose.orientation.x,
                y=current_pose.orientation.y,
                z=current_pose.orientation.z,
                w=current_pose.orientation.w,
            ),
        )

    def _wait_until_position_close(
        self,
        get_observation: GetObservationCallback,
        target_pose: Pose,
        position_tolerance: float,
        timeout_sec: float,
    ) -> bool:
        deadline = self.time_now() + Duration(seconds=timeout_sec)
        while self.time_now() < deadline:
            observation = get_observation()
            if observation is None:
                self.sleep_for(0.05)
                continue

            current_pose = observation.controller_state.tcp_pose
            dx = current_pose.position.x - target_pose.position.x
            dy = current_pose.position.y - target_pose.position.y
            dz = current_pose.position.z - target_pose.position.z
            distance = (dx * dx + dy * dy + dz * dz) ** 0.5
            if distance <= position_tolerance:
                return True
            self.sleep_for(0.05)
        return False