from __future__ import annotations

from typing import List, Optional

import math
import rclpy
from rclpy.node import Node
import tf2_ros
from aic_control_interfaces.msg import ControllerState, MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Twist, Vector3, Wrench
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import JointState
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
    MarkerArray,
)
from urdf_parser_py.urdf import URDF

from rcl_interfaces.srv import GetParameters
from aic_control_interfaces.msg import JointMotionUpdate

import tempfile
import pinocchio as pin
import numpy as np
from geometry_msgs.msg import TransformStamped

from team_policy.planner.cartesian_planner import CartesianPlanner

def tf_to_se3(transform: TransformStamped):
    t = transform.transform.translation
    q = transform.transform.rotation

    R = pin.Quaternion(q.w, q.x, q.y, q.z).toRotationMatrix()
    p = np.array([t.x, t.y, t.z])

    return pin.SE3(R, p)

def load_pinocchio_model_from_urdf(urdf_xml: str):
    print("pin")
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
        f.write(urdf_xml.encode("utf-8"))
        path = f.name

    model = pin.buildModelFromUrdf(path)
    data = model.createData()

    return model, data

def pose_to_se3(pose):
    t = pose.position
    q = pose.orientation

    return pin.SE3(
        pin.Quaternion(q.w, q.x, q.y, q.z).toRotationMatrix(),
        np.array([t.x, t.y, t.z])
    )

def solve_ik_pinocchio(model, q_init, oMdes, ee_frame):

    q = q_init.copy()
    alpha = 0.1
    damping = 1e-6
    data = model.createData()
    for _ in range(500):        

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        oMee = data.oMf[ee_frame]

        err = pin.log(oMee.actInv(oMdes)).vector

        J = pin.computeFrameJacobian(
            model,
            data,
            q,
            ee_frame,
            pin.ReferenceFrame.LOCAL
        )

        dq = np.linalg.solve(J.T @ J + 1e-6*np.eye(len(q)), J.T @ err)

        dq = np.clip(dq, -0.05, 0.05)
        q = pin.integrate(model, q, 0.1 * dq)
    
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # T = data.oMf[ee_frame]
    # print("FK result:", T)
    # print("Target:", oMdes) 
    return q




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

        self.joint_limits = {}
        self._load_joint_limits_from_robot_description()

        self.model = None
        self.data = None
        self.ee_frame = None
        self.pinocchio_ready = False

        self.controller_state_sub = self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._on_controller_state,
            10,
        )
        self.latest_joint_state = None
        self.joint_name_to_idx = {}
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10
        )
        self.q_goal = None
        self.q_goal_pub = self.create_publisher(
            JointMotionUpdate,
            "/aic_controller/joint_commands",
            10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.T_base_to_shoulder = None
        self.tf_ready = False
        self.create_timer(1.0, self._try_get_tf)

        self.T_base_to_shoulder = None
        self.tf_ready = False

        self.target_marker_pub = self.create_publisher(Marker, "/planner/target_marker", 10)
        self.waypoint_markers_pub = self.create_publisher(MarkerArray, "/planner/waypoint_markers", 10)
        self.path_pub = self.create_publisher(Path, "/planner/waypoint_path", 10)

        self.change_mode_client = self.create_client(ChangeTargetMode, "/aic_controller/change_target_mode")
        self.server = InteractiveMarkerServer(self, "planner_target")

        self.timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        self.get_logger().info("rviz_click_to_move started with a 3D interactive target marker.")
        self.get_logger().info("This version computes goal joint states and gives it directly to the aic controller.")
        self.get_logger().info("Set RViz Fixed Frame to base_link and add an InteractiveMarkers display for '/planner_target/update'.")
        
    def _on_joint_state(self, msg: JointState):

        self.latest_joint_state = msg

        if not self.joint_name_to_idx:
            self.joint_name_to_idx = {
                name: i for i, name in enumerate(msg.name)
            }

    def _try_get_tf(self):
        if self.tf_ready:
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                "world",
                "base_link",
                rclpy.time.Time()
            )

            self.T_world_base = tf_to_se3(tf_msg)
            self.tf_ready = True
            # self.compute_tcp_calibration()

            self.get_logger().info("TF base_link → shoulder_link ready")

        except Exception as e:
            self.get_logger().warn(f"TF not ready yet: {e}")

    def _on_controller_state(self, msg: ControllerState) -> None:
        self.current_state = msg
        self.current_tcp_pose = msg.tcp_pose

        if not self.pinocchio_ready and self.latest_joint_state is not None:

            urdf_xml = self.urdf_xml
            if not urdf_xml:
                return

            full_model = load_pinocchio_model_from_urdf(urdf_xml)[0]
            

            self.js_names = set(self.latest_joint_state.name)

            # Find joints NOT present in joint_states
            lock_ids = []
            removed_joints = []

            for j in range(1, full_model.njoints):
                joint = full_model.joints[j]
                name = full_model.names[j]

                if name == "universe":
                    continue

                # Only lock joints that are NOT in joint_state
                if (name not in self.js_names) or ("gripper" in name):
                    lock_ids.append(j)
                    removed_joints.append(name)

            # Build reduced model
            self.model = pin.buildReducedModel(
                full_model,
                lock_ids,
                np.zeros(full_model.nq)
            )
            self.pin_js_names =[name for name in self.model.names if name in self.js_names]

            # Set EE frame (make sure this exists in reduced model)
            self.ee_frame = self.model.getFrameId("gripper/tcp")

            self.pinocchio_ready = True

            self.get_logger().info(f"Pinocchio IK initialized.")
            self.get_logger().info(f"Removed joints: {removed_joints}")
            self.get_logger().info(f"Final DOF (nq): {self.model.nq}")
            del self.urdf_xml

       
        if not self.mode_request_sent:
            self._request_joint_mode()
        if not self.marker_initialized:
            self._initialize_interactive_target(msg.tcp_pose)
            
    def compute_tcp_calibration(self):
        if self.current_tcp_pose is None:
            self.get_logger().warn("No ROS TCP yet")
            return

        q = self.get_q_from_joint_states()
        if q is None:
            self.get_logger().warn("No joint state yet")
            return

        # --- Pinocchio FK ---
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)

        T_pin = data.oMf[self.ee_frame]

        # --- ROS TCP ---
        ros_p = self.current_tcp_pose.position
        ros_q = self.current_tcp_pose.orientation

        T_ros = pin.SE3(
            pin.Quaternion(ros_q.w, ros_q.x, ros_q.y, ros_q.z).toRotationMatrix(),
            np.array([ros_p.x, ros_p.y, ros_p.z])
        )
        print("T", T_pin)
        print("ROS",T_ros)
        print("T transformed", self.T_world_base.inverse()*T_pin)

    def _get_robot_description(self):

        client = self.create_client(
            GetParameters,
            "/robot_state_publisher/get_parameters"
        )

        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("robot_state_publisher parameter service not available")
            return ""

        request = GetParameters.Request()
        request.names = ["robot_description"]

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if not future.result():
            return ""

        return future.result().values[0].string_value
    
    def _load_joint_limits_from_robot_description(self) -> None:
        """
        Load joint limits from the URDF stored in /robot_description.
        """

        urdf_xml = self._get_robot_description()
        self.urdf_xml = urdf_xml

        if not urdf_xml:
            self.get_logger().warn(
                "robot_description is empty. Joint limits not loaded."
            )
            return

        try:
            robot = URDF.from_xml_string(urdf_xml)

            limits = {}

            for joint in robot.joints:
                if joint.limit is None:
                    continue

                limits[joint.name] = {
                    "lower": joint.limit.lower,
                    "upper": joint.limit.upper,
                    "velocity": joint.limit.velocity,
                    "effort": joint.limit.effort,
                }

            self.joint_limits = limits

            self.get_logger().info("Loaded joint limits:")

            for name, lim in limits.items():
                self.get_logger().info(
                    f"{name}: [{lim['lower']:.3f}, {lim['upper']:.3f}] "
                    f"vel={lim['velocity']} effort={lim['effort']}"
                )

        except Exception as e:
            self.get_logger().error(f"Failed to parse URDF: {e}")


    def _request_joint_mode(self) -> None:
        if not self.change_mode_client.wait_for_service(timeout_sec=0.1):
            return
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_JOINT
        self.change_mode_client.call_async(request)
        self.mode_request_sent = True
        self.get_logger().info("Requested joint target mode.")

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
            self.get_logger().warn("No state received yet.")
            return

        # 1. Compute IK
        q_goal = self.compute_joint_goal(target_pose)

        if q_goal is None:
            self.get_logger().warn("IK failed.")
            return

        # 2. Send directly to controller
        self.q_goal = q_goal
        self.send_q_goal()
        self.execution_active = True

        self.last_planned_target_pose = target_pose

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


    def _position_distance(self, pose_a: Pose, pose_b: Pose) -> float:
        dx = pose_a.position.x - pose_b.position.x
        dy = pose_a.position.y - pose_b.position.y
        dz = pose_a.position.z - pose_b.position.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    
    def get_joint_limit(self, joint_name: str):
        return self.joint_limits.get(joint_name, None)
    
    def _on_timer(self) -> None:
        if self.current_tcp_pose is None:
            return

        if self.q_goal is None:
            return
        
        # if self.execution_active:
        #     self.send_q_goal()
    
    def get_q_from_joint_states(self):

        if self.latest_joint_state is None:
            return None

        msg = self.latest_joint_state

        # map: joint_name → position
        name_to_pos = dict(zip(msg.name, msg.position))

        q = []

        for name in self.pin_js_names:
            if "gripper" in name:
                continue

            if name not in name_to_pos:
                self.get_logger().warn(f"Joint {name} not in JointState")
                return None

            q.append(name_to_pos[name])

        return np.array(q)
    
    def compute_joint_goal(self, target_pose: Pose):
        if not self.pinocchio_ready:
            return None
        
        oMdes_base = pose_to_se3(target_pose)

        q_init = self.get_q_from_joint_states()

        if q_init is None:
            self.get_logger().warn("No joint_states yet, skipping IK")
            return None
        oMdes_world = self.T_world_base * oMdes_base       

        q_goal = solve_ik_pinocchio(
            self.model,
            np.array(q_init),
            oMdes_world,
            self.ee_frame
        )    
          

        return q_goal
    
    def send_q_goal(self):

        if self.q_goal is None:
            return
        msg = MotionUpdate()

        msg = JointMotionUpdate()

        msg.target_state.positions = self.q_goal.tolist()
        # msg.target_state.positions = [0.0, -1.57, -1.57, -1.57, 1.57, 0]
        msg.target_stiffness = ([85.0] * len(self.q_goal.tolist()))
        msg.target_damping = ([75.0] * len(self.q_goal.tolist()))
        # msg.target_stiffness = [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
        # msg.target_damping = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]

        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        self.q_goal_pub.publish(msg)

        self.get_logger().info(f"Sent q_goal: {self.q_goal}")


def main() -> None:
    rclpy.init()
    node = RvizClickToMove()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()