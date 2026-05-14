import math
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

import threading
import rclpy
from izus_policy.perception.yolov12_detector import YoloV12MultiCameraDetector


# ---------------------------------------------------------------------------
# Quaternion helpers — all in wxyz convention (transforms3d standard)
# ---------------------------------------------------------------------------

def _msg_to_wxyz(q_msg):
    """geometry_msgs Quaternion (xyzw storage) -> (w, x, y, z) tuple."""
    return (float(q_msg.w), float(q_msg.x), float(q_msg.y), float(q_msg.z))


def _wxyz_to_msg(q):
    """(w, x, y, z) -> geometry_msgs Quaternion."""
    return Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))


def _quat_conjugate_wxyz(q):
    """True unit-quaternion inverse in wxyz: (w, -x, -y, -z).
    Used for coordinate-frame transforms (q_grip_to_plug sampling).
    NOT used for CheatCode pose algebra — see _cheatcode_plug_inverse_wxyz."""
    return (float(q[0]), -float(q[1]), -float(q[2]), -float(q[3]))


def _cheatcode_plug_inverse_wxyz(q_plug):
    """Exact CheatCode plug-inverse convention: (-w, x, y, z).
    Reproduces CheatCode.calc_gripper_pose line:
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
    This must not be replaced with the mathematically correct conjugate."""
    return (-float(q_plug[0]), float(q_plug[1]), float(q_plug[2]), float(q_plug[3]))


def _quat_normalize(q):
    arr = np.array(q, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return arr / n


def _quat_to_rot(q_wxyz):
    """3x3 rotation matrix from wxyz quaternion."""
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _quat_avg_wxyz(quats):
    """Robust average of unit quaternions in wxyz convention.
    Aligns signs to the first sample, averages, then normalises."""
    arr = np.array([np.array(q, dtype=np.float64) for q in quats])
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)
    ref = arr[0]
    for i in range(1, len(arr)):
        if np.dot(arr[i], ref) < 0.0:
            arr[i] = -arr[i]
    mean = arr.mean(axis=0)
    n = float(np.linalg.norm(mean))
    return mean / n if n > 1e-12 else arr[0]


def _q_delta_deg(q_a, q_b) -> float:
    """Angle in degrees between two orientations (wxyz)."""
    dot = float(min(1.0, abs(
        float(q_a[0]) * float(q_b[0]) +
        float(q_a[1]) * float(q_b[1]) +
        float(q_a[2]) * float(q_b[2]) +
        float(q_a[3]) * float(q_b[3])
    )))
    return float(np.degrees(2.0 * np.arccos(dot)))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class run_hybrid(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.010
        self._task = None
        # Frozen reference — set by _sample_stable_yolo_reference before any motion
        self._frozen_port_pos = None           # (3,) in base_link
        self._frozen_port_quat_wxyz = None     # (4,) wxyz
        self._tcp_to_plug_local = None         # (3,) plug offset in gripper-local frame
        self._q_gripper_to_plug_wxyz = None    # (4,) wxyz — relative orientation
        self._frozen_initial_tip_xy_error = None  # (2,) frozen port_xy - plug_xy at sample time
        self._log_cmd_count = 0
        super().__init__(parent_node)
        self.get_logger().info(
            "YOLO_CHEAT_CONTROL_FRAME command_pose_frame=base_link controlled_frame=gripper/tcp"
        )
        self.get_logger().info(
            "YOLO_CHEAT_MATH mode=official_cheatcode_quaternion"
        )
        self.get_logger().info(
            "YOLO_CHEAT_VERSION 2026-05-14.yolo-tf-pbvs-v2-yclamp-deepfloor"
        )
        
        # Spawn YOLO detector in the same process to avoid Zenoh IPC failure
        self.yolo_node = YoloV12MultiCameraDetector()
        self.yolo_thread = threading.Thread(target=rclpy.spin, args=(self.yolo_node,), daemon=True)
        self.yolo_thread.start()
        self.get_logger().info("Spawned internal YoloV12MultiCameraDetector thread")

    # -----------------------------------------------------------------------
    # TF helpers
    # -----------------------------------------------------------------------

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 20.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'..."
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def _lookup_transform(self, source_frame: str):
        return self._parent_node._tf_buffer.lookup_transform("base_link", source_frame, Time())

    # -----------------------------------------------------------------------
    # Task → YOLO frame names
    # -----------------------------------------------------------------------

    def _port_frame_for_task(self, task) -> str:
        port_type = str(getattr(task, "port_type", "")).strip().lower()
        port_name = str(getattr(task, "port_name", "")).strip().lower()
        if port_type == "sfp":
            return f"yolo_tri/sfp_port/{port_name}"
        if port_type == "sc":
            return f"yolo_tri/sc_port/{port_name}"
        raise ValueError(f"Unknown port_type: {port_type!r}")

    def _plug_frame_for_task(self, task) -> str:
        plug_type = str(getattr(task, "plug_type", "")).strip().lower()
        if plug_type == "sfp":
            return "yolo_tri/sfp_module/sfp_module"
        if plug_type == "sc":
            return "yolo_tri/sc_plug/sc_plug"
        raise ValueError(f"Unknown plug_type: {plug_type!r}")

    # -----------------------------------------------------------------------
    # Stable YOLO snapshot — called once while robot is stationary
    # -----------------------------------------------------------------------

    def _sample_stable_yolo_reference(
        self, port_frame: str, plug_frame: str,
        duration_s: float = 10.0, rate_hz: float = 20.0
    ) -> bool:
        self.get_logger().info(f"YOLO_CHEAT_SAMPLE_START duration={duration_s}")

        port_pos_list = []
        port_quat_list = []
        plug_pos_list = []
        tcp_to_plug_list = []
        q_gtp_list = []   # gripper-to-plug relative quaternions (correct math)

        step_s = 1.0 / rate_hz
        n_steps = int(duration_s * rate_hz)

        for _ in range(n_steps):
            try:
                port_tf = self._lookup_transform(port_frame)
                plug_tf = self._lookup_transform(plug_frame)
                grip_tf = self._lookup_transform("gripper/tcp")
            except TransformException:
                self.sleep_for(step_s)
                continue

            port_pos = np.array([
                port_tf.transform.translation.x,
                port_tf.transform.translation.y,
                port_tf.transform.translation.z,
            ])
            q_port = _msg_to_wxyz(port_tf.transform.rotation)

            plug_pos = np.array([
                plug_tf.transform.translation.x,
                plug_tf.transform.translation.y,
                plug_tf.transform.translation.z,
            ])
            q_plug = _msg_to_wxyz(plug_tf.transform.rotation)

            grip_pos = np.array([
                grip_tf.transform.translation.x,
                grip_tf.transform.translation.y,
                grip_tf.transform.translation.z,
            ])
            q_grip = _msg_to_wxyz(grip_tf.transform.rotation)

            R_grip = _quat_to_rot(q_grip)
            tcp_to_plug = R_grip.T @ (plug_pos - grip_pos)
            # Correct conjugate used here — this is a frame-relative transform, not CheatCode pose math
            q_gtp = _quat_normalize(quaternion_multiply(_quat_conjugate_wxyz(q_grip), q_plug))

            port_pos_list.append(port_pos)
            port_quat_list.append(np.array(q_port))
            plug_pos_list.append(plug_pos)
            tcp_to_plug_list.append(tcp_to_plug)
            q_gtp_list.append(q_gtp)

            self.sleep_for(step_s)

        n_raw = len(port_pos_list)
        if n_raw == 0:
            self.get_logger().error("YOLO_CHEAT_SAMPLE failed: no valid samples collected")
            return False

        port_pos_arr = np.array(port_pos_list)
        plug_pos_arr = np.array(plug_pos_list)
        tcp_arr = np.array(tcp_to_plug_list)

        # Outlier rejection: drop samples > 2 cm from their respective medians
        port_median = np.median(port_pos_arr, axis=0)
        tcp_median = np.median(tcp_arr, axis=0)
        port_mask = np.linalg.norm(port_pos_arr - port_median, axis=1) < 0.02
        tcp_mask = np.linalg.norm(tcp_arr - tcp_median, axis=1) < 0.02
        indices = np.where(port_mask & tcp_mask)[0]
        n_used = len(indices)

        self.get_logger().info(f"YOLO_CHEAT_SAMPLE_DONE n_raw={n_raw} n_used={n_used}")

        if n_used < 20:
            self.get_logger().error(
                f"YOLO_CHEAT_SAMPLE failed: only {n_used} valid samples after filtering (need 20)"
            )
            return False

        self._frozen_port_pos = np.median(port_pos_arr[indices], axis=0)
        self._frozen_port_quat_wxyz = _quat_avg_wxyz([port_quat_list[i] for i in indices])
        self._tcp_to_plug_local = np.median(tcp_arr[indices], axis=0)
        self._q_gripper_to_plug_wxyz = _quat_avg_wxyz([q_gtp_list[i] for i in indices])

        # Frozen initial XY error used during interpolation to suppress lateral drift
        frozen_plug_xy = np.median(plug_pos_arr[indices], axis=0)[:2]
        self._frozen_initial_tip_xy_error = (
            float(self._frozen_port_pos[0] - frozen_plug_xy[0]),
            float(self._frozen_port_pos[1] - frozen_plug_xy[1]),
        )

        p = self._frozen_port_pos
        q = self._frozen_port_quat_wxyz
        t = self._tcp_to_plug_local
        qg = self._q_gripper_to_plug_wxyz

        self.get_logger().info(
            f"YOLO_CHEAT_FROZEN_PORT "
            f"pos=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) "
            f"quat=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})"
        )
        self.get_logger().info(
            f"YOLO_CHEAT_FROZEN_TCP_TO_PLUG_LOCAL "
            f"offset=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f}) "
            f"norm={float(np.linalg.norm(t)):.4f}"
        )
        self.get_logger().info(
            f"YOLO_CHEAT_FROZEN_GRIPPER_TO_PLUG_QUAT "
            f"quat=({qg[0]:+.4f},{qg[1]:+.4f},{qg[2]:+.4f},{qg[3]:+.4f})"
        )
        self.get_logger().info(
            f"YOLO_CHEAT_FROZEN_INITIAL_XY_ERROR "
            f"ex={self._frozen_initial_tip_xy_error[0]:+.4f} "
            f"ey={self._frozen_initial_tip_xy_error[1]:+.4f}"
        )

        offset_norm = float(np.linalg.norm(self._tcp_to_plug_local))
        if not (0.005 <= offset_norm <= 0.12):
            self.get_logger().warn(
                f"YOLO_CHEAT_WARN tcp_to_plug_local norm={offset_norm:.4f} "
                f"outside expected range [0.005, 0.12]"
            )

        return True

    # -----------------------------------------------------------------------
    # Motion command — frozen YOLO snapshot + live gripper/tcp only
    # -----------------------------------------------------------------------

    def calc_gripper_pose_from_frozen(
        self,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.2,
        reset_xy_integrator: bool = False,
        freeze_xy_error: bool = False,
    ) -> Pose:
        assert (
            self._frozen_port_pos is not None
            and self._frozen_port_quat_wxyz is not None
            and self._tcp_to_plug_local is not None
            and self._q_gripper_to_plug_wxyz is not None
            and self._frozen_initial_tip_xy_error is not None
        ), "call _sample_stable_yolo_reference before calc_gripper_pose_from_frozen"

        # --- live gripper state (only TF lookup during motion) ---
        grip_tf = self._lookup_transform("gripper/tcp")
        grip_pos = np.array([
            grip_tf.transform.translation.x,
            grip_tf.transform.translation.y,
            grip_tf.transform.translation.z,
        ])
        q_grip = _msg_to_wxyz(grip_tf.transform.rotation)

        # --- reconstruct plug pose from frozen gripper-relative transform ---
        R_grip = _quat_to_rot(q_grip)
        plug_pos_est = grip_pos + R_grip @ self._tcp_to_plug_local
        q_plug_est = _quat_normalize(
            quaternion_multiply(q_grip, self._q_gripper_to_plug_wxyz)
        )
        self._last_plug_est = plug_pos_est.copy()

        # --- orientation: exact CheatCode algebra ---
        # CheatCode:  q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        #             q_diff = quaternion_multiply(q_port, q_plug_inv)
        #             q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_port = self._frozen_port_quat_wxyz
        q_plug_inv = _cheatcode_plug_inverse_wxyz(q_plug_est)
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        q_grip_target = quaternion_multiply(q_diff, q_grip)
        q_grip_slerp = quaternion_slerp(q_grip, q_grip_target, slerp_fraction)

        # --- position: CheatCode style ---
        # During approach rotation the reconstructed plug tip can swing laterally.
        # freeze_xy_error=True pins XY correction to the pre-motion snapshot.
        if freeze_xy_error:
            tip_x_error = self._frozen_initial_tip_xy_error[0]
            tip_y_error = self._frozen_initial_tip_xy_error[1]
        else:
            tip_x_error = float(self._frozen_port_pos[0] - plug_pos_est[0])
            tip_y_error = float(self._frozen_port_pos[1] - plug_pos_est[1])

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0


        R_target = _quat_to_rot(q_grip_slerp)

        desired_plug_pos = np.array([
            self._frozen_port_pos[0],
            self._frozen_port_pos[1],
            self._frozen_port_pos[2] + z_offset,
        ], dtype=np.float64)

        target_grip_pos = desired_plug_pos - R_target @ self._tcp_to_plug_local

        target_x = float(target_grip_pos[0])
        target_y = float(target_grip_pos[1])
        target_z = float(target_grip_pos[2])

        bx = position_fraction * target_x + (1.0 - position_fraction) * float(grip_pos[0])
        by = position_fraction * target_y + (1.0 - position_fraction) * float(grip_pos[1])
        bz = position_fraction * target_z + (1.0 - position_fraction) * float(grip_pos[2])

        delta_deg = _q_delta_deg(q_grip, q_grip_target)

        # Verbose log for first 5 commands and every 20 thereafter
        if self._log_cmd_count < 5 or self._log_cmd_count % 20 == 0:
            self.get_logger().info(
                f"YOLO_CHEAT_CMD #{self._log_cmd_count} "
                f"tcp=({float(grip_pos[0]):+.4f},{float(grip_pos[1]):+.4f},{float(grip_pos[2]):+.4f}) "
                f"plug_est=({float(plug_pos_est[0]):+.4f},{float(plug_pos_est[1]):+.4f},{float(plug_pos_est[2]):+.4f}) "
                f"frozen_port=({float(self._frozen_port_pos[0]):+.4f},{float(self._frozen_port_pos[1]):+.4f},{float(self._frozen_port_pos[2]):+.4f}) "
                f"target=({bx:+.4f},{by:+.4f},{bz:+.4f}) "
                f"xy_err=({tip_x_error:+.4f},{tip_y_error:+.4f}) "
                f"q_delta_deg={delta_deg:.2f}"
            )
        else:
            self.get_logger().info(
                f"YOLO_CHEAT_POSE pfrac={position_fraction:.3f} z_offset={z_offset:.5f} "
                f"xy_error=({tip_x_error:.4f},{tip_y_error:.4f}) q_delta_deg={delta_deg:.2f}"
            )
        self._log_cmd_count += 1

        return Pose(
            position=Point(x=bx, y=by, z=bz),
            orientation=_wxyz_to_msg(q_grip_slerp),
        )

    # -----------------------------------------------------------------------
    # Entrypoint
    # -----------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"YOLO_CHEAT_TASK {task}")
        self._task = task
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._log_cmd_count = 0

        try:
            port_frame = self._port_frame_for_task(task)
            plug_frame = self._plug_frame_for_task(task)
        except ValueError as exc:
            self.get_logger().error(f"YOLO_CHEAT_RESULT success=False reason={exc}")
            return False

        self.get_logger().info(
            f"YOLO_CHEAT_FRAMES port_frame={port_frame} plug_frame={plug_frame}"
        )

        # Wait for all frames to become available
        for frame in [port_frame, plug_frame, "gripper/tcp"]:
            if not self._wait_for_tf("base_link", frame):
                self.get_logger().error(
                    f"YOLO_CHEAT_RESULT success=False reason=tf_timeout frame={frame}"
                )
                return False

        # Collect stable YOLO snapshot while robot is stationary
        if not self._sample_stable_yolo_reference(port_frame, plug_frame):
            self.get_logger().error(
                "YOLO_CHEAT_RESULT success=False reason=sampling_failed"
            )
            return False

        # Log initial rotation delta using CheatCode inverse convention
        try:
            grip_tf = self._lookup_transform("gripper/tcp")
            q_grip_now = _msg_to_wxyz(grip_tf.transform.rotation)
            q_plug_now = _quat_normalize(
                quaternion_multiply(q_grip_now, self._q_gripper_to_plug_wxyz)
            )
            q_diff_now = quaternion_multiply(
                self._frozen_port_quat_wxyz,
                _cheatcode_plug_inverse_wxyz(q_plug_now),
            )
            q_target_now = quaternion_multiply(q_diff_now, q_grip_now)
            delta_init_deg = _q_delta_deg(q_grip_now, q_target_now)
            self.get_logger().info(
                f"YOLO_CHEAT_INITIAL_DELTA q_delta_deg={delta_init_deg:.2f}"
            )
            if delta_init_deg > 170.0:
                self.get_logger().warn(
                    f"YOLO_CHEAT_WARN large_initial_rotation_deg={delta_init_deg:.2f} "
                    f"orientation_still_large_check_yolo_tf_axes"
                )
        except TransformException:
            pass

        z_offset = 0.10

        # Interpolation: 100 steps to position above frozen port
        # freeze_xy_error=True keeps XY target stable while gripper rotates
        for t in range(100):
            interp_fraction = t / 100.0
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                    freeze_xy_error=True,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.05)

        # Settle: hold final approach orientation for 2 s so the trajectory
        # planner resolves the rotation before Z descent begins.  Without this,
        # the orientation continues converging during descent which shifts the
        # commanded TCP Y (via R_target @ tcp_to_plug_local) past the workspace
        # limit at Y≈0.254 for nic_card_mount_1.
        self.get_logger().info("YOLO_CHEAT_SETTLE start")
        for _ in range(40):
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    z_offset=z_offset,
                    freeze_xy_error=True,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during settle: {ex}")
            self.sleep_for(0.05)
        self.get_logger().info("YOLO_CHEAT_SETTLE done")

        # XY-lock: hold Z at approach height and run live xy correction until the
        # plug-port lateral error falls under 1.5 mm. Without this, descent runs
        # open-loop in XY for ~12 s and a 4 mm residual lands the plug on the
        # port rim (NIC1 case in 2026-05-14 trials), which then jams the
        # trajectory planner in Z.
        _XY_LOCK_TOL = 0.0015
        _XY_LOCK_MAX_ITERS = 60
        _xy_lock_converged = False
        for _i in range(_XY_LOCK_MAX_ITERS):
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    z_offset=z_offset,
                    freeze_xy_error=False,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during xy-lock: {ex}")
                self.sleep_for(0.05)
                continue
            self.sleep_for(0.05)
            if getattr(self, "_last_plug_est", None) is None:
                continue
            ex_live = float(self._frozen_port_pos[0] - self._last_plug_est[0])
            ey_live = float(self._frozen_port_pos[1] - self._last_plug_est[1])
            err_mag = (ex_live * ex_live + ey_live * ey_live) ** 0.5
            if err_mag < _XY_LOCK_TOL:
                self.get_logger().info(
                    f"YOLO_CHEAT_XY_LOCK converged i={_i} "
                    f"ex={ex_live:+.4f} ey={ey_live:+.4f} mag={err_mag:.4f}"
                )
                _xy_lock_converged = True
                break
        if not _xy_lock_converged:
            self.get_logger().warn(
                f"YOLO_CHEAT_XY_LOCK timeout after {_XY_LOCK_MAX_ITERS} iters; "
                f"proceeding with descent anyway"
            )

        # Record the approach-end target Y so we can clamp insertion commands.
        # This prevents any residual orientation drift from pushing TCP Y beyond
        # the workspace boundary during descent.
        _insertion_max_y = None
        try:
            _ref_pose = self.calc_gripper_pose_from_frozen(
                slerp_fraction=1.0,
                position_fraction=1.0,
                z_offset=z_offset,
                freeze_xy_error=True,
            )
            _insertion_max_y = _ref_pose.position.y + 0.003
            self.get_logger().info(
                f"YOLO_CHEAT_INSERT_CLAMP max_y={_insertion_max_y:.4f}"
            )
        except TransformException:
            self.get_logger().warn("YOLO_CHEAT_INSERT_CLAMP skipped (TF unavailable)")

        # Insertion descent floor — the deepest z_offset we command.
        #
        # z_offset is plug-Z relative to port surface (negative = inserted),
        # because calc_gripper_pose_from_frozen builds desired_plug_z =
        # port_z + z_offset and back-solves the TCP target through the captured
        # rigid TCP→plug offset. Expressing the floor as
        # (mechanical_depth + planner_lag) makes the intent explicit and
        # tunable instead of a single magic constant.
        #
        # NIC trials with -0.085 left the plug ~20 mm above the port on both
        # mounts: the trajectory planner stalls within ~20–25 mm of any
        # commanded floor near contact, so the *commanded* depth must sit
        # below the planner's observed limit for descent to continue through
        # stabilize.
        _DESIRED_PLUG_DEPTH = 0.010   # plug 10 mm below port surface (mechanical)
        _PLANNER_LAG_MARGIN = 0.100   # absorb planner undershoot near contact
        _INSERT_END = -(_DESIRED_PLUG_DEPTH + _PLANNER_LAG_MARGIN)  # = -0.110
        # Step halved to 0.5 mm (10 mm/s) so the robot trajectory planner can
        # actually follow.
        while z_offset >= _INSERT_END:
            z_offset -= 0.0005
            self.get_logger().info(f"YOLO_CHEAT_INSERT z_offset={z_offset:.5f}")
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    z_offset=z_offset,
                    freeze_xy_error=False,
                )
                if _insertion_max_y is not None and pose.position.y > _insertion_max_y:
                    self.get_logger().info(
                        f"YOLO_CHEAT_INSERT_YCLAMP "
                        f"raw_y={pose.position.y:.4f}→clamped={_insertion_max_y:.4f}"
                    )
                    pose.position.y = _insertion_max_y
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
            self.sleep_for(0.05)

        # Active stabilise: keep commanding the final insertion pose so the robot
        # continues moving toward the target while the connector settles.
        # 12 seconds at 10 Hz gives the lagging trajectory planner time to arrive.
        #
        # Some ports (e.g. nic_card_mount_1 at Y=0.2535) sit at the edge of the
        # workspace. If the TCP is still >5 cm above the frozen port z after the
        # descent loop, the arm hit a joint/workspace limit and cannot reach the
        # deep _INSERT_END target — repeating it only freezes the trajectory
        # planner. Use a moderate z_offset=-0.045 instead so the robot can still
        # coast further rather than locking in place.
        try:
            grip_tf = self._lookup_transform("gripper/tcp")
            tcp_z = grip_tf.transform.translation.z
            port_z = float(self._frozen_port_pos[2])
            near_insertion = tcp_z <= port_z + 0.07
            self.get_logger().info(
                f"YOLO_CHEAT_STABILIZE tcp_z={tcp_z:.4f} port_z={port_z:.4f} "
                f"near_insertion={near_insertion}"
            )
        except TransformException:
            near_insertion = True  # assume reachable if TF unavailable

        stabilize_z_offset = _INSERT_END if near_insertion else -0.045
        self.get_logger().info(
            f"Waiting for connector to stabilize... stabilize_z_offset={stabilize_z_offset:.3f}"
        )
        for _ in range(120):
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    z_offset=stabilize_z_offset,
                    freeze_xy_error=False,
                )
                if _insertion_max_y is not None and pose.position.y > _insertion_max_y:
                    pose.position.y = _insertion_max_y
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException:
                pass
            self.sleep_for(0.1)

        self.get_logger().info("YOLO_CHEAT_RESULT success=True")
        return True
