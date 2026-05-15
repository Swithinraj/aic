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


def _base_yaw_quat_wxyz(yaw_rad: float):
    """Pure yaw rotation about base +Z as a wxyz quaternion."""
    half = 0.5 * float(yaw_rad)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


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
#
# Base structure: run_hybrid_swithin.py — the simple generic single-path
# policy that reaches every port consistently (no SC choreography that stalls
# in mid-air). On top of that base this file adds, for SFP *and* SC:
#   * port-type-specific insertion depth
#   * descent stall detection (trial-2 stalled ~80 s on the port rim)
#   * an adaptive seating wiggle (pull-back / push-deeper + yaw + Y dither)
#     that drives the last cm after the descent stalls
#   * a stabilize hold at the deepest depth actually achieved
#   * a live detector seat-distance log for tuning
#
# All the numbers grouped in the _*_for_port_type helpers are tuning knobs.
# ---------------------------------------------------------------------------

class run_hybrid(Policy):
    # --- closed-loop insertion descent ---
    # The trajectory executor lags the pose commands. The old fixed-timer
    # descent + 30-step / 0.6 mm stall window aborted on a false "stall"
    # before the arm ever moved (every trial, 2026-05-14 run). Each descent
    # step now waits for the TCP to actually track the commanded depth; a
    # real stall = the arm neither tracks nor descends for several steps.
    _DESCENT_TRACK_TOL = 0.003        # m; TCP "arrived" within this of the command
    _DESCENT_STEP_TIMEOUT = 0.4       # s; max wait for the TCP to track one step
    _DESCENT_STALL_PROGRESS = 0.0008  # m; min TCP-Z drop/step to count as progress
    _DESCENT_STALL_STEPS = 12         # consecutive no-progress steps -> real stall
    # --- closed-loop approach settle ---
    _SETTLE_MAX_STEPS = 180   # max command repeats waiting to reach the approach pose
    _SETTLE_XY_TOL = 0.005    # m; approach XY considered reached within this
    _SETTLE_Z_TOL = 0.010     # m; approach Z considered reached within this
    # --- adaptive seating wiggle ---
    _WIGGLE_STEPS = 30      # command repeats per wiggle cycle
    _WIGGLE_SLEEP = 0.1    # s between wiggle commands
    _WIGGLE_MAX_EXTRA = 4   # extra deeper cycles past the base pattern while still improving
    _WIGGLE_EXTRA_DZ = 0.010  # m deeper per extra cycle
    _WIGGLE_IMPROVE_TOL = 0.0008  # m; plug-Z must drop more than this to count as progress
    _WIGGLE_PULLBACK_DZ = 0.012   # m to lift above the last cmd to unseat a rim catch
    _WIGGLE_PULLBACK_STEPS = 15   # command repeats for the pull-back lift
    # --- per-trial home / reset ---
    _HOME_MAX_STEPS = 120        # command repeats while driving back to the captured home pose
    _HOME_SETTLE_STEPS = 15      # extra hold once home is reached
    _HOME_TOL = 0.012            # m; "home" is reached when TCP is within this of the captured pose
    _HOME_MIN_CAPTURE_Z = 0.25   # m; only capture the trial-1 pose as home if TCP is at least this high
    _SC_PLUG_TCP_Y_OFFSET_M = 0.00975  # detector is ~1.5 mm behind along gripper/tcp +Y

    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        # Frozen reference — set by _sample_stable_yolo_reference before any motion
        self._frozen_port_pos = None           # (3,) in base_link
        self._frozen_port_quat_wxyz = None     # (4,) wxyz
        self._tcp_to_plug_local = None         # (3,) plug offset in gripper-local frame
        self._q_gripper_to_plug_wxyz = None    # (4,) wxyz — relative orientation
        self._frozen_initial_tip_xy_error = None  # (2,) frozen port_xy - plug_xy at sample time
        self._log_cmd_count = 0
        self._home_pose_wxyz = None  # (pos(3,), quat wxyz(4,)) captured on the first trial
        super().__init__(parent_node)
        self.get_logger().info(
            "HYBRID_CONTROL_FRAME command_pose_frame=base_link controlled_frame=gripper/tcp"
        )
        self.get_logger().info(
            "HYBRID_MATH mode=official_cheatcode_quaternion"
        )
        self.get_logger().info(
            "HYBRID_VERSION 2026-05-14.closedloop_z_offset_v2_rapid_speed"
        )

    # -----------------------------------------------------------------------
    # TF helpers
    # -----------------------------------------------------------------------

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
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
    # Task -> YOLO frame names  (candidate lists)
    # -----------------------------------------------------------------------
    # The detector names the SC port sc_port_0 / sc_port — never the task's
    # literal port_name (e.g. "sc_port_base"). swithin's literal lookup
    # therefore tf-timed-out on every SC trial. Resolve against a candidate
    # list so whatever the detector publishes is found.

    def _dedupe_nonempty(self, values) -> list:
        seen = set()
        out = []
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out

    def _sfp_port_frame_candidates(self, task) -> list:
        port_name = str(getattr(task, "port_name", "")).strip().lower()
        target = str(getattr(task, "target_module_name", "")).strip().lower()
        names = self._dedupe_nonempty(
            [port_name, target, "sfp_port_0", "sfp_port_1", "sfp_port"]
        )
        return self._dedupe_nonempty(
            [f"yolo_tri/sfp_port/{n}" for n in names] + [f"det_{n}" for n in names]
        )

    def _sc_port_frame_candidates(self, task) -> list:
        # Detector publishes two visible SC sockets as sc_port_0/sc_port_1.
        # With only one visible SC socket it may publish generic sc_port; use
        # that only as the single-port fallback after the task-specific target.
        port_name = str(getattr(task, "port_name", "")).strip().lower()
        target = str(getattr(task, "target_module_name", "")).strip().lower()
        if target in {"sc_port_0", "sc_port_1"}:
            other = "sc_port_1" if target == "sc_port_0" else "sc_port_0"
            names = self._dedupe_nonempty([target, "sc_port", other, port_name])
        else:
            names = self._dedupe_nonempty([target, "sc_port", "sc_port_0", "sc_port_1", port_name])
        return self._dedupe_nonempty(
            [f"yolo_tri/sc_port/{n}" for n in names] + [f"det_{n}" for n in names]
        )

    def _port_frame_candidates(self, task) -> list:
        port_type = str(getattr(task, "port_type", "")).strip().lower()
        if port_type == "sfp":
            return self._sfp_port_frame_candidates(task)
        if port_type == "sc":
            return self._sc_port_frame_candidates(task)
        return []

    def _plug_frame_candidates(self, task) -> list:
        plug_type = str(getattr(task, "plug_type", "")).strip().lower()
        if plug_type == "sfp":
            return ["yolo_tri/sfp_module/sfp_module", "det_sfp_module"]
        if plug_type == "sc":
            return ["yolo_tri/sc_plug/sc_plug", "det_sc_plug"]
        return []

    def _transform_age_seconds(self, tf_msg):
        header = getattr(tf_msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return None
        if getattr(stamp, "sec", 0) == 0 and getattr(stamp, "nanosec", 0) == 0:
            return None
        try:
            return float((self.time_now() - Time.from_msg(stamp)).nanoseconds) / 1e9
        except Exception:
            return None

    def _resolve_tf_frame(self, label, candidates, timeout_sec=10.0, max_age_sec=2.0):
        """Wait up to timeout_sec for the first available candidate TF frame."""
        candidates = self._dedupe_nonempty(candidates)
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            for frame in candidates:
                try:
                    tf_msg = self._parent_node._tf_buffer.lookup_transform(
                        "base_link", frame, Time()
                    )
                    age_s = self._transform_age_seconds(tf_msg)
                    if age_s is not None and age_s > max_age_sec:
                        continue
                    self.get_logger().info(
                        f"HYBRID_TF_RESOLVED label={label} frame={frame}"
                    )
                    return frame
                except TransformException:
                    pass
            if attempt % 20 == 0:
                self.get_logger().info(
                    f"HYBRID_TF_WAIT label={label} candidates={candidates[:6]}"
                )
            attempt += 1
            self.sleep_for(0.1)
        self.get_logger().error(
            f"HYBRID_TF_TIMEOUT label={label} candidates={candidates}"
        )
        return ""

    # -----------------------------------------------------------------------
    # Per-trial home / reset
    # -----------------------------------------------------------------------

    def _go_home(self, move_robot) -> None:
        """Drive the arm back to the trial-1 home pose before every trial.

        The harness does not reliably reset the arm between trials: in the
        2026-05-14 run trial 2 started mid-air at the trial-1 stabilize pose,
        the descent then jammed at tcp_z~0.228 and the seating wiggle could not
        move it (the plug never descended -> "no insertion"). On the first
        trial the arm IS at the rig home pose, so capture it; on later trials
        drive back to it and wait until the arm actually arrives. The lift also
        clears the cameras' view of the static SC port."""
        try:
            grip_pos, q_grip = self._current_tcp_pos_quat()
        except TransformException as ex:
            self.get_logger().warn(f"HYBRID_HOME skipped reason=tcp_tf_failed ex={ex}")
            return

        if self._home_pose_wxyz is None:
            if float(grip_pos[2]) >= self._HOME_MIN_CAPTURE_Z:
                self._home_pose_wxyz = (
                    np.array(grip_pos, dtype=np.float64),
                    np.array(q_grip, dtype=np.float64),
                )
                self.get_logger().info(
                    f"HYBRID_HOME captured "
                    f"pos=({grip_pos[0]:+.4f},{grip_pos[1]:+.4f},{grip_pos[2]:+.4f})"
                )
            else:
                self.get_logger().warn(
                    f"HYBRID_HOME not_captured tcp_z={float(grip_pos[2]):.4f} "
                    f"below {self._HOME_MIN_CAPTURE_Z:.2f} — arm may not be at rig home"
                )
            return  # first trial: already at home, nothing to drive to

        home_pos, home_q = self._home_pose_wxyz
        dist0 = float(np.linalg.norm(np.asarray(grip_pos, dtype=np.float64) - home_pos))
        self.get_logger().info(
            f"HYBRID_HOME return start "
            f"cur=({grip_pos[0]:+.4f},{grip_pos[1]:+.4f},{grip_pos[2]:+.4f}) "
            f"home=({home_pos[0]:+.4f},{home_pos[1]:+.4f},{home_pos[2]:+.4f}) dist={dist0:.4f}"
        )
        if dist0 <= self._HOME_TOL:
            self.get_logger().info("HYBRID_HOME already_home")
            return

        home_pose = Pose(
            position=Point(x=float(home_pos[0]), y=float(home_pos[1]), z=float(home_pos[2])),
            orientation=_wxyz_to_msg(home_q),
        )
        for i in range(self._HOME_MAX_STEPS):
            self.set_pose_target(move_robot=move_robot, pose=home_pose)
            self.sleep_for(0.05)
            if i % 10 == 0:
                try:
                    cur, _ = self._current_tcp_pos_quat()
                    d = float(np.linalg.norm(cur - home_pos))
                    if d <= self._HOME_TOL:
                        self.get_logger().info(f"HYBRID_HOME arrived step={i} dist={d:.4f}")
                        break
                except TransformException:
                    pass
        for _ in range(self._HOME_SETTLE_STEPS):
            self.set_pose_target(move_robot=move_robot, pose=home_pose)
            self.sleep_for(0.05)
        self.get_logger().info("HYBRID_HOME return done")

    # -----------------------------------------------------------------------
    # Port-type insertion tuning knobs
    # -----------------------------------------------------------------------

    def _insert_end_for_port_type(self, port_type: str) -> float:
        """Final descent z_offset (plug target relative to frozen port origin)."""
        return -0.015 if str(port_type).strip().lower() == "sc" else -0.085

    def _insert_step_for_port_type(self, port_type: str, z_offset: float) -> float:
        if str(port_type).strip().lower() == "sc":
            return 0.0005
        # SFP: slow down inside the precision zone. Steps are larger than the
        # original glacial values because the descent is now closed-loop (it
        # waits for the arm to track each step) instead of running open-loop
        # on a fixed timer.
        return 0.0015 if float(z_offset) > 0.06 else 0.0006

    def _insert_sleep_for_port_type(self, port_type: str, z_offset: float) -> float:
        if str(port_type).strip().lower() == "sc":
            return 0.05
        return 0.10 if float(z_offset) <= 0.06 else 0.05

    def _descent_yaw_offset_deg(self, port_type: str, z_offset: float) -> float:
        """Small clockwise base-yaw twist for SFP once inside the precision zone.
        SC keeps zero twist (kept simple, position-only)."""
        if str(port_type).strip().lower() == "sc":
            return 0.0
        return -2.0 if float(z_offset) <= 0.045 else 0.0

    def _stabilize_steps_for_port_type(self, port_type: str) -> int:
        return 13

    def _seating_wiggle_pattern(self, port_type: str):
        """Base wiggle pattern: list of (z_cmd, yaw_deg, y_dither_m).

        Unlike the old fixed pattern this is *progressively deeper* — it never
        bounces back shallow after a breakthrough — and adds a small Y dither
        to rock the plug past a port-edge detent."""
        if str(port_type).strip().lower() == "sc":
            return [
                (-0.006,  0.0,  0.000),
                (-0.014,  0.0, +0.002),
                (-0.020,  0.0, -0.002),
                (-0.026,  0.0, +0.002),
            ]
        # SFP
        return [
            (-0.075,  0.0,  0.000),
            (-0.090, -2.0, +0.002),
            (-0.098, -2.0, -0.002),
            (-0.106, +1.0, +0.002),
            (-0.112, -2.0,  0.000),
        ]

    # -----------------------------------------------------------------------
    # Live gripper / plug state
    # -----------------------------------------------------------------------

    def _current_tcp_pos_quat(self):
        grip_tf = self._lookup_transform("gripper/tcp")
        pos = np.array([
            grip_tf.transform.translation.x,
            grip_tf.transform.translation.y,
            grip_tf.transform.translation.z,
        ], dtype=np.float64)
        return pos, _msg_to_wxyz(grip_tf.transform.rotation)

    def _plug_xy_error_from_current(self):
        """Reconstruct the plug pose from the live gripper + frozen offset."""
        grip_pos, q_grip = self._current_tcp_pos_quat()
        plug_pos = grip_pos + _quat_to_rot(q_grip) @ self._tcp_to_plug_local
        err_xy = np.asarray(self._frozen_port_pos[:2], dtype=np.float64) - plug_pos[:2]
        return err_xy, float(np.linalg.norm(err_xy)), plug_pos

    # -----------------------------------------------------------------------
    # Stable YOLO snapshot — called once while robot is stationary
    # -----------------------------------------------------------------------

    def _sample_stable_yolo_reference(
        self, port_frame: str, plug_frame: str,
        duration_s: float = 10.0, rate_hz: float = 20.0
    ) -> bool:
        self.get_logger().info(f"HYBRID_SAMPLE_START duration={duration_s}")

        port_pos_list = []
        port_quat_list = []
        plug_pos_list = []
        tcp_to_plug_list = []
        q_gtp_list = []   # gripper-to-plug relative quaternions (correct math)
        plug_type = str(getattr(self._task, "plug_type", "")).strip().lower()
        apply_sc_plug_offset = plug_type == "sc"

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
            if apply_sc_plug_offset:
                plug_pos = plug_pos + R_grip @ np.array(
                    [0.0, self._SC_PLUG_TCP_Y_OFFSET_M, 0.0],
                    dtype=np.float64,
                )
            tcp_to_plug = R_grip.T @ (plug_pos - grip_pos)
            # Correct conjugate used here — frame-relative transform, not CheatCode pose math
            q_gtp = _quat_normalize(quaternion_multiply(_quat_conjugate_wxyz(q_grip), q_plug))

            port_pos_list.append(port_pos)
            port_quat_list.append(np.array(q_port))
            plug_pos_list.append(plug_pos)
            tcp_to_plug_list.append(tcp_to_plug)
            q_gtp_list.append(q_gtp)

            self.sleep_for(step_s)

        n_raw = len(port_pos_list)
        if n_raw == 0:
            self.get_logger().error("HYBRID_SAMPLE failed: no valid samples collected")
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

        self.get_logger().info(f"HYBRID_SAMPLE_DONE n_raw={n_raw} n_used={n_used}")
        if apply_sc_plug_offset:
            self.get_logger().info(
                f"HYBRID_SC_PLUG_OFFSET tcp_y_m={self._SC_PLUG_TCP_Y_OFFSET_M:+.4f}"
            )

        if n_used < 20:
            self.get_logger().error(
                f"HYBRID_SAMPLE failed: only {n_used} valid samples after filtering (need 20)"
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
            f"HYBRID_FROZEN_PORT "
            f"pos=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) "
            f"quat=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})"
        )
        self.get_logger().info(
            f"HYBRID_FROZEN_TCP_TO_PLUG_LOCAL "
            f"offset=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f}) "
            f"norm={float(np.linalg.norm(t)):.4f}"
        )
        self.get_logger().info(
            f"HYBRID_FROZEN_GRIPPER_TO_PLUG_QUAT "
            f"quat=({qg[0]:+.4f},{qg[1]:+.4f},{qg[2]:+.4f},{qg[3]:+.4f})"
        )
        self.get_logger().info(
            f"HYBRID_FROZEN_INITIAL_XY_ERROR "
            f"ex={self._frozen_initial_tip_xy_error[0]:+.4f} "
            f"ey={self._frozen_initial_tip_xy_error[1]:+.4f}"
        )

        offset_norm = float(np.linalg.norm(self._tcp_to_plug_local))
        if not (0.005 <= offset_norm <= 0.12):
            self.get_logger().warn(
                f"HYBRID_WARN tcp_to_plug_local norm={offset_norm:.4f} "
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
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
        freeze_xy_error: bool = False,
        yaw_offset_rad: float = 0.0,
        target_xy_bias=None,
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

        # --- orientation: exact CheatCode algebra ---
        # CheatCode:  q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        #             q_diff = quaternion_multiply(q_port, q_plug_inv)
        #             q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_port = self._frozen_port_quat_wxyz
        q_plug_inv = _cheatcode_plug_inverse_wxyz(q_plug_est)
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        q_grip_target = quaternion_multiply(q_diff, q_grip)
        q_grip_slerp = quaternion_slerp(q_grip, q_grip_target, slerp_fraction)

        # Optional small clockwise/counter base-yaw twist (SFP seating).
        if abs(float(yaw_offset_rad)) > 1e-9:
            q_grip_slerp = _quat_normalize(
                quaternion_multiply(_base_yaw_quat_wxyz(yaw_offset_rad), q_grip_slerp)
            )

        # --- position: CheatCode style ---
        # During approach rotation the reconstructed plug tip can swing laterally.
        # freeze_xy_error=True pins XY correction to the pre-motion snapshot.
        if freeze_xy_error:
            tip_x_error = self._frozen_initial_tip_xy_error[0]
            tip_y_error = self._frozen_initial_tip_xy_error[1]
        else:
            tip_x_error = float(self._frozen_port_pos[0] - plug_pos_est[0])
            tip_y_error = float(self._frozen_port_pos[1] - plug_pos_est[1])

        plug_tip_grip_offset_z = float(grip_pos[2] - plug_pos_est[2])

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = float(np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup, self._max_integrator_windup,
            ))
            self._tip_y_error_integrator = float(np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup, self._max_integrator_windup,
            ))

        if target_xy_bias is None:
            target_xy_bias = np.array([0.0, 0.0], dtype=np.float64)
        target_xy_bias = np.asarray(target_xy_bias, dtype=np.float64)

        i_gain = 0.15
        target_x = (
            float(self._frozen_port_pos[0])
            + i_gain * self._tip_x_error_integrator
            + float(target_xy_bias[0])
        )
        target_y = (
            float(self._frozen_port_pos[1])
            + i_gain * self._tip_y_error_integrator
            + float(target_xy_bias[1])
        )
        target_z = float(self._frozen_port_pos[2]) + z_offset - plug_tip_grip_offset_z

        bx = position_fraction * target_x + (1.0 - position_fraction) * float(grip_pos[0])
        by = position_fraction * target_y + (1.0 - position_fraction) * float(grip_pos[1])
        bz = position_fraction * target_z + (1.0 - position_fraction) * float(grip_pos[2])

        delta_deg = _q_delta_deg(q_grip, q_grip_target)
        yaw_offset_deg = float(np.degrees(yaw_offset_rad))

        # Verbose log for first 5 commands and every 20 thereafter
        if self._log_cmd_count < 5 or self._log_cmd_count % 20 == 0:
            self.get_logger().info(
                f"HYBRID_CMD #{self._log_cmd_count} "
                f"tcp=({float(grip_pos[0]):+.4f},{float(grip_pos[1]):+.4f},{float(grip_pos[2]):+.4f}) "
                f"plug_est=({float(plug_pos_est[0]):+.4f},{float(plug_pos_est[1]):+.4f},{float(plug_pos_est[2]):+.4f}) "
                f"frozen_port=({float(self._frozen_port_pos[0]):+.4f},{float(self._frozen_port_pos[1]):+.4f},{float(self._frozen_port_pos[2]):+.4f}) "
                f"target=({bx:+.4f},{by:+.4f},{bz:+.4f}) "
                f"xy_err=({tip_x_error:+.4f},{tip_y_error:+.4f}) "
                f"q_delta_deg={delta_deg:.2f} yaw_offset_deg={yaw_offset_deg:+.2f} "
                f"xy_bias=({target_xy_bias[0]:+.4f},{target_xy_bias[1]:+.4f})"
            )
        else:
            self.get_logger().info(
                f"HYBRID_POSE pfrac={position_fraction:.3f} z_offset={z_offset:.5f} "
                f"xy_error=({tip_x_error:.4f},{tip_y_error:.4f}) q_delta_deg={delta_deg:.2f} "
                f"yaw_offset_deg={yaw_offset_deg:+.2f} "
                f"xy_bias=({target_xy_bias[0]:+.4f},{target_xy_bias[1]:+.4f})"
            )
        self._log_cmd_count += 1

        return Pose(
            position=Point(x=bx, y=by, z=bz),
            orientation=_wxyz_to_msg(q_grip_slerp),
        )

    # -----------------------------------------------------------------------
    # Adaptive seating wiggle — drives the last cm after the descent stalls
    # -----------------------------------------------------------------------

    def _seating_wiggle(self, move_robot, port_type: str, insert_end: float) -> float:
        """Pull-back / push-deeper cycles with a yaw twist and a small Y dither.

        In the trial-2 log the descent hard-stalled on the SFP port rim; the
        first deeper+yaw wiggle cycle broke it (plug dropped 3.6 cm) but the
        old fixed pattern then bounced shallow and never finished the seat.
        This version only ever goes deeper, and keeps adding deeper cycles
        while the reconstructed plug-Z is still dropping.

        Returns the deepest z_offset that was commanded — used by stabilize."""
        pattern = self._seating_wiggle_pattern(port_type)
        self.get_logger().info(
            f"HYBRID_WIGGLE start port_type={port_type} cycles={len(pattern)} "
            f"insert_end={insert_end:+.4f}"
        )

        best_plug_z = None
        deepest_z_cmd = insert_end
        prev_cycle_plug_z = None
        prev_z_cmd = None

        for cycle, (z_cmd, yaw_deg, y_dither) in enumerate(pattern):
            # If the previous cycle made no progress the plug is caught on the
            # port rim — a deeper push alone will not free it. Lift a few mm
            # above the last commanded depth to unseat it, then drive deeper.
            if (
                prev_z_cmd is not None
                and prev_cycle_plug_z is not None
                and best_plug_z is not None
                and prev_cycle_plug_z > best_plug_z - self._WIGGLE_IMPROVE_TOL
            ):
                lift_z = prev_z_cmd + self._WIGGLE_PULLBACK_DZ
                self.get_logger().info(
                    f"HYBRID_WIGGLE pullback cycle={cycle} lift_z={lift_z:+.4f}"
                )
                for _ in range(self._WIGGLE_PULLBACK_STEPS):
                    try:
                        pose = self.calc_gripper_pose_from_frozen(
                            z_offset=lift_z,
                            freeze_xy_error=False,
                            yaw_offset_rad=float(np.radians(yaw_deg)),
                            target_xy_bias=np.array([0.0, y_dither], dtype=np.float64),
                        )
                        self.set_pose_target(move_robot=move_robot, pose=pose)
                    except TransformException as ex:
                        self.get_logger().warn(f"TF lookup failed during pullback: {ex}")
                    self.sleep_for(self._WIGGLE_SLEEP)

            for _ in range(self._WIGGLE_STEPS):
                try:
                    pose = self.calc_gripper_pose_from_frozen(
                        z_offset=z_cmd,
                        freeze_xy_error=False,
                        yaw_offset_rad=float(np.radians(yaw_deg)),
                        target_xy_bias=np.array([0.0, y_dither], dtype=np.float64),
                    )
                    self.set_pose_target(move_robot=move_robot, pose=pose)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during seating wiggle: {ex}")
                self.sleep_for(self._WIGGLE_SLEEP)

            prev_z_cmd = z_cmd
            try:
                _, _, plug_pos = self._plug_xy_error_from_current()
                plug_z = float(plug_pos[2])
                if best_plug_z is None or plug_z < best_plug_z:
                    best_plug_z = plug_z
                deepest_z_cmd = min(deepest_z_cmd, z_cmd)
                prev_cycle_plug_z = plug_z
                self.get_logger().info(
                    f"HYBRID_WIGGLE cycle={cycle} z_cmd={z_cmd:+.4f} yaw={yaw_deg:+.2f} "
                    f"y_dither={y_dither:+.4f} plug_z={plug_z:.4f} best_plug_z={best_plug_z:.4f}"
                )
            except TransformException:
                pass

        # Adaptive: keep pushing deeper while the plug is still descending.
        # One no-progress cycle is tolerated (the plug can ratchet past a rim
        # detent on the *next* push); only stop after two in a row.
        extra_z = deepest_z_cmd
        extra_no_progress = 0
        for extra in range(self._WIGGLE_MAX_EXTRA):
            extra_z -= self._WIGGLE_EXTRA_DZ
            prev_best = best_plug_z
            yaw_deg = self._descent_yaw_offset_deg(port_type, extra_z)
            for _ in range(self._WIGGLE_STEPS):
                try:
                    pose = self.calc_gripper_pose_from_frozen(
                        z_offset=extra_z,
                        freeze_xy_error=False,
                        yaw_offset_rad=float(np.radians(yaw_deg)),
                    )
                    self.set_pose_target(move_robot=move_robot, pose=pose)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during wiggle-extra: {ex}")
                self.sleep_for(self._WIGGLE_SLEEP)

            try:
                _, _, plug_pos = self._plug_xy_error_from_current()
                plug_z = float(plug_pos[2])
                if best_plug_z is None or plug_z < best_plug_z:
                    best_plug_z = plug_z
                if prev_best is not None and plug_z > prev_best - self._WIGGLE_IMPROVE_TOL:
                    extra_no_progress += 1
                else:
                    extra_no_progress = 0
                    deepest_z_cmd = extra_z
                self.get_logger().info(
                    f"HYBRID_WIGGLE_EXTRA #{extra} extra_z={extra_z:+.4f} "
                    f"plug_z={plug_z:.4f} best_plug_z={best_plug_z:.4f} "
                    f"no_progress={extra_no_progress}"
                )
                if extra_no_progress >= 2:
                    self.get_logger().info(
                        "HYBRID_WIGGLE_EXTRA no further progress — stopping"
                    )
                    break
            except TransformException:
                break

        self.get_logger().info(
            f"HYBRID_WIGGLE done best_plug_z={best_plug_z} deepest_z_cmd={deepest_z_cmd:+.4f}"
        )
        return deepest_z_cmd

    # -----------------------------------------------------------------------
    # Entrypoint
    # -----------------------------------------------------------------------
    def _approach_z_offset_for_port_type(self, port_type: str) -> float:
        """Approach height above the frozen port before starting descent."""
        if str(port_type).strip().lower() == "sc":
            return 0.20   # Match the known-good swithin SC approach height
        return 0.15  
        
        
             # keep current SFP behavior
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"HYBRID_TASK {task}")
        self._task = task
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._log_cmd_count = 0

        port_type = str(getattr(task, "port_type", "")).strip().lower()
        plug_type = str(getattr(task, "plug_type", "")).strip().lower()

        port_candidates = self._port_frame_candidates(task)
        plug_candidates = self._plug_frame_candidates(task)
        if not port_candidates:
            self.get_logger().error(
                f"HYBRID_RESULT success=False reason=unknown_port_type port_type={port_type!r}"
            )
            return False
        if not plug_candidates:
            self.get_logger().error(
                f"HYBRID_RESULT success=False reason=unknown_plug_type plug_type={plug_type!r}"
            )
            return False
        self.get_logger().info(
            f"HYBRID_FRAME_CANDIDATES port={port_candidates} plug={plug_candidates}"
        )

        # Reset the arm to the captured home pose before sampling. The harness
        # does not reliably reset it between trials; starting a trial mid-air
        # jams the descent (trial 2, 2026-05-14 run). This also clears the
        # cameras' view of the static SC port.
        if not self._wait_for_tf("base_link", "gripper/tcp", timeout_sec=5.0):
            self.get_logger().error(
                "HYBRID_RESULT success=False reason=tf_timeout frame=gripper/tcp before_home"
            )
            return False
        self._go_home(move_robot)

        # SC port is static -> longer wait, older transform tolerated.
        port_frame = self._resolve_tf_frame(
            f"{port_type}_port", port_candidates,
            timeout_sec=25.0 if port_type == "sc" else 12.0,
            max_age_sec=30.0 if port_type == "sc" else 2.0,
        )
        if not port_frame:
            self.get_logger().error(
                f"HYBRID_RESULT success=False reason={port_type}_port_tf_timeout"
            )
            return False
        plug_frame = self._resolve_tf_frame(
            f"{plug_type}_plug", plug_candidates,
            timeout_sec=15.0 if plug_type == "sc" else 10.0,
            max_age_sec=2.0,
        )
        if not plug_frame:
            self.get_logger().error(
                f"HYBRID_RESULT success=False reason={plug_type}_plug_tf_timeout"
            )
            return False
        if not self._wait_for_tf("base_link", "gripper/tcp"):
            self.get_logger().error(
                "HYBRID_RESULT success=False reason=tf_timeout frame=gripper/tcp"
            )
            return False
        self.get_logger().info(
            f"HYBRID_FRAMES port_frame={port_frame} plug_frame={plug_frame}"
        )

        # Collect stable YOLO snapshot while robot is stationary
        if not self._sample_stable_yolo_reference(port_frame, plug_frame):
            self.get_logger().error(
                "HYBRID_RESULT success=False reason=sampling_failed"
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
                f"HYBRID_INITIAL_DELTA q_delta_deg={delta_init_deg:.2f}"
            )
            if delta_init_deg > 170.0:
                self.get_logger().warn(
                    f"HYBRID_WARN large_initial_rotation_deg={delta_init_deg:.2f} "
                    f"orientation_still_large_check_yolo_tf_axes"
                )
        except TransformException:
            pass

        # ---------------------------------------------------------------
        # Insertion geometry (tuning knobs live in the _*_for_port_type
        # helpers above).
        # ---------------------------------------------------------------
        approach_z_offset = self._approach_z_offset_for_port_type(port_type)    
        insert_end = self._insert_end_for_port_type(port_type)
        self.get_logger().info(
            f"HYBRID_INSERT_GEOMETRY port_type={port_type} "
            f"approach_z={approach_z_offset:.3f} insert_end={insert_end:+.4f}"
        )

        if port_type == "sc":
            # SC is known-good with the simple swithin profile: approach high,
            # descend slowly, and do not run the SFP-oriented stall/wiggle
            # choreography that can pin the plug against the board.
            z_offset = approach_z_offset

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
                    self.get_logger().warn(f"TF lookup failed during SC interpolation: {ex}")
                self.sleep_for(0.05)

            while z_offset >= insert_end:
                z_offset -= self._insert_step_for_port_type(port_type, z_offset)
                self.get_logger().info(f"HYBRID_SC_INSERT z_offset={z_offset:.5f}")
                try:
                    pose = self.calc_gripper_pose_from_frozen(
                        z_offset=z_offset,
                        freeze_xy_error=False,
                    )
                    self.set_pose_target(move_robot=move_robot, pose=pose)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during SC insertion: {ex}")
                self.sleep_for(self._insert_sleep_for_port_type(port_type, z_offset))

            self.get_logger().info("HYBRID_SC stabilize_wait")
            self.sleep_for(5.0)
            self.get_logger().info("HYBRID_RESULT success=True")
            return True

        # ---------------------------------------------------------------
        # Interpolation: 100 steps to the approach pose above the port
        # freeze_xy_error=True keeps the XY target stable while the gripper
        # rotates into the approach orientation.
        # ---------------------------------------------------------------
        # Step count scales with the lateral travel so the approach speed
        # stays bounded. The old fixed 100 steps was too fast for the SC
        # port's large lateral move — the executor lagged and the arm was
        # still traversing when the descent began (trial-3 diagonal jam).
        try:
            grip_pos0, _ = self._current_tcp_pos_quat()
            xy_travel = float(np.linalg.norm(
                np.asarray(self._frozen_port_pos[:2], dtype=np.float64)
                - grip_pos0[:2]
            ))
        except TransformException:
            xy_travel = 0.10
        interp_steps = int(np.clip(xy_travel / 0.0011, 100, 300))
        self.get_logger().info(
            f"HYBRID_INTERP steps={interp_steps} xy_travel={xy_travel:.4f}"
        )
        for t in range(interp_steps):
            interp_fraction = t / float(interp_steps)
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
                    z_offset=approach_z_offset,
                    reset_xy_integrator=True,
                    freeze_xy_error=True,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.04)

        # ---------------------------------------------------------------
        # Settle: let the trajectory planner fully resolve the approach
        # rotation before descent begins.
        # ---------------------------------------------------------------
        self.get_logger().info("HYBRID_SETTLE start")
        settle_arrived = False
        for i in range(self._SETTLE_MAX_STEPS):
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    z_offset=approach_z_offset,
                    freeze_xy_error=True,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
                cur, _ = self._current_tcp_pos_quat()
                xy_err = float(np.linalg.norm(
                    cur[:2] - np.array([pose.position.x, pose.position.y])
                ))
                z_err = abs(float(cur[2]) - float(pose.position.z))
                if i % 10 == 0:
                    self.get_logger().info(
                        f"HYBRID_SETTLE step={i} xy_err={xy_err:.4f} z_err={z_err:.4f}"
                    )
                if xy_err <= self._SETTLE_XY_TOL and z_err <= self._SETTLE_Z_TOL:
                    self.get_logger().info(
                        f"HYBRID_SETTLE arrived step={i} "
                        f"xy_err={xy_err:.4f} z_err={z_err:.4f}"
                    )
                    settle_arrived = True
                    break
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during settle: {ex}")
            self.sleep_for(0.05)
        if not settle_arrived:
            self.get_logger().warn(
                f"HYBRID_SETTLE timeout after {self._SETTLE_MAX_STEPS} steps "
                f"— descending anyway"
            )
        self.get_logger().info("HYBRID_SETTLE done")

        # ---------------------------------------------------------------
        # Insertion descent — with stall detection.
        # The descent walks z_offset from the approach height down to
        # insert_end. If the TCP stops moving (hard contact on the port
        # rim — the trial-2 failure mode) we break out early and hand off
        # to the seating wiggle instead of pushing a dead command for
        # tens of seconds.
        # ---------------------------------------------------------------
        z_offset = approach_z_offset
        descent_stalled = True
        no_progress_steps = 0
        prev_tcp_z = None
        while z_offset >= insert_end:
            step = self._insert_step_for_port_type(port_type, z_offset)
            z_offset -= step
            yaw_offset_deg = self._descent_yaw_offset_deg(port_type, z_offset)
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    z_offset=z_offset,
                    freeze_xy_error=False,
                    yaw_offset_rad=float(np.radians(yaw_offset_deg)),
                )
                cmd_z = float(pose.position.z)

                # Command the step, then wait for the arm to actually track it
                # before stepping further — the executor lags the commands, so
                # stepping on a fixed timer (the old behaviour) raced ahead and
                # tripped a false stall before the arm ever moved.
                poll_sleep = self._insert_sleep_for_port_type(port_type, z_offset)
                step_timeout = Duration(seconds=self._DESCENT_STEP_TIMEOUT)
                wait_start = self.time_now()
                cur_tcp_z = prev_tcp_z
                tracked = False
                while (self.time_now() - wait_start) < step_timeout:
                    self.set_pose_target(move_robot=move_robot, pose=pose)
                    self.sleep_for(poll_sleep)
                    grip_pos, _ = self._current_tcp_pos_quat()
                    cur_tcp_z = float(grip_pos[2])
                    if abs(cur_tcp_z - cmd_z) <= self._DESCENT_TRACK_TOL:
                        tracked = True
                        break

                # A real stall = the arm could neither track the command nor
                # keep descending, sustained over several consecutive steps.
                descended = (
                    prev_tcp_z is None
                    or (prev_tcp_z - cur_tcp_z) >= self._DESCENT_STALL_PROGRESS
                )
                if tracked or descended:
                    no_progress_steps = 0
                else:
                    no_progress_steps += 1
                prev_tcp_z = cur_tcp_z

                self.get_logger().info(
                    f"HYBRID_INSERT z_offset={z_offset:.5f} tcp_z={cur_tcp_z:.4f} "
                    f"cmd_z={cmd_z:.4f} tracked={tracked} "
                    f"no_progress={no_progress_steps} yaw_offset_deg={yaw_offset_deg:+.2f}"
                )

                if no_progress_steps >= self._DESCENT_STALL_STEPS:
                    self.get_logger().info(
                        f"HYBRID_DESCENT_STALL z_offset={z_offset:.5f} "
                        f"tcp_z={cur_tcp_z:.4f} -> handing off to seating wiggle"
                    )
                    descent_stalled = True
                    break
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(self._insert_sleep_for_port_type(port_type, z_offset))
        else:
            descent_stalled = False

        if not descent_stalled:
            self.get_logger().info("HYBRID_DESCENT reached insert_end without stalling")

        # ---------------------------------------------------------------
        # Adaptive seating wiggle — finishes the last cm.
        # ---------------------------------------------------------------
        if descent_stalled:
            deepest_z_cmd = self._seating_wiggle(move_robot, port_type, insert_end)
        else:
            deepest_z_cmd = insert_end

        # ---------------------------------------------------------------
        # Active stabilise — hold the deepest depth actually achieved so
        # the lagging trajectory planner can catch up and seat the plug.
        # ---------------------------------------------------------------
        stabilize_steps = self._stabilize_steps_for_port_type(port_type)
        stabilize_yaw_deg = self._descent_yaw_offset_deg(port_type, deepest_z_cmd)
        self.get_logger().info(
            f"HYBRID_STABILIZE start z_offset={deepest_z_cmd:+.4f} "
            f"steps={stabilize_steps} yaw_offset_deg={stabilize_yaw_deg:+.2f}"
        )
        for _ in range(stabilize_steps):
            try:
                pose = self.calc_gripper_pose_from_frozen(
                    z_offset=deepest_z_cmd,
                    freeze_xy_error=False,
                    yaw_offset_rad=float(np.radians(stabilize_yaw_deg)),
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException:
                pass
            self.sleep_for(0.07)

        # ---------------------------------------------------------------
        # Live detector seat-distance log (diagnostic only — does not
        # affect the result). The detector may hold its last valid value
        # while the robot occludes the port, so treat this as advisory.
        # ---------------------------------------------------------------
        try:
            port_tf = self._lookup_transform(port_frame)
            plug_tf = self._lookup_transform(plug_frame)
            port_p = np.array([
                port_tf.transform.translation.x,
                port_tf.transform.translation.y,
                port_tf.transform.translation.z,
            ])
            plug_p = np.array([
                plug_tf.transform.translation.x,
                plug_tf.transform.translation.y,
                plug_tf.transform.translation.z,
            ])
            seat_dist = float(np.linalg.norm(port_p - plug_p))
            self.get_logger().info(
                f"HYBRID_SEAT_CHECK live port=({port_p[0]:+.4f},{port_p[1]:+.4f},{port_p[2]:+.4f}) "
                f"plug=({plug_p[0]:+.4f},{plug_p[1]:+.4f},{plug_p[2]:+.4f}) "
                f"dist={seat_dist:.4f}"
            )
        except TransformException:
            self.get_logger().info("HYBRID_SEAT_CHECK skipped reason=tf_unavailable")

        self.get_logger().info("HYBRID_RESULT success=True")
        return True
