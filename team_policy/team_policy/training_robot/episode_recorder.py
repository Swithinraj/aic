"""
Records one episode of (observation, action) pairs to HDF5.

Images are streamed directly to disk as they are recorded — only small state
fields are buffered in RAM. This keeps peak memory per episode in the low MBs
instead of 1–3 GB, which matters on 16 GB machines running Gazebo alongside.

Action stored here is the absolute TCP pose commanded by CheatCode at each step.
During convert_to_lerobot.py those are turned into delta poses (position diff +
axis-angle rotation diff) for the 6-D Cartesian twist action space.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# HDF5 file locking fails on tmpfs (/tmp) — disable it globally
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

CONTACT_FORCE_THRESHOLD_N = 0.5
SCHEMA_VERSION = "4"


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / norm


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = _quat_normalize(a)
    bx, by, bz, bw = _quat_normalize(b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _quat_normalize(q)
    return np.array([-x, -y, -z, w], dtype=np.float32)


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _quat_normalize(q)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sin_half = float(np.linalg.norm([x, y, z]))
    if sin_half < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, max(abs(float(w)), 1e-12))
    return np.array([x / sin_half * angle, y / sin_half * angle, z / sin_half * angle], dtype=np.float32)


def _pose_delta(current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Return 6D delta [dx, dy, dz, rx, ry, rz] from current pose to target pose."""
    if not np.any(target_pose):
        return np.zeros(6, dtype=np.float32)
    dp = np.asarray(target_pose[:3] - current_pose[:3], dtype=np.float32)
    dq = _quat_multiply(target_pose[3:], _quat_inverse(current_pose[3:]))
    dr = _quat_to_axis_angle(dq)
    return np.concatenate([dp, dr]).astype(np.float32)


def _compute_target_velocity_actions(cmd_poses: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Finite-difference expert commanded poses into 6D velocity-like actions."""
    T = cmd_poses.shape[0]
    velocities = np.zeros((T, 6), dtype=np.float32)
    valid = np.any(cmd_poses, axis=1)

    for i in range(T - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        dt = float(timestamps[i + 1] - timestamps[i])
        if dt <= 1e-6:
            continue
        delta = _pose_delta(cmd_poses[i], cmd_poses[i + 1])
        velocities[i] = delta / dt

    if T > 1:
        velocities[-1] = velocities[-2]
    return velocities


@dataclass
class Frame:
    """Per-frame state — images are streamed to HDF5 separately, not buffered here."""
    timestamp: float
    task_id: str
    tcp_pose: np.ndarray         # (7,) x y z qx qy qz qw
    tcp_velocity: np.ndarray     # (6,) vx vy vz wx wy wz
    tcp_error: np.ndarray        # (6,)
    joint_positions: np.ndarray  # (7,)
    joint_velocity: np.ndarray   # (7,) — measured per-joint velocity from JointState
    wrist_force: np.ndarray      # (6,) fx fy fz tx ty tz
    relative_pose: Optional[np.ndarray] = None  # (7,) target port pose in plug-tip frame
    privileged_tf: Optional[np.ndarray] = None  # (N, 7) selected TF snapshot
    privileged_tf_valid: Optional[np.ndarray] = None  # (N,)
    commanded_pose: Optional[np.ndarray] = None  # (7,) — set when CheatCode issues a command


@dataclass
class Episode:
    episode_id: int
    task_id: str
    port_type: str
    port_name: str
    privileged_tf_frame_pairs: List[str] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    success: bool = False
    start_time: float = 0.0
    end_time: float = 0.0


IMAGE_COMPRESSION_OPTS = 1  # fast gzip — keeps per-frame CPU under ~15 ms


class EpisodeRecorder:
    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._current: Optional[Episode] = None
        self._last_commanded_pose: Optional[np.ndarray] = None
        self._hf = None                              # open h5py.File during an episode
        self._partial_path: Optional[str] = None
        self._final_path: Optional[str] = None
        self._img_ds: dict = {}                      # {"left"/"center"/"right": dataset}
        self._cleanup_stale_partials()

    def _cleanup_stale_partials(self) -> None:
        """Delete leftover .partial files from a prior crash."""
        try:
            for name in os.listdir(self._output_dir):
                if name.endswith(".partial"):
                    try:
                        os.remove(os.path.join(self._output_dir, name))
                    except OSError:
                        pass
        except FileNotFoundError:
            pass

    def start_episode(self, episode_id: int, task_id: str, port_type: str, port_name: str) -> None:
        # If a previous episode was left open (crash/cancel), discard it.
        if self._hf is not None:
            self._abort_current()

        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py is required — add it to pixi.toml pypi-dependencies")

        self._current = Episode(
            episode_id=episode_id,
            task_id=task_id,
            port_type=port_type,
            port_name=port_name,
            start_time=time.time(),
        )
        self._last_commanded_pose = None

        self._final_path = os.path.join(self._output_dir, f"episode_{episode_id:05d}.hdf5")
        self._partial_path = self._final_path + ".partial"
        if os.path.exists(self._partial_path):
            os.remove(self._partial_path)

        self._hf = h5py.File(self._partial_path, "w")
        self._img_ds = {}  # created lazily on first frame so we know H and W

    def update_commanded_pose(self, pose: np.ndarray) -> None:
        """Call this every time CheatCode sends a move_robot command."""
        self._last_commanded_pose = pose.copy()

    def set_privileged_tf_frame_pairs(self, frame_pairs: List[str]) -> None:
        """Declare the selected TF snapshot fields for the current episode."""
        if self._current is not None:
            self._current.privileged_tf_frame_pairs = list(frame_pairs)

    def _init_image_datasets(self, H: int, W: int) -> None:
        """Create the three resizable image datasets once we know image shape."""
        assert self._hf is not None
        obs = self._hf.require_group("observations")
        img = obs.require_group("images")
        for name in ("left", "center", "right"):
            self._img_ds[name] = img.create_dataset(
                name,
                shape=(0, H, W, 3),
                maxshape=(None, H, W, 3),
                chunks=(1, H, W, 3),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=IMAGE_COMPRESSION_OPTS,
            )

    def record_frame(
        self,
        obs,
        relative_pose: Optional[np.ndarray] = None,
        privileged_tf: Optional[np.ndarray] = None,
        privileged_tf_valid: Optional[np.ndarray] = None,
    ) -> None:
        """
        obs is an aic_model_interfaces/msg/Observation.
        Called at ~10 Hz inside insert_cable while the episode is running.
        Images are streamed to HDF5 immediately; only small state is kept in RAM.
        """
        if self._current is None or self._hf is None:
            return

        def img_to_np(img_msg):
            arr = np.frombuffer(img_msg.data, dtype=np.uint8)
            return arr.reshape(img_msg.height, img_msg.width, 3).copy()

        left_img   = img_to_np(obs.left_image)
        center_img = img_to_np(obs.center_image)
        right_img  = img_to_np(obs.right_image)

        if not self._img_ds:
            H, W, _ = left_img.shape
            self._init_image_datasets(H, W)

        # Append the three frames to their HDF5 datasets (writes go to page cache)
        i = len(self._current.frames)
        for name, arr in (("left", left_img), ("center", center_img), ("right", right_img)):
            ds = self._img_ds[name]
            ds.resize(i + 1, axis=0)
            ds[i] = arr

        cs = obs.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js = obs.joint_states
        w = obs.wrist_wrench.wrench

        frame = Frame(
            timestamp=time.time(),
            task_id=self._current.task_id,
            tcp_pose=np.array([
                tcp.position.x, tcp.position.y, tcp.position.z,
                tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            ], dtype=np.float32),
            tcp_velocity=np.array([
                vel.linear.x, vel.linear.y, vel.linear.z,
                vel.angular.x, vel.angular.y, vel.angular.z,
            ], dtype=np.float32),
            tcp_error=np.array(list(cs.tcp_error), dtype=np.float32),
            joint_positions=np.array(list(js.position[:7]), dtype=np.float32),
            joint_velocity=np.array(list(js.velocity[:7]), dtype=np.float32),
            wrist_force=np.array([
                w.force.x, w.force.y, w.force.z,
                w.torque.x, w.torque.y, w.torque.z,
            ], dtype=np.float32),
            relative_pose=None if relative_pose is None else np.asarray(relative_pose, dtype=np.float32).copy(),
            privileged_tf=None if privileged_tf is None else np.asarray(privileged_tf, dtype=np.float32).copy(),
            privileged_tf_valid=None if privileged_tf_valid is None else np.asarray(privileged_tf_valid, dtype=np.bool_).copy(),
            commanded_pose=self._last_commanded_pose,
        )
        self._current.frames.append(frame)

    def end_episode(self, success: bool) -> Optional[str]:
        """Finalise, save to HDF5, return path or None if discarded."""
        if self._current is None or self._hf is None:
            return None

        ep = self._current
        ep.success = success
        ep.end_time = time.time()

        if len(ep.frames) < 10:
            self._abort_current()
            return None

        assert self._partial_path is not None and self._final_path is not None
        try:
            self._write_state_and_metadata(ep)
            self._hf.close()
            os.replace(self._partial_path, self._final_path)
            saved_path = self._final_path
        except Exception:
            self._abort_current()
            raise
        finally:
            self._current = None
            self._hf = None
            self._partial_path = None
            self._final_path = None
            self._img_ds = {}

        return saved_path

    def _abort_current(self) -> None:
        """Close and delete the partial file — called on discard or error."""
        if self._hf is not None:
            try:
                self._hf.close()
            except Exception:
                pass
        if self._partial_path is not None and os.path.exists(self._partial_path):
            try:
                os.remove(self._partial_path)
            except OSError:
                pass
        self._current = None
        self._hf = None
        self._partial_path = None
        self._final_path = None
        self._img_ds = {}

    def _write_state_and_metadata(self, ep: Episode) -> None:
        """Write state/action/metadata into the already-open HDF5 file."""
        import h5py

        assert self._hf is not None
        T = len(ep.frames)

        tcp_poses    = np.stack([f.tcp_pose         for f in ep.frames])
        tcp_vels     = np.stack([f.tcp_velocity     for f in ep.frames])
        tcp_errors   = np.stack([f.tcp_error        for f in ep.frames])
        joint_pos    = np.stack([f.joint_positions  for f in ep.frames])
        joint_vel    = np.stack([f.joint_velocity   for f in ep.frames])
        wrist_forces = np.stack([f.wrist_force      for f in ep.frames])
        timestamps   = np.array([f.timestamp        for f in ep.frames], dtype=np.float64)
        task_ids     = [f.task_id for f in ep.frames]

        cmd_poses = np.stack([
            f.commanded_pose if f.commanded_pose is not None else np.zeros(7, dtype=np.float32)
            for f in ep.frames
        ])
        relative_valid = np.array([f.relative_pose is not None for f in ep.frames], dtype=np.bool_)
        relative_poses = np.stack([
            f.relative_pose if f.relative_pose is not None else np.zeros(7, dtype=np.float32)
            for f in ep.frames
        ])
        tf_frame_pairs = list(ep.privileged_tf_frame_pairs)
        tf_count = len(tf_frame_pairs)
        if tf_count:
            privileged_tf = np.stack([
                f.privileged_tf if f.privileged_tf is not None else np.zeros((tf_count, 7), dtype=np.float32)
                for f in ep.frames
            ])
            privileged_tf_valid = np.stack([
                f.privileged_tf_valid if f.privileged_tf_valid is not None else np.zeros(tf_count, dtype=np.bool_)
                for f in ep.frames
            ])
        else:
            privileged_tf = np.zeros((T, 0, 7), dtype=np.float32)
            privileged_tf_valid = np.zeros((T, 0), dtype=np.bool_)
        delta_actions = np.stack([
            _pose_delta(f.tcp_pose, cmd_poses[i])
            for i, f in enumerate(ep.frames)
        ])
        velocity_actions = _compute_target_velocity_actions(cmd_poses, timestamps)

        force_mag = np.linalg.norm(wrist_forces[:, :3], axis=1)
        max_force = float(force_mag.max()) if T else 0.0
        if T > 1:
            dt = np.diff(timestamps)
            median_dt = float(np.median(dt)) if dt.size else 0.0
            frame_dt = np.concatenate([dt, np.array([median_dt], dtype=np.float64)])
            insertion_time = float(timestamps[-1] - timestamps[0])
        else:
            frame_dt = np.zeros(T, dtype=np.float64)
            insertion_time = 0.0
        contact_mask = force_mag > CONTACT_FORCE_THRESHOLD_N
        contact_duration = float(frame_dt[contact_mask].sum()) if T else 0.0
        valid_rel_indices = np.flatnonzero(relative_valid)
        final_error = (
            float(np.linalg.norm(relative_poses[valid_rel_indices[-1], :3]))
            if valid_rel_indices.size
            else float("nan")
        )

        obs = self._hf.require_group("observations")
        obs.create_dataset("task_id", data=np.asarray(task_ids, dtype=h5py.string_dtype(encoding="utf-8")))
        obs.create_dataset("tcp_pose",        data=tcp_poses)
        obs.create_dataset("tcp_velocity",    data=tcp_vels)
        obs.create_dataset("tcp_error",       data=tcp_errors)
        obs.create_dataset("joint_positions", data=joint_pos)
        obs.create_dataset("joint_velocity",  data=joint_vel)
        obs.create_dataset("wrist_force",     data=wrist_forces)
        obs.create_dataset("timestamps",      data=timestamps)
        obs.create_dataset("relative_pose",   data=relative_poses)
        obs.create_dataset("relative_pose_valid", data=relative_valid)
        tf_group = obs.create_group("privileged_tf")
        tf_group.create_dataset("transforms", data=privileged_tf)
        tf_group.create_dataset("valid", data=privileged_tf_valid)
        tf_group.create_dataset(
            "frame_pairs",
            data=np.asarray(tf_frame_pairs, dtype=h5py.string_dtype(encoding="utf-8")),
        )

        act = self._hf.create_group("actions")
        act.create_dataset("commanded_pose", data=cmd_poses)
        act.create_dataset("delta_pose", data=delta_actions)
        act.create_dataset("velocity", data=velocity_actions)

        meta = self._hf.create_group("metadata")
        meta.attrs["schema_version"] = SCHEMA_VERSION
        meta.attrs["episode_id"] = ep.episode_id
        meta.attrs["task_id"]    = ep.task_id
        meta.attrs["port_type"]  = ep.port_type
        meta.attrs["port_name"]  = ep.port_name
        meta.attrs["success"]    = int(ep.success)
        meta.attrs["num_frames"] = T
        meta.attrs["duration_s"] = ep.end_time - ep.start_time
        meta.attrs["max_force"] = max_force
        meta.attrs["final_error"] = final_error
        meta.attrs["insertion_time"] = insertion_time
        meta.attrs["contact_duration"] = contact_duration
        meta.attrs["contact_force_threshold_n"] = CONTACT_FORCE_THRESHOLD_N
