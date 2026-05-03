"""
Records one episode of (observation, action) pairs to HDF5 — Schema v9.

New in v9 vs v8:
  * fused observations/yolo_port_valid now means fresh target detection, not hold-last existence
  * fused observations/yolo_port_age (1D) stores staleness seconds for the held target port position

New in v8 vs v7:
  * target_module_onehot (7D) stored under observations/target_module_onehot
  * metadata stores the exact target_module_name and its deterministic onehot encoding

New in v7 vs v6:
  * tared_wrist_force_torque (6D) stored under observations/tared_wrist_force_torque
  * port_delta_tcp (3D) stored under observations/port_delta_tcp
  * plug_type_onehot (2D) stored under observations/plug_type_onehot
  * metadata stores the ROS task goal fields and the 6D wrist wrench tare used during collection

New in v6 vs v5:
  * Per-camera YOLO feature vectors stored under
    observations/yolo_per_camera/{left,center,right}/features  shape (T, 7)
    Each vector: [confidence, bbox_cx_norm, bbox_cy_norm, bbox_w_norm,
                  bbox_h_norm, valid_float, age_seconds]
    - valid_float = 1.0 when age < AGE_VALID_S, else 0.0
    - age_seconds clamped to [0, MAX_AGE_S]
    - bbox coordinates are normalised by the original image (H, W) at collection time
  * wrist_force (6D force/torque) is included in all prior schemas but is NOW
    explicitly incorporated into the training state via convert_to_lerobot_v2.py
  * Image dimensions stored as attributes on observations/images group

All v5 fields are preserved unchanged so episodes can still be opened with
validate_episode.py / convert_to_lerobot.py as a fallback.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCHEMA_VERSION = "9"

CONTACT_FORCE_THRESHOLD_N = 0.5
AGE_VALID_S  = 0.15   # detection fresher than this → valid_float = 1.0
MAX_AGE_S    = 10.0   # age clamped at this value

DEFAULT_MAX_FINAL_ERROR_M: dict = {
    "sfp_port":  0.003,
    "sc_port":   0.010,
    "default":   0.015,
}
DEFAULT_MAX_SUSTAINED_FORCE_DURATION_S: float = 0.5
FORCE_PENALTY_N: float = 20.0

CAMERAS = ("left", "center", "right")
TARGET_MODULE_NAMES = (
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
    "sc_port_0",
    "sc_port_1",
)
IMAGE_COMPRESSION_OPTS = 1


# ---------------------------------------------------------------------------
# Math helpers (same as v1, kept self-contained)
# ---------------------------------------------------------------------------

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = _quat_normalize(a)
    bx, by, bz, bw = _quat_normalize(b)
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float32)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _quat_normalize(q)
    return np.array([-x, -y, -z, w], dtype=np.float32)


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _quat_normalize(q)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sh = float(np.linalg.norm([x, y, z]))
    if sh < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sh, max(abs(float(w)), 1e-12))
    return np.array([x/sh*angle, y/sh*angle, z/sh*angle], dtype=np.float32)


def _pose_delta(current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    if not np.any(target_pose):
        return np.zeros(6, dtype=np.float32)
    dp = np.asarray(target_pose[:3] - current_pose[:3], dtype=np.float32)
    dq = _quat_multiply(target_pose[3:], _quat_inverse(current_pose[3:]))
    return np.concatenate([dp, _quat_to_axis_angle(dq)]).astype(np.float32)


def _compute_target_velocity_actions(cmd_poses: np.ndarray,
                                      timestamps: np.ndarray) -> np.ndarray:
    T = cmd_poses.shape[0]
    velocities = np.zeros((T, 6), dtype=np.float32)
    valid = np.any(cmd_poses, axis=1)
    for i in range(T - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        dt = float(timestamps[i + 1] - timestamps[i])
        if dt <= 1e-6:
            continue
        velocities[i] = _pose_delta(cmd_poses[i], cmd_poses[i + 1]) / dt
    return velocities


# ---------------------------------------------------------------------------
# Per-camera YOLO feature vector builder
# ---------------------------------------------------------------------------

def build_yolo_feature(
    confidence: float,
    bbox_xyxy: Optional[list],
    img_h: int,
    img_w: int,
    last_det_time: Optional[float],
    now: Optional[float] = None,
) -> np.ndarray:
    """Return 7D feature [conf, cx, cy, w, h, valid, age] for one camera.

    bbox_xyxy is in original image pixel coordinates (x1,y1,x2,y2).
    Pass bbox_xyxy=None / confidence=0 for a no-detection frame.
    """
    if now is None:
        now = time.time()
    age = min(MAX_AGE_S, now - last_det_time) if last_det_time is not None else MAX_AGE_S
    valid = 1.0 if age < AGE_VALID_S else 0.0

    if bbox_xyxy is not None and len(bbox_xyxy) == 4 and img_h > 0 and img_w > 0:
        x1, y1, x2, y2 = bbox_xyxy
        cx = float((x1 + x2) / 2) / img_w
        cy = float((y1 + y2) / 2) / img_h
        bw = float(x2 - x1) / img_w
        bh = float(y2 - y1) / img_h
    else:
        cx = cy = bw = bh = 0.0
        confidence = 0.0

    return np.array([confidence, cx, cy, bw, bh, valid, age], dtype=np.float32)


def build_plug_type_onehot(plug_type: object) -> np.ndarray:
    """Return [is_sfp, is_sc] for the task plug type."""
    value = str(plug_type).strip().lower()
    if value == "sfp":
        return np.array([1.0, 0.0], dtype=np.float32)
    if value == "sc":
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def build_target_module_onehot(target_module_name: object) -> np.ndarray:
    """Return exact onehot for supported target_module_name values."""
    value = str(target_module_name).strip().lower()
    vec = np.zeros(len(TARGET_MODULE_NAMES), dtype=np.float32)
    try:
        vec[TARGET_MODULE_NAMES.index(value)] = 1.0
    except ValueError:
        pass
    return vec


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    timestamp: float
    task_id: str
    tcp_pose: np.ndarray          # (7,)
    tcp_velocity: np.ndarray      # (6,)
    tcp_error: np.ndarray         # (6,)
    joint_positions: np.ndarray   # (7,)
    joint_velocity: np.ndarray    # (7,)
    wrist_force: np.ndarray       # (6,) fx fy fz tx ty tz
    tared_wrist_force_torque: np.ndarray  # (6,) tare-subtracted fx fy fz tx ty tz
    port_delta_tcp: np.ndarray    # (3,) fused yolo_port_xyz - tcp position in base_link
    plug_type_onehot: np.ndarray  # (2,) [is_sfp, is_sc]
    target_module_onehot: np.ndarray  # (7,) exact target_module_name onehot
    relative_pose: Optional[np.ndarray] = None
    privileged_tf: Optional[np.ndarray] = None
    privileged_tf_valid: Optional[np.ndarray] = None
    commanded_pose: Optional[np.ndarray] = None
    yolo_port_xyz: Optional[np.ndarray] = None   # (3,) fused xyz for v5 compat
    yolo_port_valid: bool = False   # fresh fused target detection flag, not hold-last existence
    yolo_port_age: float = MAX_AGE_S
    yolo_per_camera: Optional[Dict[str, np.ndarray]] = None  # {"left"/"center"/"right": (7,)}
    insertion_success: float = 0.0  # 1.0 from the frame when /scoring/insertion_event fires


@dataclass
class Episode:
    episode_id: int
    task_id: str
    cable_type: str
    cable_name: str
    port_type: str
    port_name: str
    plug_type: str
    plug_name: str
    target_module_name: str
    time_limit_s: int
    privileged_tf_frame_pairs: List[str] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    success: bool = False
    start_time: float = 0.0
    end_time: float = 0.0


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class EpisodeRecorderV2:
    """HDF5 episode recorder — writes Schema v9."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._current: Optional[Episode] = None
        self._last_commanded_pose: Optional[np.ndarray] = None
        self._commanded_pose_lock = threading.Lock()
        self._hf = None
        self._partial_path: Optional[str] = None
        self._final_path: Optional[str] = None
        self._img_ds: dict = {}
        self._img_h: int = 0
        self._img_w: int = 0
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        self._plug_type_onehot = np.zeros(2, dtype=np.float32)
        self._target_module_onehot = np.zeros(len(TARGET_MODULE_NAMES), dtype=np.float32)
        self._cleanup_stale_partials()

    def _cleanup_stale_partials(self) -> None:
        try:
            for name in os.listdir(self._output_dir):
                if name.endswith(".partial"):
                    try:
                        os.remove(os.path.join(self._output_dir, name))
                    except OSError:
                        pass
        except FileNotFoundError:
            pass

    def start_episode(
        self,
        episode_id: int,
        task_id: str,
        port_type: str,
        port_name: str,
        cable_type: str = "",
        cable_name: str = "",
        plug_type: str = "",
        plug_name: str = "",
        target_module_name: str = "",
        time_limit_s: int = 0,
    ) -> None:
        if self._hf is not None:
            self._abort_current()
        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py is required — add it to pixi.toml pypi-dependencies")

        self._current = Episode(
            episode_id=episode_id,
            task_id=task_id,
            cable_type=str(cable_type),
            cable_name=str(cable_name),
            port_type=port_type,
            port_name=port_name,
            plug_type=str(plug_type),
            plug_name=str(plug_name),
            target_module_name=str(target_module_name),
            time_limit_s=int(time_limit_s),
            start_time=time.time(),
        )
        with self._commanded_pose_lock:
            self._last_commanded_pose = None
        self._wrist_force_tare = np.zeros(6, dtype=np.float32)
        self._plug_type_onehot = build_plug_type_onehot(plug_type)
        self._target_module_onehot = build_target_module_onehot(target_module_name)

        self._final_path = os.path.join(
            self._output_dir, f"episode_{episode_id:05d}.hdf5"
        )
        self._partial_path = self._final_path + ".partial"
        if os.path.exists(self._partial_path):
            os.remove(self._partial_path)

        self._hf = h5py.File(self._partial_path, "w")
        self._img_ds = {}
        self._img_h = 0
        self._img_w = 0

    def set_wrist_force_tare(self, wrench_6d: np.ndarray) -> None:
        self._wrist_force_tare = np.asarray(wrench_6d, dtype=np.float32).reshape(6).copy()

    def update_commanded_pose(self, pose: np.ndarray) -> None:
        with self._commanded_pose_lock:
            self._last_commanded_pose = pose.copy()

    def set_privileged_tf_frame_pairs(self, frame_pairs: List[str]) -> None:
        if self._current is not None:
            self._current.privileged_tf_frame_pairs = list(frame_pairs)

    def _init_image_datasets(self, H: int, W: int) -> None:
        assert self._hf is not None
        obs = self._hf.require_group("observations")
        img = obs.require_group("images")
        img.attrs["height"] = H
        img.attrs["width"] = W
        for name in CAMERAS:
            self._img_ds[name] = img.create_dataset(
                name,
                shape=(0, H, W, 3),
                maxshape=(None, H, W, 3),
                chunks=(1, H, W, 3),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=IMAGE_COMPRESSION_OPTS,
            )
        self._img_h = H
        self._img_w = W

    def record_frame(
        self,
        obs,
        relative_pose: Optional[np.ndarray] = None,
        privileged_tf: Optional[np.ndarray] = None,
        privileged_tf_valid: Optional[np.ndarray] = None,
        yolo_port_xyz: Optional[np.ndarray] = None,
        yolo_port_valid: bool = False,
        yolo_port_age: float = MAX_AGE_S,
        yolo_per_camera: Optional[Dict[str, np.ndarray]] = None,
        insertion_success: float = 0.0,
    ) -> None:
        """Stream one frame to HDF5. yolo_per_camera = {"left"/"center"/"right": (7,)}."""
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

        i = len(self._current.frames)
        for cam_name, arr in (("left", left_img), ("center", center_img), ("right", right_img)):
            ds = self._img_ds[cam_name]
            ds.resize(i + 1, axis=0)
            ds[i] = arr

        cs = obs.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js  = obs.joint_states
        w   = obs.wrist_wrench.wrench
        raw_wrist_force = np.array([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ], dtype=np.float32)
        tared_wrist_force = raw_wrist_force - self._wrist_force_tare
        tcp_xyz = np.array([
            tcp.position.x, tcp.position.y, tcp.position.z,
        ], dtype=np.float32)
        port_delta_tcp = (
            np.asarray(yolo_port_xyz, dtype=np.float32)[:3] - tcp_xyz
            if yolo_port_xyz is not None else np.zeros(3, dtype=np.float32)
        )

        with self._commanded_pose_lock:
            cmd = self._last_commanded_pose.copy() if self._last_commanded_pose is not None else None

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
            wrist_force=raw_wrist_force,
            tared_wrist_force_torque=tared_wrist_force,
            port_delta_tcp=port_delta_tcp,
            plug_type_onehot=self._plug_type_onehot.copy(),
            target_module_onehot=self._target_module_onehot.copy(),
            relative_pose=(
                None if relative_pose is None
                else np.asarray(relative_pose, dtype=np.float32).copy()
            ),
            privileged_tf=(
                None if privileged_tf is None
                else np.asarray(privileged_tf, dtype=np.float32).copy()
            ),
            privileged_tf_valid=(
                None if privileged_tf_valid is None
                else np.asarray(privileged_tf_valid, dtype=np.bool_).copy()
            ),
            commanded_pose=cmd,
            yolo_port_xyz=(
                None if yolo_port_xyz is None
                else np.asarray(yolo_port_xyz, dtype=np.float32).copy()
            ),
            yolo_port_valid=bool(yolo_port_valid),
            yolo_port_age=float(np.clip(yolo_port_age, 0.0, MAX_AGE_S)),
            yolo_per_camera=(
                None if yolo_per_camera is None
                else {k: np.asarray(v, dtype=np.float32).copy()
                      for k, v in yolo_per_camera.items()}
            ),
            insertion_success=float(insertion_success),
        )
        self._current.frames.append(frame)

    def end_episode(
        self,
        success: bool,
        max_final_error_m: Optional[float] = None,
        max_sustained_force_duration_s: Optional[float] = DEFAULT_MAX_SUSTAINED_FORCE_DURATION_S,
        force_baseline_n: float = 0.0,
        insertion_event_data: str = "",
    ) -> Optional[str]:
        if self._current is None or self._hf is None:
            return None

        ep = self._current
        ep.success = success
        ep.end_time = time.time()

        reject_reason: Optional[str] = None

        if len(ep.frames) < 10:
            reject_reason = f"too_short ({len(ep.frames)} frames)"
        else:
            frames = ep.frames
            T = len(frames)
            timestamps = np.array([f.timestamp for f in frames], dtype=np.float64)
            wrist_forces = np.stack([f.wrist_force for f in frames])
            raw_force_mag = np.linalg.norm(wrist_forces[:, :3], axis=1)
            contact_force = np.maximum(0.0, raw_force_mag - force_baseline_n)

            if max_sustained_force_duration_s is not None:
                above = contact_force > FORCE_PENALTY_N
                if above.any() and T > 1:
                    dt = np.diff(timestamps)
                    dt_full = np.append(dt, np.median(dt))
                    penalty_duration = float(dt_full[above].sum())
                    if penalty_duration > max_sustained_force_duration_s:
                        reject_reason = (
                            f"force_penalty ({penalty_duration:.2f}s > "
                            f"{max_sustained_force_duration_s}s at >{FORCE_PENALTY_N}N, "
                            f"baseline={force_baseline_n:.1f}N)"
                        )

            if reject_reason is None and max_final_error_m is not None:
                relative_valid = np.array(
                    [f.relative_pose is not None for f in frames], dtype=np.bool_
                )
                valid_indices = np.flatnonzero(relative_valid)
                if valid_indices.size:
                    relative_poses = np.stack([
                        f.relative_pose if f.relative_pose is not None
                        else np.zeros(7, dtype=np.float32)
                        for f in frames
                    ])
                    final_err = float(np.linalg.norm(relative_poses[valid_indices[-1], :3]))
                    if final_err > max_final_error_m:
                        reject_reason = (
                            f"final_error ({final_err:.4f}m > {max_final_error_m}m)"
                        )

        if reject_reason is not None:
            self._abort_current()
            return None

        assert self._partial_path is not None and self._final_path is not None
        try:
            self._write_state_and_metadata(ep, force_baseline_n=force_baseline_n,
                                           insertion_event_data=insertion_event_data)
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

    def _write_state_and_metadata(self, ep: Episode, force_baseline_n: float = 0.0,
                                   insertion_event_data: str = "") -> None:
        import h5py

        assert self._hf is not None
        T = len(ep.frames)

        tcp_poses    = np.stack([f.tcp_pose        for f in ep.frames])
        tcp_vels     = np.stack([f.tcp_velocity    for f in ep.frames])
        tcp_errors   = np.stack([f.tcp_error       for f in ep.frames])
        joint_pos    = np.stack([f.joint_positions for f in ep.frames])
        joint_vel    = np.stack([f.joint_velocity  for f in ep.frames])
        wrist_forces = np.stack([f.wrist_force     for f in ep.frames])
        tared_wrist_forces = np.stack([f.tared_wrist_force_torque for f in ep.frames])
        port_delta_tcp = np.stack([f.port_delta_tcp for f in ep.frames])
        plug_type_onehot = np.stack([f.plug_type_onehot for f in ep.frames])
        target_module_onehot = np.stack([f.target_module_onehot for f in ep.frames])
        timestamps   = np.array([f.timestamp       for f in ep.frames], dtype=np.float64)
        task_ids     = [f.task_id for f in ep.frames]

        cmd_poses = np.stack([
            f.commanded_pose if f.commanded_pose is not None
            else np.zeros(7, dtype=np.float32)
            for f in ep.frames
        ])
        relative_valid = np.array(
            [f.relative_pose is not None for f in ep.frames], dtype=np.bool_
        )
        relative_poses = np.stack([
            f.relative_pose if f.relative_pose is not None
            else np.zeros(7, dtype=np.float32)
            for f in ep.frames
        ])
        tf_frame_pairs = list(ep.privileged_tf_frame_pairs)
        tf_count = len(tf_frame_pairs)
        if tf_count:
            privileged_tf = np.stack([
                f.privileged_tf if f.privileged_tf is not None
                else np.zeros((tf_count, 7), dtype=np.float32)
                for f in ep.frames
            ])
            privileged_tf_valid = np.stack([
                f.privileged_tf_valid if f.privileged_tf_valid is not None
                else np.zeros(tf_count, dtype=np.bool_)
                for f in ep.frames
            ])
        else:
            privileged_tf       = np.zeros((T, 0, 7), dtype=np.float32)
            privileged_tf_valid = np.zeros((T, 0), dtype=np.bool_)

        delta_actions    = np.stack([_pose_delta(f.tcp_pose, cmd_poses[i])
                                     for i, f in enumerate(ep.frames)])
        velocity_actions = _compute_target_velocity_actions(cmd_poses, timestamps)

        # --- force metrics ---
        raw_force_mag = np.linalg.norm(wrist_forces[:, :3], axis=1)
        contact_force = np.maximum(0.0, raw_force_mag - force_baseline_n)
        max_force = float(contact_force.max()) if T else 0.0
        if T > 1:
            dt = np.diff(timestamps)
            median_dt = float(np.median(dt)) if dt.size else 0.0
            frame_dt = np.concatenate([dt, np.array([median_dt], dtype=np.float64)])
            insertion_time = float(timestamps[-1] - timestamps[0])
        else:
            frame_dt = np.zeros(T, dtype=np.float64)
            insertion_time = 0.0
        contact_mask = contact_force > CONTACT_FORCE_THRESHOLD_N
        contact_duration = float(frame_dt[contact_mask].sum()) if T else 0.0
        penalty_mask = contact_force > FORCE_PENALTY_N
        sustained_penalty = float(frame_dt[penalty_mask].sum()) if T else 0.0

        valid_rel_indices = np.flatnonzero(relative_valid)
        final_error = (
            float(np.linalg.norm(relative_poses[valid_rel_indices[-1], :3]))
            if valid_rel_indices.size else float("nan")
        )

        # --- legacy fused YOLO xyz (v5 compat) ---
        yolo_port_valid = np.array([f.yolo_port_valid for f in ep.frames], dtype=np.bool_)
        yolo_port_age = np.array([f.yolo_port_age for f in ep.frames], dtype=np.float32)
        yolo_port_xyz = np.stack([
            f.yolo_port_xyz if f.yolo_port_xyz is not None
            else np.zeros(3, dtype=np.float32)
            for f in ep.frames
        ])
        yolo_valid_fraction = float(yolo_port_valid.mean()) if T else 0.0

        # --- insertion success flag ---
        insertion_success_arr = np.array(
            [f.insertion_success for f in ep.frames], dtype=np.float32
        )

        # --- per-camera YOLO features (v6) ---
        zero7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, MAX_AGE_S], dtype=np.float32)
        per_cam_features: Dict[str, np.ndarray] = {}
        for cam in CAMERAS:
            feat_list = []
            for f in ep.frames:
                if f.yolo_per_camera is not None and cam in f.yolo_per_camera:
                    feat_list.append(f.yolo_per_camera[cam])
                else:
                    feat_list.append(zero7.copy())
            per_cam_features[cam] = np.stack(feat_list)  # (T, 7)

        # --- Write HDF5 ---
        obs = self._hf.require_group("observations")
        obs.create_dataset("task_id",
                           data=np.asarray(task_ids,
                                           dtype=h5py.string_dtype(encoding="utf-8")))
        obs.create_dataset("tcp_pose",        data=tcp_poses)
        obs.create_dataset("tcp_velocity",    data=tcp_vels)
        obs.create_dataset("tcp_error",       data=tcp_errors)
        obs.create_dataset("joint_positions", data=joint_pos)
        obs.create_dataset("joint_velocity",  data=joint_vel)
        obs.create_dataset("wrist_force",     data=wrist_forces)
        obs.create_dataset("tared_wrist_force_torque", data=tared_wrist_forces)
        obs.create_dataset("timestamps",      data=timestamps)
        obs.create_dataset("relative_pose",       data=relative_poses)
        obs.create_dataset("relative_pose_valid", data=relative_valid)
        obs.create_dataset("yolo_port_xyz",   data=yolo_port_xyz)
        yolo_valid_ds = obs.create_dataset("yolo_port_valid", data=yolo_port_valid)
        yolo_valid_ds.attrs["semantics"] = "fresh_target_detection_not_hold_last_existence"
        yolo_age_ds = obs.create_dataset("yolo_port_age",   data=yolo_port_age)
        yolo_age_ds.attrs["units"] = "seconds"
        yolo_age_ds.attrs["max_age_s"] = MAX_AGE_S
        yolo_age_ds.attrs["age_valid_threshold_s"] = AGE_VALID_S
        obs.create_dataset("port_delta_tcp",  data=port_delta_tcp)
        plug_type_ds = obs.create_dataset("plug_type_onehot", data=plug_type_onehot)
        plug_type_ds.attrs["encoding"] = "is_sfp,is_sc"
        target_module_ds = obs.create_dataset("target_module_onehot", data=target_module_onehot)
        target_module_ds.attrs["encoding"] = ",".join(TARGET_MODULE_NAMES)
        ins_ds = obs.create_dataset("insertion_success", data=insertion_success_arr)
        ins_ds.attrs["semantics"] = "0=before_insertion 1=insertion_event_received"
        ins_ds.attrs["source_topic"] = "/scoring/insertion_event"

        tf_group = obs.create_group("privileged_tf")
        tf_group.create_dataset("transforms", data=privileged_tf)
        tf_group.create_dataset("valid",      data=privileged_tf_valid)
        tf_group.create_dataset(
            "frame_pairs",
            data=np.asarray(tf_frame_pairs,
                            dtype=h5py.string_dtype(encoding="utf-8")),
        )

        # Per-camera YOLO features (v6)
        pc_group = obs.create_group("yolo_per_camera")
        for cam in CAMERAS:
            cam_group = pc_group.create_group(cam)
            cam_group.create_dataset("features", data=per_cam_features[cam])
            cam_group.attrs["feature_names"] = (
                "confidence,bbox_cx_norm,bbox_cy_norm,bbox_w_norm,bbox_h_norm,"
                "valid_float,age_seconds"
            )
            cam_group.attrs["age_valid_threshold_s"] = AGE_VALID_S
            cam_group.attrs["max_age_s"] = MAX_AGE_S

        act = self._hf.create_group("actions")
        cmd_ds = act.create_dataset("commanded_pose", data=cmd_poses)
        cmd_ds.attrs["feature_names"] = "x,y,z,qx,qy,qz,qw"
        cmd_ds.attrs["description"] = "Expert commanded TCP pose sent via MotionUpdate.pose in base_link"
        delta_ds = act.create_dataset("delta_pose", data=delta_actions)
        delta_ds.attrs["feature_names"] = "dx,dy,dz,drx,dry,drz"
        delta_ds.attrs["description"] = "6D delta pose from current tcp_pose to commanded_pose"
        vel_ds = act.create_dataset("velocity", data=velocity_actions)
        vel_ds.attrs["feature_names"] = "vx,vy,vz,wx,wy,wz"
        vel_ds.attrs["description"] = "Finite-difference expert target velocity derived from commanded_pose"

        meta = self._hf.create_group("metadata")
        meta.attrs["schema_version"]     = SCHEMA_VERSION
        meta.attrs["episode_id"]         = ep.episode_id
        meta.attrs["task_id"]            = ep.task_id
        meta.attrs["cable_type"]         = ep.cable_type
        meta.attrs["cable_name"]         = ep.cable_name
        meta.attrs["port_type"]          = ep.port_type
        meta.attrs["port_name"]          = ep.port_name
        meta.attrs["plug_type"]          = ep.plug_type
        meta.attrs["plug_name"]          = ep.plug_name
        meta.attrs["target_module_name"] = ep.target_module_name
        meta.attrs["target_module_onehot_encoding"] = ",".join(TARGET_MODULE_NAMES)
        meta.attrs["time_limit_s"]       = ep.time_limit_s
        meta.attrs["success"]            = int(ep.success)
        meta.attrs["num_frames"]         = T
        meta.attrs["duration_s"]         = ep.end_time - ep.start_time
        meta.attrs["max_force"]          = max_force
        meta.attrs["final_error"]        = final_error
        meta.attrs["insertion_time"]     = insertion_time
        meta.attrs["contact_duration"]   = contact_duration
        meta.attrs["contact_force_threshold_n"] = CONTACT_FORCE_THRESHOLD_N
        meta.attrs["sustained_penalty_duration_s"] = sustained_penalty
        meta.attrs["force_penalty_threshold_n"] = FORCE_PENALTY_N
        meta.attrs["force_baseline_n"]   = force_baseline_n
        meta.attrs["wrist_force_tare"]   = self._wrist_force_tare
        meta.attrs["yolo_valid_fraction"] = yolo_valid_fraction
        meta.attrs["yolo_fresh_valid_fraction"] = yolo_valid_fraction
        meta.attrs["image_height"]       = self._img_h
        meta.attrs["image_width"]        = self._img_w
        meta.attrs["insertion_event_received"] = int(bool(insertion_event_data))
        meta.attrs["insertion_event_data"]     = str(insertion_event_data)
