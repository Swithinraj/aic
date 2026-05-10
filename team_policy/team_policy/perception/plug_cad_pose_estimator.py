"""CAD-based plug tip projection and small online grasp refinement.

This module intentionally uses only runtime information available through the
policy Observation, YOLO detections, camera_info, controller TCP pose, and TF
camera transforms.  It does not query simulator model state or scoring topics.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:  # pragma: no cover - optional runtime dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - optional runtime dependency
    import trimesh
except Exception:  # pragma: no cover
    trimesh = None


_CAMERAS = ("left", "center", "right")
_SFP_OFFSET_XYZ_RPY = (0.0, 0.015385, 0.04245, 0.4432, -0.4838, 1.3303)
_SC_OFFSET_XYZ_RPY = (0.0, 0.015385, 0.04045, 0.4432, -0.4838, 1.3303)


def _norm_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def xyz_rpy_to_matrix(values: tuple[float, float, float, float, float, float]) -> np.ndarray:
    x, y, z, roll, pitch, yaw = [float(v) for v in values]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T


def pose_to_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        q = q / n
    x, y, z, w = [float(v) for v in q]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        return (T[:3, :3] @ pts.reshape(3)) + T[:3, 3]
    return (T[:3, :3] @ pts.T).T + T[:3, 3][None, :]


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)


def _bbox_inside(uv: np.ndarray, bbox: np.ndarray, pad_frac: float = 0.30) -> bool:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 -= pad_frac * bw
    x2 += pad_frac * bw
    y1 -= pad_frac * bh
    y2 += pad_frac * bh
    return bool(x1 <= float(uv[0]) <= x2 and y1 <= float(uv[1]) <= y2)


def _clip_bbox(bbox: np.ndarray, image_hw: tuple[int, int], pad_frac: float = 0.35) -> Optional[tuple[int, int, int, int]]:
    h, w = image_hw
    if h <= 2 or w <= 2:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    x1 -= pad_frac * bw
    x2 += pad_frac * bw
    y1 -= pad_frac * bh
    y2 += pad_frac * bh
    ix1 = int(np.clip(math.floor(x1), 0, w - 2))
    iy1 = int(np.clip(math.floor(y1), 0, h - 2))
    ix2 = int(np.clip(math.ceil(x2), ix1 + 2, w))
    iy2 = int(np.clip(math.ceil(y2), iy1 + 2, h))
    if ix2 <= ix1 + 2 or iy2 <= iy1 + 2:
        return None
    return ix1, iy1, ix2, iy2


@dataclass
class PlugCadModel:
    plug_key: str
    mesh_path: Path
    keypoint_path: Path
    logger: Callable[[str], None] = lambda _msg: None
    vertices: np.ndarray = field(init=False, repr=False)
    bounds: np.ndarray = field(init=False)
    edge_samples_model: np.ndarray = field(init=False, repr=False)
    plug_tip_model: np.ndarray = field(init=False)
    plug_axis_model: np.ndarray = field(init=False)
    rear_point_model: np.ndarray = field(init=False)
    body_corners_model: np.ndarray = field(init=False, repr=False)
    front_face_corners_model: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertices = self._load_vertices()
        if self.vertices.size == 0:
            self.vertices = self._fallback_vertices()
            self.logger(f"HYBRID_CAD_WARN plug={self.plug_key} reason=mesh_unavailable using=fallback_bounds")
        self.bounds = np.asarray([np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)], dtype=np.float64)
        self._auto_semantic_points()
        self._apply_keypoint_override()
        self.edge_samples_model = self._make_edge_samples()

    def _load_vertices(self) -> np.ndarray:
        if trimesh is None or not self.mesh_path.exists():
            return np.zeros((0, 3), dtype=np.float64)
        try:
            loaded = trimesh.load(str(self.mesh_path), force="scene")
            geoms = []
            if hasattr(loaded, "geometry"):
                for geom in loaded.geometry.values():
                    if hasattr(geom, "vertices") and len(geom.vertices):
                        geoms.append(geom.copy())
            if geoms:
                mesh = trimesh.util.concatenate(geoms)
            else:
                mesh = trimesh.load(str(self.mesh_path), force="mesh")
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            return vertices.reshape(-1, 3) if vertices.size else np.zeros((0, 3), dtype=np.float64)
        except Exception as exc:
            self.logger(f"HYBRID_CAD_WARN plug={self.plug_key} reason=mesh_load_failed detail={exc}")
            return np.zeros((0, 3), dtype=np.float64)

    def _fallback_vertices(self) -> np.ndarray:
        if self.plug_key == "sc":
            lo = np.array([-0.046, -0.0125, -0.0051], dtype=np.float64)
            hi = np.array([0.01165, 0.0125, 0.0051], dtype=np.float64)
        else:
            lo = np.array([-0.0074, -0.0063, -0.02365], dtype=np.float64)
            hi = np.array([0.0074, 0.0061, 0.0328], dtype=np.float64)
        corners = []
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    corners.append([x, y, z])
        return np.asarray(corners, dtype=np.float64)

    def _auto_semantic_points(self) -> None:
        lo = self.bounds[0]
        hi = self.bounds[1]
        center = 0.5 * (lo + hi)
        ext = hi - lo
        axis_idx = int(np.argmax(ext))
        sign = 1.0 if self.plug_key == "sc" else -1.0
        self.plug_tip_model = center.copy()
        self.rear_point_model = center.copy()
        if sign > 0:
            self.plug_tip_model[axis_idx] = hi[axis_idx]
            self.rear_point_model[axis_idx] = lo[axis_idx]
        else:
            self.plug_tip_model[axis_idx] = lo[axis_idx]
            self.rear_point_model[axis_idx] = hi[axis_idx]
        axis = self.plug_tip_model - self.rear_point_model
        axis_norm = float(np.linalg.norm(axis))
        self.plug_axis_model = axis / max(axis_norm, 1e-12)
        self.body_corners_model = self._box_corners(lo, hi)
        front_lo = lo.copy()
        front_hi = hi.copy()
        if sign > 0:
            front_lo[axis_idx] = hi[axis_idx]
        else:
            front_hi[axis_idx] = lo[axis_idx]
        self.front_face_corners_model = self._box_corners(front_lo, front_hi)

    @staticmethod
    def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        out = []
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    out.append([x, y, z])
        return np.asarray(out, dtype=np.float64)

    def _apply_keypoint_override(self) -> None:
        if not self.keypoint_path.exists():
            self.logger(
                f"HYBRID_CAD_WARN plug={self.plug_key} reason=keypoint_json_missing "
                f"path={self.keypoint_path} using=mesh_bounds"
            )
            return
        try:
            data = json.loads(self.keypoint_path.read_text())
        except Exception as exc:
            self.logger(f"HYBRID_CAD_WARN plug={self.plug_key} reason=keypoint_json_invalid detail={exc}")
            return

        def _point(name: str) -> Optional[np.ndarray]:
            val = data.get(name)
            if not isinstance(val, list) or len(val) != 3:
                return None
            try:
                arr = np.asarray(val, dtype=np.float64)
            except Exception:
                return None
            return arr if np.all(np.isfinite(arr)) else None

        tip = _point("plug_tip_model")
        rear = _point("rear_point_model")
        axis = _point("plug_axis_model")
        if tip is not None:
            self.plug_tip_model = tip
        if rear is not None:
            self.rear_point_model = rear
        if axis is None:
            axis = self.plug_tip_model - self.rear_point_model
        n = float(np.linalg.norm(axis))
        if n > 1e-12:
            self.plug_axis_model = axis / n
        front = data.get("front_face_corners_model")
        if isinstance(front, list):
            try:
                arr = np.asarray(front, dtype=np.float64).reshape(-1, 3)
                if arr.shape[0] >= 4 and np.all(np.isfinite(arr)):
                    self.front_face_corners_model = arr
            except Exception:
                pass
        body = data.get("body_corners_model")
        if isinstance(body, list):
            try:
                arr = np.asarray(body, dtype=np.float64).reshape(-1, 3)
                if arr.shape[0] >= 8 and np.all(np.isfinite(arr)):
                    self.body_corners_model = arr
            except Exception:
                pass

    def _make_edge_samples(self) -> np.ndarray:
        lo = self.bounds[0]
        hi = self.bounds[1]
        corners = self._box_corners(lo, hi)
        pairs = (
            (0, 1),
            (0, 2),
            (0, 4),
            (3, 1),
            (3, 2),
            (3, 7),
            (5, 1),
            (5, 4),
            (5, 7),
            (6, 2),
            (6, 4),
            (6, 7),
        )
        samples = []
        for i, j in pairs:
            a = corners[i]
            b = corners[j]
            for alpha in np.linspace(0.0, 1.0, 12):
                samples.append((1.0 - alpha) * a + alpha * b)
        samples.extend(self.front_face_corners_model.tolist())
        samples.extend(self.body_corners_model.tolist())
        return np.asarray(samples, dtype=np.float64).reshape(-1, 3)


@dataclass
class GraspPrior:
    plug_key: str
    max_translation_m: float = 0.004
    max_rotation_rad: float = 0.08
    T_tcp_plug_nominal: np.ndarray = field(init=False)
    T_tcp_plug_current: np.ndarray = field(init=False)
    last_update_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.T_tcp_plug_nominal = xyz_rpy_to_matrix(_SC_OFFSET_XYZ_RPY if self.plug_key == "sc" else _SFP_OFFSET_XYZ_RPY)
        self.T_tcp_plug_current = self.T_tcp_plug_nominal.copy()

    def update_if_good(self, T_tcp_plug: np.ndarray, fit_score: float, min_fit_score: float) -> None:
        if not np.isfinite(fit_score) or fit_score < min_fit_score:
            return
        candidate = np.asarray(T_tcp_plug, dtype=np.float64).reshape(4, 4)
        delta = np.linalg.inv(self.T_tcp_plug_nominal) @ candidate
        trans = delta[:3, 3]
        if float(np.linalg.norm(trans)) > self.max_translation_m * 1.5:
            return
        rot_trace = float(np.trace(delta[:3, :3]))
        angle = math.acos(float(np.clip((rot_trace - 1.0) * 0.5, -1.0, 1.0)))
        if angle > self.max_rotation_rad * 1.5:
            return
        self.T_tcp_plug_current = candidate.copy()
        self.last_update_time = time.monotonic()


@dataclass
class PlugPoseEstimate:
    T_base_plug: np.ndarray
    T_tcp_plug: np.ndarray
    plug_tip_base: np.ndarray
    plug_axis_base: np.ndarray
    per_camera_projected_tip_uv: dict[str, np.ndarray]
    per_camera_projected_axis_uv: dict[str, tuple[np.ndarray, np.ndarray]]
    fit_score: float
    valid_cameras: list[str]
    timestamp: float
    source: str
    per_camera_scores: dict[str, float] = field(default_factory=dict)


class PlugCadPoseEstimator:
    def __init__(
        self,
        assets_root: str | Path,
        max_translation_m: float = 0.004,
        max_rotation_rad: float = 0.08,
        min_fit_score: float = 0.35,
        hold_s: float = 1.0,
        refine_enable: bool = True,
        logger: Callable[[str], None] = lambda _msg: None,
    ):
        self.assets_root = Path(assets_root).expanduser()
        self.max_translation_m = float(max_translation_m)
        self.max_rotation_rad = float(max_rotation_rad)
        self.min_fit_score = float(min_fit_score)
        self.hold_s = float(hold_s)
        self.refine_enable = bool(refine_enable)
        self.logger = logger
        package_dir = Path(__file__).resolve().parent
        self.models = {
            "sfp": PlugCadModel(
                "sfp",
                self.assets_root / "SFP Module" / "sfp_module_visual.glb",
                package_dir / "plug_keypoints_sfp.json",
                logger=self.logger,
            ),
            "sc": PlugCadModel(
                "sc",
                self.assets_root / "SC Plug" / "sc_plug_visual.glb",
                package_dir / "plug_keypoints_sc.json",
                logger=self.logger,
            ),
        }
        self.grasp_priors = {
            "sfp": GraspPrior("sfp", self.max_translation_m, self.max_rotation_rad),
            "sc": GraspPrior("sc", self.max_translation_m, self.max_rotation_rad),
        }
        self._last_estimates: dict[str, PlugPoseEstimate] = {}

    def resolve_plug_key(self, plug_type: str = "", plug_name: str = "") -> str:
        name = _norm_name(plug_type) or _norm_name(plug_name)
        if "sc" in name:
            return "sc"
        return "sfp"

    @staticmethod
    def image_msg_to_bgr(img_msg) -> Optional[np.ndarray]:
        if img_msg is None:
            return None
        try:
            h = int(getattr(img_msg, "height", 0))
            w = int(getattr(img_msg, "width", 0))
            step = int(getattr(img_msg, "step", 0))
            encoding = str(getattr(img_msg, "encoding", "") or "").lower()
            data = np.frombuffer(getattr(img_msg, "data", b""), dtype=np.uint8)
            if h <= 0 or w <= 0 or step <= 0 or data.size < h * step:
                return None
            row = data[: h * step].reshape(h, step)
            channels = max(1, step // max(1, w))
            img = row[:, : w * channels].reshape(h, w, channels)
            if encoding in {"bgr8", "bgra8"}:
                return img[:, :, :3].copy()
            if encoding in {"rgb8", "rgba8"}:
                return img[:, :, :3][:, :, ::-1].copy()
            if channels >= 3:
                return img[:, :, :3].copy()
        except Exception:
            return None
        return None

    def project_from_tcp(
        self,
        tcp_position: np.ndarray,
        tcp_quaternion_xyzw: np.ndarray,
        plug_type: str,
        plug_name: str,
        camera_intrinsics: dict[str, tuple[float, float, float, float]],
        T_camera_base_by_camera: dict[str, np.ndarray],
        image_hw_by_camera: dict[str, tuple[int, int]],
        T_tcp_plug: Optional[np.ndarray] = None,
    ) -> PlugPoseEstimate:
        plug_key = self.resolve_plug_key(plug_type, plug_name)
        model = self.models[plug_key]
        prior = self.grasp_priors[plug_key]
        T_base_tcp = pose_to_matrix(tcp_position, tcp_quaternion_xyzw)
        T_tcp_plug_current = prior.T_tcp_plug_current if T_tcp_plug is None else np.asarray(T_tcp_plug, dtype=np.float64)
        return self._build_estimate(
            plug_key,
            model,
            T_base_tcp,
            T_tcp_plug_current,
            camera_intrinsics,
            T_camera_base_by_camera,
            image_hw_by_camera,
            source="cad_prior_only",
            fit_score=0.20,
            per_camera_scores={},
        )

    def estimate(
        self,
        obs_msg,
        tcp_position: np.ndarray,
        tcp_quaternion_xyzw: np.ndarray,
        plug_type: str,
        plug_name: str,
        detections_by_camera: dict[str, Optional[dict]],
        camera_intrinsics: dict[str, tuple[float, float, float, float]],
        T_camera_base_by_camera: dict[str, np.ndarray],
        image_hw_by_camera: dict[str, tuple[int, int]],
        refine_enable: Optional[bool] = None,
    ) -> PlugPoseEstimate:
        plug_key = self.resolve_plug_key(plug_type, plug_name)
        model = self.models[plug_key]
        prior = self.grasp_priors[plug_key]
        T_base_tcp = pose_to_matrix(tcp_position, tcp_quaternion_xyzw)
        use_refine = self.refine_enable if refine_enable is None else bool(refine_enable)
        T_tcp_plug = prior.T_tcp_plug_current.copy()
        source = "cad_prior_only"
        fit_score = 0.20
        per_camera_scores: dict[str, float] = {}

        if use_refine:
            candidate, candidate_score, per_camera_scores = self._refine_small_delta(
                obs_msg,
                plug_key,
                model,
                T_base_tcp,
                T_tcp_plug,
                detections_by_camera,
                camera_intrinsics,
                T_camera_base_by_camera,
                image_hw_by_camera,
            )
            if candidate is not None and candidate_score >= self.min_fit_score:
                T_tcp_plug = candidate
                fit_score = float(candidate_score)
                source = "cad_edge_refine"
                prior.update_if_good(T_tcp_plug, fit_score, self.min_fit_score)
            elif candidate is not None:
                fit_score = float(candidate_score)

        estimate = self._build_estimate(
            plug_key,
            model,
            T_base_tcp,
            T_tcp_plug,
            camera_intrinsics,
            T_camera_base_by_camera,
            image_hw_by_camera,
            source=source,
            fit_score=fit_score,
            per_camera_scores=per_camera_scores,
        )
        if estimate.valid_cameras:
            self._last_estimates[plug_key] = estimate
            return estimate
        last = self._last_estimates.get(plug_key)
        if last is not None and time.monotonic() - last.timestamp <= self.hold_s:
            held = PlugPoseEstimate(
                T_base_plug=last.T_base_plug.copy(),
                T_tcp_plug=last.T_tcp_plug.copy(),
                plug_tip_base=last.plug_tip_base.copy(),
                plug_axis_base=last.plug_axis_base.copy(),
                per_camera_projected_tip_uv={k: v.copy() for k, v in last.per_camera_projected_tip_uv.items()},
                per_camera_projected_axis_uv={
                    k: (v[0].copy(), v[1].copy()) for k, v in last.per_camera_projected_axis_uv.items()
                },
                fit_score=last.fit_score,
                valid_cameras=list(last.valid_cameras),
                timestamp=time.monotonic(),
                source="last_hold",
                per_camera_scores=dict(last.per_camera_scores),
            )
            return held
        return estimate

    def _build_estimate(
        self,
        plug_key: str,
        model: PlugCadModel,
        T_base_tcp: np.ndarray,
        T_tcp_plug: np.ndarray,
        camera_intrinsics: dict[str, tuple[float, float, float, float]],
        T_camera_base_by_camera: dict[str, np.ndarray],
        image_hw_by_camera: dict[str, tuple[int, int]],
        source: str,
        fit_score: float,
        per_camera_scores: dict[str, float],
    ) -> PlugPoseEstimate:
        del plug_key
        T_base_plug = T_base_tcp @ T_tcp_plug
        tip_base = transform_points(T_base_plug, model.plug_tip_model)
        axis_point_model = model.plug_tip_model + 0.012 * model.plug_axis_model
        axis_point_base = transform_points(T_base_plug, axis_point_model)
        axis_base = axis_point_base - tip_base
        n = float(np.linalg.norm(axis_base))
        if n > 1e-12:
            axis_base = axis_base / n
        tip_uv: dict[str, np.ndarray] = {}
        axis_uv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        valid: list[str] = []
        for cam in _CAMERAS:
            intr = camera_intrinsics.get(cam)
            T_camera_base = T_camera_base_by_camera.get(cam)
            if intr is None or T_camera_base is None:
                continue
            uv_tip = self._project_point(T_camera_base, intr, tip_base)
            uv_axis = self._project_point(T_camera_base, intr, axis_point_base)
            if uv_tip is None:
                continue
            tip_uv[cam] = uv_tip
            if uv_axis is not None:
                axis_uv[cam] = (uv_tip, uv_axis)
            h, w = image_hw_by_camera.get(cam, (0, 0))
            margin = 2.0
            if h > 0 and w > 0 and margin <= uv_tip[0] <= w - margin and margin <= uv_tip[1] <= h - margin:
                valid.append(cam)
        return PlugPoseEstimate(
            T_base_plug=T_base_plug,
            T_tcp_plug=np.asarray(T_tcp_plug, dtype=np.float64).copy(),
            plug_tip_base=tip_base,
            plug_axis_base=axis_base,
            per_camera_projected_tip_uv=tip_uv,
            per_camera_projected_axis_uv=axis_uv,
            fit_score=float(fit_score),
            valid_cameras=valid,
            timestamp=time.monotonic(),
            source=source,
            per_camera_scores=dict(per_camera_scores),
        )

    @staticmethod
    def _project_point(
        T_camera_base: np.ndarray,
        intr: tuple[float, float, float, float],
        point_base: np.ndarray,
    ) -> Optional[np.ndarray]:
        point_cam = transform_points(np.asarray(T_camera_base, dtype=np.float64), np.asarray(point_base, dtype=np.float64))
        X, Y, Z = [float(v) for v in point_cam[:3]]
        if Z <= 1e-6:
            return None
        fx, fy, cx, cy = [float(v) for v in intr]
        return np.array([fx * X / Z + cx, fy * Y / Z + cy], dtype=np.float64)

    def _refine_small_delta(
        self,
        obs_msg,
        plug_key: str,
        model: PlugCadModel,
        T_base_tcp: np.ndarray,
        T_tcp_plug_start: np.ndarray,
        detections_by_camera: dict[str, Optional[dict]],
        camera_intrinsics: dict[str, tuple[float, float, float, float]],
        T_camera_base_by_camera: dict[str, np.ndarray],
        image_hw_by_camera: dict[str, tuple[int, int]],
    ) -> tuple[Optional[np.ndarray], float, dict[str, float]]:
        if cv2 is None:
            return None, 0.0, {}
        camera_contexts = self._build_edge_contexts(
            obs_msg,
            detections_by_camera,
            camera_intrinsics,
            T_camera_base_by_camera,
            image_hw_by_camera,
        )
        if not camera_contexts:
            return None, 0.0, {}

        def _candidate(delta: np.ndarray) -> np.ndarray:
            T_delta = xyz_rpy_to_matrix(
                (
                    float(delta[0]),
                    float(delta[1]),
                    float(delta[2]),
                    float(delta[3]),
                    float(delta[4]),
                    float(delta[5]),
                )
            )
            return T_tcp_plug_start @ T_delta

        def _score(delta: np.ndarray) -> tuple[float, dict[str, float]]:
            return self._fit_score(
                model,
                T_base_tcp @ _candidate(delta),
                camera_contexts,
                plug_key,
            )

        delta = np.zeros(6, dtype=np.float64)
        best_score, best_per_camera = _score(delta)
        steps = np.array(
            [
                self.max_translation_m * 0.5,
                self.max_translation_m * 0.5,
                self.max_translation_m * 0.5,
                self.max_rotation_rad * 0.5,
                self.max_rotation_rad * 0.5,
                self.max_rotation_rad * 0.5,
            ],
            dtype=np.float64,
        )
        bounds = np.array(
            [
                self.max_translation_m,
                self.max_translation_m,
                self.max_translation_m,
                self.max_rotation_rad,
                self.max_rotation_rad,
                self.max_rotation_rad,
            ],
            dtype=np.float64,
        )
        for _level in range(3):
            improved = True
            while improved:
                improved = False
                for dim in range(6):
                    for sign in (-1.0, 1.0):
                        trial = delta.copy()
                        trial[dim] = float(np.clip(trial[dim] + sign * steps[dim], -bounds[dim], bounds[dim]))
                        score, per_camera = _score(trial)
                        if score > best_score + 1e-4:
                            delta = trial
                            best_score = score
                            best_per_camera = per_camera
                            improved = True
            steps *= 0.5
        return _candidate(delta), float(best_score), best_per_camera

    def _build_edge_contexts(
        self,
        obs_msg,
        detections_by_camera: dict[str, Optional[dict]],
        camera_intrinsics: dict[str, tuple[float, float, float, float]],
        T_camera_base_by_camera: dict[str, np.ndarray],
        image_hw_by_camera: dict[str, tuple[int, int]],
    ) -> dict[str, dict]:
        contexts: dict[str, dict] = {}
        for cam in _CAMERAS:
            det = detections_by_camera.get(cam)
            intr = camera_intrinsics.get(cam)
            T_camera_base = T_camera_base_by_camera.get(cam)
            image_hw = image_hw_by_camera.get(cam, (0, 0))
            if det is None or intr is None or T_camera_base is None:
                continue
            bbox_val = det.get("bbox_xyxy_feature") or det.get("bbox_xyxy") or det.get("bbox_xyxy_raw")
            if bbox_val is None or len(bbox_val) < 4:
                continue
            bbox = np.asarray(bbox_val[:4], dtype=np.float64)
            image = self.image_msg_to_bgr(getattr(obs_msg, f"{cam}_image", None))
            if image is None:
                continue
            h, w = image.shape[:2]
            image_hw = image_hw if image_hw[0] > 0 and image_hw[1] > 0 else (h, w)
            roi = _clip_bbox(bbox, image_hw, pad_frac=0.40)
            if roi is None:
                continue
            x1, y1, x2, y2 = roi
            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(gray, 40, 140)
            edge_density = float(np.count_nonzero(edges)) / max(1.0, float(edges.size))
            if edge_density < 0.005:
                continue
            dist = cv2.distanceTransform((255 - edges).astype(np.uint8), cv2.DIST_L2, 3)
            near_border = self._bbox_border_penalty(bbox, image_hw)
            try:
                conf = float(det.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            weight = float(np.clip(conf, 0.0, 1.0)) * float(np.clip(edge_density * 25.0, 0.0, 1.0)) * near_border
            if weight <= 0.01:
                continue
            contexts[cam] = {
                "bbox": bbox,
                "roi": roi,
                "dist": dist,
                "intr": intr,
                "T_camera_base": T_camera_base,
                "image_hw": image_hw,
                "weight": weight,
                "edge_density": edge_density,
            }
        return contexts

    @staticmethod
    def _bbox_border_penalty(bbox: np.ndarray, image_hw: tuple[int, int]) -> float:
        h, w = image_hw
        if h <= 1 or w <= 1:
            return 0.0
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        margin = min(x1, y1, w - x2, h - y2)
        return float(np.clip(margin / 25.0, 0.15, 1.0))

    def _fit_score(
        self,
        model: PlugCadModel,
        T_base_plug: np.ndarray,
        camera_contexts: dict[str, dict],
        plug_key: str,
    ) -> tuple[float, dict[str, float]]:
        del plug_key
        points_base = transform_points(T_base_plug, model.edge_samples_model)
        total_num = 0.0
        total_den = 0.0
        per_camera: dict[str, float] = {}
        for cam, ctx in camera_contexts.items():
            intr = ctx["intr"]
            T_camera_base = ctx["T_camera_base"]
            projected = self._project_points(T_camera_base, intr, points_base)
            if projected.size == 0:
                continue
            bbox = ctx["bbox"]
            in_crop = np.array([_bbox_inside(uv, bbox, pad_frac=0.45) for uv in projected], dtype=bool)
            if not np.any(in_crop):
                continue
            pts = projected[in_crop]
            x1, y1, _x2, _y2 = ctx["roi"]
            dist = ctx["dist"]
            h, w = dist.shape[:2]
            uu = np.clip(np.round(pts[:, 0] - x1).astype(np.int32), 0, w - 1)
            vv = np.clip(np.round(pts[:, 1] - y1).astype(np.int32), 0, h - 1)
            d = dist[vv, uu]
            mean_dist = float(np.mean(np.clip(d, 0.0, 30.0)))
            inside_frac = float(np.mean(in_crop))
            bbox_center = _bbox_center(bbox)
            proj_center = np.mean(pts, axis=0)
            center_dist = float(np.linalg.norm(proj_center - bbox_center))
            diag = float(np.linalg.norm([bbox[2] - bbox[0], bbox[3] - bbox[1]]))
            center_score = max(0.0, 1.0 - center_dist / max(1.0, diag))
            edge_score = math.exp(-mean_dist / 8.0)
            score = float(np.clip(0.65 * edge_score + 0.25 * inside_frac + 0.10 * center_score, 0.0, 1.0))
            weighted = score * float(ctx["weight"])
            total_num += weighted
            total_den += float(ctx["weight"])
            per_camera[cam] = score
        if total_den <= 1e-9:
            return 0.0, per_camera
        return float(total_num / total_den), per_camera

    @staticmethod
    def _project_points(
        T_camera_base: np.ndarray,
        intr: tuple[float, float, float, float],
        points_base: np.ndarray,
    ) -> np.ndarray:
        points_cam = transform_points(np.asarray(T_camera_base, dtype=np.float64), points_base)
        z = points_cam[:, 2]
        good = z > 1e-6
        if not np.any(good):
            return np.zeros((0, 2), dtype=np.float64)
        pts = points_cam[good]
        fx, fy, cx, cy = [float(v) for v in intr]
        uv = np.empty((pts.shape[0], 2), dtype=np.float64)
        uv[:, 0] = fx * pts[:, 0] / pts[:, 2] + cx
        uv[:, 1] = fy * pts[:, 1] / pts[:, 2] + cy
        return uv
