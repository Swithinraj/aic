"""CameraInfo, TF, and projection helpers for image-space servoing.

The functions here keep projection math tied to robot/camera geometry only.
YOLO detections stay in image space; TF is used only to move points between
base_link and camera optical frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:  # pragma: no cover - runtime optional in stripped test envs
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


TransformLookup = Callable[[str, str], object | None]


@dataclass
class CameraGeometry:
    name: str
    camera_info_frame: str
    tf_frame: str
    K: np.ndarray
    D: np.ndarray
    distortion_model: str
    width: int
    height: int
    T_camera_base: np.ndarray
    T_base_camera: np.ndarray
    optical_ok: bool
    reason: str = "ok"

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def R_base_camera(self) -> np.ndarray:
        return self.T_base_camera[:3, :3]


@dataclass
class CameraGeometryResult:
    geometry: Optional[CameraGeometry]
    reason: str
    camera_info_frame: str = ""
    tf_frame: str = ""
    distortion_model: str = ""
    D: tuple[float, ...] = ()
    K: tuple[float, ...] = ()
    optical_ok: bool = False


def frame_is_optical(frame_id: str) -> bool:
    frame = str(frame_id or "").strip().lower()
    if not frame:
        return False
    return frame.endswith("_optical_frame") or frame.endswith("/optical_frame") or "optical" in frame.split("/")[-1]


def matrix_from_tf(tf_msg) -> np.ndarray:
    transform = tf_msg.transform
    trans = transform.translation
    rot = transform.rotation
    q = np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float64)
    qn = float(np.linalg.norm(q))
    if qn <= 1e-12:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        q = q / qn
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
    T[:3, 3] = np.array([trans.x, trans.y, trans.z], dtype=np.float64)
    return T


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


def camera_info_to_K_D(info) -> tuple[Optional[np.ndarray], np.ndarray, str, int, int]:
    if info is None:
        return None, np.zeros(0, dtype=np.float64), "", 0, 0
    k = getattr(info, "k", None)
    p = getattr(info, "p", None)
    K = None
    try:
        if k is not None and len(k) >= 9 and abs(float(k[0])) > 1e-9 and abs(float(k[4])) > 1e-9:
            K = np.asarray(k[:9], dtype=np.float64).reshape(3, 3)
        elif p is not None and len(p) >= 12 and abs(float(p[0])) > 1e-9 and abs(float(p[5])) > 1e-9:
            K = np.array(
                [[float(p[0]), 0.0, float(p[2])], [0.0, float(p[5]), float(p[6])], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
    except Exception:
        K = None
    d_raw = getattr(info, "d", None)
    try:
        D = np.asarray(list(d_raw) if d_raw is not None else [], dtype=np.float64).reshape(-1)
    except Exception:
        D = np.zeros(0, dtype=np.float64)
    model = str(getattr(info, "distortion_model", "") or "")
    width = int(getattr(info, "width", 0) or 0)
    height = int(getattr(info, "height", 0) or 0)
    return K, D, model, width, height


def camera_info_frame(info, image_msg=None) -> str:
    for msg in (info, image_msg):
        header = getattr(msg, "header", None)
        frame_id = str(getattr(header, "frame_id", "") or "").strip()
        if frame_id:
            return frame_id
    return ""


def optical_frame_candidates(frame_id: str, camera_name: str) -> list[str]:
    frame = str(frame_id or "").strip()
    out: list[str] = []
    if frame:
        out.append(frame)
        if not frame_is_optical(frame):
            out.extend(
                [
                    f"{frame}_optical_frame",
                    frame.replace("_frame", "_optical_frame"),
                    frame.replace("_link", "_optical_frame"),
                ]
            )
    out.extend(
        [
            f"{camera_name}_camera_optical_frame",
            f"{camera_name}_camera/color_optical_frame",
            f"{camera_name}_camera_link_optical_frame",
        ]
    )
    deduped: list[str] = []
    for candidate in out:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


class CameraGeometryCache:
    def __init__(self, camera_names: tuple[str, ...] = ("left", "center", "right"), base_frame: str = "base_link"):
        self.camera_names = camera_names
        self.base_frame = base_frame
        self.last_results: dict[str, CameraGeometryResult] = {}

    def geometry_from_observation(
        self,
        obs_msg,
        camera: str,
        lookup_transform: TransformLookup,
    ) -> CameraGeometryResult:
        image_msg = getattr(obs_msg, f"{camera}_image", None) if obs_msg is not None else None
        info = getattr(obs_msg, f"{camera}_camera_info", None) if obs_msg is not None else None
        if image_msg is None:
            result = CameraGeometryResult(None, "no_current_image")
            self.last_results[camera] = result
            return result
        if info is None:
            result = CameraGeometryResult(None, "no_camera_info")
            self.last_results[camera] = result
            return result
        K, D, model, width, height = camera_info_to_K_D(info)
        info_frame = camera_info_frame(info, image_msg)
        if width <= 0:
            width = int(getattr(image_msg, "width", 0) or 0)
        if height <= 0:
            height = int(getattr(image_msg, "height", 0) or 0)
        if K is None:
            result = CameraGeometryResult(
                None,
                "invalid_camera_info",
                camera_info_frame=info_frame,
                distortion_model=model,
                D=tuple(float(v) for v in D),
            )
            self.last_results[camera] = result
            return result

        chosen_tf = None
        chosen_frame = ""
        optical_ok = False
        for frame in optical_frame_candidates(info_frame, camera):
            if not frame_is_optical(frame):
                continue
            tf_msg = lookup_transform(frame, self.base_frame)
            if tf_msg is not None:
                chosen_tf = tf_msg
                chosen_frame = frame
                optical_ok = True
                break
        if chosen_tf is None and info_frame and frame_is_optical(info_frame):
            chosen_tf = lookup_transform(info_frame, self.base_frame)
            chosen_frame = info_frame if chosen_tf is not None else ""
            optical_ok = chosen_tf is not None
        if chosen_tf is None:
            result = CameraGeometryResult(
                None,
                "no_valid_tf_camera_transform",
                camera_info_frame=info_frame,
                tf_frame=chosen_frame or info_frame,
                distortion_model=model,
                D=tuple(float(v) for v in D),
                K=tuple(float(v) for v in K.reshape(-1)),
                optical_ok=False,
            )
            self.last_results[camera] = result
            return result
        try:
            T_camera_base = matrix_from_tf(chosen_tf)
            T_base_camera = np.linalg.inv(T_camera_base)
        except Exception:
            result = CameraGeometryResult(
                None,
                "invalid_tf_matrix",
                camera_info_frame=info_frame,
                tf_frame=chosen_frame,
                distortion_model=model,
                D=tuple(float(v) for v in D),
                K=tuple(float(v) for v in K.reshape(-1)),
                optical_ok=optical_ok,
            )
            self.last_results[camera] = result
            return result
        geom = CameraGeometry(
            name=camera,
            camera_info_frame=info_frame,
            tf_frame=chosen_frame,
            K=K,
            D=D,
            distortion_model=model,
            width=int(width),
            height=int(height),
            T_camera_base=T_camera_base,
            T_base_camera=T_base_camera,
            optical_ok=optical_ok,
        )
        result = CameraGeometryResult(
            geom,
            "ok",
            camera_info_frame=info_frame,
            tf_frame=chosen_frame,
            distortion_model=model,
            D=tuple(float(v) for v in D),
            K=tuple(float(v) for v in K.reshape(-1)),
            optical_ok=optical_ok,
        )
        self.last_results[camera] = result
        return result

    def all_from_observation(self, obs_msg, lookup_transform: TransformLookup) -> dict[str, CameraGeometryResult]:
        return {cam: self.geometry_from_observation(obs_msg, cam, lookup_transform) for cam in self.camera_names}


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        return T[:3, :3] @ pts.reshape(3) + T[:3, 3]
    return (T[:3, :3] @ pts.T).T + T[:3, 3][None, :]


def project_points_camera(points_camera: np.ndarray, geom: CameraGeometry) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    z = pts[:, 2].copy()
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64), z
    use_cv_distortion = (
        cv2 is not None
        and geom.distortion_model in {"plumb_bob", "rational_polynomial"}
        and geom.D.size > 0
        and bool(np.any(np.abs(geom.D) > 1e-12))
    )
    if use_cv_distortion:
        uv, _ = cv2.projectPoints(
            pts.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            geom.K,
            geom.D,
        )
        return uv.reshape(-1, 2).astype(np.float64), z
    uv = np.empty((pts.shape[0], 2), dtype=np.float64)
    good_z = np.abs(pts[:, 2]) > 1e-12
    uv[:] = np.nan
    uv[good_z, 0] = geom.K[0, 0] * pts[good_z, 0] / pts[good_z, 2] + geom.K[0, 2]
    uv[good_z, 1] = geom.K[1, 1] * pts[good_z, 1] / pts[good_z, 2] + geom.K[1, 2]
    return uv, z


def project_points_base(points_base: np.ndarray, geom: CameraGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_cam = transform_points(geom.T_camera_base, np.asarray(points_base, dtype=np.float64))
    uv, z = project_points_camera(pts_cam, geom)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    return uv, z, valid


def inside_image(uv: np.ndarray, width: int, height: int, margin: float = 0.0) -> bool:
    if width <= 0 or height <= 0:
        return False
    u = float(uv[0])
    v = float(uv[1])
    return margin <= u <= float(width) - margin and margin <= v <= float(height) - margin


def far_outside_image(uv: np.ndarray, width: int, height: int, margin_scale: float = 0.75) -> bool:
    if width <= 0 or height <= 0:
        return True
    margin = margin_scale * float(max(width, height))
    u = float(uv[0])
    v = float(uv[1])
    if not math.isfinite(u) or not math.isfinite(v):
        return True
    return u < -margin or u > float(width) + margin or v < -margin or v > float(height) + margin
