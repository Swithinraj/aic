"""Nominal CAD plug-tip projection from TCP pose and gripper-offset prior."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from team_policy.camera_geometry import (
    CameraGeometry,
    far_outside_image,
    project_points_base,
    transform_points,
)


SFP_OFFSET_XYZ_RPY = (0.0, 0.015385, 0.04245, 0.4432, -0.4838, 1.3303)
SC_OFFSET_XYZ_RPY = (0.0, 0.015385, 0.04045, 0.4432, -0.4838, 1.3303)


def _norm_name(value: object) -> str:
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
    q = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
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


@dataclass
class PlugKeypoints:
    plug_key: str
    plug_tip_model: np.ndarray
    rear_point_model: np.ndarray
    plug_axis_model: np.ndarray
    front_face_corners_model: np.ndarray
    body_corners_model: np.ndarray


@dataclass
class CameraCADProjection:
    camera: str
    valid: bool
    reason: str
    tip_uv: Optional[np.ndarray] = None
    tip_camera: Optional[np.ndarray] = None
    axis_uv: Optional[tuple[np.ndarray, np.ndarray]] = None
    front_face_uv: Optional[np.ndarray] = None
    body_corners_uv: Optional[np.ndarray] = None
    geometry: Optional[CameraGeometry] = None


@dataclass
class CADProjection:
    plug_key: str
    T_base_tcp: np.ndarray
    T_tcp_plug: np.ndarray
    T_base_plug: np.ndarray
    plug_tip_base: np.ndarray
    plug_axis_base: np.ndarray
    camera_projections: dict[str, CameraCADProjection]
    source: str
    fit_score: float
    per_camera_scores: dict[str, float] = field(default_factory=dict)

    @property
    def valid_cameras(self) -> list[str]:
        return [cam for cam, proj in self.camera_projections.items() if proj.valid]

    @property
    def per_camera_projected_tip_uv(self) -> dict[str, np.ndarray]:
        return {
            cam: proj.tip_uv.copy()
            for cam, proj in self.camera_projections.items()
            if proj.tip_uv is not None
        }

    @property
    def per_camera_projected_axis_uv(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            cam: (proj.axis_uv[0].copy(), proj.axis_uv[1].copy())
            for cam, proj in self.camera_projections.items()
            if proj.axis_uv is not None
        }


class CADPlugProjector:
    def __init__(
        self,
        assets_root: str | Path = "",
        offset_refine_enable: bool = False,
        max_translation_correction_m: float = 0.003,
        max_rotation_correction_rad: float = 0.04,
    ):
        self.assets_root = Path(assets_root).expanduser() if str(assets_root).strip() else Path()
        self.offset_refine_enable = bool(offset_refine_enable)
        self.max_translation_correction_m = float(max_translation_correction_m)
        self.max_rotation_correction_rad = float(max_rotation_correction_rad)
        self._offset_corrections = {
            "sfp": np.zeros(6, dtype=np.float64),
            "sc": np.zeros(6, dtype=np.float64),
        }
        self.keypoints = {
            "sfp": self._load_keypoints("sfp"),
            "sc": self._load_keypoints("sc"),
        }

    def resolve_plug_key(self, plug_type: str = "", plug_name: str = "") -> str:
        name = _norm_name(plug_type) or _norm_name(plug_name)
        if "sc" in name:
            return "sc"
        return "sfp"

    def nominal_T_tcp_plug(self, plug_key: str) -> np.ndarray:
        return xyz_rpy_to_matrix(SC_OFFSET_XYZ_RPY if plug_key == "sc" else SFP_OFFSET_XYZ_RPY)

    def set_offset_correction(self, plug_key: str, correction_xyz_rpy: np.ndarray) -> None:
        if not self.offset_refine_enable:
            return
        key = "sc" if plug_key == "sc" else "sfp"
        corr = np.asarray(correction_xyz_rpy, dtype=np.float64).reshape(6).copy()
        corr[:3] = np.clip(
            corr[:3],
            -abs(self.max_translation_correction_m),
            abs(self.max_translation_correction_m),
        )
        corr[3:] = np.clip(
            corr[3:],
            -abs(self.max_rotation_correction_rad),
            abs(self.max_rotation_correction_rad),
        )
        self._offset_corrections[key] = corr

    def current_T_tcp_plug(self, plug_key: str) -> np.ndarray:
        T_nominal = self.nominal_T_tcp_plug(plug_key)
        corr = self._offset_corrections.get(plug_key, np.zeros(6, dtype=np.float64))
        if not self.offset_refine_enable or float(np.linalg.norm(corr)) <= 1e-12:
            return T_nominal
        return T_nominal @ xyz_rpy_to_matrix(tuple(float(v) for v in corr))

    def project(
        self,
        tcp_position: np.ndarray,
        tcp_quaternion_xyzw: np.ndarray,
        plug_type: str,
        plug_name: str,
        camera_geometries: dict[str, CameraGeometry],
        T_tcp_plug: Optional[np.ndarray] = None,
    ) -> CADProjection:
        plug_key = self.resolve_plug_key(plug_type, plug_name)
        keypoints = self.keypoints[plug_key]
        T_base_tcp = pose_to_matrix(tcp_position, tcp_quaternion_xyzw)
        T_tcp = self.current_T_tcp_plug(plug_key) if T_tcp_plug is None else np.asarray(T_tcp_plug, dtype=np.float64)
        T_base_plug = T_base_tcp @ T_tcp
        tip_base = transform_points(T_base_plug, keypoints.plug_tip_model)
        rear_base = transform_points(T_base_plug, keypoints.rear_point_model)
        axis_base = tip_base - rear_base
        axis_norm = float(np.linalg.norm(axis_base))
        if axis_norm > 1e-12:
            axis_base = axis_base / axis_norm
        axis_point_base = tip_base + 0.012 * axis_base
        front_base = transform_points(T_base_plug, keypoints.front_face_corners_model)
        body_base = transform_points(T_base_plug, keypoints.body_corners_model)
        cam_projects: dict[str, CameraCADProjection] = {}
        per_scores: dict[str, float] = {}
        for cam, geom in camera_geometries.items():
            uv, z, valid = project_points_base(np.asarray([tip_base], dtype=np.float64), geom)
            if uv.shape[0] == 0 or not bool(valid[0]):
                cam_projects[cam] = CameraCADProjection(cam, False, "plug_tip_behind_camera", geometry=geom)
                per_scores[cam] = 0.0
                continue
            tip_uv = uv[0]
            point_cam = transform_points(geom.T_camera_base, tip_base)
            # Strict control validity: tip must be inside image bounds
            u_val, v_val = float(tip_uv[0]), float(tip_uv[1])
            _in_image = (
                math.isfinite(u_val) and math.isfinite(v_val)
                and geom.width > 0 and geom.height > 0
                and 0.0 <= u_val < float(geom.width)
                and 0.0 <= v_val < float(geom.height)
            )
            if not _in_image:
                reason = "projected_tip_outside_image" if not far_outside_image(tip_uv, geom.width, geom.height) else "projected_point_far_outside_image"
                cam_projects[cam] = CameraCADProjection(
                    cam,
                    False,
                    reason,
                    tip_uv=tip_uv,
                    tip_camera=point_cam,
                    geometry=geom,
                )
                per_scores[cam] = 0.0
                continue
            axis_uv_arr, _axis_z, axis_valid = project_points_base(np.asarray([tip_base, axis_point_base]), geom)
            axis_uv = None
            if axis_uv_arr.shape[0] >= 2 and bool(axis_valid[0]) and bool(axis_valid[1]):
                axis_uv = (axis_uv_arr[0].copy(), axis_uv_arr[1].copy())
            front_uv, _front_z, front_valid = project_points_base(front_base, geom)
            body_uv, _body_z, body_valid = project_points_base(body_base, geom)
            front_out = front_uv[front_valid] if front_uv.size else None
            body_out = body_uv[body_valid] if body_uv.size else None
            cam_projects[cam] = CameraCADProjection(
                cam,
                True,
                "ok",
                tip_uv=tip_uv,
                tip_camera=point_cam,
                axis_uv=axis_uv,
                front_face_uv=front_out,
                body_corners_uv=body_out,
                geometry=geom,
            )
            per_scores[cam] = 1.0
        fit_score = 0.60 if any(p.valid for p in cam_projects.values()) else 0.0
        source = "cad_grasp_prior" if not self.offset_refine_enable else "cad_grasp_prior_offset_refine_enabled"
        return CADProjection(
            plug_key=plug_key,
            T_base_tcp=T_base_tcp,
            T_tcp_plug=T_tcp,
            T_base_plug=T_base_plug,
            plug_tip_base=tip_base,
            plug_axis_base=axis_base,
            camera_projections=cam_projects,
            source=source,
            fit_score=fit_score,
            per_camera_scores=per_scores,
        )

    def _load_keypoints(self, plug_key: str) -> PlugKeypoints:
        package_dir = Path(__file__).resolve().parent
        path = package_dir / "perception" / f"plug_keypoints_{plug_key}.json"
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except Exception:
                data = {}
        fallback = self._fallback_keypoints(plug_key)

        def point(name: str) -> np.ndarray:
            val = data.get(name)
            if isinstance(val, list) and len(val) == 3:
                try:
                    arr = np.asarray(val, dtype=np.float64)
                    if np.all(np.isfinite(arr)):
                        return arr
                except Exception:
                    pass
            return fallback[name].copy()

        def points(name: str) -> np.ndarray:
            val = data.get(name)
            if isinstance(val, list):
                try:
                    arr = np.asarray(val, dtype=np.float64).reshape(-1, 3)
                    if arr.shape[0] >= 4 and np.all(np.isfinite(arr)):
                        return arr
                except Exception:
                    pass
            return fallback[name].copy()

        tip = point("plug_tip_model")
        rear = point("rear_point_model")
        axis = point("plug_axis_model")
        n = float(np.linalg.norm(axis))
        if n <= 1e-12:
            axis = tip - rear
            n = float(np.linalg.norm(axis))
        if n > 1e-12:
            axis = axis / n
        return PlugKeypoints(
            plug_key=plug_key,
            plug_tip_model=tip,
            rear_point_model=rear,
            plug_axis_model=axis,
            front_face_corners_model=points("front_face_corners_model"),
            body_corners_model=points("body_corners_model"),
        )

    @staticmethod
    def _fallback_keypoints(plug_key: str) -> dict[str, np.ndarray]:
        if plug_key == "sc":
            lo = np.array([-0.0457523, -0.0125, -0.00505], dtype=np.float64)
            hi = np.array([0.01165, 0.0125, 0.00505], dtype=np.float64)
            tip = np.array([hi[0], 0.0, 0.0], dtype=np.float64)
            rear = np.array([lo[0], 0.0, 0.0], dtype=np.float64)
            front_x = hi[0]
            front = np.array(
                [
                    [front_x, lo[1], lo[2]],
                    [front_x, lo[1], hi[2]],
                    [front_x, hi[1], lo[2]],
                    [front_x, hi[1], hi[2]],
                ],
                dtype=np.float64,
            )
        else:
            lo = np.array([-0.007375, -0.00625, -0.02365], dtype=np.float64)
            hi = np.array([0.007375, 0.00597563, 0.0327711], dtype=np.float64)
            tip = np.array([0.0, 0.0, lo[2]], dtype=np.float64)
            rear = np.array([0.0, 0.0, hi[2]], dtype=np.float64)
            front_z = lo[2]
            front = np.array(
                [
                    [lo[0], lo[1], front_z],
                    [lo[0], hi[1], front_z],
                    [hi[0], lo[1], front_z],
                    [hi[0], hi[1], front_z],
                ],
                dtype=np.float64,
            )
        body = []
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    body.append([x, y, z])
        return {
            "plug_tip_model": tip,
            "rear_point_model": rear,
            "plug_axis_model": tip - rear,
            "front_face_corners_model": front,
            "body_corners_model": np.asarray(body, dtype=np.float64),
        }
