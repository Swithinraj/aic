"""Deterministic multi-camera image-based visual servo command synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IBVSCameraError:
    camera: str
    err_uv: np.ndarray
    weight: float
    depth_m: float
    fx: float
    fy: float
    R_base_camera: np.ndarray


@dataclass
class IBVSCommand:
    robot_delta: np.ndarray
    per_camera_delta_base: dict[str, np.ndarray]
    total_weight: float
    reason: str = "ok"


class MultiCamIBVS:
    def __init__(
        self,
        gain: float = 0.45,
        max_step_m: float = 0.0015,
        max_z_step_m: float = 0.0010,
    ):
        self.gain = float(gain)
        self.max_step_m = float(max_step_m)
        self.max_z_step_m = float(max_z_step_m)

    def compute(self, errors: list[IBVSCameraError], gain: Optional[float] = None) -> IBVSCommand:
        if not errors:
            return IBVSCommand(np.zeros(3, dtype=np.float64), {}, 0.0, reason="no_valid_cameras")
        total_w = 0.0
        delta_sum = np.zeros(3, dtype=np.float64)
        per_camera: dict[str, np.ndarray] = {}
        for obs in errors:
            err = np.asarray(obs.err_uv, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(err)):
                continue
            fx = max(1e-6, abs(float(obs.fx)))
            fy = max(1e-6, abs(float(obs.fy)))
            z = max(0.02, float(obs.depth_m))
            # Raw pixel residual is port_uv - projected_tip_uv. Moving the TCP
            # by this camera-plane delta moves the projected CAD tip toward the
            # port mouth without consuming any object 3D pose.
            delta_camera = np.array([err[0] * z / fx, err[1] * z / fy, 0.0], dtype=np.float64)
            delta_base = np.asarray(obs.R_base_camera, dtype=np.float64).reshape(3, 3) @ delta_camera
            w = max(1e-6, float(obs.weight))
            per_camera[obs.camera] = delta_base
            delta_sum += w * delta_base
            total_w += w
        if total_w <= 1e-9:
            return IBVSCommand(np.zeros(3, dtype=np.float64), per_camera, 0.0, reason="zero_weight")
        delta = delta_sum / total_w
        delta *= self.gain if gain is None else float(gain)
        if self.max_z_step_m >= 0.0:
            delta[2] = float(np.clip(delta[2], -self.max_z_step_m, self.max_z_step_m))
        norm = float(np.linalg.norm(delta))
        if self.max_step_m > 0.0 and norm > self.max_step_m:
            delta *= self.max_step_m / max(1e-12, norm)
        return IBVSCommand(delta, per_camera, total_w)
