from __future__ import annotations

import math
from typing import Dict, Optional, Set

import numpy as np


def _norm_name(name: str) -> str:
    return str(name).strip().lower().replace('-', '_').replace(' ', '_')


def _strip_numeric_suffix(name: str) -> str:
    parts = _norm_name(name).split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return _norm_name(name)


def _matches_any_name(det: Dict, allowed_names: Set[str]) -> bool:
    for key in ('class_name', 'raw_class_name', 'base_class_name', 'instance_name'):
        value = det.get(key, '')
        norm = _norm_name(value)
        base = _strip_numeric_suffix(norm)
        for allowed in allowed_names:
            if norm == allowed or base == allowed or norm.startswith(f'{allowed}_'):
                return True
    return False


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = qx/n, qy/n, qz/n, qw/n
    return np.asarray([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)],
    ], dtype=np.float64)


class SfpModuleGroundAngleEstimator:
    def __init__(self, detection_listener, sfp_module_classes: Set[str]):
        self._listener = detection_listener
        self._sfp_module_classes = set(sfp_module_classes)

    def _extract_axis_and_angle(self, det: Dict) -> Optional[Dict]:
        pose = det.get('pose_base_link')
        if not isinstance(pose, dict):
            return None
        ori = pose.get('orientation')
        if not isinstance(ori, dict):
            return None
        try:
            R = _quat_to_rot(
                float(ori.get('x', 0.0)),
                float(ori.get('y', 0.0)),
                float(ori.get('z', 0.0)),
                float(ori.get('w', 1.0)),
            )
        except Exception:
            return None
        # SFP module long axis assumed local X axis.
        axis = R @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            return None
        axis = axis / axis_norm
        dot = float(np.clip(abs(axis[2]), 0.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        return {
            'angle_deg': angle_deg,
            'axis_base': axis.tolist(),
            'class_name': str(det.get('instance_name', det.get('class_name', ''))),
            'confidence': float(det.get('confidence', 0.0)),
        }

    def estimate(self) -> Dict:
        per_camera: Dict[str, Dict] = {}
        for cam in ('left', 'center', 'right'):
            dets = self._listener.get_camera_detections(cam, freshness_sec=1.0)
            cands = [d for d in dets if _matches_any_name(d, self._sfp_module_classes)]
            cands.sort(key=lambda d: float(d.get('confidence', 0.0)), reverse=True)
            for det in cands:
                item = self._extract_axis_and_angle(det)
                if item is not None:
                    per_camera[cam] = item
                    break

        if not per_camera:
            dets = self._listener.get_all_detections(freshness_sec=1.0)
            cands = [d for d in dets if _matches_any_name(d, self._sfp_module_classes)]
            cands.sort(key=lambda d: float(d.get('confidence', 0.0)), reverse=True)
            for det in cands:
                item = self._extract_axis_and_angle(det)
                if item is not None:
                    per_camera['fused'] = item
                    break

        fused = None
        if per_camera:
            angles = [float(v['angle_deg']) for v in per_camera.values()]
            fused_angle = float(np.median(np.asarray(angles, dtype=np.float64)))
            spread = float(max(angles) - min(angles)) if len(angles) > 1 else 0.0
            fused = {
                'angle_deg': fused_angle,
                'spread_deg': spread,
                'cameras': list(per_camera.keys()),
            }
        return {'per_camera': per_camera, 'fused': fused}