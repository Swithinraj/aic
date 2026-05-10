"""Image-space port-mouth refinement from YOLO port detections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional in static checks
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class PortMouthRefinerConfig:
    roi_scale: float = 1.6
    canny_low: int = 40
    canny_high: int = 140
    min_area_px: float = 40.0
    max_area_ratio: float = 0.90
    min_aspect: float = 1.1
    max_aspect: float = 8.0
    center_gate_px: float = 45.0
    min_quality: float = 0.20
    fallback_to_yolo: bool = True


@dataclass
class PortMouthEstimate:
    uv: np.ndarray
    quality: float
    source: str
    bbox_xyxy: np.ndarray
    confidence: float
    corners_uv: Optional[np.ndarray] = None
    angle_rad: Optional[float] = None
    reason: str = "ok"


class PortMouthRefiner:
    def __init__(self, config: Optional[PortMouthRefinerConfig] = None):
        self.config = config or PortMouthRefinerConfig()

    def refine(self, image_bgr: Optional[np.ndarray], detection: dict) -> Optional[PortMouthEstimate]:
        bbox = self._bbox(detection)
        if bbox is None:
            return None
        conf = self._confidence(detection)
        yolo_center = self._detection_mouth_uv(detection)
        source = "yolo_edge" if yolo_center is not None else "yolo_center"
        if yolo_center is None:
            yolo_center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)

        edge_est = self._refine_from_canny(image_bgr, bbox, yolo_center, conf)
        if edge_est is not None:
            return edge_est
        if not self.config.fallback_to_yolo:
            return None
        return PortMouthEstimate(
            uv=yolo_center,
            quality=max(0.05, min(1.0, conf * 0.55)),
            source=source,
            bbox_xyxy=bbox,
            confidence=conf,
            corners_uv=None,
            angle_rad=self._detection_angle(detection),
            reason="fallback_yolo",
        )

    @staticmethod
    def _bbox(detection: dict) -> Optional[np.ndarray]:
        for key in ("bbox_xyxy_feature", "bbox_xyxy", "bbox_xyxy_raw"):
            val = detection.get(key)
            if val is None or len(val) < 4:
                continue
            try:
                bbox = np.asarray(val[:4], dtype=np.float64)
            except Exception:
                continue
            if np.all(np.isfinite(bbox)) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                return bbox
        return None

    @staticmethod
    def _confidence(detection: dict) -> float:
        try:
            return float(np.clip(float(detection.get("confidence", 0.0)), 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _uv_field(detection: dict, *keys: str) -> Optional[np.ndarray]:
        for key in keys:
            val = detection.get(key)
            if isinstance(val, list) and len(val) >= 2:
                try:
                    uv = np.asarray(val[:2], dtype=np.float64)
                except Exception:
                    continue
                if np.all(np.isfinite(uv)):
                    return uv
        return None

    def _detection_mouth_uv(self, detection: dict) -> Optional[np.ndarray]:
        uv = self._uv_field(
            detection,
            "mouth_center_uv",
            "sc_port_center_uv",
            "axis_center_uv",
            "center_uv_feature",
            "center_uv",
        )
        return uv

    @staticmethod
    def _detection_angle(detection: dict) -> Optional[float]:
        for key in ("mouth_angle_rad", "sc_port_axis_angle_rad", "axis_angle_rad"):
            if key in detection:
                try:
                    return float(detection[key])
                except Exception:
                    pass
        return None

    @staticmethod
    def _expand_bbox(bbox: np.ndarray, img_w: int, img_h: int, scale: float) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = 0.5 * (x2 - x1) * scale
        half_h = 0.5 * (y2 - y1) * scale
        return np.array(
            [
                max(0.0, cx - half_w),
                max(0.0, cy - half_h),
                min(float(img_w - 1), cx + half_w),
                min(float(img_h - 1), cy + half_h),
            ],
            dtype=np.float64,
        )

    def _refine_from_canny(
        self,
        image_bgr: Optional[np.ndarray],
        bbox: np.ndarray,
        yolo_center: np.ndarray,
        yolo_conf: float,
    ) -> Optional[PortMouthEstimate]:
        if cv2 is None or image_bgr is None:
            return None
        cfg = self.config
        try:
            img_h, img_w = image_bgr.shape[:2]
            roi_box = self._expand_bbox(bbox, img_w, img_h, cfg.roi_scale)
            x1, y1, x2, y2 = [int(round(v)) for v in roi_box]
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                return None
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, int(cfg.canny_low), int(cfg.canny_high))
            kernel = np.ones((3, 3), dtype=np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
            best = None
            best_score = float("-inf")
            for contour in contours:
                if len(contour) < 4:
                    continue
                rect = cv2.minAreaRect(contour)
                (cx, cy), (rw, rh), angle_deg = rect
                short_side = max(1e-6, min(float(rw), float(rh)))
                long_side = max(float(rw), float(rh))
                area = max(float(cv2.contourArea(contour)), float(rw * rh))
                aspect = long_side / short_side
                center_full = np.array([x1 + cx, y1 + cy], dtype=np.float64)
                center_dist = float(np.linalg.norm(center_full - yolo_center))
                if area < cfg.min_area_px:
                    continue
                if area > cfg.max_area_ratio * roi_area:
                    continue
                if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
                    continue
                if center_dist > cfg.center_gate_px:
                    continue
                box = cv2.boxPoints(rect)
                box[:, 0] += x1
                box[:, 1] += y1
                area_score = min(1.0, area / max(cfg.min_area_px, 1.0))
                aspect_mid = 0.5 * (cfg.min_aspect + cfg.max_aspect)
                aspect_span = max(1e-6, 0.5 * (cfg.max_aspect - cfg.min_aspect))
                aspect_score = max(0.0, 1.0 - abs(aspect - aspect_mid) / aspect_span)
                center_score = max(0.0, 1.0 - center_dist / max(1.0, cfg.center_gate_px))
                score = 0.45 * area_score + 0.35 * aspect_score + 0.20 * center_score
                if score > best_score:
                    best_score = score
                    long_vec = None
                    if box.shape == (4, 2):
                        lengths = []
                        for i in range(4):
                            a = box[i]
                            b = box[(i + 1) % 4]
                            lengths.append((float(np.linalg.norm(b - a)), a, b))
                        long_edge = max(lengths, key=lambda item: item[0])
                        long_vec = long_edge[2] - long_edge[1]
                    angle = None
                    if long_vec is not None and float(np.linalg.norm(long_vec)) > 1e-6:
                        angle = float(np.arctan2(float(long_vec[1]), float(long_vec[0])))
                    else:
                        angle = float(np.deg2rad(angle_deg))
                    best = PortMouthEstimate(
                        uv=center_full,
                        quality=float(np.clip(score, 0.0, 1.0)),
                        source="canny",
                        bbox_xyxy=bbox,
                        confidence=yolo_conf,
                        corners_uv=box.astype(np.float64),
                        angle_rad=angle,
                    )
            if best is not None and best.quality >= cfg.min_quality:
                return best
        except Exception:
            return None
        return None
