from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import math
import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("OpenCV (cv2) is required for team_policy.planner.angle_estimator") from exc

from aic_model_interfaces.msg import Observation
from sensor_msgs.msg import Image


@dataclass
class CameraRoiConfig:
    x_min_frac: float
    x_max_frac: float
    y_min_frac: float = 0.70
    y_max_frac: float = 1.00
    center_bias: float = 1.0


@dataclass
class AngleEstimatorConfig:
    camera_order: Tuple[str, ...] = ("center", "left", "right")
    center_camera_roi: CameraRoiConfig = field(
        default_factory=lambda: CameraRoiConfig(x_min_frac=0.40, x_max_frac=0.60, y_min_frac=0.70, y_max_frac=1.00, center_bias=2.0)
    )
    side_camera_roi: CameraRoiConfig = field(
        default_factory=lambda: CameraRoiConfig(x_min_frac=0.28, x_max_frac=0.72, y_min_frac=0.70, y_max_frac=1.00, center_bias=1.0)
    )
    min_contour_area: int = 80
    min_elongation: float = 1.5
    binary_blur_kernel: int = 5
    morph_kernel: int = 5
    dark_threshold_bias: int = 0
    gripper_reference_angle_deg: float = 90.0
    suppress_bottom_band_frac: float = 0.12


@dataclass
class AngleEstimate:
    valid: bool
    camera_name: str = ""
    absolute_relative_angle_deg: float = 0.0
    signed_relative_angle_deg: float = 0.0
    plug_axis_angle_deg: float = 0.0
    gripper_axis_angle_deg: float = 0.0
    confidence: float = 0.0
    contour_area: float = 0.0
    elongation: float = 0.0
    roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    contour_center_xy: Tuple[float, float] = (0.0, 0.0)
    debug_reason: str = ""


PSEUDO_ALGORITHM = """
Upgrade path for robust runtime angle estimation

1. Detect the plug body or keypoints in all three wrist cameras.
2. Use the bottom-30-percent prior only as a coarse search window.
3. Use CameraInfo intrinsics and TF extrinsics to estimate plug pose with PnP.
4. Recover the plug axis in gripper/tcp.
5. Compute the relative angle between the plug axis and the gripper axis.
6. Keep the current image-plane estimator only as a fallback / debug tool.
"""


class PlugAngleEstimator:
    def __init__(self, config: Optional[AngleEstimatorConfig] = None):
        self.config = config or AngleEstimatorConfig()
        self._last_debug_bgr: Dict[str, np.ndarray] = {}

    def estimate_from_observation(self, observation: Observation) -> AngleEstimate:
        self._last_debug_bgr = {}
        first_valid: Optional[AngleEstimate] = None
        for camera_name in self.config.camera_order:
            image_msg = self._get_image(observation, camera_name)
            if image_msg is None:
                continue
            estimate = self._estimate_from_image(image_msg=image_msg, camera_name=camera_name)
            if estimate.valid:
                first_valid = estimate
                break
            if first_valid is None:
                first_valid = estimate
        return first_valid if first_valid is not None else AngleEstimate(valid=False, debug_reason="no_camera_image")

    def get_debug_images(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._last_debug_bgr.items()}

    def _get_image(self, observation: Observation, camera_name: str) -> Optional[Image]:
        if camera_name == "center":
            return observation.center_image
        if camera_name == "left":
            return observation.left_image
        if camera_name == "right":
            return observation.right_image
        return None

    def _estimate_from_image(self, image_msg: Image, camera_name: str) -> AngleEstimate:
        bgr = self._ros_image_to_bgr(image_msg)
        if bgr is None:
            return AngleEstimate(valid=False, camera_name=camera_name, debug_reason="bad_image")

        roi_cfg = self.config.center_camera_roi if camera_name == "center" else self.config.side_camera_roi
        x0, y0, x1, y1 = self._compute_roi(bgr.shape[1], bgr.shape[0], roi_cfg)
        roi_bgr = bgr[y0:y1, x0:x1]
        if roi_bgr.size == 0:
            return AngleEstimate(valid=False, camera_name=camera_name, debug_reason="empty_roi")

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur_k = max(3, int(self.config.binary_blur_kernel) | 1)
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_value = max(0, min(255, int(otsu_val) + int(self.config.dark_threshold_bias)))
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        suppress_rows = int(binary.shape[0] * self.config.suppress_bottom_band_frac)
        if suppress_rows > 0:
            binary[-suppress_rows:, :] = 0

        morph_k = max(3, int(self.config.morph_kernel) | 1)
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        debug = bgr.copy()
        cv2.rectangle(debug, (x0, y0), (x1, y1), (255, 255, 0), 2)
        if camera_name == "center":
            center_x = int((x0 + x1) * 0.5)
            cv2.line(debug, (center_x, y0), (center_x, y1), (0, 255, 255), 1)

        best: Optional[AngleEstimate] = None
        best_score = -1.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_contour_area:
                continue
            if len(contour) < 8:
                continue

            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            major = max(float(width), float(height), 1e-6)
            minor = max(min(float(width), float(height)), 1e-6)
            elongation = major / minor
            if elongation < self.config.min_elongation:
                continue

            vx, vy, cx, cy = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = self._scalar(vx)
            vy = self._scalar(vy)
            cx = self._scalar(cx)
            cy = self._scalar(cy)

            plug_axis_angle_deg = self._normalize_axis_angle_deg(math.degrees(math.atan2(vy, vx)))
            signed_relative = self._signed_axis_difference_deg(
                plug_axis_angle_deg,
                self.config.gripper_reference_angle_deg,
            )
            absolute_relative = abs(signed_relative)

            roi_width = max(1.0, float(roi_bgr.shape[1]))
            roi_height = max(1.0, float(roi_bgr.shape[0]))
            x_center_norm = abs(cx - 0.5 * roi_width) / roi_width
            y_upper_preference = 1.0 - min(1.0, cy / roi_height)

            score = area * elongation
            score *= 1.0 / (1.0 + roi_cfg.center_bias * 8.0 * x_center_norm)
            score *= 0.75 + 0.5 * y_upper_preference

            if score > best_score:
                best_score = score
                best = AngleEstimate(
                    valid=True,
                    camera_name=camera_name,
                    absolute_relative_angle_deg=absolute_relative,
                    signed_relative_angle_deg=signed_relative,
                    plug_axis_angle_deg=plug_axis_angle_deg,
                    gripper_axis_angle_deg=self.config.gripper_reference_angle_deg,
                    confidence=score,
                    contour_area=area,
                    elongation=elongation,
                    roi=(x0, y0, x1, y1),
                    contour_center_xy=(x0 + cx, y0 + cy),
                    debug_reason="selected_best_contour",
                )

            color = (0, 120, 255)
            if camera_name == "center" and x_center_norm < 0.08:
                color = (255, 0, 255)
            contour_global = contour + np.array([[[x0, y0]]], dtype=contour.dtype)
            cv2.drawContours(debug, [contour_global], -1, color, 1)

        if best is not None:
            line_len = 120
            cxg, cyg = best.contour_center_xy
            angle_rad = math.radians(best.plug_axis_angle_deg)
            dx = math.cos(angle_rad) * line_len
            dy = math.sin(angle_rad) * line_len
            p1 = (int(cxg - dx), int(cyg - dy))
            p2 = (int(cxg + dx), int(cyg + dy))
            cv2.line(debug, p1, p2, (0, 255, 0), 2)
            cv2.circle(debug, (int(cxg), int(cyg)), 4, (0, 0, 255), -1)
            cv2.putText(
                debug,
                f"{camera_name} angle={best.signed_relative_angle_deg:+.1f} area={best.contour_area:.0f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                debug,
                f"{camera_name} no_valid_contour",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        self._last_debug_bgr[camera_name] = debug

        if best is None:
            return AngleEstimate(valid=False, camera_name=camera_name, roi=(x0, y0, x1, y1), debug_reason="no_valid_contour")
        return best

    def _compute_roi(self, width: int, height: int, roi_cfg: CameraRoiConfig) -> Tuple[int, int, int, int]:
        x0 = max(0, int(width * roi_cfg.x_min_frac))
        x1 = min(width, int(width * roi_cfg.x_max_frac))
        y0 = max(0, int(height * roi_cfg.y_min_frac))
        y1 = min(height, int(height * roi_cfg.y_max_frac))
        return x0, y0, x1, y1

    def _ros_image_to_bgr(self, image_msg: Image) -> Optional[np.ndarray]:
        height = int(image_msg.height)
        width = int(image_msg.width)
        step = int(image_msg.step)
        encoding = image_msg.encoding.lower()
        if height <= 0 or width <= 0 or step <= 0:
            return None

        data = np.frombuffer(image_msg.data, dtype=np.uint8)

        if encoding in ("rgb8", "bgr8"):
            img = data.reshape((height, step // 3, 3))[:, :width, :]
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else img.copy()

        if encoding in ("rgba8", "bgra8"):
            img = data.reshape((height, step // 4, 4))[:, :width, :]
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if encoding == "rgba8" else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if encoding == "mono8":
            img = data.reshape((height, step))[:, :width]
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return None

    def _scalar(self, value) -> float:
        return float(np.asarray(value).reshape(-1)[0])

    def _normalize_axis_angle_deg(self, angle_deg: float) -> float:
        angle = angle_deg % 180.0
        if angle < 0.0:
            angle += 180.0
        return angle

    def _signed_axis_difference_deg(self, plug_angle_deg: float, ref_angle_deg: float) -> float:
        return (plug_angle_deg - ref_angle_deg + 90.0) % 180.0 - 90.0