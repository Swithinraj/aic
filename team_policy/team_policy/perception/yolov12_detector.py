from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from ultralytics import YOLO

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional runtime dependency
    linear_sum_assignment = None


_CAMERAS = ("left", "center", "right")
_FAMILY_ORDER = {
    "task_board": 0,
    "nic_card": 1,
    "sc_port": 2,
    "sfp_port": 3,
    "sfp_module": 4,
    "sc_plug": 5,
}
_ALLOWED_TF_NAMES = {
    "task_board",
    "nic_card",
    "sc_port",
    "sfp_port",
    "sfp_port_0",
    "sfp_port_1",
    "sfp_module",
    "sc_plug",
}


@dataclass
class CanonicalDetection:
    family: str
    class_id: int
    raw_class_name: str
    confidence: float
    bbox_xyxy: np.ndarray
    center_uv: np.ndarray
    stamp_sec: float
    camera_name: str
    image_width: int
    image_height: int
    extra: Dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Dict, family: str, camera_name: str) -> "CanonicalDetection":
        bbox = np.asarray(raw.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])[:4], dtype=np.float64)
        center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)
        known = {
            "class_id",
            "raw_class_name",
            "base_class_name",
            "class_name",
            "instance_name",
            "confidence",
            "bbox_xyxy",
            "center_uv",
            "stamp_sec",
            "camera_name",
            "image_width",
            "image_height",
        }
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(
            family=family,
            class_id=int(raw.get("class_id", -1)),
            raw_class_name=str(raw.get("raw_class_name", raw.get("class_name", ""))),
            confidence=float(raw.get("confidence", 0.0)),
            bbox_xyxy=bbox,
            center_uv=center,
            stamp_sec=float(raw.get("stamp_sec", 0.0)),
            camera_name=camera_name,
            image_width=int(raw.get("image_width", 0) or 0),
            image_height=int(raw.get("image_height", 0) or 0),
            extra=extra,
        )


class RoiFeatureTrack:
    def __init__(
        self,
        owner: "YoloV12MultiCameraDetector",
        family: str,
        instance_name: str,
        parent_track_id: int,
        bbox_xyxy: np.ndarray,
        frame_bgr: np.ndarray,
        now: float,
    ):
        self.owner = owner
        self.feature_track_id = int(parent_track_id)
        self.family = family
        self.instance_name = instance_name
        self.parent_track_id = int(parent_track_id)
        self.last_gray_roi = None
        self.last_full_gray = None
        self.last_keypoints_uv = np.zeros((0, 2), dtype=np.float32)
        self.last_descriptors = None
        self.active_points_uv = np.zeros((0, 2), dtype=np.float32)
        self.point_ids: List[int] = []
        self.next_point_id = 1
        self.bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float64).copy()
        self.center_uv = self._bbox_center(self.bbox_xyxy)
        self.affine_2x3 = np.eye(2, 3, dtype=np.float64)
        self.quality_score = 0.0
        self.tracked_count = 0
        self.inlier_count = 0
        self.lost_count = 0
        self.last_update_time = float(now)
        self.mode = "yolo_reinit"
        self.jump_rejects = 0
        self.refresh_count = 0
        self._last_motion_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._init_from_bbox(frame_bgr, self.bbox_xyxy, now, reason="init")

    @staticmethod
    def _bbox_center(bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)

    @staticmethod
    def _bbox_wh(bbox: np.ndarray) -> np.ndarray:
        return np.array([max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])], dtype=np.float64)

    def _gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.owner.feature_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        return gray

    def _clip_roi(self, bbox: np.ndarray, shape_hw: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
        h, w = shape_hw
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)
        pad = float(self.owner.feature_roi_pad)
        x1 -= pad * bw
        x2 += pad * bw
        y1 -= pad * bh
        y2 += pad * bh
        ix1 = int(np.clip(np.floor(x1), 0, max(0, w - 1)))
        iy1 = int(np.clip(np.floor(y1), 0, max(0, h - 1)))
        ix2 = int(np.clip(np.ceil(x2), ix1 + 1, w))
        iy2 = int(np.clip(np.ceil(y2), iy1 + 1, h))
        if ix2 <= ix1 + 2 or iy2 <= iy1 + 2:
            return None
        return ix1, iy1, ix2, iy2

    def _feature_image(self, gray_roi: np.ndarray) -> np.ndarray:
        mode = str(self.owner.feature_edge_preprocess).strip().lower()
        if mode == "canny":
            return cv2.Canny(gray_roi, 40, 140)
        if mode == "sobel":
            gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            return np.clip(mag * (255.0 / max(1e-6, float(np.max(mag)))), 0, 255).astype(np.uint8)
        return gray_roi

    def _detect_points(self, gray: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, List[int]]:
        roi = self._clip_roi(bbox, gray.shape[:2])
        if roi is None:
            return np.zeros((0, 2), dtype=np.float32), []
        x1, y1, x2, y2 = roi
        roi_gray = gray[y1:y2, x1:x2]
        feature_img = self._feature_image(roi_gray)
        pts = cv2.goodFeaturesToTrack(
            feature_img,
            maxCorners=int(self.owner.feature_max_corners),
            qualityLevel=float(self.owner.feature_quality_level),
            minDistance=float(self.owner.feature_min_distance),
            blockSize=5,
        )
        if pts is None or len(pts) < int(self.owner.feature_min_corners):
            pts = cv2.goodFeaturesToTrack(
                roi_gray,
                maxCorners=int(self.owner.feature_max_corners),
                qualityLevel=max(1e-4, float(self.owner.feature_quality_level) * 0.5),
                minDistance=float(self.owner.feature_min_distance),
                blockSize=5,
            )
        if pts is None:
            return np.zeros((0, 2), dtype=np.float32), []
        pts = pts.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += float(x1)
        pts[:, 1] += float(y1)
        ids = list(range(self.next_point_id, self.next_point_id + len(pts)))
        self.next_point_id += len(pts)
        return pts, ids

    def _compute_orb(self, gray: np.ndarray, bbox: np.ndarray):
        if not self.owner.feature_orb_enable:
            return np.zeros((0, 2), dtype=np.float32), None
        roi = self._clip_roi(bbox, gray.shape[:2])
        if roi is None:
            return np.zeros((0, 2), dtype=np.float32), None
        x1, y1, x2, y2 = roi
        orb = cv2.ORB_create(nfeatures=int(self.owner.feature_orb_nfeatures))
        keypoints, descriptors = orb.detectAndCompute(gray[y1:y2, x1:x2], None)
        if not keypoints or descriptors is None:
            return np.zeros((0, 2), dtype=np.float32), None
        pts = np.array([[kp.pt[0] + x1, kp.pt[1] + y1] for kp in keypoints], dtype=np.float32)
        return pts, descriptors

    def _init_from_bbox(self, frame_bgr: np.ndarray, bbox: np.ndarray, now: float, reason: str) -> None:
        gray = self._gray(frame_bgr)
        self.last_full_gray = gray
        self.bbox_xyxy = np.asarray(bbox, dtype=np.float64).copy()
        self.center_uv = self._bbox_center(self.bbox_xyxy)
        roi = self._clip_roi(self.bbox_xyxy, gray.shape[:2])
        self.last_gray_roi = None if roi is None else gray[roi[1]:roi[3], roi[0]:roi[2]].copy()
        self.active_points_uv, self.point_ids = self._detect_points(gray, self.bbox_xyxy)
        self.last_keypoints_uv, self.last_descriptors = self._compute_orb(gray, self.bbox_xyxy)
        self.quality_score = min(1.0, len(self.active_points_uv) / max(1.0, float(self.owner.feature_max_corners)))
        self.tracked_count = int(len(self.active_points_uv))
        self.inlier_count = int(len(self.active_points_uv))
        self.lost_count = 0
        self.last_update_time = float(now)
        self.mode = "yolo_reinit"
        self.refresh_count = 0
        self._last_motion_pairs = []
        if self.owner.feature_debug:
            self.owner.get_logger().info(
                f"FEATURE_REINIT camera={self.owner._current_feature_camera} "
                f"track={self.instance_name}#{self.parent_track_id} reason={reason}"
            )

    def update_lk(self, frame_bgr: np.ndarray, predicted_bbox: np.ndarray, now: float):
        if self.last_full_gray is None or len(self.active_points_uv) < self.owner.feature_min_points:
            self.mode = "kalman_only"
            self.lost_count += 1
            return None
        gray = self._gray(frame_bgr)
        old = self.active_points_uv.astype(np.float32).reshape(-1, 1, 2)
        lk_params = dict(
            winSize=(int(self.owner.feature_lk_win_size), int(self.owner.feature_lk_win_size)),
            maxLevel=int(self.owner.feature_lk_max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        new, status, _ = cv2.calcOpticalFlowPyrLK(self.last_full_gray, gray, old, None, **lk_params)
        if new is None or status is None:
            self.last_full_gray = gray
            self.mode = "kalman_only"
            self.lost_count += 1
            return None
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.last_full_gray, new, None, **lk_params)
        if back is None or back_status is None:
            self.last_full_gray = gray
            self.mode = "kalman_only"
            self.lost_count += 1
            return None
        old2 = old.reshape(-1, 2)
        new2 = new.reshape(-1, 2)
        back2 = back.reshape(-1, 2)
        fb = np.linalg.norm(old2 - back2, axis=1)
        good = (status.reshape(-1) > 0) & (back_status.reshape(-1) > 0) & (fb <= float(self.owner.feature_fb_thresh_px))
        old_good = old2[good]
        new_good = new2[good]
        ids_good = [pid for pid, ok in zip(self.point_ids, good.tolist()) if ok]
        self.tracked_count = int(len(new_good))
        if len(new_good) < self.owner.feature_min_points:
            self.active_points_uv = new_good.astype(np.float32)
            self.point_ids = ids_good
            self.last_full_gray = gray
            self.mode = "kalman_only"
            self.lost_count += 1
            return None
        A, inliers = cv2.estimateAffinePartial2D(
            old_good,
            new_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.owner.feature_ransac_thresh_px),
            maxIters=2000,
            confidence=0.98,
        )
        if A is None or inliers is None:
            self.active_points_uv = new_good.astype(np.float32)
            self.point_ids = ids_good
            self.last_full_gray = gray
            self.mode = "kalman_only"
            self.lost_count += 1
            return None
        inlier_mask = inliers.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inlier_mask))
        ratio = inlier_count / max(1.0, float(len(new_good)))
        self.inlier_count = inlier_count
        self.quality_score = float(min(1.0, ratio * min(1.0, inlier_count / max(1.0, self.owner.feature_min_inliers))))
        if inlier_count < self.owner.feature_min_inliers or ratio < self.owner.feature_min_inlier_ratio:
            self.active_points_uv = new_good.astype(np.float32)
            self.point_ids = ids_good
            self.last_full_gray = gray
            self.mode = "kalman_only"
            self.lost_count += 1
            return None
        corners = np.array(
            [
                [self.bbox_xyxy[0], self.bbox_xyxy[1]],
                [self.bbox_xyxy[2], self.bbox_xyxy[1]],
                [self.bbox_xyxy[2], self.bbox_xyxy[3]],
                [self.bbox_xyxy[0], self.bbox_xyxy[3]],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.transform(corners, A).reshape(-1, 2)
        bbox = np.array(
            [np.min(warped[:, 0]), np.min(warped[:, 1]), np.max(warped[:, 0]), np.max(warped[:, 1])],
            dtype=np.float64,
        )
        bbox = self._clip_bbox_to_frame(bbox, gray.shape[:2])
        self.affine_2x3 = np.asarray(A, dtype=np.float64)
        self.bbox_xyxy = bbox
        self.center_uv = self._bbox_center(bbox)
        self.active_points_uv = new_good[inlier_mask].astype(np.float32)
        self.point_ids = [pid for pid, ok in zip(ids_good, inlier_mask.tolist()) if ok]
        self._last_motion_pairs = [(a.copy(), b.copy()) for a, b in zip(old_good[inlier_mask], new_good[inlier_mask])][:20]
        self.last_full_gray = gray
        self.last_update_time = float(now)
        self.mode = "lk"
        self.lost_count = 0
        self.refresh_count += 1
        return bbox

    def try_orb_reid(self, frame_bgr: np.ndarray, predicted_bbox: np.ndarray, now: float):
        if not self.owner.feature_orb_enable or self.last_descriptors is None or len(self.last_keypoints_uv) < 4:
            return None
        gray = self._gray(frame_bgr)
        cur_pts, cur_desc = self._compute_orb(gray, predicted_bbox)
        if cur_desc is None or len(cur_pts) < self.owner.feature_orb_min_matches:
            self.mode = "kalman_only"
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            knn = matcher.knnMatch(self.last_descriptors, cur_desc, k=2)
        except Exception:
            return None
        matches = []
        ratio = float(self.owner.feature_orb_match_ratio)
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                matches.append(m)
        if len(matches) < self.owner.feature_orb_min_matches:
            self.mode = "kalman_only"
            return None
        src = np.asarray([self.last_keypoints_uv[m.queryIdx] for m in matches], dtype=np.float32)
        dst = np.asarray([cur_pts[m.trainIdx] for m in matches], dtype=np.float32)
        A, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.owner.feature_ransac_thresh_px),
            maxIters=2000,
            confidence=0.98,
        )
        inlier_count = 0 if inliers is None else int(np.count_nonzero(inliers.reshape(-1)))
        accepted = A is not None and inlier_count >= self.owner.feature_orb_min_matches
        if self.owner.feature_debug:
            self.owner.get_logger().info(
                f"FEATURE_ORB_REID camera={self.owner._current_feature_camera} "
                f"track={self.instance_name}#{self.parent_track_id} matches={len(matches)} "
                f"inliers={inlier_count} accepted={str(accepted).lower()}"
            )
        if not accepted:
            self.mode = "kalman_only"
            return None
        corners = np.array(
            [
                [self.bbox_xyxy[0], self.bbox_xyxy[1]],
                [self.bbox_xyxy[2], self.bbox_xyxy[1]],
                [self.bbox_xyxy[2], self.bbox_xyxy[3]],
                [self.bbox_xyxy[0], self.bbox_xyxy[3]],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.transform(corners, A).reshape(-1, 2)
        bbox = np.array(
            [np.min(warped[:, 0]), np.min(warped[:, 1]), np.max(warped[:, 0]), np.max(warped[:, 1])],
            dtype=np.float64,
        )
        bbox = self._clip_bbox_to_frame(bbox, gray.shape[:2])
        self.bbox_xyxy = bbox
        self.center_uv = self._bbox_center(bbox)
        self.affine_2x3 = np.asarray(A, dtype=np.float64)
        self.quality_score = float(min(1.0, inlier_count / max(1.0, len(matches))))
        self.tracked_count = len(matches)
        self.inlier_count = inlier_count
        self.mode = "orb_reid"
        self.lost_count = 0
        self.active_points_uv, self.point_ids = self._detect_points(gray, bbox)
        self.last_keypoints_uv, self.last_descriptors = self._compute_orb(gray, bbox)
        self.last_full_gray = gray
        self.last_update_time = float(now)
        return bbox

    def refresh_if_needed(self, frame_bgr: np.ndarray, bbox: np.ndarray, now: float, reason: str = "") -> None:
        if len(self.active_points_uv) < self.owner.feature_min_points:
            self._init_from_bbox(frame_bgr, bbox, now, reason or "min_points")
            return
        if self.quality_score < self.owner.feature_min_inlier_ratio:
            self._init_from_bbox(frame_bgr, bbox, now, reason or "inlier_ratio")
            return
        if self.refresh_count >= self.owner.feature_refresh_frames:
            self._init_from_bbox(frame_bgr, bbox, now, reason or "refresh")

    def _clip_bbox_to_frame(self, bbox: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
        h, w = shape_hw
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return np.array(
            [
                float(np.clip(x1, 0, max(0, w - 1))),
                float(np.clip(y1, 0, max(0, h - 1))),
                float(np.clip(x2, 1, max(1, w))),
                float(np.clip(y2, 1, max(1, h))),
            ],
            dtype=np.float64,
        )


@dataclass
class CanonicalTrack:
    track_id: int
    family: str
    instance_name: str
    bbox_xyxy: np.ndarray
    center_uv: np.ndarray
    velocity_uv: np.ndarray
    size_wh: np.ndarray
    confidence_ema: float
    raw_confidence: float
    hit_count: int
    miss_count: int
    age: int
    confirmed: bool
    last_update_time: float
    last_raw_detection: CanonicalDetection
    class_id: int
    raw_class_name: str
    base_class_name: str
    extra: Dict = field(default_factory=dict)
    last_published_confidence: float = 0.0
    feature_track: Optional[RoiFeatureTrack] = None
    feature_bbox_xyxy: Optional[np.ndarray] = None
    feature_center_uv: Optional[np.ndarray] = None
    yolo_jump_count: int = 0

    @classmethod
    def create(cls, track_id: int, det: CanonicalDetection, now: float, frame_bgr: Optional[np.ndarray] = None, owner=None) -> "CanonicalTrack":
        w = max(1.0, float(det.bbox_xyxy[2] - det.bbox_xyxy[0]))
        h = max(1.0, float(det.bbox_xyxy[3] - det.bbox_xyxy[1]))
        track = cls(
            track_id=track_id,
            family=det.family,
            instance_name=det.family,
            bbox_xyxy=det.bbox_xyxy.copy(),
            center_uv=det.center_uv.copy(),
            velocity_uv=np.zeros(2, dtype=np.float64),
            size_wh=np.array([w, h], dtype=np.float64),
            confidence_ema=float(det.confidence),
            raw_confidence=float(det.confidence),
            hit_count=1,
            miss_count=0,
            age=1,
            confirmed=False,
            last_update_time=float(now),
            last_raw_detection=det,
            class_id=int(det.class_id),
            raw_class_name=det.raw_class_name,
            base_class_name=det.family,
            extra=dict(det.extra),
        )
        if frame_bgr is not None and owner is not None and owner.enable_feature_tracking:
            track.feature_track = RoiFeatureTrack(owner, det.family, det.family, track_id, det.bbox_xyxy, frame_bgr, now)
            track.feature_bbox_xyxy = track.feature_track.bbox_xyxy.copy()
            track.feature_center_uv = track.feature_track.center_uv.copy()
        return track

    def predict(self, now: float, dt_default: float = 1.0 / 15.0) -> None:
        dt = max(1e-3, float(now - self.last_update_time))
        if dt > 1.0:
            dt = dt_default
        self.center_uv = self.center_uv + self.velocity_uv * dt
        self._refresh_bbox()
        self.age += 1

    def mark_missed(self) -> None:
        self.miss_count += 1

    def update(
        self,
        det: CanonicalDetection,
        now: float,
        alpha: float,
        beta: float,
        size_ema: float,
        conf_ema: float,
        confirm_hits: int,
        feature_bbox: Optional[np.ndarray] = None,
        owner=None,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> None:
        dt = max(1e-3, float(now - self.last_update_time))
        predicted = self.center_uv + self.velocity_uv * dt
        yolo_bbox = det.bbox_xyxy.copy()
        yolo_center = det.center_uv.copy()
        final_bbox = yolo_bbox.copy()
        feature_good = False
        if feature_bbox is not None:
            feature_bbox = np.asarray(feature_bbox, dtype=np.float64)
            feature_center = self._bbox_center(feature_bbox)
            feature_quality = 0.0 if self.feature_track is None else float(self.feature_track.quality_score)
            feature_good = feature_quality >= 0.45
            jump = float(np.linalg.norm(yolo_center - feature_center))
            if owner is not None and jump > owner.yolo_jump_gate_px and feature_good:
                self.yolo_jump_count += 1
                if self.yolo_jump_count < owner.yolo_jump_confirm_frames:
                    yolo_bbox = feature_bbox.copy()
                    yolo_center = feature_center.copy()
                    self.feature_track.jump_rejects += 1
                    owner.get_logger().info(
                        f"FEATURE_YOLO_OUTLIER camera={owner._current_feature_camera} family={self.family} "
                        f"track_id={self.track_id} jump={jump:.1f}px feature_quality={feature_quality:.2f}"
                    )
                else:
                    self.yolo_jump_count = 0
            else:
                self.yolo_jump_count = 0
            if feature_good and feature_bbox is not None:
                publish_conf = (
                    owner._family_publish_thresholds(self.family)[0]
                    if owner is not None
                    else 0.70
                )
                if owner is not None and det.confidence < publish_conf:
                    final_bbox = feature_bbox.copy()
                elif float(np.linalg.norm(det.center_uv - predicted)) > (owner.track_center_gate_px if owner is not None else 90.0):
                    final_bbox = 0.80 * feature_bbox + 0.20 * yolo_bbox
                else:
                    final_bbox = 0.55 * feature_bbox + 0.45 * yolo_bbox
            else:
                final_bbox = 0.20 * feature_bbox + 0.80 * yolo_bbox
        else:
            final_bbox = yolo_bbox.copy()
        final_center = self._bbox_center(final_bbox)
        residual = final_center - predicted
        self.center_uv = predicted + alpha * residual
        self.velocity_uv = self.velocity_uv + beta * residual / dt
        meas_wh = self._bbox_wh(final_bbox)
        self.size_wh = size_ema * meas_wh + (1.0 - size_ema) * self.size_wh
        effective_conf = float(det.confidence)
        if owner is not None and feature_good:
            effective_conf = max(effective_conf, float(owner.feature_confirmed_publish_conf))
        self.confidence_ema = conf_ema * effective_conf + (1.0 - conf_ema) * self.confidence_ema
        self.raw_confidence = float(det.confidence)
        self.hit_count += 1
        self.miss_count = 0
        self.age += 1
        self.confirmed = self.hit_count >= confirm_hits
        self.last_update_time = float(now)
        self.last_raw_detection = det
        self.class_id = int(det.class_id)
        self.raw_class_name = det.raw_class_name
        self.base_class_name = det.family
        self.extra = dict(det.extra)
        self._refresh_bbox()
        if self.feature_track is not None:
            self.feature_track.instance_name = self.instance_name
            self.feature_bbox_xyxy = self.feature_track.bbox_xyxy.copy()
            self.feature_center_uv = self.feature_track.center_uv.copy()
            if frame_bgr is not None and feature_good:
                self.feature_track.refresh_if_needed(frame_bgr, self.bbox_xyxy, now, reason="yolo_agree")

    def _refresh_bbox(self) -> None:
        half = 0.5 * self.size_wh
        self.bbox_xyxy = np.array(
            [
                self.center_uv[0] - half[0],
                self.center_uv[1] - half[1],
                self.center_uv[0] + half[0],
                self.center_uv[1] + half[1],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _bbox_center(bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)

    @staticmethod
    def _bbox_wh(bbox: np.ndarray) -> np.ndarray:
        return np.array([max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])], dtype=np.float64)

    def publishable(
        self,
        publish_conf: float,
        min_raw_for_publish: float,
        owner=None,
    ) -> tuple[bool, str]:
        if not self.confirmed:
            return False, "not_confirmed"
        feature_good = False
        if owner is not None and self.feature_track is not None:
            feature_good = (
                self.feature_track.quality_score >= owner.feature_confirmed_quality_min
                and self.feature_track.inlier_count >= owner.feature_min_inliers
            )
        hold_s = 0.0
        if owner is not None:
            if self.family == "sfp_module":
                hold_s = owner.module_feature_hold_s
            elif self.family == "sc_plug":
                hold_s = owner.plug_feature_hold_s
        hold_ok = (
            feature_good
            and self.miss_count > 0
            and (time.monotonic() - float(self.last_update_time)) <= hold_s
        )
        if self.confidence_ema < publish_conf and not (feature_good and self.confidence_ema >= (owner.feature_confirmed_publish_conf if owner is not None else publish_conf)):
            return False, "publish_conf_low"
        if self.raw_confidence < min_raw_for_publish and not (feature_good or hold_ok):
            return False, "raw_conf_low"
        if self.miss_count == 0:
            return True, ""
        if hold_ok:
            return True, "hold"
        if self.miss_count <= 1 and self.last_published_confidence >= publish_conf:
            return True, ""
        return False, "missed"

    def to_public_detection(self, stamp_sec: float) -> Dict:
        anchor_fields = self._servo_anchor_fields()
        det = {
            "class_id": int(self.class_id),
            "raw_class_name": self.raw_class_name,
            "base_class_name": self.base_class_name,
            "class_name": self.instance_name,
            "instance_name": self.instance_name,
            "confidence": float(self.confidence_ema),
            "raw_confidence": float(self.raw_confidence),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy.tolist()],
            "center_uv": [float(v) for v in self.center_uv.tolist()],
            "bbox_xyxy_raw": [float(v) for v in self.last_raw_detection.bbox_xyxy.tolist()],
            "center_uv_raw": [float(v) for v in self.last_raw_detection.center_uv.tolist()],
            "bbox_xyxy_feature": [] if self.feature_bbox_xyxy is None else [float(v) for v in self.feature_bbox_xyxy.tolist()],
            "center_uv_feature": [] if self.feature_center_uv is None else [float(v) for v in self.feature_center_uv.tolist()],
            "camera_name": self.last_raw_detection.camera_name,
            "track_id": int(self.track_id),
            "track_age": int(self.age),
            "track_hit_count": int(self.hit_count),
            "track_miss_count": int(self.miss_count),
            "track_confirmed": bool(self.confirmed),
            "feature_track_id": int(self.feature_track.feature_track_id) if self.feature_track is not None else -1,
            "feature_quality_score": float(self.feature_track.quality_score) if self.feature_track is not None else 0.0,
            "feature_tracked_count": int(self.feature_track.tracked_count) if self.feature_track is not None else 0,
            "feature_inlier_count": int(self.feature_track.inlier_count) if self.feature_track is not None else 0,
            "feature_mode": str(self.feature_track.mode) if self.feature_track is not None else "kalman_only",
            "feature_point_ids_sample": list(self.feature_track.point_ids[:10]) if self.feature_track is not None else [],
            "stamp_sec": float(stamp_sec if stamp_sec > 0.0 else self.last_raw_detection.stamp_sec),
            "image_width": int(self.last_raw_detection.image_width),
            "image_height": int(self.last_raw_detection.image_height),
        }
        det.update(anchor_fields)
        for key in ("obb_cxcywh_deg", "obb_corners_uv", "mask_polygon_uv", "mask_area_px"):
            if key in self.extra:
                det[key] = self.extra[key]
        return det

    @staticmethod
    def _rect_points_from_bbox(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)

    def _rect_points_for_anchor(self) -> np.ndarray:
        if "obb_corners_uv" in self.extra:
            try:
                corners = np.asarray(self.extra["obb_corners_uv"], dtype=np.float64).reshape(4, 2)
                if np.all(np.isfinite(corners)):
                    return corners
            except Exception:
                pass
        bbox = self.feature_bbox_xyxy if self.feature_bbox_xyxy is not None else self.bbox_xyxy
        return self._rect_points_from_bbox(np.asarray(bbox, dtype=np.float64))

    @staticmethod
    def _axis_fields_from_rect(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[tuple[float, np.ndarray, np.ndarray]]]:
        center = np.mean(corners, axis=0)
        edges: list[tuple[float, np.ndarray, np.ndarray]] = []
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            edges.append((float(np.linalg.norm(b - a)), a, b))
        edges.sort(key=lambda item: item[0])
        long_edge = edges[-1]
        vec = long_edge[2] - long_edge[1]
        length = float(np.linalg.norm(vec))
        if length <= 1e-6:
            vec = np.array([1.0, 0.0], dtype=np.float64)
            length = 1.0
        axis = vec / length
        left = center - 0.5 * length * axis
        right = center + 0.5 * length * axis
        if float(left[0]) > float(right[0]):
            left, right = right, left
        return center, left, right, float(np.arctan2(axis[1], axis[0])), edges

    def _servo_anchor_fields(self) -> Dict:
        corners = self._rect_points_for_anchor()
        center, left, right, angle, edges = self._axis_fields_from_rect(corners)
        ft = self.feature_track
        quality = float(ft.quality_score) if ft is not None else 0.0
        source = str(ft.mode) if ft is not None else "bbox"
        common = {
            "servo_anchor_valid": True,
            "servo_anchor_source": source,
            "servo_anchor_quality": quality,
        }
        if self.family in {"sfp_port", "sc_port"}:
            fields = {
                **common,
                "mouth_center_uv": [float(v) for v in center.tolist()],
                "mouth_left_uv": [float(v) for v in left.tolist()],
                "mouth_right_uv": [float(v) for v in right.tolist()],
                "mouth_angle_rad": float(angle),
            }
            if self.family == "sc_port":
                fields.update(
                    {
                        "sc_port_center_uv": fields["mouth_center_uv"],
                        "sc_port_axis_left_uv": fields["mouth_left_uv"],
                        "sc_port_axis_right_uv": fields["mouth_right_uv"],
                        "sc_port_axis_angle_rad": float(angle),
                    }
                )
            return fields

        short_edges = edges[:2]
        front = short_edges[0]
        # Without the target port in this node, choose a stable image-space front
        # edge; run_hybrid reselects the side nearest the port when needed.
        mids = [(0.5 * (item[1] + item[2]), item) for item in short_edges]
        mids.sort(key=lambda item: (float(item[0][0]), float(item[0][1])))
        front = mids[0][1]
        a, b = front[1], front[2]
        front_center = 0.5 * (a + b)
        front_left, front_right = (a, b) if float(a[0]) <= float(b[0]) else (b, a)
        vec = front_right - front_left
        front_angle = float(np.arctan2(vec[1], vec[0])) if float(np.linalg.norm(vec)) > 1e-6 else angle
        fields = {
            **common,
            "tip_uv": [float(v) for v in front_center.tolist()],
            "front_center_uv": [float(v) for v in front_center.tolist()],
            "front_left_uv": [float(v) for v in front_left.tolist()],
            "front_right_uv": [float(v) for v in front_right.tolist()],
            "front_angle_rad": float(front_angle),
        }
        if self.family == "sc_plug":
            fields.update(
                {
                    "sc_plug_tip_uv": fields["tip_uv"],
                    "sc_plug_axis_left_uv": fields["front_left_uv"],
                    "sc_plug_axis_right_uv": fields["front_right_uv"],
                    "sc_plug_axis_angle_rad": float(front_angle),
                }
            )
        return fields


class PerCameraCanonicalTracker:
    def __init__(self, owner: "YoloV12MultiCameraDetector", camera_name: str):
        self.owner = owner
        self.camera_name = camera_name
        self.tracks: List[CanonicalTrack] = []
        self._next_track_id = 1
        self._sfp_number_map: Dict[int, int] = {}
        self._sfp_candidate_map: Optional[Dict[int, int]] = None
        self._sfp_candidate_frames = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._sfp_number_map.clear()
        self._sfp_candidate_map = None
        self._sfp_candidate_frames = 0

    def update(self, detections: List[CanonicalDetection], now: float, frame_bgr: Optional[np.ndarray] = None) -> List[Dict]:
        for track in self.tracks:
            track.predict(now)
            self._update_feature_prediction(track, frame_bgr, now)

        detections_by_family: Dict[str, List[CanonicalDetection]] = {}
        for det in detections:
            if det.confidence >= self.owner.track_low_conf:
                detections_by_family.setdefault(det.family, []).append(det)

        matched_track_ids = set()
        matched_det_ids = set()

        for family in _FAMILY_ORDER:
            family_tracks = [t for t in self.tracks if t.family == family]
            family_dets = detections_by_family.get(family, [])
            matches = self._associate_family(family_tracks, family_dets)
            for t_idx, d_idx in matches:
                track = family_tracks[t_idx]
                det = family_dets[d_idx]
                if frame_bgr is not None and self._feature_enabled_for(track.family) and track.feature_track is None:
                    track.feature_track = RoiFeatureTrack(
                        self.owner,
                        track.family,
                        track.instance_name,
                        track.track_id,
                        track.bbox_xyxy,
                        frame_bgr,
                        now,
                    )
                    track.feature_bbox_xyxy = track.feature_track.bbox_xyxy.copy()
                    track.feature_center_uv = track.feature_track.center_uv.copy()
                track.update(
                    det,
                    now,
                    self.owner.track_alpha,
                    self.owner.track_beta,
                    self.owner.track_size_ema,
                    self.owner.track_conf_ema,
                    self.owner.track_confirm_hits,
                    feature_bbox=track.feature_bbox_xyxy,
                    owner=self.owner,
                    frame_bgr=frame_bgr,
                )
                if frame_bgr is not None and track.feature_track is not None:
                    if (
                        track.feature_track.mode == "kalman_only"
                        or len(track.feature_track.active_points_uv) < self.owner.feature_min_points
                        or track.feature_track.quality_score < self.owner.feature_min_inlier_ratio
                    ):
                        track.feature_track.refresh_if_needed(frame_bgr, track.bbox_xyxy, now, reason="yolo_recover")
                        track.feature_bbox_xyxy = track.feature_track.bbox_xyxy.copy()
                        track.feature_center_uv = track.feature_track.center_uv.copy()
                matched_track_ids.add(track.track_id)
                matched_det_ids.add(id(det))

        for track in self.tracks:
            if track.track_id not in matched_track_ids:
                track.mark_missed()
                if track.feature_bbox_xyxy is not None and self._feature_quality_good(track):
                    self._apply_feature_only_update(track, now)

        for det in detections:
            if id(det) in matched_det_ids:
                continue
            if det.confidence < self.owner.track_new_conf:
                continue
            track = CanonicalTrack.create(
                self._next_track_id,
                det,
                now,
                frame_bgr=frame_bgr if self._feature_enabled_for(det.family) else None,
                owner=self.owner,
            )
            track.confirmed = track.hit_count >= self.owner.track_confirm_hits
            self._next_track_id += 1
            self.tracks.append(track)

        survivors = []
        for track in self.tracks:
            if track.miss_count > self.owner.track_max_misses:
                if self.owner.track_debug:
                    self.owner.get_logger().info(
                        f"TRACK_DELETE camera={self.camera_name} family={track.family} "
                        f"track_id={track.track_id} misses={track.miss_count}"
                    )
                continue
            survivors.append(track)
        self.tracks = survivors

        return self._public_detections(now)

    def _feature_enabled_for(self, family: str) -> bool:
        if not self.owner.enable_feature_tracking:
            return False
        if self.owner.feature_method in {"", "none", "off", "false"}:
            return False
        return family in self.owner.feature_families

    def _update_feature_prediction(self, track: CanonicalTrack, frame_bgr: Optional[np.ndarray], now: float) -> None:
        if frame_bgr is None or not self._feature_enabled_for(track.family):
            return
        if track.feature_track is None:
            track.feature_track = RoiFeatureTrack(
                self.owner,
                track.family,
                track.instance_name,
                track.track_id,
                track.bbox_xyxy,
                frame_bgr,
                now,
            )
        feature_bbox = track.feature_track.update_lk(frame_bgr, track.bbox_xyxy, now)
        if feature_bbox is None:
            should_try_orb = (
                self.owner.feature_orb_enable
                and track.feature_track.lost_count > 0
                and len(track.feature_track.active_points_uv) < self.owner.feature_min_points
            )
            if should_try_orb:
                feature_bbox = track.feature_track.try_orb_reid(frame_bgr, track.bbox_xyxy, now)
        if feature_bbox is not None:
            track.feature_bbox_xyxy = np.asarray(feature_bbox, dtype=np.float64).copy()
            track.feature_center_uv = track.feature_track.center_uv.copy()
        else:
            track.feature_bbox_xyxy = None
            track.feature_center_uv = None

    def _feature_quality_good(self, track: CanonicalTrack) -> bool:
        ft = track.feature_track
        if ft is None:
            return False
        return (
            track.feature_bbox_xyxy is not None
            and ft.quality_score >= self.owner.feature_min_inlier_ratio
            and ft.inlier_count >= self.owner.feature_min_inliers
        )

    def _apply_feature_only_update(self, track: CanonicalTrack, now: float) -> None:
        del now
        if track.feature_bbox_xyxy is None:
            return
        bbox = np.asarray(track.feature_bbox_xyxy, dtype=np.float64)
        center = CanonicalTrack._bbox_center(bbox)
        wh = CanonicalTrack._bbox_wh(bbox)
        track.center_uv = 0.70 * center + 0.30 * track.center_uv
        track.size_wh = 0.70 * wh + 0.30 * track.size_wh
        track._refresh_bbox()

    def _associate_family(self, tracks: List[CanonicalTrack], detections: List[CanonicalDetection]) -> List[tuple[int, int]]:
        if not tracks or not detections:
            return []
        costs = np.full((len(tracks), len(detections)), 1e6, dtype=np.float64)
        for ti, track in enumerate(tracks):
            for di, det in enumerate(detections):
                iou = self._bbox_iou(track.bbox_xyxy, det.bbox_xyxy)
                center_dist = float(np.linalg.norm(track.center_uv - det.center_uv))
                if iou < self.owner.track_iou_gate and center_dist > self.owner.track_center_gate_px:
                    continue
                norm_dist = min(1.0, center_dist / max(1.0, self.owner.track_center_gate_px))
                conf_penalty = 1.0 - float(det.confidence)
                costs[ti, di] = 0.55 * norm_dist + 0.35 * (1.0 - iou) + 0.10 * conf_penalty

        matches: List[tuple[int, int]] = []
        used_t = set()
        used_d = set()
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(costs)
            for r, c in zip(rows, cols):
                if costs[r, c] >= 1e5:
                    continue
                matches.append((int(r), int(c)))
        else:
            candidates = [
                (float(costs[ti, di]), ti, di)
                for ti in range(costs.shape[0])
                for di in range(costs.shape[1])
                if costs[ti, di] < 1e5
            ]
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            for _, ti, di in candidates:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                matches.append((ti, di))
        return matches

    def _public_detections(self, now: float) -> List[Dict]:
        del now
        out: List[Dict] = []
        single_families = ("task_board", "nic_card", "sc_port", "sfp_module", "sc_plug")
        for family in single_families:
            candidates = self._publishable_tracks(family)
            if candidates:
                chosen = self._rank_tracks(candidates)[0]
                chosen.instance_name = family
                if chosen.feature_track is not None:
                    chosen.feature_track.instance_name = family
                det = chosen.to_public_detection(chosen.last_raw_detection.stamp_sec)
                self.owner._adjust_public_detection_confidence(det, chosen)
                chosen.last_published_confidence = float(det["confidence"])
                out.append(det)

        sfp_tracks = self._rank_tracks(self._publishable_tracks("sfp_port"))[:2]
        if len(sfp_tracks) == 1:
            sfp_tracks[0].instance_name = "sfp_port"
            if sfp_tracks[0].feature_track is not None:
                sfp_tracks[0].feature_track.instance_name = "sfp_port"
            det = sfp_tracks[0].to_public_detection(sfp_tracks[0].last_raw_detection.stamp_sec)
            self.owner._adjust_public_detection_confidence(det, sfp_tracks[0])
            sfp_tracks[0].last_published_confidence = float(det["confidence"])
            out.append(det)
        elif len(sfp_tracks) >= 2:
            numbered = self._assign_sfp_numbers(sfp_tracks[:2], out)
            for track, name, source, edge_score in numbered:
                track.instance_name = name
                if track.feature_track is not None:
                    track.feature_track.instance_name = name
                det = track.to_public_detection(track.last_raw_detection.stamp_sec)
                det["sfp_numbering_source"] = source
                if edge_score is not None:
                    det["sfp_edge_score"] = float(edge_score)
                self.owner._adjust_public_detection_confidence(det, track)
                track.last_published_confidence = float(det["confidence"])
                out.append(det)

        return self.owner._sort_output_detections(out)

    def _publishable_tracks(self, family: str) -> List[CanonicalTrack]:
        good = []
        publish_conf, min_raw = self.owner._family_publish_thresholds(family)
        for track in self.tracks:
            if track.family != family:
                continue
            ok, reason = track.publishable(publish_conf, min_raw, owner=self.owner)
            if ok:
                good.append(track)
            elif self.owner.track_debug and self.owner._should_debug_log(self.camera_name):
                if reason == "publish_conf_low":
                    self.owner.get_logger().info(
                        f"TRACK_DROP camera={self.camera_name} family={family} reason={reason} "
                        f"raw={track.raw_confidence:.2f} ema={track.confidence_ema:.2f}"
                    )
                elif reason == "not_confirmed":
                    self.owner.get_logger().info(
                        f"TRACK_DROP camera={self.camera_name} family={family} reason={reason} hits={track.hit_count}"
                    )
                else:
                    self.owner.get_logger().info(
                        f"TRACK_DROP camera={self.camera_name} family={family} reason={reason} "
                        f"track_id={track.track_id} raw={track.raw_confidence:.2f} ema={track.confidence_ema:.2f} misses={track.miss_count}"
                    )
        return good

    @staticmethod
    def _rank_tracks(tracks: List[CanonicalTrack]) -> List[CanonicalTrack]:
        return sorted(
            tracks,
            key=lambda t: (
                -float(t.confidence_ema),
                -int(t.hit_count),
                int(t.miss_count),
                -int(t.age),
                int(t.track_id),
            ),
        )

    def _assign_sfp_numbers(
        self,
        tracks: List[CanonicalTrack],
        current_public: List[Dict],
    ) -> List[tuple[CanonicalTrack, str, str, Optional[float]]]:
        source = "image_order"
        edge_scores: Dict[int, Optional[float]] = {t.track_id: None for t in tracks}
        bbox = None
        board = self._best_public_family(current_public, "task_board")
        nic = self._best_public_family(current_public, "nic_card")
        if board is not None:
            source = "task_board_edges"
            bbox = board.get("bbox_xyxy")
        elif nic is not None:
            source = "nic_card_bbox"
            bbox = nic.get("bbox_xyxy")

        if bbox is not None:
            scored = []
            for track in tracks:
                score = self._two_nearest_bbox_edge_sum(track.center_uv, bbox)
                edge_scores[track.track_id] = score
                scored.append((score, track))
            scored.sort(key=lambda item: item[0])
            nearer = scored[0][1]
            farther = scored[1][1]
            margin = abs(float(scored[1][0] - scored[0][0]))
            # Preserve the previous detector convention: after sorting by
            # nearest board/nic edges, the order was reversed before assigning
            # sfp_port_0/1. Thus the farther of the two selected ports remains
            # sfp_port_0 and the nearer one remains sfp_port_1.
            proposed = {farther.track_id: 0, nearer.track_id: 1}
            if margin < self.owner.sfp_number_margin_px and self._sfp_number_map:
                proposed = {tid: idx for tid, idx in self._sfp_number_map.items() if tid in proposed}
                for tid in [track.track_id for track in tracks]:
                    proposed.setdefault(tid, 1 if 0 in proposed.values() else 0)
        else:
            ordered = sorted(tracks, key=lambda t: (float(t.center_uv[1]), float(t.center_uv[0]), t.track_id))
            proposed = {ordered[0].track_id: 0, ordered[1].track_id: 1}

        track_ids = {t.track_id for t in tracks}
        if set(self._sfp_number_map.keys()) != track_ids:
            self._sfp_number_map = {}
            self._sfp_candidate_map = None
            self._sfp_candidate_frames = 0

        if self._sfp_number_map and proposed != self._sfp_number_map:
            if proposed == self._sfp_candidate_map:
                self._sfp_candidate_frames += 1
            else:
                self._sfp_candidate_map = dict(proposed)
                self._sfp_candidate_frames = 1
            if self._sfp_candidate_frames >= self.owner.sfp_number_switch_frames:
                self._sfp_number_map = dict(proposed)
                self._sfp_candidate_map = None
                self._sfp_candidate_frames = 0
        elif not self._sfp_number_map:
            self._sfp_number_map = dict(proposed)
        else:
            self._sfp_candidate_map = None
            self._sfp_candidate_frames = 0

        output = []
        for track in sorted(tracks, key=lambda t: self._sfp_number_map.get(t.track_id, 99)):
            idx = self._sfp_number_map.get(track.track_id, 0)
            output.append((track, f"sfp_port_{idx}", source, edge_scores.get(track.track_id)))
        return output

    @staticmethod
    def _best_public_family(public: List[Dict], family: str) -> Optional[Dict]:
        items = [det for det in public if det.get("base_class_name") == family or det.get("class_name") == family]
        if not items:
            return None
        return max(items, key=lambda det: float(det.get("confidence", 0.0)))

    @staticmethod
    def _two_nearest_bbox_edge_sum(point_uv: np.ndarray, bbox_xyxy) -> float:
        if bbox_xyxy is None or len(bbox_xyxy) != 4:
            return 1e9
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
        px, py = [float(v) for v in np.asarray(point_uv, dtype=np.float64).reshape(2)]
        dists = sorted([abs(px - x1), abs(px - x2), abs(py - y1), abs(py - y2)])
        return float(dists[0] + dists[1])

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 1e-9 else 0.0


class YoloV12MultiCameraDetector(Node):
    """Canonical YOLO detector with per-camera temporal tracking."""

    def __init__(self):
        super().__init__("yolov12_multicamera_detector")

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        default_model_path = Path(__file__).resolve().parents[1] / "models" / "yolov12.pt"
        self._model_path = os.environ.get("YOLOV12_MODEL_PATH", str(default_model_path))
        self._device_request = os.environ.get("YOLOV12_DEVICE", "auto").strip().lower()
        self._device = self._resolve_device(self._device_request)

        def param_env_float(param_name: str, env_name: str, default: float) -> float:
            env_value = os.environ.get(env_name)
            declared_default = float(env_value) if env_value is not None else float(default)
            try:
                return float(self.declare_parameter(param_name, declared_default).value)
            except Exception:
                try:
                    return float(self.get_parameter(param_name).value)
                except Exception:
                    return declared_default

        self._raw_conf = float(os.environ.get("YOLOV12_RAW_CONF", "0.25"))
        self.publish_conf = float(os.environ.get("YOLOV12_PUBLISH_CONF", "0.70"))
        self.track_new_conf = float(os.environ.get("YOLOV12_TRACK_NEW_CONF", "0.85"))
        self.track_low_conf = float(os.environ.get("YOLOV12_TRACK_LOW_CONF", "0.25"))
        self.track_confirm_hits = int(os.environ.get("YOLOV12_TRACK_CONFIRM_HITS", "2"))
        self.track_max_misses = int(os.environ.get("YOLOV12_TRACK_MAX_MISSES", "5"))
        self.track_iou_gate = float(os.environ.get("YOLOV12_TRACK_IOU_GATE", "0.10"))
        self.track_center_gate_px = float(os.environ.get("YOLOV12_TRACK_CENTER_GATE_PX", "90.0"))
        self.track_alpha = float(os.environ.get("YOLOV12_TRACK_ALPHA", "0.45"))
        self.track_beta = float(os.environ.get("YOLOV12_TRACK_BETA", "0.12"))
        self.track_size_ema = float(os.environ.get("YOLOV12_TRACK_SIZE_EMA", "0.35"))
        self.track_conf_ema = float(os.environ.get("YOLOV12_TRACK_CONF_EMA", "0.30"))
        self.track_min_raw_for_publish = float(os.environ.get("YOLOV12_TRACK_MIN_RAW_FOR_PUBLISH", "0.70"))
        self.task_board_publish_conf = param_env_float("task_board_publish_conf", "YOLOV12_TASK_BOARD_PUBLISH_CONF", 0.85)
        self.port_publish_conf = param_env_float("port_publish_conf", "YOLOV12_PORT_PUBLISH_CONF", 0.72)
        self.module_publish_conf = param_env_float("module_publish_conf", "YOLOV12_MODULE_PUBLISH_CONF", 0.45)
        self.plug_publish_conf = param_env_float("plug_publish_conf", "YOLOV12_PLUG_PUBLISH_CONF", 0.45)
        self.module_feature_hold_s = param_env_float("module_feature_hold_s", "YOLOV12_MODULE_FEATURE_HOLD_S", 1.0)
        self.plug_feature_hold_s = param_env_float("plug_feature_hold_s", "YOLOV12_PLUG_FEATURE_HOLD_S", 1.0)
        self.feature_confirmed_quality_min = param_env_float("feature_confirmed_quality_min", "YOLOV12_FEATURE_CONFIRMED_QUALITY_MIN", 0.65)
        self.feature_confirmed_publish_conf = param_env_float("feature_confirmed_publish_conf", "YOLOV12_FEATURE_CONFIRMED_PUBLISH_CONF", 0.45)
        self.track_debug = self._env_bool("YOLOV12_TRACK_DEBUG", True)
        self.track_debug_every = max(1, int(os.environ.get("YOLOV12_TRACK_DEBUG_EVERY", "30")))
        self.sfp_number_switch_frames = max(1, int(os.environ.get("YOLOV12_SFP_NUMBER_SWITCH_FRAMES", "8")))
        self.sfp_number_margin_px = float(os.environ.get("YOLOV12_SFP_NUMBER_MARGIN_PX", "15.0"))
        self.use_ultralytics_track = self._env_bool("YOLOV12_USE_ULTRALYTICS_TRACK", False)
        self._tracker_yaml = os.environ.get("YOLOV12_TRACKER_YAML", "botsort.yaml")
        self.draw_raw_debug = self._env_bool("YOLOV12_DRAW_RAW_DEBUG", False)
        self.enable_feature_tracking = self._env_bool("YOLOV12_ENABLE_FEATURE_TRACKING", True)
        self.feature_method = os.environ.get("YOLOV12_FEATURE_METHOD", "lk_orb").strip().lower()
        self.feature_roi_pad = float(os.environ.get("YOLOV12_FEATURE_ROI_PAD", "0.20"))
        self.feature_max_corners = int(os.environ.get("YOLOV12_FEATURE_MAX_CORNERS", "80"))
        self.feature_min_corners = int(os.environ.get("YOLOV12_FEATURE_MIN_CORNERS", "12"))
        self.feature_quality_level = float(os.environ.get("YOLOV12_FEATURE_QUALITY_LEVEL", "0.01"))
        self.feature_min_distance = float(os.environ.get("YOLOV12_FEATURE_MIN_DISTANCE", "5"))
        self.feature_lk_win_size = int(os.environ.get("YOLOV12_FEATURE_LK_WIN_SIZE", "21"))
        self.feature_lk_max_level = int(os.environ.get("YOLOV12_FEATURE_LK_MAX_LEVEL", "3"))
        self.feature_fb_thresh_px = float(os.environ.get("YOLOV12_FEATURE_FB_THRESH_PX", "2.0"))
        self.feature_min_points = int(os.environ.get("YOLOV12_FEATURE_MIN_POINTS", "8"))
        self.feature_min_inliers = int(os.environ.get("YOLOV12_FEATURE_MIN_INLIERS", "6"))
        self.feature_min_inlier_ratio = float(os.environ.get("YOLOV12_FEATURE_MIN_INLIER_RATIO", "0.45"))
        self.feature_ransac_thresh_px = float(os.environ.get("YOLOV12_FEATURE_RANSAC_THRESH_PX", "4.0"))
        self.feature_refresh_frames = max(1, int(os.environ.get("YOLOV12_FEATURE_REFRESH_FRAMES", "10")))
        self.feature_clahe = self._env_bool("YOLOV12_FEATURE_CLAHE", True)
        self.feature_edge_preprocess = os.environ.get("YOLOV12_FEATURE_EDGE_PREPROCESS", "sobel").strip().lower()
        self.feature_orb_enable = self._env_bool("YOLOV12_FEATURE_ORB_ENABLE", True)
        self.feature_orb_nfeatures = int(os.environ.get("YOLOV12_FEATURE_ORB_NFEATURES", "250"))
        self.feature_orb_match_ratio = float(os.environ.get("YOLOV12_FEATURE_ORB_MATCH_RATIO", "0.75"))
        self.feature_orb_min_matches = int(os.environ.get("YOLOV12_FEATURE_ORB_MIN_MATCHES", "8"))
        self.yolo_jump_confirm_frames = max(1, int(os.environ.get("YOLOV12_YOLO_JUMP_CONFIRM_FRAMES", "3")))
        self.yolo_jump_gate_px = float(os.environ.get("YOLOV12_YOLO_JUMP_GATE_PX", "60.0"))
        self.feature_debug = self._env_bool("YOLOV12_FEATURE_DEBUG", True)
        self.feature_debug_every = max(1, int(os.environ.get("YOLOV12_FEATURE_DEBUG_EVERY", "30")))
        self.draw_feature_debug = self._env_bool("YOLOV12_DRAW_FEATURE_DEBUG", False)
        self.feature_families = self._parse_name_set(
            os.environ.get(
                "YOLOV12_FEATURE_FAMILIES",
                "sfp_port,sfp_module,sc_port,sc_plug,task_board,nic_card",
            )
        )
        self._current_feature_camera = ""
        self._iou = float(os.environ.get("YOLOV12_IOU", "0.45"))
        self._imgsz = int(os.environ.get("YOLOV12_IMGSZ", "640"))
        self._max_hz = max(0.1, float(os.environ.get("YOLOV12_MAX_HZ", "15.0")))
        self._min_period = 1.0 / self._max_hz
        self._frame_count: Dict[str, int] = {cam: 0 for cam in _CAMERAS}

        self._family_aliases = {
            "task_board": self._parse_name_set(
                os.environ.get("YOLOV12_TASKBOARD_CLASSES", "task_board,task board,taskboard,board")
            ),
            "nic_card": self._parse_name_set(os.environ.get("YOLOV12_NIC_CLASSES", "nic_card,nic card,nic")),
            "sc_port": self._parse_name_set(os.environ.get("YOLOV12_SC_CLASSES", "sc_port,sc port")),
            "sfp_port": self._parse_name_set(
                os.environ.get("YOLOV12_SFP_PORT_CLASSES", "sfp_port,sfp port,sfp_port_0,sfp_port_1")
            ),
            "sfp_module": self._parse_name_set(
                os.environ.get(
                    "YOLOV12_SFP_MODULE_CLASSES",
                    "sfp_module,sfp module,sfp-module,sfpmodule,transceiver",
                )
            ),
            "sc_plug": self._parse_name_set(
                os.environ.get("YOLOV12_SC_PLUG_CLASSES", "sc_plug,sc plug,sc_connector,sc connector")
            ),
        }

        self._last_infer_time = {cam: 0.0 for cam in _CAMERAS}
        self._latest_frames: Dict[str, Optional[Image]] = {cam: None for cam in _CAMERAS}
        self._latest_infos: Dict[str, Optional[CameraInfo]] = {cam: None for cam in _CAMERAS}
        self._trackers = {cam: PerCameraCanonicalTracker(self, cam) for cam in _CAMERAS}

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(f"YOLO model not found: {self._model_path}")
        self._model = YOLO(self._model_path)
        try:
            self._model.to(self._device)
        except Exception as exc:
            if self._device != "cpu":
                self.get_logger().warn(f"Falling back to CPU because moving model to {self._device} failed: {exc}")
                self._device = "cpu"
                self._model.to(self._device)
            else:
                raise

        self._image_subs = {
            "left": self.create_subscription(Image, "/left_camera/image", lambda msg: self._image_cb("left", msg), 10),
            "center": self.create_subscription(Image, "/center_camera/image", lambda msg: self._image_cb("center", msg), 10),
            "right": self.create_subscription(Image, "/right_camera/image", lambda msg: self._image_cb("right", msg), 10),
        }
        self._info_subs = {
            "left": self.create_subscription(CameraInfo, "/left_camera/camera_info", lambda msg: self._info_cb("left", msg), 10),
            "center": self.create_subscription(CameraInfo, "/center_camera/camera_info", lambda msg: self._info_cb("center", msg), 10),
            "right": self.create_subscription(CameraInfo, "/right_camera/camera_info", lambda msg: self._info_cb("right", msg), 10),
        }

        self._annotated_pubs = {
            "left": self.create_publisher(Image, "/left_camera/yolo/annotated", 10),
            "center": self.create_publisher(Image, "/center_camera/yolo/annotated", 10),
            "right": self.create_publisher(Image, "/right_camera/yolo/annotated", 10),
        }
        self._json_pubs = {
            "left": self.create_publisher(String, "/left_camera/yolo/detections_json", 10),
            "center": self.create_publisher(String, "/center_camera/yolo/detections_json", 10),
            "right": self.create_publisher(String, "/right_camera/yolo/detections_json", 10),
        }
        self._classes_pubs = {
            "left": self.create_publisher(String, "/left_camera/yolo/classes", 10),
            "center": self.create_publisher(String, "/center_camera/yolo/classes", 10),
            "right": self.create_publisher(String, "/right_camera/yolo/classes", 10),
        }
        self._mask_overlay_pubs = {
            "left": self.create_publisher(Image, "/left_camera/yolo/mask_overlay", 10),
            "center": self.create_publisher(Image, "/center_camera/yolo/mask_overlay", 10),
            "right": self.create_publisher(Image, "/right_camera/yolo/mask_overlay", 10),
        }
        self._mask_binary_pubs = {
            "left": self.create_publisher(Image, "/left_camera/yolo/taskboard_mask", 10),
            "center": self.create_publisher(Image, "/center_camera/yolo/taskboard_mask", 10),
            "right": self.create_publisher(Image, "/right_camera/yolo/taskboard_mask", 10),
        }
        self._mask_status_pubs = {
            "left": self.create_publisher(String, "/left_camera/yolo/mask_status", 10),
            "center": self.create_publisher(String, "/center_camera/yolo/mask_status", 10),
            "right": self.create_publisher(String, "/right_camera/yolo/mask_status", 10),
        }

        self._timer = self.create_timer(0.02, self._tick)

        self.get_logger().info("YOLOv12 tracked canonical multi-camera detector started")
        self.get_logger().info(f"Model: {self._model_path}")
        self.get_logger().info(f"Device request: {self._device_request}")
        self.get_logger().info(f"Resolved device: {self._device}")
        self.get_logger().info(f"Raw/public confidence: {self._raw_conf:.2f}/{self.publish_conf:.2f}")
        self.get_logger().info(f"Inference rate limit per camera: {self._max_hz:.2f} Hz")

    @staticmethod
    def _norm_name(name: object) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    def _parse_name_set(self, text: str) -> set[str]:
        return {self._norm_name(x) for x in str(text).split(",") if str(x).strip()}

    def _family_publish_thresholds(self, family: str) -> tuple[float, float]:
        if family == "task_board":
            return self.task_board_publish_conf, max(self.track_min_raw_for_publish, 0.80)
        if family in {"sfp_port", "sc_port"}:
            return self.port_publish_conf, min(self.track_min_raw_for_publish, 0.72)
        if family == "sfp_module":
            return self.module_publish_conf, self.feature_confirmed_publish_conf
        if family == "sc_plug":
            return self.plug_publish_conf, self.feature_confirmed_publish_conf
        return self.publish_conf, self.track_min_raw_for_publish

    def _adjust_public_detection_confidence(self, det: Dict, track: CanonicalTrack) -> None:
        ft = track.feature_track
        feature_good = (
            ft is not None
            and ft.quality_score >= self.feature_confirmed_quality_min
            and ft.inlier_count >= self.feature_min_inliers
        )
        if track.family in {"sfp_module", "sc_plug"} and feature_good:
            det["confidence"] = float(max(float(det.get("confidence", 0.0)), self.feature_confirmed_publish_conf))
            if track.miss_count > 0:
                det["feature_mode"] = "hold"

    def _env_bool(self, key: str, default: bool) -> bool:
        value = str(os.environ.get(key, "1" if default else "0")).strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _resolve_device(self, requested: str) -> str:
        if requested in {"", "auto"}:
            if torch.cuda.is_available():
                return "cuda:0"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda"):
            return requested if torch.cuda.is_available() else "cpu"
        if requested == "mps":
            return "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
        return "cpu"

    def _image_cb(self, camera_name: str, msg: Image) -> None:
        with self._lock:
            self._latest_frames[camera_name] = msg

    def _info_cb(self, camera_name: str, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_infos[camera_name] = msg

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            frames = dict(self._latest_frames)
        for camera_name, msg in frames.items():
            if msg is None:
                continue
            if now - self._last_infer_time[camera_name] < self._min_period:
                continue
            try:
                result = self._process_camera_frame(camera_name, msg)
            except Exception as exc:
                self.get_logger().error(f"{camera_name} inference failed: {exc}")
                continue
            self._last_infer_time[camera_name] = now
            self._publish_camera_outputs(camera_name, msg, *result[:6])

    def _process_camera_frame(self, camera_name: str, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._current_feature_camera = camera_name
        raw = self._infer_raw_detections(frame, msg, camera_name)
        canonical = self._canonicalize_raw_detections(raw, camera_name)
        detections = self._trackers[camera_name].update(canonical, time.monotonic(), frame)
        classes = [str(det.get("instance_name", det.get("class_name", ""))) for det in detections]
        annotated = self._draw_filtered_detections(frame, detections)
        if self.draw_feature_debug:
            annotated = self._draw_feature_debug(annotated, camera_name)
        if self.draw_raw_debug:
            annotated = self._draw_raw_debug(annotated, canonical)
        mask_overlay = frame.copy()
        mask_binary = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask_status = "disabled_simple_tracked_yolo"
        self._log_track_summary(camera_name, canonical, detections)
        return annotated, detections, classes, mask_overlay, mask_binary, mask_status, canonical

    def _publish_camera_outputs(
        self,
        camera_name: str,
        image_msg: Image,
        annotated: np.ndarray,
        detections: List[Dict],
        classes: List[str],
        mask_overlay: np.ndarray,
        mask_binary: np.ndarray,
        mask_status: str,
    ) -> None:
        annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_msg.header = image_msg.header
        self._annotated_pubs[camera_name].publish(annotated_msg)

        json_msg = String()
        json_msg.data = json.dumps(detections, separators=(",", ":"))
        self._json_pubs[camera_name].publish(json_msg)

        classes_msg = String()
        classes_msg.data = ",".join(classes)
        self._classes_pubs[camera_name].publish(classes_msg)

        overlay_msg = self._bridge.cv2_to_imgmsg(mask_overlay, encoding="bgr8")
        overlay_msg.header = image_msg.header
        self._mask_overlay_pubs[camera_name].publish(overlay_msg)

        mask_msg = self._bridge.cv2_to_imgmsg(mask_binary, encoding="mono8")
        mask_msg.header = image_msg.header
        self._mask_binary_pubs[camera_name].publish(mask_msg)

        status_msg = String()
        status_msg.data = mask_status
        self._mask_status_pubs[camera_name].publish(status_msg)

    def _infer_raw_detections(self, frame: np.ndarray, msg: Image, camera_name: str) -> List[Dict]:
        predict = self._model.track if self.use_ultralytics_track else self._model.predict
        kwargs = {
            "source": frame,
            "device": self._device,
            "conf": self._raw_conf,
            "iou": self._iou,
            "imgsz": self._imgsz,
            "verbose": False,
        }
        if self.use_ultralytics_track:
            kwargs.update({"persist": True, "tracker": self._tracker_yaml})
        results = predict(**kwargs)
        result = results[0]
        names = result.names if hasattr(result, "names") else self._model.names
        seg_geoms = self._extract_segmentation_geometries(result, frame.shape[:2])
        boxes = result.boxes
        detections: List[Dict] = []
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((0,), dtype=np.float32)
        clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((0,), dtype=int)
        ids = None
        if getattr(boxes, "id", None) is not None:
            try:
                ids = boxes.id.detach().cpu().numpy().astype(int)
            except Exception:
                ids = None
        stamp_sec = self._stamp_to_sec(msg.header.stamp)
        img_h, img_w = frame.shape[:2]
        for idx, (box, conf, cls_idx) in enumerate(zip(xyxy, confs, clss)):
            cls_name = str(names[int(cls_idx)]) if names is not None else str(int(cls_idx))
            det = {
                "class_id": int(cls_idx),
                "raw_class_name": cls_name,
                "base_class_name": cls_name,
                "class_name": cls_name,
                "instance_name": cls_name,
                "confidence": float(conf),
                "bbox_xyxy": [float(v) for v in np.asarray(box, dtype=np.float32).tolist()],
                "camera_name": camera_name,
                "stamp_sec": float(stamp_sec),
                "image_width": int(img_w),
                "image_height": int(img_h),
            }
            if ids is not None and idx < len(ids):
                det["ultralytics_track_id"] = int(ids[idx])
            if idx < len(seg_geoms) and isinstance(seg_geoms[idx], dict):
                det.update(seg_geoms[idx])
            detections.append(det)
        return detections

    def _extract_segmentation_geometries(self, result, image_shape) -> List[Dict]:
        h, w = int(image_shape[0]), int(image_shape[1])
        geoms: List[Dict] = []
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None or getattr(masks_obj, "data", None) is None:
            return geoms
        try:
            masks = masks_obj.data.detach().cpu().numpy()
        except Exception:
            return geoms
        for mask in masks:
            m = np.asarray(mask, dtype=np.float32)
            if m.ndim != 2:
                geoms.append({})
                continue
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mb = np.where(m > 0.5, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                geoms.append({})
                continue
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            if area < 8.0:
                geoms.append({})
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            corners = cv2.boxPoints(rect).astype(np.float32)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * max(peri, 1.0), True)
            poly = approx.reshape(-1, 2).astype(np.float32) if approx is not None and len(approx) >= 3 else contour.reshape(-1, 2).astype(np.float32)
            geoms.append(
                {
                    "mask_area_px": area,
                    "obb_cxcywh_deg": [float(cx), float(cy), float(rw), float(rh), float(angle)],
                    "obb_corners_uv": [[float(x), float(y)] for x, y in corners.tolist()],
                    "mask_polygon_uv": [[float(x), float(y)] for x, y in poly.tolist()],
                }
            )
        return geoms

    def _family_for_name(self, class_name: object) -> Optional[str]:
        name = self._norm_name(class_name)
        for family in ("task_board", "nic_card", "sc_port", "sfp_module", "sfp_port", "sc_plug"):
            if name in self._family_aliases[family]:
                return family
        return None

    def _canonicalize_raw_detections(self, detections: List[Dict], camera_name: str) -> List[CanonicalDetection]:
        out: List[CanonicalDetection] = []
        for raw in detections:
            family = self._family_for_name(raw.get("raw_class_name", raw.get("class_name", "")))
            if family is None:
                continue
            out.append(CanonicalDetection.from_raw(raw, family, camera_name))
        return out

    def _sort_output_detections(self, detections: List[Dict]) -> List[Dict]:
        return sorted(
            detections,
            key=lambda det: (
                _FAMILY_ORDER.get(str(det.get("base_class_name", det.get("class_name", ""))).replace("_0", "").replace("_1", ""), 99),
                self._instance_sort_index(det.get("instance_name", det.get("class_name", ""))),
                -float(det.get("confidence", 0.0)),
                int(det.get("track_id", 0)),
            ),
        )

    def _instance_sort_index(self, name: object) -> int:
        text = self._norm_name(name)
        if text.endswith("_0"):
            return 0
        if text.endswith("_1"):
            return 1
        return 9

    def _detection_family_name(self, det: Dict) -> Optional[str]:
        base = str(det.get("base_class_name", det.get("class_name", "")))
        if base in _FAMILY_ORDER:
            return base
        if str(det.get("class_name", "")).startswith("sfp_port"):
            return "sfp_port"
        return self._family_for_name(base)

    def _detection_anchor_point(self, det: Dict, family_key: str = "GENERIC") -> np.ndarray:
        if isinstance(det.get("center_uv"), list) and len(det["center_uv"]) >= 2:
            return np.asarray(det["center_uv"][:2], dtype=np.float32)
        obb = det.get("obb_cxcywh_deg")
        if isinstance(obb, list) and len(obb) >= 2:
            return np.array([float(obb[0]), float(obb[1])], dtype=np.float32)
        x1, y1, x2, y2 = [float(v) for v in det.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])[:4]]
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    def _draw_filtered_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        out = image.copy()
        colors = {
            "task_board": (255, 0, 0),
            "nic_card": (0, 255, 255),
            "sc_port": (255, 255, 0),
            "sfp_port": (0, 165, 255),
            "sfp_module": (180, 180, 180),
            "sc_plug": (0, 128, 255),
        }
        for det in detections:
            x1, y1, x2, y2 = [int(round(float(v))) for v in det.get("bbox_xyxy", [0, 0, 0, 0])]
            family = self._detection_family_name(det)
            color = colors.get(family, (0, 255, 0))
            corners = det.get("obb_corners_uv")
            if isinstance(corners, list) and len(corners) == 4:
                pts = np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            center = self._detection_anchor_point(det)
            cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
            label = str(det.get("instance_name", det.get("class_name", "")))
            text = (
                f"{label} id={int(det.get('track_id', -1))} "
                f"conf={float(det.get('confidence', 0.0)):.2f} raw={float(det.get('raw_confidence', 0.0)):.2f} "
                f"feat={float(det.get('feature_quality_score', 0.0)):.2f} pts={int(det.get('feature_tracked_count', 0))}"
            )
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            tx = max(0, x1)
            ty = max(th + 6, y1 + th + 4)
            cv2.rectangle(out, (tx, ty - th - 6), (tx + tw + 8, ty + baseline - 2), color, -1, cv2.LINE_AA)
            cv2.putText(out, text, (tx + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        return out

    def _draw_feature_debug(self, image: np.ndarray, camera_name: str) -> np.ndarray:
        out = image.copy()
        tracker = self._trackers.get(camera_name)
        if tracker is None:
            return out
        for track in tracker.tracks:
            ft = track.feature_track
            if ft is None:
                continue
            for old, new in ft._last_motion_pairs[:20]:
                p0 = (int(round(float(old[0]))), int(round(float(old[1]))))
                p1 = (int(round(float(new[0]))), int(round(float(new[1]))))
                cv2.line(out, p0, p1, (0, 180, 255), 1, cv2.LINE_AA)
            for idx, (pt, pid) in enumerate(zip(ft.active_points_uv[:40], ft.point_ids[:40])):
                p = (int(round(float(pt[0]))), int(round(float(pt[1]))))
                cv2.circle(out, p, 2, (0, 255, 0), -1, cv2.LINE_AA)
                if idx < 10:
                    cv2.putText(
                        out,
                        str(pid),
                        (p[0] + 3, p[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
        return out

    def _draw_raw_debug(self, image: np.ndarray, raw: List[CanonicalDetection]) -> np.ndarray:
        out = image.copy()
        for det in raw:
            x1, y1, x2, y2 = [int(round(float(v))) for v in det.bbox_xyxy]
            cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(
                out,
                f"raw {det.family} {det.confidence:.2f}",
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )
        return out

    def _log_track_summary(self, camera_name: str, raw: List[CanonicalDetection], published: List[Dict]) -> None:
        self._frame_count[camera_name] = self._frame_count.get(camera_name, 0) + 1
        if self._should_debug_log(camera_name):
            raw_summary = ",".join(f"{det.family}:{det.confidence:.2f}" for det in raw)
            pub_summary = ",".join(
                f"{det.get('instance_name')}#{det.get('track_id')}:{float(det.get('confidence', 0.0)):.2f}"
                for det in published
            )
            self.get_logger().info(f"TRACK_YOLO camera={camera_name} raw={raw_summary} published={pub_summary}")
        self._log_feature_summary(camera_name, published)

    def _should_debug_log(self, camera_name: str) -> bool:
        return self.track_debug and self._frame_count.get(camera_name, 0) % self.track_debug_every == 0

    def _should_feature_debug_log(self, camera_name: str) -> bool:
        return self.feature_debug and self._frame_count.get(camera_name, 0) % self.feature_debug_every == 0

    def _log_feature_summary(self, camera_name: str, published: List[Dict]) -> None:
        if not self._should_feature_debug_log(camera_name):
            return
        parts = []
        for det in published:
            parts.append(
                f"{det.get('instance_name')}#{det.get('track_id')} "
                f"conf={float(det.get('confidence', 0.0)):.2f} raw={float(det.get('raw_confidence', 0.0)):.2f} "
                f"feat={float(det.get('feature_quality_score', 0.0)):.2f} "
                f"pts={int(det.get('feature_tracked_count', 0))} "
                f"inliers={int(det.get('feature_inlier_count', 0))} "
                f"mode={det.get('feature_mode', 'kalman_only')}"
            )
        jump_rejects = 0
        tracker = self._trackers.get(camera_name)
        if tracker is not None:
            for track in tracker.tracks:
                if track.feature_track is not None:
                    jump_rejects += int(track.feature_track.jump_rejects)
        summary = ";".join(parts)
        if summary:
            summary += f" jump_rejects={jump_rejects}"
        else:
            summary = f"none jump_rejects={jump_rejects}"
        self.get_logger().info(f"FEATURE_TRACK camera={camera_name} published={summary}")

    def _stamp_to_sec(self, stamp) -> float:
        return float(getattr(stamp, "sec", 0)) + 1e-9 * float(getattr(stamp, "nanosec", 0))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloV12MultiCameraDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
