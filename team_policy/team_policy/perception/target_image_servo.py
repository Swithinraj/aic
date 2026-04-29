"""
target_image_servo.py
---------------------
Target-image visual servoing backend.
Matching priority per camera:
  1. EfficientLoFTR  (if available and confident)
  2. ORB/AKAZE       (fast feature matching)
  3. ECC             (photometric template matching)
  4. Phase correlation on edge maps (last resort)
"""

import json
import math
import os

import cv2
import numpy as np


_MIN_CONF_LOFTR  = 0.05   # anything below this is treated as loftr failure
_MIN_ORB_INLIERS = 6      # minimum inliers after RANSAC homography

# Minimum weight per camera – even a bad match contributes a tiny signal
MIN_WEIGHT = {"center": 0.20, "left": 0.05, "right": 0.05}


class TargetImageServo:
    def __init__(self, target_img_dir: str, logger, use_loftr: bool = True):
        self.target_img_dir = target_img_dir
        self.logger = logger
        self.use_loftr = use_loftr

        self.targets: dict   = {}  # cam -> BGR ndarray
        self.rois:    dict   = {}  # cam -> [x1,y1,x2,y2]  (fixed target ROI)
        self.trackers: dict  = {}  # cam -> {"type": "loftr"|"orb", "obj": ...}

        # Per-camera ORB/BF matcher (reused across calls)
        self._orb   = cv2.ORB_create(nfeatures=2000)
        self._akaze = cv2.AKAZE_create()
        self._bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Target descriptors (computed once)
        self._target_kp:  dict = {}
        self._target_des: dict = {}
        self._target_gray: dict = {}

        self._load_targets()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_targets(self):
        cameras = ["center", "left", "right"]

        default_rois = {
            "center": [350, 250, 780, 760],
            "left":   [300, 220, 850, 820],
            "right":  [280, 220, 850, 850],
        }

        config_path = os.path.join(self.target_img_dir, "target_align_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                for k, v in cfg.items():
                    if k in default_rois and "roi_xyxy" in v:
                        default_rois[k] = v["roi_xyxy"]
            except Exception as exc:
                self.logger.warn(f"[target_image_servo] Failed to load config: {exc}")

        for cam in cameras:
            img_path = os.path.join(self.target_img_dir, f"{cam}_align.png")
            if not os.path.exists(img_path):
                self.logger.warn(f"[target_image_servo] Missing target image: {img_path}")
                continue
            img = cv2.imread(img_path)
            if img is None:
                self.logger.warn(f"[target_image_servo] Failed to read: {img_path}")
                continue

            self.targets[cam] = img
            self.rois[cam]    = default_rois.get(cam, [200, 200, 800, 800])

            # Pre-compute ORB descriptors on the target ROI
            roi = self.rois[cam]
            x1, y1, x2, y2 = map(int, roi)
            t_crop = img[y1:y2, x1:x2]
            t_gray = cv2.cvtColor(t_crop, cv2.COLOR_BGR2GRAY)
            kp, des = self._orb.detectAndCompute(t_gray, None)
            self._target_gray[cam] = t_gray
            self._target_kp[cam]   = kp
            self._target_des[cam]  = des

            # LoFTR tracker
            if self.use_loftr:
                try:
                    from team_policy.perception.efficientloftr_tracker import EfficientLoFTRTracker
                    tracker = EfficientLoFTRTracker()
                    if tracker.available():
                        tracker.set_reference(img, self.rois[cam])
                        self.trackers[cam] = {"type": "loftr", "obj": tracker}
                        self.logger.info(f"[target_image_servo] LoFTR loaded for {cam}")
                        continue
                except Exception as exc:
                    self.logger.warn(f"[target_image_servo] LoFTR init failed for {cam}: {exc}")

            self.trackers[cam] = {"type": "orb", "obj": None}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return len(self.targets) > 0

    def get_target_cameras(self):
        return list(self.targets.keys())

    def get_target_roi(self, camera_name: str):
        return self.rois.get(camera_name)

    def match_camera(self, camera_name: str, current_img: np.ndarray,
                     current_roi=None) -> dict | None:
        """
        Returns dict with keys:
            method, matches, inliers, e_c (np 2-vec), err, conf, residual
        or None if completely failed.
        """
        if camera_name not in self.targets:
            return None

        # Always use the fixed target ROI for current image matching.
        # YOLO-based dynamic ROIs cause drift; the fixed ROI is consistent.
        target_roi = self.rois[camera_name]

        # Try LoFTR
        ti = self.trackers.get(camera_name, {"type": "orb"})
        if ti["type"] == "loftr":
            result = self._try_loftr(camera_name, current_img, target_roi)
            if result is not None and result["conf"] >= _MIN_CONF_LOFTR:
                return result

        # Try ORB with RANSAC
        result = self._try_orb(camera_name, current_img, target_roi)
        if result is not None and result["inliers"] >= _MIN_ORB_INLIERS:
            return result

        # Try ECC (photometric)
        result = self._try_ecc(camera_name, current_img, target_roi)
        if result is not None:
            return result

        # Try phase correlation on edges
        result = self._try_phase_corr(camera_name, current_img, target_roi)
        return result  # may be None

    # ------------------------------------------------------------------
    # Matching backends
    # ------------------------------------------------------------------
    def _try_loftr(self, cam, current_img, roi):
        ti = self.trackers.get(cam)
        if ti is None or ti["type"] != "loftr":
            return None
        try:
            res = ti["obj"].track(current_img, roi)
            if not res.get("success", False):
                return None
            e_c  = np.asarray(res["shift"], dtype=np.float64)
            err  = float(np.linalg.norm(e_c))
            mad  = float(res.get("mad",  10.0))
            score = float(res.get("score", 0.5))
            n_matches = int(res.get("matches", 0))
            # confidence: high score + low MAD + many matches
            conf = score * math.exp(-mad / 8.0) * min(1.0, n_matches / 20.0)
            conf = float(np.clip(conf, 0.01, 1.0))
            return {
                "method":   "loftr",
                "matches":  n_matches,
                "inliers":  n_matches,
                "e_c":      e_c,
                "err":      err,
                "conf":     conf,
                "residual": mad,
            }
        except Exception:
            return None

    def _try_orb(self, cam, current_img, roi):
        if cam not in self._target_des or self._target_des[cam] is None:
            return None
        try:
            x1, y1, x2, y2 = map(int, roi)
            c_crop = current_img[y1:y2, x1:x2]
            if c_crop.size == 0:
                return None
            c_gray = cv2.cvtColor(c_crop, cv2.COLOR_BGR2GRAY)

            kp_c, des_c = self._orb.detectAndCompute(c_gray, None)
            des_t = self._target_des[cam]
            kp_t  = self._target_kp[cam]

            if des_t is None or des_c is None or len(des_t) < 8 or len(des_c) < 8:
                return None

            # kNN match k=2 for Lowe's ratio test
            matches_raw = self._bf_hamming.knnMatch(des_t, des_c, k=2)
            good = []
            for pair in matches_raw:
                if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance:
                    good.append(pair[0])
            if len(good) < _MIN_ORB_INLIERS:
                return None

            pts_t = np.float32([kp_t[m.queryIdx].pt for m in good])
            pts_c = np.float32([kp_c[m.trainIdx].pt for m in good])

            # RANSAC homography to filter outliers
            inlier_mask = None
            if len(good) >= 8:
                _, mask = cv2.findHomography(pts_t, pts_c, cv2.RANSAC, 5.0)
                if mask is not None:
                    inlier_mask = mask.ravel().astype(bool)

            if inlier_mask is not None and inlier_mask.sum() >= _MIN_ORB_INLIERS:
                pts_t_in = pts_t[inlier_mask]
                pts_c_in = pts_c[inlier_mask]
                n_in = int(inlier_mask.sum())
            else:
                pts_t_in = pts_t
                pts_c_in = pts_c
                n_in = len(good)

            diff = pts_c_in - pts_t_in
            e_c  = np.median(diff, axis=0).astype(np.float64)
            err  = float(np.linalg.norm(e_c))
            residuals = np.linalg.norm(diff - e_c, axis=1)
            res_med = float(np.median(residuals))

            # confidence: inlier ratio + residual decay
            inlier_ratio = n_in / max(len(good), 1)
            conf = inlier_ratio * math.exp(-res_med / 12.0)
            conf = float(np.clip(conf, 0.01, 1.0))

            return {
                "method":   "orb",
                "matches":  len(good),
                "inliers":  n_in,
                "e_c":      e_c,
                "err":      err,
                "conf":     conf,
                "residual": res_med,
            }
        except Exception:
            return None

    def _try_ecc(self, cam, current_img, roi):
        """ECC template alignment (translation model)."""
        try:
            x1, y1, x2, y2 = map(int, roi)
            t_gray = self._target_gray.get(cam)
            if t_gray is None:
                return None
            c_crop = current_img[y1:y2, x1:x2]
            if c_crop.size == 0:
                return None
            c_gray = cv2.cvtColor(c_crop, cv2.COLOR_BGR2GRAY)

            # Resize to same shape if needed
            if t_gray.shape != c_gray.shape:
                c_gray = cv2.resize(c_gray, (t_gray.shape[1], t_gray.shape[0]))

            warp_mode   = cv2.MOTION_TRANSLATION
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4)

            ecc_val, warp_out = cv2.findTransformECC(
                t_gray.astype(np.float32),
                c_gray.astype(np.float32),
                warp_matrix, warp_mode, criteria,
            )
            dx = float(warp_out[0, 2])
            dy = float(warp_out[1, 2])
            e_c  = np.array([dx, dy], dtype=np.float64)
            err  = float(np.linalg.norm(e_c))
            # ECC value is correlation coefficient in [-1,1]; map to conf
            conf = float(np.clip((ecc_val + 1.0) / 2.0 * 0.5, 0.01, 0.50))
            return {
                "method":   "ecc",
                "matches":  0,
                "inliers":  0,
                "e_c":      e_c,
                "err":      err,
                "conf":     conf,
                "residual": 0.0,
            }
        except Exception:
            return None

    def _try_phase_corr(self, cam, current_img, roi):
        """Phase correlation on Canny edge maps – last resort."""
        try:
            x1, y1, x2, y2 = map(int, roi)
            t_gray = self._target_gray.get(cam)
            if t_gray is None:
                return None
            c_crop = current_img[y1:y2, x1:x2]
            if c_crop.size == 0:
                return None
            c_gray = cv2.cvtColor(c_crop, cv2.COLOR_BGR2GRAY)

            # Edge maps
            t_edge = cv2.Canny(t_gray, 40, 120).astype(np.float32)
            c_gray_r = cv2.resize(c_gray, (t_gray.shape[1], t_gray.shape[0]))
            c_edge = cv2.Canny(c_gray_r, 40, 120).astype(np.float32)

            if t_edge.sum() < 10 or c_edge.sum() < 10:
                return None

            (dx, dy), response = cv2.phaseCorrelate(t_edge, c_edge)
            e_c  = np.array([dx, dy], dtype=np.float64)
            err  = float(np.linalg.norm(e_c))
            # response in [0,1], very low → unreliable
            conf = float(np.clip(response * 0.3, 0.01, 0.30))
            return {
                "method":   "phase_corr",
                "matches":  0,
                "inliers":  0,
                "e_c":      e_c,
                "err":      err,
                "conf":     conf,
                "residual": 0.0,
            }
        except Exception:
            return None
