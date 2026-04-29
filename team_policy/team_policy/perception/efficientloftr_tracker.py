import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

class EfficientLoFTRTracker:
    def __init__(self, model_name: str = "zju-community/efficientloftr", device: Optional[str] = None, threshold: float = 0.2):
        self.available_flag = False
        self.threshold = threshold
        self.processor = None
        self.model = None
        self.device = None
        self.reference_image_pil = None
        self.reference_roi_xyxy = None
        self.reference_center_uv = None
        
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForKeypointMatching
            from PIL import Image
            self.Image = Image
            self.torch = torch
            
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
                
            # Lazy load happens here if available
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForKeypointMatching.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.available_flag = True
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to initialize EfficientLoFTR: {e}")

    def available(self) -> bool:
        return self.available_flag

    def set_reference(self, image_bgr: np.ndarray, roi_xyxy: list) -> bool:
        if not self.available():
            return False
            
        try:
            # Crop to ROI with some margin
            h, w = image_bgr.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in roi_xyxy]
            
            # Add margin
            margin_x = int((x2 - x1) * 0.25)
            margin_y = int((y2 - y1) * 0.25)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            
            if x2 <= x1 or y2 <= y1:
                return False
                
            crop_bgr = image_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            self.reference_image_pil = self.Image.fromarray(crop_rgb)
            self.reference_roi_xyxy = [x1, y1, x2, y2]
            self.reference_center_uv = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            return True
        except Exception:
            return False

    def track(self, image_bgr: np.ndarray, current_roi_xyxy: list) -> Dict[str, Any]:
        """Tracks the reference crop to the current image near the current ROI."""
        res = {"success": False, "shift": np.zeros(2), "matches": 0, "score": 0.0, "mad": 0.0}
        
        if not self.available() or self.reference_image_pil is None:
            return res
            
        try:
            h, w = image_bgr.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in current_roi_xyxy]
            
            # Enlarge search ROI significantly
            margin_x = int((x2 - x1) * 0.5)
            margin_y = int((y2 - y1) * 0.5)
            
            cx1 = max(0, x1 - margin_x)
            cy1 = max(0, y1 - margin_y)
            cx2 = min(w, x2 + margin_x)
            cy2 = min(h, y2 + margin_y)
            
            if cx2 <= cx1 or cy2 <= cy1:
                return res
                
            crop_bgr = image_bgr[cy1:cy2, cx1:cx2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            current_image_pil = self.Image.fromarray(crop_rgb)
            
            images = [self.reference_image_pil, current_image_pil]
            inputs = self.processor(images, return_tensors="pt").to(self.device)
            
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
                
            image_sizes = [[(self.reference_image_pil.height, self.reference_image_pil.width), 
                            (current_image_pil.height, current_image_pil.width)]]
            matches = self.processor.post_process_keypoint_matching(outputs, image_sizes, threshold=self.threshold)
            
            if not matches or len(matches) == 0:
                return res
                
            m = matches[0]
            kp0 = m["keypoints0"].cpu().numpy()
            kp1 = m["keypoints1"].cpu().numpy()
            scores = m["matching_scores"].cpu().numpy()
            
            n_matches = len(kp0)
            if n_matches < 4:
                return res
                
            # Shift vectors between reference crop and current crop
            shifts = kp1 - kp0
            
            # Median shift
            median_shift = np.median(shifts, axis=0)
            
            # Median absolute deviation (MAD) to gauge consistency
            mad = np.median(np.abs(shifts - median_shift), axis=0)
            mad_norm = float(np.linalg.norm(mad))
            
            mean_score = float(np.mean(scores))
            
            # Reject if too noisy
            if mad_norm > 15.0:
                return res
                
            # Compute full image shift
            # In crop space, feature went from kp0 to kp1. 
            # kp0 was relative to self.reference_roi_xyxy top-left.
            # kp1 is relative to current search crop top-left.
            # Point in original image: p_ref = kp0 + [ref_x1, ref_y1]
            # Mapped point in new image: p_cur = kp1 + [cx1, cy1]
            # Total shift in absolute image coordinates:
            # Shift = p_cur - p_ref = kp1 + [cx1, cy1] - (kp0 + [ref_x1, ref_y1])
            # Shift = (kp1 - kp0) + [cx1 - ref_x1, cy1 - ref_y1]
            
            ref_x1, ref_y1 = self.reference_roi_xyxy[0], self.reference_roi_xyxy[1]
            abs_shift = median_shift + np.array([cx1 - ref_x1, cy1 - ref_y1])
            
            res["success"] = True
            res["shift"] = abs_shift
            res["matches"] = n_matches
            res["score"] = mean_score
            res["mad"] = mad_norm
            return res
            
        except Exception as e:
            return res
