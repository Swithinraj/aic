import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class YoloV12MultiCameraDetector(Node):
    def __init__(self):
        super().__init__("yolov12_multicamera_detector")

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        pkg_dir = Path(__file__).resolve().parent
        default_model_path = Path(__file__).resolve().parents[1] / "models" / "yolov12.pt"
        default_mask_path = pkg_dir / "task_board_mask.png"
        default_features_path = pkg_dir / "taskboard_features.json"

        self._model_path = os.environ.get("YOLOV12_MODEL_PATH", str(default_model_path))
        self._mask_path = os.environ.get("YOLOV12_TASKBOARD_MASK_PATH", str(default_mask_path))
        self._features_path = os.environ.get("YOLOV12_TASKBOARD_FEATURES_PATH", str(default_features_path))

        self._device_request = os.environ.get("YOLOV12_DEVICE", "auto").strip().lower()
        self._device = self._resolve_device(self._device_request)

        self._conf = float(os.environ.get("YOLOV12_CONF", "0.25"))
        self._iou = float(os.environ.get("YOLOV12_IOU", "0.45"))
        self._imgsz = int(os.environ.get("YOLOV12_IMGSZ", "640"))
        self._max_hz = max(0.1, float(os.environ.get("YOLOV12_MAX_HZ", "5.0")))
        self._min_period = 1.0 / self._max_hz
        self._mask_alpha = float(os.environ.get("YOLOV12_MASK_ALPHA", "0.55"))

        self._taskboard_classes = self._parse_name_set(
            os.environ.get("YOLOV12_TASKBOARD_CLASSES", "taskboard,task_board,task board,board")
        )
        self._fix_classes = self._parse_name_set(
            os.environ.get("YOLOV12_FIX_CLASSES", "fix")
        )

        self._pink_h1_min = int(os.environ.get("YOLOV12_PINK_H1_MIN", "145"))
        self._pink_h1_max = int(os.environ.get("YOLOV12_PINK_H1_MAX", "160"))
        self._pink_s_min = int(os.environ.get("YOLOV12_PINK_S_MIN", "40"))
        self._pink_v_min = int(os.environ.get("YOLOV12_PINK_V_MIN", "40"))
        self._fix_h_tol = int(os.environ.get("YOLOV12_FIX_H_TOL", "10"))
        self._fix_s_tol = int(os.environ.get("YOLOV12_FIX_S_TOL", "110"))
        self._fix_v_tol = int(os.environ.get("YOLOV12_FIX_V_TOL", "110"))

        self._last_infer_time = {"left": 0.0, "center": 0.0, "right": 0.0}
        self._latest_frames: Dict[str, Optional[Image]] = {"left": None, "center": None, "right": None}

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(f"YOLO model not found: {self._model_path}")
        if not os.path.isfile(self._mask_path):
            raise FileNotFoundError(f"Taskboard mask not found: {self._mask_path}")
        if not os.path.isfile(self._features_path):
            raise FileNotFoundError(f"Taskboard features json not found: {self._features_path}")

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

        self._mask_rgba = cv2.imread(self._mask_path, cv2.IMREAD_UNCHANGED)
        if self._mask_rgba is None:
            raise FileNotFoundError(f"Could not load taskboard mask image: {self._mask_path}")

        if self._mask_rgba.ndim == 2:
            gray = self._mask_rgba.copy()
            self._mask_rgba = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
            self._mask_rgba[:, :, 3] = np.where(gray > 0, 255, 0).astype(np.uint8)
        elif self._mask_rgba.shape[2] == 3:
            alpha = np.where(np.any(self._mask_rgba > 0, axis=2), 255, 0).astype(np.uint8)
            self._mask_rgba = np.dstack([self._mask_rgba, alpha])

        self._mask_bgr = self._mask_rgba[:, :, :3].copy()

        with open(self._features_path, "r") as f:
            features_data = json.load(f)

        self._feature_boxes_norm = {}
        for ann in features_data.get("annotations", []):
            label = str(ann.get("label", "")).strip()
            bbox = ann.get("bbox_xyxy", None)
            if label and isinstance(bbox, list) and len(bbox) == 4:
                self._feature_boxes_norm[self._norm_name(label)] = [int(v) for v in bbox]

        self._mask_fix_hint = self._feature_boxes_norm.get("fix")
        self._base_mask_board_quad = self._compute_mask_board_quad()
        self._base_sc_rail_quads = self._load_feature_quads("sc_rail")
        self._base_nic_rail_quads = self._load_feature_quads("nic_rail")

        self._mask_fix_mean_hsv = np.array(
            [
                0.5 * float(self._pink_h1_min + self._pink_h1_max),
                0.5 * float(self._pink_s_min + 255.0),
                0.5 * float(self._pink_v_min + 255.0),
            ],
            dtype=np.float32,
        )
        self._mask_fix_shape_contour = None

        self._mask_fix_component = self._extract_mask_fix_component()
        if self._mask_fix_component is None:
            self._mask_fix_component = self._extract_component_from_threshold(self._mask_bgr, self._mask_fix_hint)
        if self._mask_fix_component is None:
            raise RuntimeError("Could not find fix feature in task_board_mask.png")

        self._mask_fix_mean_hsv = self._mask_fix_component["mean_hsv"]
        self._mask_fix_shape_contour = self._mask_fix_component["contour"]

        refined = self._extract_mask_fix_component()
        if refined is not None:
            self._mask_fix_component = refined
            self._mask_fix_mean_hsv = refined["mean_hsv"]
            self._mask_fix_shape_contour = refined["contour"]

        self._rotated_states = self._build_rotated_states()
        if len(self._rotated_states) == 0:
            raise RuntimeError("Could not build rotated mask states")

        self._image_subs = {
            "left": self.create_subscription(Image, "/left_camera/image", lambda msg: self._image_cb("left", msg), 10),
            "center": self.create_subscription(Image, "/center_camera/image", lambda msg: self._image_cb("center", msg), 10),
            "right": self.create_subscription(Image, "/right_camera/image", lambda msg: self._image_cb("right", msg), 10),
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

        self.get_logger().info("YOLOv12 multi-camera detector started")
        self.get_logger().info(f"Model: {self._model_path}")
        self.get_logger().info(f"Mask: {self._mask_path}")
        self.get_logger().info(f"Features: {self._features_path}")
        self.get_logger().info(f"Device request: {self._device_request}")
        self.get_logger().info(f"Resolved device: {self._device}")
        self.get_logger().info(f"Inference rate limit per camera: {self._max_hz:.2f} Hz")

    def _parse_name_set(self, text: str) -> set:
        return {self._norm_name(x) for x in text.split(",") if x.strip()}

    def _norm_name(self, name: str) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

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

    def _compute_mask_board_quad(self) -> np.ndarray:
        edge1 = self._feature_boxes_norm.get("edge_1")
        edge3 = self._feature_boxes_norm.get("edge_3")
        edge2 = self._feature_boxes_norm.get("edge_2")
        edge0 = self._feature_boxes_norm.get("edge_0")
        if edge1 is None or edge3 is None or edge2 is None or edge0 is None:
            h, w = self._mask_bgr.shape[:2]
            quad = np.array([[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]], dtype=np.float32)
            return self._order_points(quad)
        quad = np.array(
            [
                [float(edge1[0]), float(edge1[1])],
                [float(edge3[2]), float(edge3[1])],
                [float(edge2[2]), float(edge2[3])],
                [float(edge0[0]), float(edge0[3])],
            ],
            dtype=np.float32,
        )
        return self._order_points(quad)

    def _box_to_quad(self, box: List[int]) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box]
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    def _load_feature_quads(self, prefix: str):
        out = []
        prefix_norm = self._norm_name(prefix)
        for label, box in self._feature_boxes_norm.items():
            if label.startswith(prefix_norm):
                out.append((label, self._box_to_quad(box)))
        out.sort(key=lambda item: self._feature_sort_key(item[0]))
        return out

    def _feature_sort_key(self, label: str):
        parts = label.split("_")
        try:
            return int(parts[-1])
        except Exception:
            return 0

    def _image_cb(self, camera_name: str, msg: Image) -> None:
        with self._lock:
            self._latest_frames[camera_name] = msg

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
                annotated, detections, classes, mask_overlay, mask_binary, mask_status = self._run_inference(msg)
            except Exception as exc:
                self.get_logger().error(f"{camera_name} inference failed: {exc}")
                continue
            self._last_infer_time[camera_name] = now

            annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self._annotated_pubs[camera_name].publish(annotated_msg)

            json_msg = String()
            json_msg.data = json.dumps(detections, separators=(",", ":"))
            self._json_pubs[camera_name].publish(json_msg)

            classes_msg = String()
            classes_msg.data = ",".join(classes)
            self._classes_pubs[camera_name].publish(classes_msg)

            overlay_msg = self._bridge.cv2_to_imgmsg(mask_overlay, encoding="bgr8")
            overlay_msg.header = msg.header
            self._mask_overlay_pubs[camera_name].publish(overlay_msg)

            mask_msg = self._bridge.cv2_to_imgmsg(mask_binary, encoding="mono8")
            mask_msg.header = msg.header
            self._mask_binary_pubs[camera_name].publish(mask_msg)

            status_msg = String()
            status_msg.data = mask_status
            self._mask_status_pubs[camera_name].publish(status_msg)

    def _run_inference(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self._model.predict(
            source=frame,
            device=self._device,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()
        detections: List[Dict] = []
        classes: List[str] = []
        names = result.names if hasattr(result, "names") else self._model.names
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((0,), dtype=np.float32)
            clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((0,), dtype=int)
            for box, conf, cls_idx in zip(xyxy, confs, clss):
                cls_name = str(names[int(cls_idx)]) if names is not None else str(int(cls_idx))
                detections.append({
                    "class_id": int(cls_idx),
                    "class_name": cls_name,
                    "confidence": float(conf),
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                })
                classes.append(cls_name)
        mask_overlay, mask_binary, mask_status = self._fit_rails(frame, detections)
        return annotated, detections, classes, mask_overlay, mask_binary, mask_status

    def _fit_rails(self, frame: np.ndarray, detections: List[Dict]):
        h, w = frame.shape[:2]
        observed = self._detect_observed_fix_component(frame, detections)
        if observed is None:
            return frame.copy(), np.zeros((h, w), dtype=np.uint8), "missing_fix_hsv"

        best = self._estimate_fix_only_transform(frame, observed)
        if best is None:
            return frame.copy(), np.zeros((h, w), dtype=np.uint8), "rails_alignment_failed"

        state, M, rotation_name, score = best
        overlay_rgba, binary = self._render_transformed_rails(state, M, (h, w))
        overlay = self._compose_overlay(frame, overlay_rgba)

        cv2.polylines(overlay, [np.round(observed["quad"]).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
        proj_fix_quad = self._transform_quad_affine(M, state["fix_quad"])
        cv2.polylines(overlay, [np.round(proj_fix_quad).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)

        obs_center = self._mask_centroid(observed["mask"])
        if obs_center is None:
            obs_center = observed["quad"].mean(axis=0).astype(np.float32)
        cv2.circle(overlay, tuple(np.round(obs_center).astype(int)), 5, (255, 255, 0), -1, cv2.LINE_AA)

        proj_center = self._mask_centroid(cv2.warpAffine(state["fix_mask"], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        if proj_center is not None:
            cv2.circle(overlay, tuple(np.round(proj_center).astype(int)), 5, (0, 255, 0), -1, cv2.LINE_AA)

        for _, rail_quad in state["sc_rails"]:
            q = self._transform_quad_affine(M, rail_quad)
            cv2.polylines(overlay, [np.round(q).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
        for _, rail_quad in state["nic_rails"]:
            q = self._transform_quad_affine(M, rail_quad)
            cv2.polylines(overlay, [np.round(q).astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)

        return overlay, binary, f"rails_from_fix_only_{rotation_name}_{score:.4f}"
    def _render_transformed_rails(self, state, M: np.ndarray, shape_hw: Tuple[int, int]):
        h, w = shape_hw
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        binary = np.zeros((h, w), dtype=np.uint8)

        for _, rail_quad in state["sc_rails"]:
            q = self._transform_quad_affine(M, rail_quad)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 255)
            rgba[mask > 0, 0] = 255
            rgba[mask > 0, 2] = 255
            rgba[mask > 0, 3] = 200
            binary = cv2.bitwise_or(binary, mask)

        for _, rail_quad in state["nic_rails"]:
            q = self._transform_quad_affine(M, rail_quad)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 255)
            rgba[mask > 0, 1] = 255
            rgba[mask > 0, 2] = 255
            rgba[mask > 0, 3] = 200
            binary = cv2.bitwise_or(binary, mask)

        return rgba, binary
    def _estimate_fix_only_transform(self, frame: np.ndarray, observed):
        h, w = frame.shape[:2]
        observed_mask = observed["mask"]
        observed_quad = observed["quad"]
        observed_center = self._mask_centroid(observed_mask)
        if observed_center is None:
            observed_center = observed_quad.mean(axis=0).astype(np.float32)

        best = None
        best_score = -1e18

        for state in self._rotated_states:
            src_pts = []
            dst_pts = []
            self._append_point_pairs(src_pts, dst_pts, state["fix_quad"], observed_quad, repeat=6)
            src_pts.append(state["fix_center"].astype(np.float32))
            dst_pts.append(observed_center.astype(np.float32))

            src_pts_np = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
            dst_pts_np = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)

            M, _ = cv2.estimateAffinePartial2D(src_pts_np, dst_pts_np, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000, confidence=0.99, refineIters=50)
            if M is None:
                M, _ = cv2.estimateAffine2D(src_pts_np, dst_pts_np, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000, confidence=0.99, refineIters=50)
            if M is None:
                M = cv2.getAffineTransform(
                    np.asarray([state["fix_quad"][0], state["fix_quad"][1], state["fix_quad"][3]], dtype=np.float32),
                    np.asarray([observed_quad[0], observed_quad[1], observed_quad[3]], dtype=np.float32),
                )
            if M is None:
                continue
            M = M.astype(np.float32)

            warped_fix = cv2.warpAffine(state["fix_mask"], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            proj_center = self._mask_centroid(warped_fix)
            if proj_center is not None:
                delta = observed_center - proj_center
                M[:, 2] += delta.astype(np.float32)
                warped_fix = cv2.warpAffine(state["fix_mask"], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            fix_iou = self._binary_iou(warped_fix, observed_mask)
            proj_fix_quad = self._transform_quad_affine(M, state["fix_quad"])
            fix_quad_err = float(np.mean(np.linalg.norm(proj_fix_quad - observed_quad, axis=1)))
            proj_fix_center = self._mask_centroid(warped_fix)
            fix_center_err = float(np.linalg.norm(proj_fix_center - observed_center)) if proj_fix_center is not None else 1e6

            score = 20.0 * fix_iou - 0.06 * fix_quad_err - 0.04 * fix_center_err
            if score > best_score:
                best_score = score
                best = (state, M, state["rotation_name"], fix_iou)

        return best
    def _extract_board_edge_features(self, quad: np.ndarray, fix_center: np.ndarray):
        quad = self._order_points(quad)
        edges = [
            np.vstack([quad[0], quad[1]]).astype(np.float32),
            np.vstack([quad[1], quad[2]]).astype(np.float32),
            np.vstack([quad[2], quad[3]]).astype(np.float32),
            np.vstack([quad[3], quad[0]]).astype(np.float32),
        ]
        lengths = [self._segment_length(edge) for edge in edges]
        if 0.5 * (lengths[0] + lengths[2]) <= 0.5 * (lengths[1] + lengths[3]):
            short_ids = [0, 2]
            long_ids = [1, 3]
        else:
            short_ids = [1, 3]
            long_ids = [0, 2]

        short_mids = [self._edge_midpoint(edges[idx]) for idx in short_ids]
        long_mids = [self._edge_midpoint(edges[idx]) for idx in long_ids]
        d_short = [float(np.linalg.norm(mid - fix_center)) for mid in short_mids]
        d_long = [float(np.linalg.norm(mid - fix_center)) for mid in long_mids]

        bottom_short_id = short_ids[int(np.argmax(d_short))]
        near_short_id = short_ids[1 - int(np.argmax(d_short))]
        near_long_id = long_ids[int(np.argmin(d_long))]
        far_long_id = long_ids[1 - int(np.argmin(d_long))]

        long_axis_dir = self._normalize_vector(self._edge_midpoint(edges[bottom_short_id]) - self._edge_midpoint(edges[near_short_id]))
        width_axis_dir = self._normalize_vector(self._edge_midpoint(edges[far_long_id]) - self._edge_midpoint(edges[near_long_id]))

        bottom_edge = self._order_edge_by_axis(edges[bottom_short_id], width_axis_dir)
        long_edge_near_fix = self._order_edge_by_axis(edges[near_long_id], long_axis_dir)

        return {
            "quad": quad.astype(np.float32),
            "bottom_edge": bottom_edge.astype(np.float32),
            "long_edge_near_fix": long_edge_near_fix.astype(np.float32),
        }

    def _order_edge_by_axis(self, edge: np.ndarray, axis: np.ndarray):
        p0 = np.asarray(edge[0], dtype=np.float32)
        p1 = np.asarray(edge[1], dtype=np.float32)
        axis = self._normalize_vector(axis)
        if float(np.dot(p0, axis)) <= float(np.dot(p1, axis)):
            return np.vstack([p0, p1]).astype(np.float32)
        return np.vstack([p1, p0]).astype(np.float32)

    def _append_point_pairs(self, src_pts: List[np.ndarray], dst_pts: List[np.ndarray], src_arr: np.ndarray, dst_arr: np.ndarray, repeat: int = 1):
        src_arr = np.asarray(src_arr, dtype=np.float32).reshape(-1, 2)
        dst_arr = np.asarray(dst_arr, dtype=np.float32).reshape(-1, 2)
        for _ in range(int(max(1, repeat))):
            for s, d in zip(src_arr, dst_arr):
                src_pts.append(s.astype(np.float32))
                dst_pts.append(d.astype(np.float32))

    def _build_rotated_states(self):
        states = []
        h0, w0 = self._mask_bgr.shape[:2]
        for k in range(4):
            fix_mask = self._rotate_image90(self._mask_fix_component["mask"], k)
            fix_quad = self._component_quad_from_binary(fix_mask)
            contour = self._largest_contour(fix_mask)
            if fix_quad is None or contour is None:
                continue
            board_quad = self._rotate_points_image(self._base_mask_board_quad, w0, h0, k)
            board_quad = self._order_points(board_quad)
            fix_center = self._mask_centroid(fix_mask)
            if fix_center is None:
                fix_center = fix_quad.mean(axis=0).astype(np.float32)
            edge_features = self._extract_board_edge_features(board_quad, fix_center)
            sc_rails = [(label, self._order_points(self._rotate_points_image(quad, w0, h0, k))) for label, quad in self._base_sc_rail_quads]
            nic_rails = [(label, self._order_points(self._rotate_points_image(quad, w0, h0, k))) for label, quad in self._base_nic_rail_quads]
            states.append({
                "k": k,
                "rotation_name": f"r{k * 90}",
                "fix_mask": fix_mask,
                "fix_quad": fix_quad,
                "fix_contour": contour,
                "board_quad": board_quad,
                "fix_center": fix_center,
                "edge_features": edge_features,
                "sc_rails": sc_rails,
                "nic_rails": nic_rails,
            })
        return states

    def _find_best_detection(self, detections: List[Dict], allowed_names: set) -> Optional[Dict]:
        best = None
        best_conf = -1.0
        for det in detections:
            name = self._norm_name(det["class_name"])
            if name in allowed_names:
                conf = float(det["confidence"])
                if conf > best_conf:
                    best_conf = conf
                    best = det
        return best

    def _extract_mask_fix_component(self):
        full_mask = self._hsv_match_mask(self._mask_bgr)
        ref_contour = getattr(self, "_mask_fix_shape_contour", None)
        comp = self._select_best_component(self._mask_bgr, full_mask, self._mask_fix_hint, ref_contour, prefer_hint=True)
        if comp is not None:
            return comp
        return self._extract_component_from_threshold(self._mask_bgr, self._mask_fix_hint)

    def _detect_observed_fix_component(self, frame: np.ndarray, detections: List[Dict]):
        fix_det = self._find_best_detection(detections, self._fix_classes)
        hint_box = None
        if fix_det is not None:
            hint_box = self._clip_box(fix_det["bbox_xyxy"], frame.shape[1], frame.shape[0])

        masks = []
        if hint_box is not None:
            masks.append((self._hsv_match_mask(frame, hint_box), hint_box, True))
        masks.append((self._hsv_match_mask(frame, None), hint_box, False))

        best = None
        best_score = None
        for mask, hint, prefer_hint in masks:
            comp = self._select_best_component(frame, mask, hint, self._mask_fix_shape_contour, prefer_hint=prefer_hint)
            if comp is None:
                continue
            score = self._component_selection_score(comp, hint, prefer_hint, self._mask_fix_shape_contour)
            if best_score is None or score > best_score:
                best = comp
                best_score = score
        return best

    def _hsv_match_mask(self, image_bgr: np.ndarray, hint_box=None):
        h, w = image_bgr.shape[:2]
        if hint_box is None:
            x1, y1, x2, y2 = 0, 0, w - 1, h - 1
        else:
            x1, y1, x2, y2 = [int(v) for v in hint_box]
            pad_x = max(12, int(0.15 * max(1, x2 - x1)))
            pad_y = max(12, int(0.15 * max(1, y2 - y1)))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w - 1, x2 + pad_x)
            y2 = min(h - 1, y2 + pad_y)

        roi = image_bgr[y1:y2 + 1, x1:x2 + 1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_thr = cv2.inRange(
            hsv,
            np.array([self._pink_h1_min, self._pink_s_min, self._pink_v_min], dtype=np.uint8),
            np.array([self._pink_h1_max, 255, 255], dtype=np.uint8),
        ) > 0

        ref_h, ref_s, ref_v = [float(v) for v in self._mask_fix_mean_hsv]
        hue = hsv[:, :, 0].astype(np.float32)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        dh = np.minimum(np.abs(hue - ref_h), 180.0 - np.abs(hue - ref_h))
        ds = np.abs(sat - ref_s)
        dv = np.abs(val - ref_v)
        mask_sim = (dh <= float(self._fix_h_tol)) & (ds <= float(self._fix_s_tol)) & (dv <= float(self._fix_v_tol))
        mask = np.where(mask_sim & mask_thr, 255, 0).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        full = np.zeros((h, w), dtype=np.uint8)
        full[y1:y2 + 1, x1:x2 + 1] = mask
        return full

    def _extract_component_from_threshold(self, image_bgr: np.ndarray, hint_box=None):
        h, w = image_bgr.shape[:2]
        if hint_box is None:
            x1, y1, x2, y2 = 0, 0, w - 1, h - 1
        else:
            x1, y1, x2, y2 = [int(v) for v in hint_box]
            pad_x = max(12, int(0.15 * max(1, x2 - x1)))
            pad_y = max(12, int(0.15 * max(1, y2 - y1)))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w - 1, x2 + pad_x)
            y2 = min(h - 1, y2 + pad_y)
        roi = image_bgr[y1:y2 + 1, x1:x2 + 1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self._pink_h1_min, self._pink_s_min, self._pink_v_min], dtype=np.uint8),
            np.array([self._pink_h1_max, 255, 255], dtype=np.uint8),
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        full = np.zeros((h, w), dtype=np.uint8)
        full[y1:y2 + 1, x1:x2 + 1] = mask
        return self._select_best_component(image_bgr, full, hint_box, None, prefer_hint=hint_box is not None)

    def _select_best_component(self, image_bgr: np.ndarray, binary_mask: np.ndarray, hint_box, ref_contour, prefer_hint: bool):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            return None
        best = None
        best_score = None
        for label_id in range(1, num_labels):
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            if area < 20.0:
                continue
            comp = np.where(labels == label_id, 255, 0).astype(np.uint8)
            contour = self._largest_contour(comp)
            if contour is None:
                continue
            quad = self._component_quad_from_binary(comp)
            if quad is None:
                continue
            mean_hsv = self._mean_hsv_on_mask(image_bgr, comp)
            bbox = self._quad_to_bbox(quad)
            comp_info = {
                "mask": comp,
                "contour": contour,
                "quad": quad,
                "bbox": bbox,
                "mean_hsv": mean_hsv,
                "area": area,
            }
            score = self._component_selection_score(comp_info, hint_box, prefer_hint, ref_contour)
            if best_score is None or score > best_score:
                best = comp_info
                best_score = score
        return best

    def _component_selection_score(self, comp_info, hint_box, prefer_hint: bool, ref_contour=None):
        area = float(comp_info["area"])
        bbox = comp_info["bbox"]
        mean_hsv = comp_info["mean_hsv"]
        shape_bonus = 0.0
        if ref_contour is not None:
            try:
                shape_bonus = -float(cv2.matchShapes(ref_contour, comp_info["contour"], cv2.CONTOURS_MATCH_I1, 0.0))
            except Exception:
                shape_bonus = 0.0
        hsv_dist = self._hsv_distance(mean_hsv, self._mask_fix_mean_hsv)
        hint_iou = self._bbox_iou_xyxy(bbox, hint_box) if hint_box is not None else 0.0
        hint_cover = self._bbox_intersection_fraction(bbox, hint_box) if hint_box is not None else 0.0
        score = 0.001 * area - 0.015 * hsv_dist + 4.0 * shape_bonus
        if prefer_hint and hint_box is not None:
            score += 3.0 * hint_cover + 2.0 * hint_iou
        return score

    def _rotate_image90(self, image: np.ndarray, k: int):
        k = int(k) % 4
        if k == 0:
            return image.copy()
        if k == 1:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if k == 2:
            return cv2.rotate(image, cv2.ROTATE_180)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _rotate_points_image(self, pts: np.ndarray, width: int, height: int, k: int):
        k = int(k) % 4
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        out = np.zeros_like(arr)
        if k == 0:
            out = arr.copy()
        elif k == 1:
            out[:, 0] = float(height - 1) - arr[:, 1]
            out[:, 1] = arr[:, 0]
        elif k == 2:
            out[:, 0] = float(width - 1) - arr[:, 0]
            out[:, 1] = float(height - 1) - arr[:, 1]
        else:
            out[:, 0] = arr[:, 1]
            out[:, 1] = float(width - 1) - arr[:, 0]
        return out.astype(np.float32)

    def _component_quad_from_binary(self, comp: np.ndarray):
        contour = self._largest_contour(comp)
        if contour is None:
            return None
        rect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rect).astype(np.float32)
        return self._order_points(pts)

    def _largest_contour(self, comp: np.ndarray):
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _mean_hsv_on_mask(self, image_bgr: np.ndarray, mask: np.ndarray):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vals = hsv[ys, xs].astype(np.float32)
        hue = vals[:, 0]
        angles = hue * (2.0 * np.pi / 180.0)
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        if mean_angle < 0.0:
            mean_angle += 2.0 * np.pi
        mean_h = mean_angle * (180.0 / (2.0 * np.pi))
        mean_s = float(np.mean(vals[:, 1]))
        mean_v = float(np.mean(vals[:, 2]))
        return np.array([mean_h, mean_s, mean_v], dtype=np.float32)

    def _hsv_distance(self, a: np.ndarray, b: np.ndarray):
        ah, as_, av = [float(v) for v in a]
        bh, bs, bv = [float(v) for v in b]
        dh = min(abs(ah - bh), 180.0 - abs(ah - bh))
        ds = abs(as_ - bs)
        dv = abs(av - bv)
        return float(np.sqrt(dh * dh + 0.01 * ds * ds + 0.01 * dv * dv))

    def _detect_taskboard_plane_quad(self, frame: np.ndarray, board_box: List[int]) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self._expand_box(board_box, w, h, 0.08, 0.08)
        roi = frame[ry1:ry2 + 1, rx1:rx2 + 1]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        masks = []
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(otsu_mask)
        percentile_value = int(np.percentile(gray, 38))
        _, percentile_mask = cv2.threshold(gray, percentile_value, 255, cv2.THRESH_BINARY_INV)
        masks.append(percentile_mask)

        roi_area = float(roi.shape[0] * roi.shape[1])
        target_center = np.array([(board_box[0] + board_box[2]) * 0.5 - rx1, (board_box[1] + board_box[3]) * 0.5 - ry1], dtype=np.float32)
        target_ratio = float(self._mask_bgr.shape[1]) / float(max(1, self._mask_bgr.shape[0]))
        best_quad = None
        best_score = None

        for mask in masks:
            work = mask.copy()
            k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k_close, iterations=2)
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, k_open, iterations=1)
            contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < 0.22 * roi_area:
                    continue
                hull = cv2.convexHull(cnt)
                peri = float(cv2.arcLength(hull, True))
                approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                if len(approx) == 4:
                    quad = approx.reshape(4, 2).astype(np.float32)
                    rect = cv2.minAreaRect(hull)
                    rect_area = float(max(1.0, rect[1][0] * rect[1][1]))
                else:
                    rect = cv2.minAreaRect(hull)
                    rect_area = float(max(1.0, rect[1][0] * rect[1][1]))
                    quad = cv2.boxPoints(rect).astype(np.float32)
                quad = self._order_points(quad)
                quad_center = quad.mean(axis=0)
                center_penalty = float(np.linalg.norm((quad_center - target_center) / np.array([max(1.0, roi.shape[1]), max(1.0, roi.shape[0])], dtype=np.float32)))
                top_w = float(np.linalg.norm(quad[1] - quad[0]))
                bot_w = float(np.linalg.norm(quad[2] - quad[3]))
                left_h = float(np.linalg.norm(quad[3] - quad[0]))
                right_h = float(np.linalg.norm(quad[2] - quad[1]))
                quad_w = max(1.0, 0.5 * (top_w + bot_w))
                quad_h = max(1.0, 0.5 * (left_h + right_h))
                quad_ratio = quad_w / quad_h
                ratio_penalty = abs(np.log(max(1e-6, quad_ratio) / max(1e-6, target_ratio)))
                fill_ratio = area / rect_area
                area_ratio = rect_area / roi_area
                cand_box = [float(quad[:, 0].min() + rx1), float(quad[:, 1].min() + ry1), float(quad[:, 0].max() + rx1), float(quad[:, 1].max() + ry1)]
                iou = self._bbox_iou_xyxy(cand_box, board_box)
                score = 1.8 * iou + 0.45 * area_ratio + 0.35 * fill_ratio - 0.55 * center_penalty - 0.30 * ratio_penalty
                if best_score is None or score > best_score:
                    best_score = score
                    best_quad = quad.copy()
        if best_quad is None:
            return None
        best_quad[:, 0] += rx1
        best_quad[:, 1] += ry1
        return best_quad.astype(np.float32)

    def _expand_box(self, box: List[int], width: int, height: int, frac_x: float, frac_y: float) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(v) for v in box]
        pad_x = max(4, int((x2 - x1) * frac_x))
        pad_y = max(4, int((y2 - y1) * frac_y))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width - 1, x2 + pad_x)
        y2 = min(height - 1, y2 + pad_y)
        return x1, y1, x2, y2

    def _compose_overlay(self, frame: np.ndarray, mask_rgba: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        mask_bgr = mask_rgba[:, :, :3]
        alpha = (mask_rgba[:, :, 3].astype(np.float32) / 255.0) * self._mask_alpha
        alpha_3 = np.dstack([alpha, alpha, alpha])
        overlay = (overlay.astype(np.float32) * (1.0 - alpha_3) + mask_bgr.astype(np.float32) * alpha_3).astype(np.uint8)
        return overlay

    def _binary_iou(self, a: np.ndarray, b: np.ndarray):
        aa = a > 0
        bb = b > 0
        inter = np.logical_and(aa, bb).sum()
        union = np.logical_or(aa, bb).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

    def _mask_centroid(self, binary_mask: np.ndarray):
        ys, xs = np.where(binary_mask > 0)
        if xs.size == 0:
            return None
        return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)

    def _transform_quad_affine(self, M: np.ndarray, quad: np.ndarray) -> np.ndarray:
        arr = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
        out = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1) @ M.T
        return self._order_points(out.astype(np.float32))

    def _segment_length(self, edge: np.ndarray):
        edge = np.asarray(edge, dtype=np.float32).reshape(2, 2)
        return float(np.linalg.norm(edge[1] - edge[0]))

    def _edge_midpoint(self, edge: np.ndarray):
        edge = np.asarray(edge, dtype=np.float32).reshape(2, 2)
        return 0.5 * (edge[0] + edge[1])

    def _edge_distance(self, a: np.ndarray, b: np.ndarray):
        a = np.asarray(a, dtype=np.float32).reshape(2, 2)
        b = np.asarray(b, dtype=np.float32).reshape(2, 2)
        return 0.5 * (float(np.linalg.norm(a[0] - b[0])) + float(np.linalg.norm(a[1] - b[1])))

    def _normalize_vector(self, v: np.ndarray):
        v = np.asarray(v, dtype=np.float32).reshape(2)
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _clip_box(self, box, width: int, height: int):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _quad_to_bbox(self, quad: np.ndarray):
        quad = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
        return [float(np.min(quad[:, 0])), float(np.min(quad[:, 1])), float(np.max(quad[:, 0])), float(np.max(quad[:, 1]))]

    def _bbox_iou_xyxy(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
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
        union = max(1e-6, area_a + area_b - inter)
        return float(inter / union)

    def _bbox_intersection_fraction(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        return float(inter / area_a)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(d)]
        ordered[3] = pts[np.argmax(d)]
        return ordered


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