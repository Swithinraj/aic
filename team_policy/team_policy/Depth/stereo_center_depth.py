from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import message_filters
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class StereoCenterDepthResult:
    camera_name: str
    stamp: object
    frame_id: str
    depth_meters: np.ndarray
    overlay_bgr: np.ndarray
    depth_vis_bgr: np.ndarray
    disparity_vis_bgr: np.ndarray
    rectified_pair_debug_bgr: np.ndarray
    center_camera_info: CameraInfo
    depth_msg: Image
    depth_vis_msg: Image
    overlay_msg: Image
    disparity_vis_msg: Image
    rectified_pair_debug_msg: Image
    num_matches: int
    num_inliers: int
    coverage: float
    baseline_m: float
    sources_used: str


class StereoCenterDepth:
    def __init__(
        self,
        node: Node,
        left_image_topic: str = "/left_camera/image",
        left_info_topic: str = "/left_camera/camera_info",
        center_image_topic: str = "/center_camera/image",
        center_info_topic: str = "/center_camera/camera_info",
        right_image_topic: str = "/right_camera/image",
        right_info_topic: str = "/right_camera/camera_info",
        queue_size: int = 10,
        sync_slop: float = 0.05,
        publish_outputs: bool = False,
    ):
        self._node = node
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._publish_outputs = publish_outputs
        self._latest_results: Dict[str, Optional[StereoCenterDepthResult]] = {
            "left": None,
            "center": None,
            "right": None,
        }
        self._pending: Dict[str, Optional[tuple[Image, CameraInfo, float]]] = {
            "left": None,
            "center": None,
            "right": None,
        }
        self._last_processed_at = {"left": 0.0, "center": 0.0, "right": 0.0}
        self._last_total_inference_time = 0.0
        self._last_status_log_time = 0.0
        self._worker_stop = False
        self._worker_cond = threading.Condition()
        self._round_robin = ["center", "left", "right"]
        self._round_robin_index = 0

        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._PILImage = None
        self._model_loaded = False
        self._model_load_failed = False

        self._max_side_cpu = int(os.environ.get("DEPTH_ANYTHING_V2_MAX_SIDE_CPU", "448"))
        self._max_side_gpu = int(os.environ.get("DEPTH_ANYTHING_V2_MAX_SIDE_GPU", "700"))
        self._max_hz_per_camera = float(os.environ.get("DEPTH_ANYTHING_V2_MAX_HZ_PER_CAMERA", "5.0"))
        self._max_total_hz = float(os.environ.get("DEPTH_ANYTHING_V2_MAX_TOTAL_HZ", "5.0"))
        self._use_fp16 = os.environ.get("DEPTH_ANYTHING_V2_USE_FP16", "1") != "0"
        self._gpu_min_vram_gb = float(os.environ.get("DEPTH_ANYTHING_V2_GPU_MIN_VRAM_GB", "6.0"))
        self._device_pref = os.environ.get("DEPTH_ANYTHING_V2_DEVICE", "auto").strip().lower()
        self._model_id_or_path = os.environ.get(
            "DEPTH_ANYTHING_V2_MODEL_ID_OR_PATH",
            "depth-anything/Depth-Anything-V2-Small-hf",
        )
        self._local_only = os.environ.get("DEPTH_ANYTHING_V2_LOCAL_ONLY", "0") == "1"

        self._sync_left = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(node, Image, left_image_topic),
                message_filters.Subscriber(node, CameraInfo, left_info_topic),
            ],
            queue_size,
            sync_slop,
        )
        self._sync_center = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(node, Image, center_image_topic),
                message_filters.Subscriber(node, CameraInfo, center_info_topic),
            ],
            queue_size,
            sync_slop,
        )
        self._sync_right = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(node, Image, right_image_topic),
                message_filters.Subscriber(node, CameraInfo, right_info_topic),
            ],
            queue_size,
            sync_slop,
        )
        self._sync_left.registerCallback(lambda image_msg, info_msg: self._sync_callback("left", image_msg, info_msg))
        self._sync_center.registerCallback(lambda image_msg, info_msg: self._sync_callback("center", image_msg, info_msg))
        self._sync_right.registerCallback(lambda image_msg, info_msg: self._sync_callback("right", image_msg, info_msg))

        self._pubs = {}
        if publish_outputs:
            for camera_name in ("left", "center", "right"):
                ns = f"/{camera_name}_camera/stereo_depth"
                self._pubs[camera_name] = {
                    "depth": node.create_publisher(Image, f"{ns}/image", 10),
                    "depth_vis": node.create_publisher(Image, f"{ns}/vis", 10),
                    "overlay": node.create_publisher(Image, f"{ns}/overlay", 10),
                    "camera_info": node.create_publisher(CameraInfo, f"{ns}/camera_info", 10),
                }

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def destroy(self):
        with self._worker_cond:
            self._worker_stop = True
            self._worker_cond.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def has_result(self, camera_name: str = "center") -> bool:
        with self._lock:
            return self._latest_results.get(camera_name) is not None

    def get_latest_result(self, camera_name: str = "center"):
        with self._lock:
            return self._latest_results.get(camera_name)

    def get_latest_depth_array(self, camera_name: str = "center"):
        with self._lock:
            result = self._latest_results.get(camera_name)
            return None if result is None else result.depth_meters.copy()

    def get_latest_overlay_array(self, camera_name: str = "center"):
        with self._lock:
            result = self._latest_results.get(camera_name)
            return None if result is None else result.overlay_bgr.copy()

    def is_calibrating(self) -> bool:
        return False

    def start_calibration(self, duration_s: Optional[float] = None, interval_s: Optional[float] = None, move_distance_m: Optional[float] = None):
        return None

    def stop_calibration(self):
        return None

    def _sync_callback(self, camera_name: str, image_msg: Image, info_msg: CameraInfo):
        with self._worker_cond:
            self._pending[camera_name] = (image_msg, info_msg, time.monotonic())
            self._worker_cond.notify_all()

    def _worker_loop(self):
        while True:
            with self._worker_cond:
                if self._worker_stop:
                    return
                ready = self._select_ready_camera_locked()
                if ready is None:
                    self._worker_cond.wait(timeout=0.1)
                    continue
                camera_name = ready
                image_msg, info_msg, _ = self._pending[camera_name]
                self._pending[camera_name] = None

            try:
                self._ensure_model_loaded()
                result = self._compute(camera_name, image_msg, info_msg)
            except Exception as exc:
                self._log_throttled(f"Depth Anything V2 multi-camera depth failed for {camera_name}: {exc}")
                continue

            with self._lock:
                self._latest_results[camera_name] = result
                self._last_processed_at[camera_name] = time.monotonic()
                self._last_total_inference_time = self._last_processed_at[camera_name]

            if self._publish_outputs:
                pubs = self._pubs[camera_name]
                pubs["depth"].publish(result.depth_msg)
                pubs["depth_vis"].publish(result.depth_vis_msg)
                pubs["overlay"].publish(result.overlay_msg)
                depth_info = CameraInfo()
                depth_info.header = info_msg.header
                depth_info.height = info_msg.height
                depth_info.width = info_msg.width
                depth_info.distortion_model = info_msg.distortion_model
                depth_info.d = list(info_msg.d)
                depth_info.k = list(info_msg.k)
                depth_info.r = list(info_msg.r)
                depth_info.p = list(info_msg.p)
                pubs["camera_info"].publish(depth_info)

    def _select_ready_camera_locked(self):
        now = time.monotonic()
        min_total_dt = 0.0 if self._max_total_hz <= 0.0 else 1.0 / self._max_total_hz
        if now - self._last_total_inference_time < min_total_dt:
            return None

        min_camera_dt = 0.0 if self._max_hz_per_camera <= 0.0 else 1.0 / self._max_hz_per_camera
        for offset in range(len(self._round_robin)):
            idx = (self._round_robin_index + offset) % len(self._round_robin)
            camera_name = self._round_robin[idx]
            pending = self._pending.get(camera_name)
            if pending is None:
                continue
            if now - self._last_processed_at[camera_name] < min_camera_dt:
                continue
            self._round_robin_index = (idx + 1) % len(self._round_robin)
            return camera_name
        return None

    def _compute(self, camera_name: str, image_msg: Image, info_msg: CameraInfo) -> StereoCenterDepthResult:
        if not self._model_loaded or self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError("Depth Anything V2 model is not loaded")

        bgr = self._to_bgr(image_msg)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = self._resize_for_inference(rgb)
        pil_image = self._pil_from_rgb(resized_rgb)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        if self._device.type == "cuda" and self._dtype is not None:
            for key, value in list(inputs.items()):
                if hasattr(value, "dtype") and getattr(value.dtype, "is_floating_point", False):
                    inputs[key] = value.to(self._dtype)

        try:
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
                predicted_depth = outputs.predicted_depth
                prediction = self._torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=bgr.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1).squeeze(0)
        except RuntimeError as exc:
            if self._device is not None and self._device.type == "cuda" and "out of memory" in str(exc).lower():
                self._node.get_logger().warn("CUDA OOM during Depth Anything V2 inference; switching to CPU")
                self._switch_model_to_cpu()
                return self._compute(camera_name, image_msg, info_msg)
            raise

        depth = prediction.detach().float().cpu().numpy().astype(np.float32)
        depth = np.maximum(depth, 0.0)
        positive = np.isfinite(depth) & (depth > 0.0)
        if not np.any(positive):
            raise RuntimeError("Model returned no positive depth values")

        depth_vis = self._make_depth_vis(depth)
        overlay = bgr.copy()
        blended = cv2.addWeighted(bgr, 0.65, depth_vis, 0.35, 0.0)
        overlay[positive] = blended[positive]
        support_vis = depth_vis.copy()
        debug_vis = bgr.copy()

        depth_msg = self._bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header = image_msg.header
        depth_vis_msg = self._bridge.cv2_to_imgmsg(depth_vis, encoding="bgr8")
        depth_vis_msg.header = image_msg.header
        overlay_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = image_msg.header
        disparity_vis_msg = self._bridge.cv2_to_imgmsg(support_vis, encoding="bgr8")
        disparity_vis_msg.header = image_msg.header
        rectified_pair_debug_msg = self._bridge.cv2_to_imgmsg(debug_vis, encoding="bgr8")
        rectified_pair_debug_msg.header = image_msg.header

        coverage = float(np.count_nonzero(positive)) / float(positive.size)

        return StereoCenterDepthResult(
            camera_name=camera_name,
            stamp=image_msg.header.stamp,
            frame_id=info_msg.header.frame_id if info_msg.header.frame_id else image_msg.header.frame_id,
            depth_meters=depth,
            overlay_bgr=overlay,
            depth_vis_bgr=depth_vis,
            disparity_vis_bgr=support_vis,
            rectified_pair_debug_bgr=debug_vis,
            center_camera_info=info_msg,
            depth_msg=depth_msg,
            depth_vis_msg=depth_vis_msg,
            overlay_msg=overlay_msg,
            disparity_vis_msg=disparity_vis_msg,
            rectified_pair_debug_msg=rectified_pair_debug_msg,
            num_matches=0,
            num_inliers=0,
            coverage=coverage,
            baseline_m=0.0,
            sources_used=f"depth-anything-v2-small-{camera_name}",
        )

    def _ensure_model_loaded(self):
        if self._model_loaded:
            return
        if self._model_load_failed:
            raise RuntimeError("Depth Anything V2 model previously failed to load")

        try:
            import torch
            from PIL import Image as PILImage
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            self._model_load_failed = True
            raise RuntimeError("Missing dependencies. Install transformers, pillow, and safetensors inside pixi before running.") from exc

        self._torch = torch
        self._PILImage = PILImage
        self._device = self._select_device(torch)
        self._dtype = torch.float16 if self._device.type == "cuda" and self._use_fp16 else torch.float32

        kwargs = {"local_files_only": self._local_only}
        self._processor = AutoImageProcessor.from_pretrained(self._model_id_or_path, **kwargs)
        self._model = AutoModelForDepthEstimation.from_pretrained(self._model_id_or_path, **kwargs)
        self._model.to(self._device)
        if self._device.type == "cuda" and self._dtype == torch.float16:
            self._model.to(dtype=torch.float16)
        self._model.eval()
        self._model_loaded = True
        self._node.get_logger().info(
            f"Depth Anything V2 Small loaded from {self._model_id_or_path} on {self._device} dtype={self._dtype}"
        )

    def _switch_model_to_cpu(self):
        if self._torch is None or self._model is None:
            return
        try:
            self._model.to(self._torch.device("cpu"), dtype=self._torch.float32)
        except Exception:
            self._model.to(self._torch.device("cpu"))
        self._device = self._torch.device("cpu")
        self._dtype = self._torch.float32
        if hasattr(self._torch.cuda, "empty_cache"):
            self._torch.cuda.empty_cache()

    def _select_device(self, torch):
        def _cuda_supported() -> bool:
            """Return True only if the current GPU's compute capability is
            actually in the list of capabilities this PyTorch build supports."""
            if not torch.cuda.is_available():
                return False
            major, minor = torch.cuda.get_device_capability(0)
            gpu_sm = f"sm_{major}{minor}"
            arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else [gpu_sm]
            return gpu_sm in arch_list

        cuda_ok = _cuda_supported()
        pref = self._device_pref

        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            if not cuda_ok:
                raise RuntimeError("DEPTH_ANYTHING_V2_DEVICE=cuda but CUDA compute capability is not supported or CUDA is not available")
            return torch.device("cuda")

        if cuda_ok:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            total_gb = float(total_bytes) / float(1024 ** 3)
            if total_gb >= self._gpu_min_vram_gb:
                return torch.device("cuda")
            self._node.get_logger().warn(
                f"CUDA VRAM {total_gb:.2f} GB is below DEPTH_ANYTHING_V2_GPU_MIN_VRAM_GB={self._gpu_min_vram_gb:.2f}; using CPU"
            )
        elif torch.cuda.is_available() and not cuda_ok:
            self._node.get_logger().warn(
                "GPU detected but its compute capability is not supported by this PyTorch build. "
                "Falling back to CPU."
            )
        return torch.device("cpu")

    def _resize_for_inference(self, rgb: np.ndarray) -> np.ndarray:
        max_side = self._max_side_gpu if self._device is not None and self._device.type == "cuda" else self._max_side_cpu
        h, w = rgb.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale >= 1.0:
            return rgb
        new_w = max(32, int(round(w * scale / 14.0) * 14))
        new_h = max(32, int(round(h * scale / 14.0) * 14))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _pil_from_rgb(self, rgb: np.ndarray):
        return self._PILImage.fromarray(rgb)

    def _to_bgr(self, image_msg: Image) -> np.ndarray:
        image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        if image.ndim == 2:
            image = image if image.dtype == np.uint8 else self._normalize_to_u8(image)
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype != np.uint8:
            image = self._normalize_to_u8(image)
        return image

    def _normalize_to_u8(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        mn = float(np.min(image))
        mx = float(np.max(image))
        if mx - mn < 1e-6:
            return np.zeros(image.shape[:2], dtype=np.uint8) if image.ndim == 2 else np.zeros(image.shape, dtype=np.uint8)
        return ((image - mn) / (mx - mn) * 255.0).astype(np.uint8)

    def _make_depth_vis(self, depth: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 0.0)
        vis = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            inv = 1.0 / np.clip(depth[valid], 1e-6, None)
            lo = float(np.percentile(inv, 2.0))
            hi = float(np.percentile(inv, 98.0))
            if hi - lo < 1e-6:
                vis[valid] = 255
            else:
                vis_vals = np.clip((inv - lo) / (hi - lo), 0.0, 1.0)
                vis[valid] = (vis_vals * 255.0).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)

    def _log_throttled(self, message: str, period_s: float = 2.0):
        now = time.monotonic()
        if now - self._last_status_log_time >= period_s:
            self._node.get_logger().error(message)
            self._last_status_log_time = now
