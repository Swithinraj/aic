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

        default_model_path = Path(__file__).resolve().parents[1] / "models" / "yolov12.pt"
        self._model_path = os.environ.get("YOLOV12_MODEL_PATH", str(default_model_path))
        self._device_request = os.environ.get("YOLOV12_DEVICE", "auto").strip().lower()
        self._device = self._resolve_device(self._device_request)
        self._conf = float(os.environ.get("YOLOV12_CONF", "0.25"))
        self._iou = float(os.environ.get("YOLOV12_IOU", "0.45"))
        self._imgsz = int(os.environ.get("YOLOV12_IMGSZ", "640"))
        self._max_hz = max(0.1, float(os.environ.get("YOLOV12_MAX_HZ", "5.0")))
        self._min_period = 1.0 / self._max_hz

        self._last_infer_time = {"left": 0.0, "center": 0.0, "right": 0.0}
        self._latest_frames: Dict[str, Optional[Image]] = {"left": None, "center": None, "right": None}

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

        self._timer = self.create_timer(0.02, self._tick)

        self.get_logger().info("YOLOv12 multi-camera detector started")
        self.get_logger().info(f"Model: {self._model_path}")
        self.get_logger().info(f"Device request: {self._device_request}")
        self.get_logger().info(f"Resolved device: {self._device}")
        self.get_logger().info(f"Inference rate limit per camera: {self._max_hz:.2f} Hz")

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
        if requested == "cpu":
            return "cpu"
        return "cpu"

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
                annotated, detections, classes = self._run_inference(msg)
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

    def _run_inference(self, msg: Image) -> Tuple[np.ndarray, List[Dict], List[str]]:
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
                detections.append(
                    {
                        "class_id": int(cls_idx),
                        "class_name": cls_name,
                        "confidence": float(conf),
                        "bbox_xyxy": [float(v) for v in box.tolist()],
                    }
                )
                classes.append(cls_name)

        return annotated, detections, classes


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