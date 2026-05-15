#!/usr/bin/env python3

import sys
import tty
import termios
import select
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ThreeCameraDatasetCollector(Node):
    def __init__(self):
        super().__init__("three_camera_dataset_collector")

        self.dataset_dir = Path("/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/Data_Collection/dataset")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.left_msg = None
        self.center_msg = None
        self.right_msg = None
        self.stop_event = threading.Event()

        self.create_subscription(Image, "/left_camera/image", self.left_callback, 10)
        self.create_subscription(Image, "/center_camera/image", self.center_callback, 10)
        self.create_subscription(Image, "/right_camera/image", self.right_callback, 10)

        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info("Listening to /left_camera/image, /center_camera/image, /right_camera/image")
        self.get_logger().info("Press SPACE to save images")
        self.get_logger().info("Press q to quit")

    def left_callback(self, msg):
        with self.lock:
            self.left_msg = msg

    def center_callback(self, msg):
        with self.lock:
            self.center_msg = msg

    def right_callback(self, msg):
        with self.lock:
            self.right_msg = msg

    def image_msg_to_cv2(self, msg):
        encoding = msg.encoding.lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)

        if encoding == "bgr8":
            return raw[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()

        if encoding == "rgb8":
            img = raw[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if encoding == "bgra8":
            img = raw[:, : msg.width * 4].reshape(msg.height, msg.width, 4).copy()
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if encoding == "rgba8":
            img = raw[:, : msg.width * 4].reshape(msg.height, msg.width, 4).copy()
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if encoding == "mono8":
            return raw[:, : msg.width].reshape(msg.height, msg.width).copy()

        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    def save_images(self):
        with self.lock:
            left_msg = self.left_msg
            center_msg = self.center_msg
            right_msg = self.right_msg

        if left_msg is None or center_msg is None or right_msg is None:
            self.get_logger().warn("Images from all three cameras are not available yet")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_dir = self.dataset_dir / timestamp
        sample_dir.mkdir(parents=True, exist_ok=True)

        left_img = self.image_msg_to_cv2(left_msg)
        center_img = self.image_msg_to_cv2(center_msg)
        right_img = self.image_msg_to_cv2(right_msg)

        cv2.imwrite(str(sample_dir / "left.png"), left_img)
        cv2.imwrite(str(sample_dir / "center.png"), center_img)
        cv2.imwrite(str(sample_dir / "right.png"), right_img)

        self.get_logger().info(f"Saved images to {sample_dir}")

    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while not self.stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue

                key = sys.stdin.read(1)

                if key == " ":
                    self.save_images()
                elif key.lower() == "q":
                    self.stop_event.set()
                    if rclpy.ok():
                        rclpy.shutdown()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    rclpy.init()
    node = ThreeCameraDatasetCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()