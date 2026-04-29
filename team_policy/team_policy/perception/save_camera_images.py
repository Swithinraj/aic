#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import sys
import tty
import termios
from threading import Thread
import time

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.images = {'left': None, 'center': None, 'right': None}
        
        # Subscribe to the three camera topics
        self.create_subscription(Image, '/left_camera/image', lambda msg: self.image_cb('left', msg), 10)
        self.create_subscription(Image, '/center_camera/image', lambda msg: self.image_cb('center', msg), 10)
        self.create_subscription(Image, '/right_camera/image', lambda msg: self.image_cb('right', msg), 10)
        
        # Save directory is the same folder as this script
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        self.get_logger().info(f"Initialized. Images will be saved to: {self.save_dir}")
        self.get_logger().info("Press SPACE BAR to save images, or 'q' to quit.")

    def image_cb(self, camera_name, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.images[camera_name] = cv_img
        except Exception as e:
            self.get_logger().error(f"Error converting {camera_name} image: {e}")

    def save_images(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_count = 0
        for camera_name, img in self.images.items():
            if img is not None:
                filename = os.path.join(self.save_dir, f"{camera_name}_{timestamp}.png")
                cv2.imwrite(filename, img)
                saved_count += 1
                self.get_logger().info(f"Saved: {filename}")
            else:
                self.get_logger().warn(f"No image received yet for {camera_name}")
        
        if saved_count > 0:
            self.get_logger().info(f"Successfully saved {saved_count} images.")
        else:
            self.get_logger().warn("Failed to save any images. Make sure the cameras are publishing.")

def getch():
    """Reads a single character from the standard input without echoing."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()

    # Run ROS spin in a separate thread so we can capture keyboard input
    spin_thread = Thread(target=rclpy.spin, args=(image_saver,))
    spin_thread.daemon = True
    spin_thread.start()

    try:
        while rclpy.ok():
            ch = getch()
            if ch == ' ':
                image_saver.save_images()
            elif ch.lower() == 'q' or ch == '\x03':  # 'q' or Ctrl+C
                break
    except KeyboardInterrupt:
        pass
    finally:
        image_saver.get_logger().info("Shutting down...")
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
