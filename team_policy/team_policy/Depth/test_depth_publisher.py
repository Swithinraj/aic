import numpy as np
import rclpy
from rclpy.node import Node

from .stereo_center_depth import StereoCenterDepth


class StereoCenterDepthPublisher(Node):
    def __init__(self):
        super().__init__("stereo_center_depth_publisher")
        self._depth = StereoCenterDepth(self, publish_outputs=True)
        self._status_timer = self.create_timer(2.0, self._tick)

        self.get_logger().info("Depth Anything V2 Small three-camera depth publisher started")
        self.get_logger().info("Inputs:")
        self.get_logger().info("  /left_camera/image")
        self.get_logger().info("  /left_camera/camera_info")
        self.get_logger().info("  /center_camera/image")
        self.get_logger().info("  /center_camera/camera_info")
        self.get_logger().info("  /right_camera/image")
        self.get_logger().info("  /right_camera/camera_info")
        self.get_logger().info("Publishers:")
        for camera_name in ("left", "center", "right"):
            self.get_logger().info(f"  /{camera_name}_camera/stereo_depth/image")
            self.get_logger().info(f"  /{camera_name}_camera/stereo_depth/vis")
            self.get_logger().info(f"  /{camera_name}_camera/stereo_depth/overlay")
            self.get_logger().info(f"  /{camera_name}_camera/stereo_depth/camera_info")
        self.get_logger().info("Rate limits:")
        self.get_logger().info("  DEPTH_ANYTHING_V2_MAX_HZ_PER_CAMERA default = 5.0")
        self.get_logger().info("  DEPTH_ANYTHING_V2_MAX_TOTAL_HZ default = 5.0")

    def _tick(self):
        missing = []
        for camera_name in ("left", "center", "right"):
            result = self._depth.get_latest_result(camera_name)
            if result is None:
                missing.append(camera_name)
                continue
            valid = np.isfinite(result.depth_meters) & (result.depth_meters > 0.0)
            if valid.any():
                depth_min = float(result.depth_meters[valid].min())
                depth_max = float(result.depth_meters[valid].max())
                self.get_logger().info(
                    f"camera={camera_name} source={result.sources_used} coverage={result.coverage:.4f} depth_min={depth_min:.4f} depth_max={depth_max:.4f}"
                )
            else:
                self.get_logger().warn(f"camera={camera_name} has no valid depth pixels")
        if missing:
            self.get_logger().warn("Waiting for model inference on: " + ", ".join(missing))

    def destroy_node(self):
        self._depth.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCenterDepthPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
