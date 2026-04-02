from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass
class RoiConfig:
    x_min_frac: float
    x_max_frac: float
    y_min_frac: float
    y_max_frac: float


@dataclass
class CameraAxisResult:
    valid: bool
    camera_name: str
    angle_deg: float = 0.0
    confidence: float = 0.0
    gripper_center_px: Tuple[float, float] = (0.0, 0.0)
    plug_center_px: Tuple[float, float] = (0.0, 0.0)
    num_plug_points: int = 0
    gripper_axis_name: str = ""
    debug_reason: str = ""


class DepthPlugGripperAngleEstimator:
    def __init__(self, node: Node):
        self._node = node
        self._bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node, spin_thread=True)

        self._latest_image: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._latest_info: Dict[str, Optional[CameraInfo]] = {"left": None, "right": None}
        self._results: Dict[str, CameraAxisResult] = {
            "left": CameraAxisResult(valid=False, camera_name="left", debug_reason="waiting"),
            "right": CameraAxisResult(valid=False, camera_name="right", debug_reason="waiting"),
        }

        self._plug_roi = {
            "left": RoiConfig(0.36, 0.72, 0.42, 0.86),
            "right": RoiConfig(0.22, 0.62, 0.42, 0.86),
        }
        self._gripper_frame_candidates = ["gripper/tcp", "tcp", "tool0", "ee_link", "gripper"]
        self._axis_len_m = 0.035

        self._left_image_sub = node.create_subscription(Image, "/left_camera/image", self._left_image_cb, 10)
        self._right_image_sub = node.create_subscription(Image, "/right_camera/image", self._right_image_cb, 10)
        self._left_info_sub = node.create_subscription(CameraInfo, "/left_camera/camera_info", self._left_info_cb, 10)
        self._right_info_sub = node.create_subscription(CameraInfo, "/right_camera/camera_info", self._right_info_cb, 10)

        self._left_overlay_pub = node.create_publisher(Image, "/left_camera/plug_gripper_angle/overlay", 10)
        self._right_overlay_pub = node.create_publisher(Image, "/right_camera/plug_gripper_angle/overlay", 10)
        self._left_angle_pub = node.create_publisher(Float32, "/left_camera/plug_gripper_angle/value_deg", 10)
        self._right_angle_pub = node.create_publisher(Float32, "/right_camera/plug_gripper_angle/value_deg", 10)
        self._fused_angle_pub = node.create_publisher(Float32, "/plug_gripper_angle/fused_deg", 10)

        self._timer = node.create_timer(0.2, self._tick)

    def get_latest_result(self, camera_name: str) -> CameraAxisResult:
        return self._results[camera_name]

    def get_fused_angle_deg(self) -> Optional[float]:
        vals = []
        weights = []
        for name in ("left", "right"):
            res = self._results[name]
            if res.valid:
                vals.append(res.angle_deg)
                weights.append(max(1e-6, res.confidence))
        if not vals:
            return None
        return float(np.average(np.asarray(vals, dtype=np.float32), weights=np.asarray(weights, dtype=np.float32)))

    def _left_image_cb(self, msg: Image):
        self._latest_image["left"] = self._to_bgr(msg)

    def _right_image_cb(self, msg: Image):
        self._latest_image["right"] = self._to_bgr(msg)

    def _left_info_cb(self, msg: CameraInfo):
        self._latest_info["left"] = msg

    def _right_info_cb(self, msg: CameraInfo):
        self._latest_info["right"] = msg

    def _tick(self):
        for camera_name in ("left", "right"):
            image = self._latest_image[camera_name]
            info = self._latest_info[camera_name]
            if image is None or info is None:
                continue
            result, overlay = self._process_camera(camera_name, image, info)
            self._results[camera_name] = result
            overlay_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_msg.header = info.header
            if camera_name == "left":
                self._left_overlay_pub.publish(overlay_msg)
                if result.valid:
                    self._left_angle_pub.publish(Float32(data=float(result.angle_deg)))
            else:
                self._right_overlay_pub.publish(overlay_msg)
                if result.valid:
                    self._right_angle_pub.publish(Float32(data=float(result.angle_deg)))

        fused = self.get_fused_angle_deg()
        if fused is not None:
            self._fused_angle_pub.publish(Float32(data=float(fused)))

    def _process_camera(self, camera_name: str, image_bgr: np.ndarray, info: CameraInfo):
        vis = image_bgr.copy()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        plug_roi = self._roi_to_pixels(self._plug_roi[camera_name], w, h)
        self._draw_roi(vis, plug_roi, (0, 255, 255))

        gripper_fit = self._project_gripper_axis_and_candidates(info)
        if gripper_fit is None:
            reason = "tf_gripper_axis_failed"
            cv2.putText(vis, f"{camera_name} {reason}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return CameraAxisResult(valid=False, camera_name=camera_name, debug_reason=reason), vis

        gripper_ctr, gripper_axis, gripper_axis_name, gripper_conf, candidate_axes = gripper_fit
        self._draw_axis(vis, gripper_ctr, gripper_axis, 90, (255, 0, 0), f"gripper {gripper_axis_name}")

        plug_mask, selected_axis_name = self._segment_plug(gray, plug_roi, gripper_ctr, gripper_axis, candidate_axes)
        self._draw_mask_outline(vis, plug_mask, (0, 255, 0))

        plug_fit = self._fit_axis_2d(plug_mask, gripper_ctr)
        if plug_fit is None:
            reason = "bad_plug_fit"
            cv2.putText(vis, f"{camera_name} {reason}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return CameraAxisResult(valid=False, camera_name=camera_name, gripper_center_px=(float(gripper_ctr[0]), float(gripper_ctr[1])), gripper_axis_name=selected_axis_name, debug_reason=reason), vis

        plug_ctr, plug_axis, plug_pts, plug_conf = plug_fit
        to_plug = plug_ctr - gripper_ctr
        if float(np.dot(plug_axis, to_plug)) < 0.0:
            plug_axis = -plug_axis

        dot_val = float(np.clip(np.dot(gripper_axis, plug_axis), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(dot_val)))
        confidence = float(min(plug_conf, gripper_conf))

        self._draw_axis(vis, plug_ctr, plug_axis, 70, (0, 255, 0), "plug")
        cv2.putText(vis, f"{camera_name} angle={angle_deg:.2f} deg", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"plug_pts={plug_pts} conf={confidence:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        result = CameraAxisResult(
            valid=True,
            camera_name=camera_name,
            angle_deg=angle_deg,
            confidence=confidence,
            gripper_center_px=(float(gripper_ctr[0]), float(gripper_ctr[1])),
            plug_center_px=(float(plug_ctr[0]), float(plug_ctr[1])),
            num_plug_points=plug_pts,
            gripper_axis_name=gripper_axis_name,
        )
        return result, vis

    def _segment_plug(
        self,
        gray: np.ndarray,
        roi: Tuple[int, int, int, int],
        gripper_ctr: np.ndarray,
        gripper_axis: np.ndarray,
        candidate_axes: List[Tuple[str, np.ndarray]],
    ) -> Tuple[np.ndarray, str]:
        x0, y0, x1, y1 = roi
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return mask, ""

        blur = cv2.GaussianBlur(crop, (5, 5), 0)
        q = float(np.percentile(blur, 78.0))
        otsu_ret, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(q, float(otsu_ret))
        local = np.zeros_like(crop, dtype=np.uint8)
        local[blur >= thr] = 255

        kernel = np.ones((3, 3), np.uint8)
        local = cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel)
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel)

        gx = float(gripper_ctr[0] - x0)
        gy = float(gripper_ctr[1] - y0)
        yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]

        best_axis_name = ""
        best_label_mask = None
        best_score = -1e18

        for axis_name, axis in candidate_axes:
            ax = float(axis[0])
            ay = float(axis[1])
            base_x = gx + 10.0 * ax
            base_y = gy + 10.0 * ay
            forward = (xx - base_x) * ax + (yy - base_y) * ay
            lateral = np.abs((xx - base_x) * (-ay) + (yy - base_y) * ax)
            corridor = (forward > 0.0) & (forward < 120.0) & (lateral < 30.0)
            corridor_u8 = corridor.astype(np.uint8) * 255
            local_axis = cv2.bitwise_and(local, corridor_u8)
            local_axis = cv2.morphologyEx(local_axis, cv2.MORPH_OPEN, kernel)
            local_axis = cv2.morphologyEx(local_axis, cv2.MORPH_CLOSE, kernel)
            local_axis = cv2.dilate(local_axis, kernel, iterations=1)

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(local_axis, connectivity=8)
            for label in range(1, num):
                area = float(stats[label, cv2.CC_STAT_AREA])
                if area < 20.0 or area > 6000.0:
                    continue
                bw = float(stats[label, cv2.CC_STAT_WIDTH])
                bh = float(stats[label, cv2.CC_STAT_HEIGHT])
                cx, cy = centroids[label]
                component_mask = labels == label
                comp_forward = forward[component_mask]
                comp_lateral = lateral[component_mask]
                if comp_forward.size == 0:
                    continue
                mean_forward = float(np.mean(comp_forward))
                max_forward = float(np.max(comp_forward))
                mean_lateral = float(np.mean(comp_lateral))
                min_dist_to_base = float(np.min(np.hypot(np.where(component_mask)[1] - gx, np.where(component_mask)[0] - gy)))
                elong = max(bw, bh) / max(1.0, min(bw, bh))
                behind_penalty = max(0.0, -float(np.min(comp_forward)))
                area_penalty = 0.002 * max(0.0, area - 2000.0)
                score = (
                    0.06 * area
                    + 16.0 * max(0.0, elong - 1.15)
                    + 0.7 * mean_forward
                    + 0.2 * max_forward
                    - 1.2 * mean_lateral
                    - 2.0 * min_dist_to_base
                    - 15.0 * behind_penalty
                    - area_penalty
                )
                if score > best_score:
                    best_score = score
                    best_axis_name = axis_name
                    best_label_mask = component_mask.copy()

        if best_label_mask is None:
            return mask, best_axis_name

        chosen = best_label_mask.astype(np.uint8) * 255
        mask[y0:y1, x0:x1] = chosen
        return mask, best_axis_name

    def _fit_axis_2d(self, mask: np.ndarray, gripper_ctr: np.ndarray):
        ys, xs = np.where(mask > 0)
        if xs.size < 20:
            return None
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        ctr = np.mean(pts, axis=0)
        centered = pts - ctr[None, :]
        cov = centered.T @ centered / max(1, pts.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        axis = eigvecs[:, 0].astype(np.float32)
        axis = self._normalize_2d(axis)
        if float(np.linalg.norm(axis)) < 1e-6:
            return None
        to_ctr = self._normalize_2d(ctr - gripper_ctr)
        if float(np.dot(axis, to_ctr)) < 0.0:
            axis = -axis
        anisotropy = float(eigvals[0] / max(eigvals[1], 1e-8)) if eigvals.size > 1 else 1.0
        confidence = float(min(1.0, np.log1p(max(0.0, anisotropy - 1.0)) / 2.5) * min(1.0, pts.shape[0] / 600.0))
        return ctr.astype(np.float32), axis, int(pts.shape[0]), confidence

    def _project_gripper_axis_and_candidates(self, info: CameraInfo):
        camera_frame = info.header.frame_id
        if not camera_frame:
            return None
        transform = None
        used_gripper_frame = ""
        for frame_name in self._gripper_frame_candidates:
            try:
                transform = self._tf_buffer.lookup_transform(camera_frame, frame_name, Time())
                used_gripper_frame = frame_name
                break
            except TransformException:
                continue
        if transform is None:
            return None

        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if abs(fx) < 1e-6 or abs(fy) < 1e-6:
            return None

        r = self._quat_to_rot(
            float(transform.transform.rotation.x),
            float(transform.transform.rotation.y),
            float(transform.transform.rotation.z),
            float(transform.transform.rotation.w),
        )
        t = np.array(
            [
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            ],
            dtype=np.float32,
        )
        ctr = self._project_point(t, fx, fy, cx, cy)
        if ctr is None:
            return None

        raw_candidates = [
            ("+x", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("-x", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
            ("+y", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ("-y", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
            ("+z", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("-z", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
        ]
        candidates_2d: List[Tuple[str, np.ndarray]] = []
        for axis_name, axis_local in raw_candidates:
            p_cam = t + (r @ (axis_local * self._axis_len_m).astype(np.float64)).astype(np.float32)
            uv = self._project_point(p_cam, fx, fy, cx, cy)
            if uv is None:
                continue
            axis2 = uv - ctr
            pix_len = float(np.linalg.norm(axis2))
            if pix_len < 5.0:
                continue
            axis2 = axis2 / pix_len
            candidates_2d.append((f"{used_gripper_frame}:{axis_name}", axis2.astype(np.float32)))

        if not candidates_2d:
            return None

        best_name, best_axis = max(candidates_2d, key=lambda item: abs(float(item[1][1])))
        return ctr.astype(np.float32), best_axis.astype(np.float32), best_name, 1.0, candidates_2d

    def _project_point(self, p: np.ndarray, fx: float, fy: float, cx: float, cy: float):
        if float(p[2]) <= 1e-6:
            return None
        u = fx * float(p[0]) / float(p[2]) + cx
        v = fy * float(p[1]) / float(p[2]) + cy
        return np.array([u, v], dtype=np.float32)

    def _to_bgr(self, msg: Image) -> np.ndarray:
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if image.ndim == 2:
            if image.dtype != np.uint8:
                image = self._normalize_to_u8(image)
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

    def _normalize_2d(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return v.astype(np.float32)
        return (v / n).astype(np.float32)

    def _quat_to_rot(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        n = x * x + y * y + z * z + w * w
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        s = 2.0 / n
        xx = x * x * s
        yy = y * y * s
        zz = z * z * s
        xy = x * y * s
        xz = x * z * s
        yz = y * z * s
        wx = w * x * s
        wy = w * y * s
        wz = w * z * s
        return np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dtype=np.float64,
        )

    def _draw_axis(self, image: np.ndarray, center_px: np.ndarray, axis_px: np.ndarray, half_len: int, color: Tuple[int, int, int], label: str):
        p0 = (center_px - axis_px * float(half_len)).astype(np.int32)
        p1 = (center_px + axis_px * float(half_len)).astype(np.int32)
        cv2.arrowedLine(image, tuple(p0), tuple(p1), color, 3, cv2.LINE_AA, tipLength=0.18)
        cv2.circle(image, tuple(center_px.astype(np.int32)), 4, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (int(center_px[0]) + 8, int(center_px[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _draw_mask_outline(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)

    def _draw_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int], color: Tuple[int, int, int]):
        x0, y0, x1, y1 = roi
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

    def _roi_to_pixels(self, roi: RoiConfig, width: int, height: int) -> Tuple[int, int, int, int]:
        x0 = max(0, min(width - 1, int(round(roi.x_min_frac * width))))
        x1 = max(x0 + 1, min(width, int(round(roi.x_max_frac * width))))
        y0 = max(0, min(height - 1, int(round(roi.y_min_frac * height))))
        y1 = max(y0 + 1, min(height, int(round(roi.y_max_frac * height))))
        return x0, y0, x1, y1


class DepthAngleEstimatorNode(Node):
    def __init__(self):
        super().__init__("depth_plug_gripper_angle_estimator")
        self._estimator = DepthPlugGripperAngleEstimator(self)
        self._status_timer = self.create_timer(1.0, self._status)
        self.get_logger().info("RGB plug-gripper angle estimator started")
        self.get_logger().info("Inputs:")
        self.get_logger().info("  /left_camera/image")
        self.get_logger().info("  /left_camera/camera_info")
        self.get_logger().info("  /right_camera/image")
        self.get_logger().info("  /right_camera/camera_info")
        self.get_logger().info("  TF for gripper/tcp projection")
        self.get_logger().info("Publishers:")
        self.get_logger().info("  /left_camera/plug_gripper_angle/overlay")
        self.get_logger().info("  /right_camera/plug_gripper_angle/overlay")
        self.get_logger().info("  /left_camera/plug_gripper_angle/value_deg")
        self.get_logger().info("  /right_camera/plug_gripper_angle/value_deg")
        self.get_logger().info("  /plug_gripper_angle/fused_deg")

    def _status(self):
        left = self._estimator.get_latest_result("left")
        right = self._estimator.get_latest_result("right")
        fused = self._estimator.get_fused_angle_deg()
        if left.valid:
            self.get_logger().info(
                f"left angle={left.angle_deg:.2f} deg conf={left.confidence:.3f} plug_pts={left.num_plug_points} axis={left.gripper_axis_name}"
            )
        else:
            self.get_logger().warn(f"left invalid: {left.debug_reason}")
        if right.valid:
            self.get_logger().info(
                f"right angle={right.angle_deg:.2f} deg conf={right.confidence:.3f} plug_pts={right.num_plug_points} axis={right.gripper_axis_name}"
            )
        else:
            self.get_logger().warn(f"right invalid: {right.debug_reason}")
        if fused is not None:
            self.get_logger().info(f"fused angle={fused:.2f} deg")


def main(args=None):
    rclpy.init(args=args)
    node = DepthAngleEstimatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
