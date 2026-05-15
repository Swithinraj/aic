from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.time import Time
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


CAMERAS = ("left", "center", "right")
BASE_FRAME = "base_link"
CAMERA_OPTICAL_FRAMES = {cam: f"{cam}_camera/optical" for cam in CAMERAS}

# Optional TF aliases republished by this detector for policy-side debugging/control.
# We do NOT rebroadcast the original child frame (for example, gripper/tcp) to avoid
# TF authority conflicts. Instead, we publish stable aliases under yolo_tri/gripper/*.
GRIPPER_TF_SOURCE_FRAMES = ("gripper/tcp",)
GRIPPER_TF_ALIAS_PREFIX = "yolo_tri/gripper"
GRIPPER_TF_LOG_PERIOD_S = 2.0
FAMILY_ORDER = {"task_board": 0, "nic_card": 1, "sc_port": 2, "sfp_port": 3, "sfp_module": 4, "sc_plug": 5}
PARAMS = dict(raw_conf=0.15, publish_conf=0.70, track_new_conf=0.85, track_low_conf=0.25, track_confirm_hits=2, track_max_misses=5, track_iou_gate=0.10, track_center_gate_px=90.0, track_alpha=0.45, track_beta=0.12, track_size_ema=0.35, track_conf_ema=0.30, track_min_raw_for_publish=0.70, task_board_publish_conf=0.35, port_publish_conf=0.45, module_publish_conf=0.20, plug_publish_conf=0.45, feature_confirmed_publish_conf=0.45, feature_confirmed_quality_min=0.65, sfp_number_switch_frames=8, sfp_number_margin_px=15.0, byte_kalman_enable=True, byte_kalman_process_var=20.0, byte_kalman_measurement_var=25.0, draw_raw_debug=False, draw_feature_debug=False, enable_feature_tracking=True, feature_method="lk_orb", feature_roi_pad=0.20, feature_max_corners=80, feature_quality_level=0.01, feature_min_distance=5.0, feature_lk_win_size=21, feature_lk_max_level=3, feature_fb_thresh_px=2.0, feature_min_points=8, feature_min_inliers=6, feature_min_inlier_ratio=0.45, feature_ransac_thresh_px=4.0, feature_refresh_frames=10, feature_clahe=True, feature_edge_preprocess="sobel", feature_orb_enable=True, feature_orb_nfeatures=250, feature_orb_match_ratio=0.75, feature_orb_min_matches=8, yolo_jump_gate_px=60.0, iou=0.45, imgsz=640, max_hz=20.0, sc_max_ports=2, sc_order_axis="x", sfp_consensus_hold_s=1.5, sfp_consensus_min_cams=2, sfp_left_default_label=1, sfp_all_cameras_override_s=10.0, sc_consensus_hold_s=1.5, sc_consensus_min_cams=2, sc_consensus_flip_frames=5, enable_sfp_module_gripper_orientation=True, sfp_module_gripper_orientation_min_offset_m=0.005, enable_sc_plug_pnp_orientation=True, sc_plug_pnp_width_m=0.012, sc_plug_pnp_height_m=0.006, sc_plug_pnp_log_period_s=1.0, enable_sc_plug_gripper_prior=True, sc_plug_gripper_offset_xyz=(0.0, 0.015385, 0.04045), sc_plug_gripper_offset_rpy=(0.4432, -0.4838, 1.3303), sc_plug_visual_fusion_alpha=0.25, sc_plug_visual_fusion_max_correction_m=0.020, sc_plug_pnp_max_reproj_px=3.0, sc_plug_pnp_max_delta_deg=35.0)
TRIANGULATION_MIN_CAMS = 2
TRIANGULATION_MAX_STAMP_SPREAD_S = 0.200
TRIANGULATION_MIN_RAY_ANGLE_DEG = 0.35
TRIANGULATION_MAX_RAY_ERROR_M = 0.025
TRIANGULATION_MAX_REPROJ_ERROR_PX = 12.0
TRIANGULATION_MAX_CONDITION = 1.0e8
TRIANGULATION_MIN_DEPTH_M = 0.03
TRIANGULATION_MAX_DEPTH_M = 3.0
TRIANGULATION_MODULE_MAX_STAMP_SPREAD_S = 0.250
TRIANGULATION_MODULE_MAX_RAY_ERROR_M = 0.045
TRIANGULATION_MODULE_MAX_REPROJ_ERROR_PX = 22.0
FEATURE_FAMILIES = {"sfp_port", "sfp_module", "sc_port", "sc_plug", "task_board", "nic_card"}
FAMILY_ALIASES = {"task_board": {"task_board", "taskboard", "board"}, "nic_card": {"nic_card", "nic", "nic_card_0", "nic_card_1", "nic_card_2", "nic_card_3", "nic_card_4"}, "sc_port": {"sc_port", "sc_port_0", "sc_port_1", "sc_port_2", "sc_port_3", "sc_port_4"}, "sfp_port": {"sfp_port", "sfp_port_0", "sfp_port_1", "sfp_port_2", "sfp_port_3"}, "sfp_module": {"sfp_module", "sfpmodule", "transceiver"}, "sc_plug": {"sc_plug", "sc_connector"}}


def norm_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def to_numpy(value):
    if value is None:
        return None
    try:
        return value.detach().cpu().numpy()
    except Exception:
        try:
            return value.cpu().numpy()
        except Exception:
            return np.asarray(value)


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)


def bbox_size(bbox: np.ndarray) -> np.ndarray:
    return np.asarray([max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])], dtype=np.float64)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def rect_corners_from_bbox(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def clip_bbox(bbox: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.asarray(bbox, dtype=np.float64).copy()
    out[0] = np.clip(out[0], 0, max(0, w - 1))
    out[1] = np.clip(out[1], 0, max(0, h - 1))
    out[2] = np.clip(out[2], 1, max(1, w))
    out[3] = np.clip(out[3], 1, max(1, h))
    if out[2] <= out[0] + 1:
        out[2] = min(float(w), out[0] + 2.0)
    if out[3] <= out[1] + 1:
        out[3] = min(float(h), out[1] + 2.0)
    return out


def class_name_from_names(names, cls_idx: int) -> str:
    try:
        idx = int(cls_idx)
        if isinstance(names, dict):
            return str(names.get(idx, idx))
        return str(names[idx])
    except Exception:
        return str(int(cls_idx))


def safe_frame_token(value: object) -> str:
    text = norm_name(value)
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return safe or "det"


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return np.asarray([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_inverse_xyzw(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_angle_deg_xyzw(a: np.ndarray, b: np.ndarray) -> float:
    qa = quat_normalize(a)
    qb = quat_normalize(b)
    dot = abs(float(np.dot(qa, qb)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return quat_normalize(np.asarray([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float64))


def quat_slerp_xyzw(q_old: np.ndarray, q_new: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized-lerp toward q_new with sign correction for shortest-path interpolation.

    alpha=0 returns q_old unchanged; alpha=1 returns q_new unchanged.
    """
    q_o = quat_normalize(q_old)
    q_n = quat_normalize(q_new)
    if float(np.dot(q_o, q_n)) < 0.0:
        q_n = -q_n
    return quat_normalize((1.0 - alpha) * q_o + alpha * q_n)


def quat_to_rpy_xyzw(q: np.ndarray) -> Tuple[float, float, float]:
    x, y, z, w = [float(v) for v in quat_normalize(q)]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = float(np.copysign(np.pi / 2.0, sinp))
    else:
        pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def quat_yaw_align_blue_axis_to_vector_xyzw(q: np.ndarray, target_vector_base: np.ndarray) -> np.ndarray:
    """Rotate the frame about base-link Z so the blue/Z axis yaw points toward target_vector_base.

    This is intentionally a yaw-only correction for SC plug. It preserves the existing
    3D tilt as much as possible and does not force the blue axis to exactly equal the
    full 3D gripper→plug vector.
    """
    q = quat_normalize(q)
    target = np.asarray(target_vector_base, dtype=np.float64).reshape(3)
    target_xy = target[:2].copy()
    if not np.all(np.isfinite(target_xy)) or float(np.linalg.norm(target_xy)) < 1e-9:
        return q

    R = quat_to_matrix(q)
    blue_xy = R[:2, 2].copy()
    if not np.all(np.isfinite(blue_xy)) or float(np.linalg.norm(blue_xy)) < 1e-9:
        return q

    target_yaw = float(np.arctan2(target_xy[1], target_xy[0]))
    blue_yaw = float(np.arctan2(blue_xy[1], blue_xy[0]))
    yaw_delta = float(np.arctan2(np.sin(target_yaw - blue_yaw), np.cos(target_yaw - blue_yaw)))
    q_delta = rpy_to_quat_xyzw(0.0, 0.0, yaw_delta)
    return quat_normalize(quat_multiply_xyzw(q_delta, q))


def quat_local_blue_twist_make_green_perpendicular_xyzw(
    q_sc_plug: np.ndarray,
    q_gripper_tcp: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    """Rotate only around the SC plug's own local blue/Z axis.

    The SC plug blue/Z direction is preserved.  The twist angle is selected so the
    resulting SC plug green/Y axis is perpendicular to the gripper TCP green/Y axis,
    both expressed in base_link coordinates.

    Returns (new_quat_xyzw, twist_rad, green_dot_before, green_dot_after).
    """
    q_sc = quat_normalize(q_sc_plug)
    q_tcp = quat_normalize(q_gripper_tcp)
    R_sc = quat_to_matrix(q_sc)
    R_tcp = quat_to_matrix(q_tcp)

    sc_x = R_sc[:, 0].copy()
    sc_y = R_sc[:, 1].copy()
    tcp_y = R_tcp[:, 1].copy()

    before_dot = float(np.dot(sc_y, tcp_y))
    a = before_dot
    b = float(np.dot(sc_x, tcp_y))
    if not np.all(np.isfinite([a, b])) or float(np.hypot(a, b)) < 1e-9:
        return q_sc, 0.0, before_dot, before_dot

    # Local Z twist means q_new = q_sc * Rz(theta). Therefore:
    #   y_new = -sin(theta) * x_old + cos(theta) * y_old
    # Want dot(y_new, tcp_y) = 0 -> a*cos(theta) - b*sin(theta) = 0.
    theta = float(np.arctan2(a, b))

    # theta + pi is the other valid perpendicular solution. Pick the smaller
    # absolute local twist to avoid unnecessary frame spin.
    theta = float(np.arctan2(np.sin(theta), np.cos(theta)))
    theta_alt = float(np.arctan2(np.sin(theta + np.pi), np.cos(theta + np.pi)))
    if abs(theta_alt) < abs(theta):
        theta = theta_alt

    q_twist_local_z = rpy_to_quat_xyzw(0.0, 0.0, theta)
    q_new = quat_normalize(quat_multiply_xyzw(q_sc, q_twist_local_z))
    after_dot = float(np.dot(quat_to_matrix(q_new)[:, 1], tcp_y))
    return q_new, theta, before_dot, after_dot



def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return quat_normalize(np.asarray([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s], dtype=np.float64))
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        return quat_normalize(np.asarray([0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s], dtype=np.float64))
    if r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        return quat_normalize(np.asarray([(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s], dtype=np.float64))
    s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
    return quat_normalize(np.asarray([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s], dtype=np.float64))


def ensure_right_handed_axes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize three axes and flip Y if det([x,y,z]) < 0 to ensure a proper rotation matrix."""
    x = np.asarray(x_axis, dtype=np.float64).reshape(3)
    y = np.asarray(y_axis, dtype=np.float64).reshape(3)
    z = np.asarray(z_axis, dtype=np.float64).reshape(3)
    x = x / max(1e-12, float(np.linalg.norm(x)))
    y = y / max(1e-12, float(np.linalg.norm(y)))
    z = z / max(1e-12, float(np.linalg.norm(z)))
    if float(np.dot(x, np.cross(y, z))) < 0.0:
        y = -y
    return x, y, z


def build_frame_from_z_and_x_hint(
    z_axis: np.ndarray,
    x_hint: np.ndarray,
    fallbacks: Optional[List[np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Build a right-handed ROS quaternion from a primary Z axis and a lateral X hint.

    Shared convention for SFP port and SFP module so CheatCode quaternion matching works:
      Z / blue  = insertion / approaching axis  (kept as given, normalized)
      X / red   = lateral axis  (x_hint projected onto the plane ⊥ Z, normalized)
      Y / green = cross(Z, X)   →  det(R) = +1  (valid ROS rotation)
    Re-orthogonalize X = cross(Y, Z) to remove numerical drift.

    If x_hint is degenerate against Z, each fallback is tried in order.
    Returns normalized xyzw quaternion, or None if all hints are degenerate.
    """
    z = np.asarray(z_axis, dtype=np.float64).reshape(3)
    z_n = float(np.linalg.norm(z))
    if not np.isfinite(z_n) or z_n < 1e-9:
        return None
    z = z / z_n

    hints: List[np.ndarray] = [np.asarray(x_hint, dtype=np.float64).reshape(3)]
    if fallbacks:
        hints.extend(np.asarray(f, dtype=np.float64).reshape(3) for f in fallbacks)

    x: Optional[np.ndarray] = None
    for h in hints:
        proj = h - z * float(np.dot(h, z))
        proj_n = float(np.linalg.norm(proj))
        if proj_n > 1e-9:
            x = proj / proj_n
            break
    if x is None:
        return None

    y = np.cross(z, x)
    y_n = float(np.linalg.norm(y))
    if y_n < 1e-12:
        return None
    y = y / y_n
    x = np.cross(y, z)
    x = x / max(1e-12, float(np.linalg.norm(x)))
    return matrix_to_quat(np.column_stack([x, y, z]))


def build_frame_from_z_and_y_hint(
    z_axis: np.ndarray,
    y_hint: np.ndarray,
    fallbacks: Optional[List[np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Build a right-handed ROS quaternion from primary Z and desired Y/green hint.

    This is used only for the SFP module orientation.  It preserves the existing
    module convention that Z / blue is the TCP→plug direction, but it chooses
    the Y / green axis from the gripper TCP green axis.  The hint is projected
    onto the plane perpendicular to Z, then X is recomputed as Y × Z so that
    [X Y Z] is a valid right-handed rotation matrix.
    """
    z = np.asarray(z_axis, dtype=np.float64).reshape(3)
    z_n = float(np.linalg.norm(z))
    if not np.isfinite(z_n) or z_n < 1e-9:
        return None
    z = z / z_n

    hints: List[np.ndarray] = [np.asarray(y_hint, dtype=np.float64).reshape(3)]
    if fallbacks:
        hints.extend(np.asarray(f, dtype=np.float64).reshape(3) for f in fallbacks)

    y: Optional[np.ndarray] = None
    for h in hints:
        proj = h - z * float(np.dot(h, z))
        proj_n = float(np.linalg.norm(proj))
        if proj_n > 1e-9:
            y = proj / proj_n
            break
    if y is None:
        return None

    x = np.cross(y, z)
    x_n = float(np.linalg.norm(x))
    if x_n < 1e-12:
        return None
    x = x / x_n
    y = np.cross(z, x)
    y = y / max(1e-12, float(np.linalg.norm(y)))
    return matrix_to_quat(np.column_stack([x, y, z]))


def compute_sfp_port_orientation_from_pair(
    sfp0_pos: np.ndarray,
    sfp1_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build SFP port frame from the two triangulated port positions.

    Convention (right-handed, shared with SFP module for CheatCode quaternion matching):
      Z / blue  = downward base_link direction, orthogonalized against X (insertion axis)
      X / red   = direction from sfp_port_0 to sfp_port_1 (lateral, used as x_hint)
      Y / green = cross(Z, X)  →  det(R) = +1  (valid ROS rotation)

    Returns (quat_xyzw, x_axis_normalized).
    x_axis is the normalized sfp0→sfp1 vector, returned so the module orientation builder
    can reuse it as its x_hint to guarantee both frames share the same lateral reference.
    Falls back to identity quat + base-X if the two positions coincide.
    """
    sfp0 = np.asarray(sfp0_pos, dtype=np.float64).reshape(3)
    sfp1 = np.asarray(sfp1_pos, dtype=np.float64).reshape(3)
    _id_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    x_hint = sfp1 - sfp0
    x_norm = float(np.linalg.norm(x_hint))
    if x_norm < 1e-9:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64), _id_x
    x_hint = x_hint / x_norm

    # Z = insertion/downward axis (primary); X = lateral (hint projected ⊥ Z)
    z_primary = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    z_fallbacks: List[np.ndarray] = [np.asarray([0.0, -1.0, 0.0], dtype=np.float64)]
    # build_frame_from_z_and_x_hint treats Z as primary and X as hint, so swap roles:
    # we want X = sfp0→sfp1 preserved, Z = downward projected ⊥ X.
    # Achieve this by calling with z=downward as primary and x_hint=sfp0→sfp1.
    quat = build_frame_from_z_and_x_hint(z_primary, x_hint, fallbacks=z_fallbacks)
    if quat is None:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64), x_hint

    # Extract the actual X column from the built matrix so the caller gets the
    # re-orthogonalized lateral axis (very close to the original sfp0→sfp1 unit vector).
    R = quat_to_matrix(quat)
    x_axis_out = R[:, 0].copy()
    return quat, x_axis_out

def ray_pose_quaternion(ray_base: np.ndarray, base_R_camera: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(ray_base, dtype=np.float64).reshape(3)
    z_axis /= max(1e-12, float(np.linalg.norm(z_axis)))
    x_hint = base_R_camera @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_hint - z_axis * float(np.dot(x_hint, z_axis))
    if float(np.linalg.norm(x_axis)) < 1e-9:
        y_hint = base_R_camera @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = y_hint - z_axis * float(np.dot(y_hint, z_axis))
    x_axis /= max(1e-12, float(np.linalg.norm(x_axis)))
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(1e-12, float(np.linalg.norm(y_axis)))
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(1e-12, float(np.linalg.norm(x_axis)))
    return matrix_to_quat(np.column_stack([x_axis, y_axis, z_axis]))


def transform_to_arrays(tf_msg: TransformStamped) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    trans = np.asarray([float(t.x), float(t.y), float(t.z)], dtype=np.float64)
    quat = quat_normalize(np.asarray([float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64))
    return trans, quat, quat_to_matrix(quat)


def compute_sfp_module_orientation_matching_port(
    gripper_pos: np.ndarray,
    gripper_quat_xyzw: np.ndarray,
    plug_pos: np.ndarray,
    port_x_axis: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build SFP module/plug frame while matching the gripper TCP green axis.

    Only the SFP module orientation is changed.  The position source, TF setup,
    port orientation, triangulation, tracking, and existing axis convention are
    left intact.

    Convention:
      Z / blue  = normalized vector from gripper/TCP to triangulated plug position.
      Y / green = -gripper TCP local Y axis, projected perpendicular to Z.
      X / red   = recomputed as Y × Z to keep [X Y Z] right-handed.

    The negative gripper-Y is intentional because the observed module green axis
    must follow the opposite TCP green direction in this setup.
    """
    del port_x_axis  # kept in the signature to avoid changing call sites/API
    g = np.asarray(gripper_pos, dtype=np.float64).reshape(3)
    p = np.asarray(plug_pos, dtype=np.float64).reshape(3)
    tcp_to_plug = p - g
    norm = float(np.linalg.norm(tcp_to_plug))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    z_axis = tcp_to_plug / norm

    Rg = quat_to_matrix(gripper_quat_xyzw)
    gripper_green_minus = -Rg[:, 1].copy()
    fallbacks: List[np.ndarray] = [
        Rg[:, 1].copy(),
        Rg[:, 0].copy(),
        -Rg[:, 0].copy(),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    ]
    return build_frame_from_z_and_y_hint(z_axis, gripper_green_minus, fallbacks=fallbacks)


def compute_sc_port_orientation_from_axis(
    axis_left_uv: np.ndarray,
    axis_right_uv: np.ndarray,
    k: np.ndarray,
    base_R_camera: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build SC port frame from the detected OBB mouth axis projected into base_link.

    Convention (right-handed, matching SFP port for CheatCode quaternion compatibility):
      Z / blue  = base_link downward [0, 0, -1], orthogonalized against X
      X / red   = SC port mouth direction (left -> right) projected into base_link
      Y / green = cross(Z, X)  ->  det(R) = +1

    Returns (quat_xyzw, x_axis_normalized) or None if the axis is degenerate.
    """
    left = np.asarray(axis_left_uv, dtype=np.float64).reshape(2)
    right = np.asarray(axis_right_uv, dtype=np.float64).reshape(2)
    axis_2d = right - left
    norm_2d = float(np.linalg.norm(axis_2d))
    if norm_2d < 1e-9:
        return None

    fx, fy = float(k[0, 0]), float(k[1, 1])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        return None
    dx, dy = axis_2d[0] / norm_2d, axis_2d[1] / norm_2d
    # Map image-plane direction to camera-space direction (perpendicular to optical axis).
    axis_cam = np.asarray([dx / fx, dy / fy, 0.0], dtype=np.float64)
    axis_cam_n = float(np.linalg.norm(axis_cam))
    if axis_cam_n < 1e-12:
        return None
    axis_cam = axis_cam / axis_cam_n

    # Negate the OBB axis so Y = cross(Z_down, -OBB) = cross(OBB, Z_down):
    # this is the left-hand-rule Y with Z facing down, det(R) = +1 (valid ROS TF).
    # Equivalent to the old post-hoc 180-deg yaw flip, but done directly here.
    x_hint = -(np.asarray(base_R_camera, dtype=np.float64).reshape(3, 3) @ axis_cam)
    z_primary = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    quat = build_frame_from_z_and_x_hint(
        z_primary, x_hint,
        fallbacks=[np.asarray([0.0, -1.0, 0.0], dtype=np.float64)],
    )
    if quat is None:
        return None

    R = quat_to_matrix(quat)
    return quat, R[:, 0].copy()


def compute_sc_plug_orientation_matching_sc_port(
    gripper_pos: np.ndarray,
    gripper_quat_xyzw: np.ndarray,
    plug_pos: np.ndarray,
    sc_port_x_axis: Optional[np.ndarray] = None,
    z_axis_hint: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build SC plug frame matching the SC port convention.

    Convention (right-handed, same object-frame as SC port):
      Z / blue  = z_axis_hint when provided — the plug's forward axis derived from the
                  calibrated RPY grasp prior (qualification_phase.md: roll=0.4432,
                  pitch=-0.4838, yaw=1.3303 applied to gripper orientation). This gives
                  the physically correct insertion tilt (~28 deg pitch). Falls back to
                  normalize(plug_pos - gripper_pos) if the hint is absent or degenerate.
      X / red   = SC port X axis projected perpendicular to Z  (gripper local X fallback)
      Y / green = cross(Z, X),  then X = cross(Y, Z)  right-handed

    Compatible with SC port convention so CheatCode quaternion matching works.
    """
    g = np.asarray(gripper_pos, dtype=np.float64).reshape(3)
    p = np.asarray(plug_pos, dtype=np.float64).reshape(3)

    z_axis: Optional[np.ndarray] = None
    if z_axis_hint is not None:
        zh = np.asarray(z_axis_hint, dtype=np.float64).reshape(3)
        zh_n = float(np.linalg.norm(zh))
        if np.isfinite(zh_n) and zh_n > 1e-9:
            z_axis = zh / zh_n
    if z_axis is None:
        tcp_to_plug = p - g
        norm = float(np.linalg.norm(tcp_to_plug))
        if not np.isfinite(norm) or norm < 1e-9:
            return None
        z_axis = tcp_to_plug / norm

    Rg = quat_to_matrix(gripper_quat_xyzw)
    fallbacks: List[np.ndarray] = [
        Rg[:, 0].copy(),
        -Rg[:, 0].copy(),
        Rg[:, 1].copy(),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    ]
    x_hint = sc_port_x_axis if sc_port_x_axis is not None else fallbacks[0]
    return build_frame_from_z_and_x_hint(z_axis, x_hint, fallbacks=fallbacks)


def order_quad_points_tl_tr_br_bl(points_uv: np.ndarray) -> Optional[np.ndarray]:
    """Return four image points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != 4 or not np.all(np.isfinite(pts)):
        return None
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    # np.argsort gives cyclic order. Rotate so first point is top-left.
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    # Ensure clockwise TL,TR,BR,BL in image coordinates.
    # If the second point is bottom-left rather than top-right, reverse the cycle.
    if ordered[1, 0] < ordered[3, 0]:
        ordered = np.asarray([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float64)
    return ordered.astype(np.float64)


def image_quad_from_sc_plug_detection(det: Dict) -> Optional[np.ndarray]:
    """Get a stable SC-plug 2D quadrilateral from OBB corners, falling back to bbox corners."""
    corners = det.get("obb_corners_uv")
    if isinstance(corners, list) and len(corners) == 4:
        ordered = order_quad_points_tl_tr_br_bl(np.asarray(corners, dtype=np.float64))
        if ordered is not None:
            return ordered
    bbox = det.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return order_quad_points_tl_tr_br_bl(rect_corners_from_bbox(np.asarray(bbox[:4], dtype=np.float64)))
    return None


def sc_plug_object_points_tl_tr_br_bl(width_m: float, height_m: float) -> np.ndarray:
    """Planar SC-plug front-face model points matching TL,TR,BR,BL image order.

    The exact scale mainly affects estimated translation. Orientation is the intended
    output here; final TF position still comes from multi-camera triangulation.
    """
    w = max(1e-6, float(width_m))
    h = max(1e-6, float(height_m))
    return np.asarray([
        [-0.5 * w, -0.5 * h, 0.0],
        [+0.5 * w, -0.5 * h, 0.0],
        [+0.5 * w, +0.5 * h, 0.0],
        [-0.5 * w, +0.5 * h, 0.0],
    ], dtype=np.float64)


def solve_sc_plug_pnp_quat_base(
    det: Dict,
    k: np.ndarray,
    base_R_camera: np.ndarray,
    width_m: float,
    height_m: float,
) -> Optional[Tuple[np.ndarray, float, str]]:
    """Estimate SC-plug orientation from one camera using planar PnP.

    Returns (quat_xyzw_in_base_link, reprojection_error_px, method_name), or None.
    Position from PnP is intentionally ignored; triangulated position remains the TF origin.
    """
    image_points = image_quad_from_sc_plug_detection(det)
    if image_points is None:
        return None
    object_points = sc_plug_object_points_tl_tr_br_bl(width_m, height_m)
    camera_matrix = np.asarray(k, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    candidates: List[Tuple[float, np.ndarray, str]] = []
    flags_to_try: List[Tuple[int, str]] = []
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        flags_to_try.append((cv2.SOLVEPNP_IPPE, "IPPE"))
    flags_to_try.append((cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"))

    for flag, method_name in flags_to_try:
        try:
            if method_name == "IPPE" and hasattr(cv2, "solvePnPGeneric"):
                ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=flag,
                )
                if not ok:
                    continue
                for idx, rvec in enumerate(rvecs):
                    tvec = np.asarray(tvecs[idx], dtype=np.float64).reshape(3, 1)
                    if float(tvec[2, 0]) <= 0.0:
                        continue
                    err = float(np.asarray(reproj).reshape(-1)[idx]) if reproj is not None and len(np.asarray(reproj).reshape(-1)) > idx else 1e9
                    r_cam_obj, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
                    r_base_obj = np.asarray(base_R_camera, dtype=np.float64).reshape(3, 3) @ r_cam_obj
                    candidates.append((err, matrix_to_quat(r_base_obj), method_name))
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=flag,
                )
                if not ok or float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
                    continue
                projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
                projected = projected.reshape(-1, 2)
                err = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
                r_cam_obj, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
                r_base_obj = np.asarray(base_R_camera, dtype=np.float64).reshape(3, 3) @ r_cam_obj
                candidates.append((err, matrix_to_quat(r_base_obj), method_name))
        except cv2.error:
            continue
        except Exception:
            continue

    if not candidates:
        return None
    err, quat, method_name = min(candidates, key=lambda item: item[0])
    return quat_normalize(quat), float(err), method_name


@dataclass
class Detection:
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


class FeatureTrack:
    def __init__(self, owner, track_id: int, family: str, bbox: np.ndarray, frame: np.ndarray, now: float):
        self.owner = owner
        self.track_id = int(track_id)
        self.family = family
        self.bbox_xyxy = np.asarray(bbox, dtype=np.float64).copy()
        self.center_uv = bbox_center(self.bbox_xyxy)
        self.prev_gray: Optional[np.ndarray] = None
        self.points = np.zeros((0, 2), dtype=np.float32)
        self.point_ids: List[int] = []
        self.next_point_id = 1
        self.orb_points = np.zeros((0, 2), dtype=np.float32)
        self.orb_descriptors = None
        self.quality_score = 0.0
        self.tracked_count = 0
        self.inlier_count = 0
        self.mode = "init"
        self.lost_count = 0
        self.jump_rejects = 0
        self.refresh_count = 0
        self.last_update_time = float(now)
        self.last_motion_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self.reinit(frame, self.bbox_xyxy, now, "init")

    def gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.owner.feature_clahe:
            return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return gray

    def roi(self, bbox: np.ndarray, shape_hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        h, w = shape_hw
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
        pad = float(self.owner.feature_roi_pad)
        x1, x2 = x1 - pad * bw, x2 + pad * bw
        y1, y2 = y1 - pad * bh, y2 + pad * bh
        ix1 = int(np.clip(np.floor(x1), 0, max(0, w - 1)))
        iy1 = int(np.clip(np.floor(y1), 0, max(0, h - 1)))
        ix2 = int(np.clip(np.ceil(x2), ix1 + 1, w))
        iy2 = int(np.clip(np.ceil(y2), iy1 + 1, h))
        if ix2 <= ix1 + 2 or iy2 <= iy1 + 2:
            return None
        return ix1, iy1, ix2, iy2

    def feature_image(self, gray_roi: np.ndarray) -> np.ndarray:
        mode = self.owner.feature_edge_preprocess
        if mode == "canny":
            return cv2.Canny(gray_roi, 40, 140)
        if mode == "sobel":
            gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            return np.clip(mag * 255.0 / max(1e-6, float(np.max(mag))), 0, 255).astype(np.uint8)
        return gray_roi

    def detect_points(self, gray: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        roi = self.roi(bbox, gray.shape[:2])
        if roi is None:
            return np.zeros((0, 2), dtype=np.float32), []
        x1, y1, x2, y2 = roi
        img = self.feature_image(gray[y1:y2, x1:x2])
        pts = cv2.goodFeaturesToTrack(
            img,
            maxCorners=self.owner.feature_max_corners,
            qualityLevel=self.owner.feature_quality_level,
            minDistance=self.owner.feature_min_distance,
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

    def compute_orb(self, gray: np.ndarray, bbox: np.ndarray):
        if not self.owner.feature_orb_enable:
            return np.zeros((0, 2), dtype=np.float32), None
        roi = self.roi(bbox, gray.shape[:2])
        if roi is None:
            return np.zeros((0, 2), dtype=np.float32), None
        x1, y1, x2, y2 = roi
        orb = cv2.ORB_create(nfeatures=self.owner.feature_orb_nfeatures)
        keypoints, desc = orb.detectAndCompute(gray[y1:y2, x1:x2], None)
        if not keypoints or desc is None:
            return np.zeros((0, 2), dtype=np.float32), None
        pts = np.asarray([[kp.pt[0] + x1, kp.pt[1] + y1] for kp in keypoints], dtype=np.float32)
        return pts, desc

    def reinit(self, frame: np.ndarray, bbox: np.ndarray, now: float, reason: str) -> None:
        gray = self.gray(frame)
        self.prev_gray = gray
        self.bbox_xyxy = np.asarray(bbox, dtype=np.float64).copy()
        self.center_uv = bbox_center(self.bbox_xyxy)
        self.points, self.point_ids = self.detect_points(gray, self.bbox_xyxy)
        self.orb_points, self.orb_descriptors = self.compute_orb(gray, self.bbox_xyxy)
        self.tracked_count = int(len(self.points))
        self.inlier_count = int(len(self.points))
        self.quality_score = min(1.0, len(self.points) / max(1.0, float(self.owner.feature_max_corners)))
        self.mode = f"reinit_{reason}"
        self.lost_count = 0
        self.refresh_count = 0
        self.last_motion_pairs = []
        self.last_update_time = float(now)

    def update(self, frame: np.ndarray, predicted_bbox: np.ndarray, now: float) -> Optional[np.ndarray]:
        gray = self.gray(frame)
        if self.prev_gray is None or len(self.points) < self.owner.feature_min_points:
            self.lost_count += 1
            self.prev_gray = gray
            self.mode = "kalman_only"
            return None

        old = self.points.reshape(-1, 1, 2).astype(np.float32)
        new, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            old,
            None,
            winSize=(self.owner.feature_lk_win_size, self.owner.feature_lk_win_size),
            maxLevel=self.owner.feature_lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if new is None or st is None:
            self.lost_count += 1
            self.prev_gray = gray
            self.mode = "lk_lost"
            return None

        back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.prev_gray,
            new,
            None,
            winSize=(self.owner.feature_lk_win_size, self.owner.feature_lk_win_size),
            maxLevel=self.owner.feature_lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        old2 = old.reshape(-1, 2)
        new2 = new.reshape(-1, 2)
        back2 = old2 if back is None else back.reshape(-1, 2)
        good = (st.reshape(-1) == 1) & (np.linalg.norm(old2 - back2, axis=1) <= self.owner.feature_fb_thresh_px)
        old_good = old2[good]
        new_good = new2[good]
        ids_good = [pid for pid, ok in zip(self.point_ids, good.tolist()) if ok]
        self.tracked_count = int(len(new_good))
        self.last_motion_pairs = [(o.copy(), n.copy()) for o, n in zip(old_good[:40], new_good[:40])]

        if len(new_good) < self.owner.feature_min_points:
            self.points = new_good.astype(np.float32)
            self.point_ids = ids_good
            self.prev_gray = gray
            self.lost_count += 1
            self.mode = "lk_few_points"
            return self.try_orb(frame, predicted_bbox, now)

        affine, inliers = cv2.estimateAffinePartial2D(
            old_good,
            new_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.owner.feature_ransac_thresh_px,
            maxIters=2000,
            confidence=0.99,
        )
        if affine is None or inliers is None:
            self.points = new_good.astype(np.float32)
            self.point_ids = ids_good
            self.prev_gray = gray
            self.lost_count += 1
            self.mode = "lk_no_affine"
            return self.try_orb(frame, predicted_bbox, now)

        mask = inliers.reshape(-1).astype(bool)
        inlier_old = old_good[mask]
        inlier_new = new_good[mask]
        self.inlier_count = int(len(inlier_new))
        ratio = len(inlier_new) / max(1, len(new_good))
        self.quality_score = float(np.clip(0.5 * ratio + 0.5 * min(1.0, len(inlier_new) / self.owner.feature_max_corners), 0.0, 1.0))

        if len(inlier_new) < self.owner.feature_min_inliers or ratio < self.owner.feature_min_inlier_ratio:
            self.points = new_good.astype(np.float32)
            self.point_ids = ids_good
            self.prev_gray = gray
            self.lost_count += 1
            self.mode = "lk_low_inlier"
            return self.try_orb(frame, predicted_bbox, now)

        corners = rect_corners_from_bbox(self.bbox_xyxy).astype(np.float64)
        moved = cv2.transform(corners.reshape(1, -1, 2), affine).reshape(-1, 2)
        h, w = gray.shape[:2]
        new_bbox = clip_bbox(np.asarray([moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()]), w, h)
        self.bbox_xyxy = new_bbox
        self.center_uv = bbox_center(new_bbox)
        self.points = inlier_new.astype(np.float32)
        self.point_ids = [pid for pid, ok in zip(ids_good, mask.tolist()) if ok]
        self.prev_gray = gray
        self.lost_count = 0
        self.mode = "lk_affine"
        self.last_update_time = float(now)
        self.refresh_count += 1
        if self.refresh_count >= self.owner.feature_refresh_frames:
            self.reinit(frame, self.bbox_xyxy, now, "refresh")
        return self.bbox_xyxy.copy()

    def try_orb(self, frame: np.ndarray, predicted_bbox: np.ndarray, now: float) -> Optional[np.ndarray]:
        if not self.owner.feature_orb_enable or self.orb_descriptors is None or len(self.orb_points) < self.owner.feature_orb_min_matches:
            return None
        gray = self.gray(frame)
        new_points, new_desc = self.compute_orb(gray, predicted_bbox)
        if new_desc is None or len(new_points) < self.owner.feature_orb_min_matches:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            matches = matcher.knnMatch(self.orb_descriptors, new_desc, k=2)
        except Exception:
            return None
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < self.owner.feature_orb_match_ratio * n.distance]
        if len(good) < self.owner.feature_orb_min_matches:
            return None
        old = np.asarray([self.orb_points[m.queryIdx] for m in good], dtype=np.float32)
        new = np.asarray([new_points[m.trainIdx] for m in good], dtype=np.float32)
        affine, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=self.owner.feature_ransac_thresh_px)
        if affine is None or inliers is None or int(np.sum(inliers)) < self.owner.feature_orb_min_matches:
            return None
        moved = cv2.transform(rect_corners_from_bbox(self.bbox_xyxy).reshape(1, -1, 2), affine).reshape(-1, 2)
        h, w = gray.shape[:2]
        self.bbox_xyxy = clip_bbox(np.asarray([moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()]), w, h)
        self.center_uv = bbox_center(self.bbox_xyxy)
        self.prev_gray = gray
        self.points, self.point_ids = self.detect_points(gray, self.bbox_xyxy)
        self.orb_points, self.orb_descriptors = new_points, new_desc
        self.tracked_count = int(len(self.points))
        self.inlier_count = int(np.sum(inliers))
        self.quality_score = min(1.0, self.inlier_count / max(1.0, float(len(good))))
        self.mode = "orb_reid"
        self.lost_count = 0
        self.last_update_time = float(now)
        return self.bbox_xyxy.copy()


@dataclass
class Track:
    track_id: int
    family: str
    instance_name: str
    bbox_xyxy: np.ndarray
    center_uv: np.ndarray
    size_wh: np.ndarray
    velocity_uv: np.ndarray
    confidence: float
    raw_confidence: float
    class_id: int
    raw_class_name: str
    camera_name: str
    stamp_sec: float
    image_width: int
    image_height: int
    extra: Dict
    age: int = 1
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    last_time: float = 0.0
    feature: Optional[FeatureTrack] = None
    kalman_x: Optional[np.ndarray] = None
    kalman_p: Optional[np.ndarray] = None
    kalman_q: float = 1.0
    # Persistent smoothed SC port mouth-axis anchor (only used when family == "sc_port").
    stable_anchor_axis_uv: Optional[np.ndarray] = None
    stable_anchor_center_uv: Optional[np.ndarray] = None
    stable_anchor_length_px: float = 0.0

    @classmethod
    def create(cls, track_id: int, det: Detection, now: float, owner, frame: Optional[np.ndarray]) -> "Track":
        feature = None
        if frame is not None and owner.feature_enabled(det.family):
            feature = FeatureTrack(owner, track_id, det.family, det.bbox_xyxy, frame, now)
        center = det.center_uv.copy()
        size = bbox_size(det.bbox_xyxy)
        track = cls(
            track_id=track_id,
            family=det.family,
            instance_name=det.family,
            bbox_xyxy=det.bbox_xyxy.copy(),
            center_uv=center,
            size_wh=size,
            velocity_uv=np.zeros(2, dtype=np.float64),
            confidence=float(det.confidence),
            raw_confidence=float(det.confidence),
            class_id=int(det.class_id),
            raw_class_name=det.raw_class_name,
            camera_name=det.camera_name,
            stamp_sec=float(det.stamp_sec),
            image_width=int(det.image_width),
            image_height=int(det.image_height),
            extra=dict(det.extra),
            confirmed=1 >= owner.family_confirm_hits(det.family),
            last_time=float(now),
            feature=feature,
        )
        track.kalman_x = np.asarray([center[0], center[1], 0.0, 0.0, size[0], size[1]], dtype=np.float64)
        track.kalman_p = np.eye(6, dtype=np.float64) * 50.0
        track.kalman_q = float(getattr(owner, "byte_kalman_process_var", 20.0))
        return track

    def predict(self, now: float) -> None:
        dt = min(1.0, max(1e-3, float(now - self.last_time)))
        if getattr(self, "kalman_x", None) is not None and getattr(self, "kalman_p", None) is not None:
            F = np.eye(6, dtype=np.float64)
            F[0, 2] = dt
            F[1, 3] = dt
            q = max(1e-6, float(getattr(self, "kalman_q", 20.0)))
            self.kalman_x = F @ self.kalman_x
            self.kalman_p = F @ self.kalman_p @ F.T + np.eye(6, dtype=np.float64) * q
            self.center_uv = self.kalman_x[:2].copy()
            self.velocity_uv = self.kalman_x[2:4].copy()
            self.size_wh = np.maximum(1.0, self.kalman_x[4:6]).copy()
        else:
            self.center_uv = self.center_uv + self.velocity_uv * dt
        self.refresh_bbox()
        self.age += 1

    def refresh_bbox(self) -> None:
        half = 0.5 * self.size_wh
        self.bbox_xyxy = np.asarray(
            [self.center_uv[0] - half[0], self.center_uv[1] - half[1], self.center_uv[0] + half[0], self.center_uv[1] + half[1]],
            dtype=np.float64,
        )

    def update(self, det: Detection, now: float, owner, frame: Optional[np.ndarray], feature_bbox: Optional[np.ndarray]) -> None:
        dt = min(1.0, max(1e-3, float(now - self.last_time)))
        measured_bbox = det.bbox_xyxy.copy()
        if feature_bbox is not None:
            feature_center = bbox_center(feature_bbox)
            jump = float(np.linalg.norm(det.center_uv - feature_center))
            if jump > owner.yolo_jump_gate_px and self.feature is not None and self.feature.quality_score >= owner.feature_min_inlier_ratio:
                self.feature.jump_rejects += 1
                measured_bbox = feature_bbox.copy()
            elif det.confidence < owner.family_threshold(self.family)[0]:
                measured_bbox = 0.70 * feature_bbox + 0.30 * det.bbox_xyxy
        measured_center = bbox_center(measured_bbox)
        measured_size = bbox_size(measured_bbox)
        if bool(getattr(owner, "byte_kalman_enable", True)) and self.kalman_x is not None and self.kalman_p is not None:
            H = np.zeros((4, 6), dtype=np.float64)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[2, 4] = 1.0
            H[3, 5] = 1.0
            z = np.asarray([measured_center[0], measured_center[1], measured_size[0], measured_size[1]], dtype=np.float64)
            r = float(getattr(owner, "byte_kalman_measurement_var", 25.0))
            R = np.eye(4, dtype=np.float64) * max(1e-6, r)
            y = z - H @ self.kalman_x
            S = H @ self.kalman_p @ H.T + R
            K = self.kalman_p @ H.T @ np.linalg.inv(S)
            self.kalman_x = self.kalman_x + K @ y
            self.kalman_p = (np.eye(6, dtype=np.float64) - K @ H) @ self.kalman_p
            self.center_uv = self.kalman_x[:2].copy()
            self.velocity_uv = self.kalman_x[2:4].copy()
            self.size_wh = np.maximum(1.0, self.kalman_x[4:6]).copy()
        else:
            residual = measured_center - self.center_uv
            self.center_uv = self.center_uv + owner.track_alpha * residual
            self.velocity_uv = self.velocity_uv + owner.track_beta * residual / dt
            self.size_wh = owner.track_size_ema * measured_size + (1.0 - owner.track_size_ema) * self.size_wh
        self.refresh_bbox()
        self.confidence = owner.track_conf_ema * float(det.confidence) + (1.0 - owner.track_conf_ema) * self.confidence
        self.raw_confidence = float(det.confidence)
        self.class_id = int(det.class_id)
        self.raw_class_name = det.raw_class_name
        self.camera_name = det.camera_name
        self.stamp_sec = float(det.stamp_sec)
        self.image_width = int(det.image_width)
        self.image_height = int(det.image_height)
        self.extra = dict(det.extra)
        self.hits += 1
        self.misses = 0
        self.confirmed = self.hits >= owner.family_confirm_hits(self.family)
        self.last_time = float(now)
        if frame is not None and self.feature is None and owner.feature_enabled(self.family):
            self.feature = FeatureTrack(owner, self.track_id, self.family, self.bbox_xyxy, frame, now)
        if frame is not None and self.feature is not None and self.feature.tracked_count < owner.feature_min_points:
            self.feature.reinit(frame, self.bbox_xyxy, now, "recover")

    def mark_missed(self, feature_bbox: Optional[np.ndarray]) -> None:
        self.misses += 1
        if feature_bbox is not None:
            self.bbox_xyxy = 0.70 * feature_bbox + 0.30 * self.bbox_xyxy
            self.center_uv = bbox_center(self.bbox_xyxy)
            self.size_wh = bbox_size(self.bbox_xyxy)

    def anchor_fields(self) -> Dict:
        corners = None
        if isinstance(self.extra.get("obb_corners_uv"), list) and len(self.extra["obb_corners_uv"]) == 4:
            try:
                corners = np.asarray(self.extra["obb_corners_uv"], dtype=np.float64).reshape(4, 2)
            except Exception:
                corners = None
        if corners is None:
            corners = rect_corners_from_bbox(self.bbox_xyxy)
        center = np.mean(corners, axis=0)
        edges = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            edges.append((float(np.linalg.norm(b - a)), a, b))
        edges.sort(key=lambda item: item[0])
        long = edges[-1]
        vec = long[2] - long[1]
        length = max(1e-6, float(np.linalg.norm(vec)))
        axis = vec / length
        left, right = center - 0.5 * length * axis, center + 0.5 * length * axis
        if left[0] > right[0]:
            left, right = right, left
        angle = float(np.arctan2(axis[1], axis[0]))
        quality = 0.0 if self.feature is None else float(self.feature.quality_score)
        mode = "bbox" if self.feature is None else self.feature.mode
        common = {"servo_anchor_valid": True, "servo_anchor_source": mode, "servo_anchor_quality": quality}
        if self.family == "sfp_port":
            data = {
                **common,
                "mouth_center_uv": [float(v) for v in center.tolist()],
                "mouth_left_uv": [float(v) for v in left.tolist()],
                "mouth_right_uv": [float(v) for v in right.tolist()],
                "mouth_angle_rad": angle,
            }
            return data
        if self.family == "sc_port":
            # --- Stable SC port mouth-axis with two-edge averaging + EMA smoothing ---
            # Step 1: extract raw axis from the two longest OBB edges.
            if corners is not None:
                all_edges = []
                for i in range(4):
                    a_e, b_e = corners[i], corners[(i + 1) % 4]
                    ev = b_e - a_e
                    el = float(np.linalg.norm(ev))
                    all_edges.append((el, ev / max(el, 1e-6)))
                all_edges.sort(key=lambda item: item[0], reverse=True)
                raw_length = 0.5 * (all_edges[0][0] + all_edges[1][0])
                d0 = all_edges[0][1]
                d1 = all_edges[1][1]
                # Align d1 sign so both long edges point the same half-space.
                if float(np.dot(d0, d1)) < 0.0:
                    d1 = -d1
                raw_axis = d0 + d1
                n = float(np.linalg.norm(raw_axis))
                raw_axis = raw_axis / max(n, 1e-6)
            else:
                # Fallback: use horizontal bbox axis.
                raw_axis = np.asarray([1.0, 0.0], dtype=np.float64)
                raw_length = float(self.size_wh[0])
            # Step 2: sign consistency against stored stable axis.
            if self.stable_anchor_axis_uv is not None and float(np.dot(raw_axis, self.stable_anchor_axis_uv)) < 0.0:
                raw_axis = -raw_axis
            # Step 3: EMA smoothing (alpha = 0.15 toward new measurement).
            ema_alpha = 0.15
            if self.stable_anchor_axis_uv is not None:
                blended = (1.0 - ema_alpha) * self.stable_anchor_axis_uv + ema_alpha * raw_axis
                n = float(np.linalg.norm(blended))
                smoothed_axis = blended / max(n, 1e-6)
                smoothed_center = (1.0 - ema_alpha) * self.stable_anchor_center_uv + ema_alpha * self.center_uv
                smoothed_length = (1.0 - ema_alpha) * self.stable_anchor_length_px + ema_alpha * raw_length
            else:
                smoothed_axis = raw_axis
                smoothed_center = self.center_uv.copy()
                smoothed_length = raw_length
            # Step 4: persist smoothed values back onto the track.
            self.stable_anchor_axis_uv = smoothed_axis.copy()
            self.stable_anchor_center_uv = smoothed_center.copy()
            self.stable_anchor_length_px = float(smoothed_length)
            # Step 5: derive left/right endpoints; guarantee left is the lower-x end.
            half = 0.5 * smoothed_length
            sc_left = smoothed_center - half * smoothed_axis
            sc_right = smoothed_center + half * smoothed_axis
            if sc_left[0] > sc_right[0]:
                sc_left, sc_right = sc_right, sc_left
            sc_angle = float(np.arctan2(smoothed_axis[1], smoothed_axis[0]))
            data = {
                **common,
                "mouth_center_uv": [float(v) for v in smoothed_center.tolist()],
                "mouth_left_uv": [float(v) for v in sc_left.tolist()],
                "mouth_right_uv": [float(v) for v in sc_right.tolist()],
                "mouth_angle_rad": sc_angle,
                "sc_port_center_uv": [float(v) for v in smoothed_center.tolist()],
                "sc_port_axis_left_uv": [float(v) for v in sc_left.tolist()],
                "sc_port_axis_right_uv": [float(v) for v in sc_right.tolist()],
                "sc_port_axis_angle_rad": sc_angle,
            }
            return data
        short_edges = edges[:2]
        mids = [(0.5 * (edge[1] + edge[2]), edge) for edge in short_edges]
        mids.sort(key=lambda item: (float(item[0][0]), float(item[0][1])))
        front = mids[0][1]
        a, b = front[1], front[2]
        if a[0] <= b[0]:
            front_left, front_right = a, b
        else:
            front_left, front_right = b, a
        front_center = 0.5 * (front_left + front_right)
        vec = front_right - front_left
        front_angle = float(np.arctan2(vec[1], vec[0])) if np.linalg.norm(vec) > 1e-6 else angle
        data = {
            **common,
            "tip_uv": [float(v) for v in front_center.tolist()],
            "front_center_uv": [float(v) for v in front_center.tolist()],
            "front_left_uv": [float(v) for v in front_left.tolist()],
            "front_right_uv": [float(v) for v in front_right.tolist()],
            "front_angle_rad": front_angle,
        }
        if self.family == "sc_plug":
            data.update({
                "sc_plug_tip_uv": data["tip_uv"],
                "sc_plug_axis_left_uv": data["front_left_uv"],
                "sc_plug_axis_right_uv": data["front_right_uv"],
                "sc_plug_axis_angle_rad": front_angle,
            })
        return data

    def to_dict(self) -> Dict:
        feature_bbox = [] if self.feature is None else [float(v) for v in self.feature.bbox_xyxy.tolist()]
        feature_center = [] if self.feature is None else [float(v) for v in self.feature.center_uv.tolist()]
        data = {
            "class_id": int(self.class_id),
            "raw_class_name": self.raw_class_name,
            "base_class_name": self.family,
            "class_name": self.instance_name,
            "instance_name": self.instance_name,
            "confidence": float(self.confidence),
            "raw_confidence": float(self.raw_confidence),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy.tolist()],
            "center_uv": [float(v) for v in self.center_uv.tolist()],
            "bbox_xyxy_feature": feature_bbox,
            "center_uv_feature": feature_center,
            "camera_name": self.camera_name,
            "track_id": int(self.track_id),
            "track_age": int(self.age),
            "track_hit_count": int(self.hits),
            "track_miss_count": int(self.misses),
            "track_confirmed": bool(self.confirmed),
            "feature_track_id": int(self.track_id if self.feature is not None else -1),
            "feature_quality_score": 0.0 if self.feature is None else float(self.feature.quality_score),
            "feature_tracked_count": 0 if self.feature is None else int(self.feature.tracked_count),
            "feature_inlier_count": 0 if self.feature is None else int(self.feature.inlier_count),
            "feature_mode": "none" if self.feature is None else self.feature.mode,
            "feature_point_ids_sample": [] if self.feature is None else list(self.feature.point_ids[:10]),
            "stamp_sec": float(self.stamp_sec),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
        }
        data.update(self.anchor_fields())
        for key in ("obb_cxcywh_deg", "obb_corners_uv", "mask_area_px"):
            if key in self.extra:
                data[key] = self.extra[key]
        return data


class PerCameraTracker:
    def __init__(self, owner, camera_name: str):
        self.owner = owner
        self.camera_name = camera_name
        self.tracks: List[Track] = []
        self.next_id = 1
        self.sfp_number_map: Dict[int, int] = {}
        self.sfp_candidate_map: Optional[Dict[int, int]] = None
        self.sfp_candidate_frames = 0

    def update(self, detections: List[Detection], frame: np.ndarray, now: float) -> List[Dict]:
        feature_boxes: Dict[int, Optional[np.ndarray]] = {}
        for track in self.tracks:
            track.predict(now)
            feature_boxes[track.track_id] = None
            if track.feature is not None and self.owner.feature_enabled(track.family):
                feature_boxes[track.track_id] = track.feature.update(frame, track.bbox_xyxy, now)

        matched_tracks, matched_dets = set(), set()
        detections_by_family: Dict[str, List[Detection]] = {}
        for det in detections:
            if det.confidence >= self.owner.family_track_low_conf(det.family):
                detections_by_family.setdefault(det.family, []).append(det)

        for family in FAMILY_ORDER:
            tracks = [t for t in self.tracks if t.family == family]
            dets = detections_by_family.get(family, [])
            for ti, di in self.associate(tracks, dets):
                track, det = tracks[ti], dets[di]
                track.update(det, now, self.owner, frame, feature_boxes.get(track.track_id))
                matched_tracks.add(track.track_id)
                matched_dets.add(id(det))

        for track in self.tracks:
            if track.track_id not in matched_tracks:
                track.mark_missed(feature_boxes.get(track.track_id))

        for det in detections:
            if id(det) in matched_dets or det.confidence < self.owner.family_track_new_conf(det.family):
                continue
            self.tracks.append(Track.create(self.next_id, det, now, self.owner, frame))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.owner.track_max_misses]
        return self.public_detections()

    def associate(self, tracks: List[Track], dets: List[Detection]) -> List[Tuple[int, int]]:
        if not tracks or not dets:
            return []
        costs = np.full((len(tracks), len(dets)), 1e6, dtype=np.float64)
        for ti, track in enumerate(tracks):
            for di, det in enumerate(dets):
                iou = bbox_iou(track.bbox_xyxy, det.bbox_xyxy)
                dist = float(np.linalg.norm(track.center_uv - det.center_uv))
                if iou < self.owner.track_iou_gate and dist > self.owner.track_center_gate_px:
                    continue
                costs[ti, di] = 0.55 * min(1.0, dist / max(1.0, self.owner.track_center_gate_px)) + 0.35 * (1.0 - iou) + 0.10 * (1.0 - det.confidence)
        matches = []
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(costs)
            for r, c in zip(rows, cols):
                if costs[r, c] < 1e5:
                    matches.append((int(r), int(c)))
            return matches
        used_t, used_d = set(), set()
        candidates = sorted((float(costs[ti, di]), ti, di) for ti in range(costs.shape[0]) for di in range(costs.shape[1]) if costs[ti, di] < 1e5)
        for _, ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        return matches

    def publishable(self, family: str) -> List[Track]:
        publish_conf, min_raw = self.owner.family_threshold(family)
        out = []
        for track in self.tracks:
            if track.family != family or not track.confirmed:
                continue
            feature_good = track.feature is not None and track.feature.quality_score >= self.owner.feature_confirmed_quality_min
            if track.confidence >= publish_conf and (track.raw_confidence >= min_raw or feature_good or track.misses <= 1):
                out.append(track)
        return sorted(out, key=lambda t: (-t.confidence, -t.hits, t.misses, t.track_id))

    def public_detections(self) -> List[Dict]:
        out: List[Dict] = []
        for family in ("task_board", "nic_card", "sfp_module", "sc_plug"):
            tracks = self.publishable(family)
            if tracks:
                tracks[0].instance_name = family
                out.append(tracks[0].to_dict())

        sc_tracks = self.order_sc(self.publishable("sc_port"))[:1]
        for track in sc_tracks:
            track.instance_name = "sc_port_0"
            data = track.to_dict()
            data["class_name"] = "sc_port_0"
            data["sc_numbering_source"] = "tracked_single_sc_port_0"
            out.append(data)

        sfp_tracks = self.publishable("sfp_port")[:2]
        if len(sfp_tracks) == 1:
            sfp_tracks[0].instance_name = "sfp_port"
            data = sfp_tracks[0].to_dict()
            data["sfp_tracker_assigned_label"] = "sfp_port"
            out.append(data)
        elif len(sfp_tracks) >= 2:
            for track, name, source, score in self.assign_sfp(sfp_tracks[:2], out):
                track.instance_name = name
                data = track.to_dict()
                data["sfp_numbering_source"] = source
                data["sfp_tracker_assigned_label"] = name
                if score is not None:
                    data["sfp_edge_score"] = float(score)
                out.append(data)
        return sorted(out, key=lambda d: (FAMILY_ORDER.get(str(d.get("base_class_name", "")), 99), self.owner.instance_sort_index(d.get("instance_name", "")), -float(d.get("confidence", 0.0))))

    def order_sc(self, tracks: List[Track]) -> List[Track]:
        if self.owner.sc_order_axis == "y":
            return sorted(tracks, key=lambda t: (float(t.center_uv[1]), float(t.center_uv[0]), t.track_id))
        return sorted(tracks, key=lambda t: (float(t.center_uv[0]), float(t.center_uv[1]), t.track_id))

    def assign_sfp(self, tracks: List[Track], public: List[Dict]) -> List[Tuple[Track, str, str, Optional[float]]]:
        source = "image_order"
        scores = {t.track_id: None for t in tracks}
        ref = self.best_public(public, "task_board") or self.best_public(public, "nic_card")
        if ref is not None and len(ref.get("bbox_xyxy", [])) == 4:
            source = "task_board_edges" if ref.get("base_class_name") == "task_board" else "nic_card_bbox"
            scored = []
            for track in tracks:
                score = self.two_nearest_edge_sum(track.center_uv, ref["bbox_xyxy"])
                scores[track.track_id] = score
                scored.append((score, track))
            scored.sort(key=lambda item: item[0])
            proposed = {scored[1][1].track_id: 0, scored[0][1].track_id: 1}
            if abs(scored[1][0] - scored[0][0]) < self.owner.sfp_number_margin_px and self.sfp_number_map:
                proposed = {tid: idx for tid, idx in self.sfp_number_map.items() if tid in {t.track_id for t in tracks}}
                for track in tracks:
                    proposed.setdefault(track.track_id, 1 if 0 in proposed.values() else 0)
        else:
            ordered = sorted(tracks, key=lambda t: (float(t.center_uv[1]), float(t.center_uv[0]), t.track_id))
            proposed = {ordered[0].track_id: 0, ordered[1].track_id: 1}

        ids = {t.track_id for t in tracks}
        if set(self.sfp_number_map.keys()) != ids:
            self.sfp_number_map = {}
            self.sfp_candidate_map = None
            self.sfp_candidate_frames = 0
        if self.sfp_number_map and proposed != self.sfp_number_map:
            if proposed == self.sfp_candidate_map:
                self.sfp_candidate_frames += 1
            else:
                self.sfp_candidate_map = dict(proposed)
                self.sfp_candidate_frames = 1
            if self.sfp_candidate_frames >= self.owner.sfp_number_switch_frames:
                self.sfp_number_map = dict(proposed)
                self.sfp_candidate_map = None
                self.sfp_candidate_frames = 0
        elif not self.sfp_number_map:
            self.sfp_number_map = dict(proposed)
        else:
            self.sfp_candidate_map = None
            self.sfp_candidate_frames = 0
        return [(track, f"sfp_port_{self.sfp_number_map.get(track.track_id, 0)}", source, scores.get(track.track_id)) for track in sorted(tracks, key=lambda t: self.sfp_number_map.get(t.track_id, 99))]

    @staticmethod
    def best_public(public: List[Dict], family: str) -> Optional[Dict]:
        items = [d for d in public if d.get("base_class_name") == family]
        return max(items, key=lambda d: float(d.get("confidence", 0.0))) if items else None

    @staticmethod
    def two_nearest_edge_sum(point_uv: np.ndarray, bbox) -> float:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        px, py = [float(v) for v in point_uv[:2]]
        return float(sum(sorted([abs(px - x1), abs(px - x2), abs(py - y1), abs(py - y2)])[:2]))


class YoloV12MultiCameraDetector(Node):
    def __init__(self):
        super().__init__("yolov12_multicamera_detector")
        # Defer heavy imports until node initialization to avoid discovery-time delays
        try:
            import torch
        except Exception:
            torch = None
        try:
            from ultralytics import YOLO
        except Exception:
            YOLO = None
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.model_path = str(Path(__file__).resolve().parents[1] / "models" / "yolo_v26_obb.pt")
        self.device = self.resolve_device("auto")
        for name, value in PARAMS.items():
            setattr(self, name, value)
        self.min_period = 1.0 / self.max_hz
        self.sfp_consensus_last_time = 0.0
        self.sfp_consensus_signature: Optional[Tuple[Tuple[str, int], ...]] = None
        self.sfp_left_default_label: int = int(getattr(self, "sfp_left_default_label", 1))
        self.sfp_all_cameras_override_s: float = float(getattr(self, "sfp_all_cameras_override_s", 10.0))
        self.sc_consensus_labels: List[str] = []
        self.sc_consensus_candidate_labels: Optional[List[str]] = None
        self.sc_consensus_last_time = 0.0
        self.sc_consensus_flip_count = 0
        self.feature_families = FEATURE_FAMILIES
        self.family_aliases = FAMILY_ALIASES
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_frames: Dict[str, Optional[Image]] = {cam: None for cam in CAMERAS}
        self.camera_k: Dict[str, Optional[np.ndarray]] = {cam: None for cam in CAMERAS}
        self.camera_info_frame_ids: Dict[str, str] = {cam: "" for cam in CAMERAS}
        self.camera_info_sizes: Dict[str, Tuple[int, int]] = {cam: (0, 0) for cam in CAMERAS}
        self.camera_info_logged = {cam: False for cam in CAMERAS}
        self.last_infer_time = {cam: 0.0 for cam in CAMERAS}
        self.trackers = {cam: PerCameraTracker(self, cam) for cam in CAMERAS}
        self.latest_outputs: Dict[str, Dict] = {}
        self.last_triangulated_poses: Dict[Tuple[str, str], Dict] = {}
        self.sfp_last_log_labels: Dict[Tuple[str, int], str] = {}
        self.sfp_consensus_left_label: Optional[int] = int(self.sfp_left_default_label)
        self.sfp_override_candidate_label: Optional[int] = None
        self.sfp_override_candidate_start: Optional[float] = None
        self.module_tri_last_log_time = 0.0
        self._sfp_port_orientations: Dict[str, np.ndarray] = {}  # child_frame_id -> xyzw quat
        self._object_orientations: Dict[str, np.ndarray] = {}
        self._sfp_orient_last_log_time = 0.0
        self._gripper_tf_last_log_time = 0.0
        self._gripper_tf_warned_missing: set[str] = set()
        self._sfp_module_gripper_orient_last_log_time = 0.0
        self._sc_plug_pnp_last_log_time = 0.0
        self._sc_plug_prior_last_log_time = 0.0
        # Latest SFP port axes stored when both ports are triangulated.
        # Used as the preferred x_hint for module orientation so both frames
        # share the same lateral reference (matching port convention).
        self._sfp_port_x_axis: Optional[np.ndarray] = None   # lateral  (red)
        self._sfp_port_z_axis: Optional[np.ndarray] = None   # insertion (blue)
        self._sfp_port_quat: Optional[np.ndarray] = None
        # SC port axes — persisted across frames so SC plug can always reference them.
        self._sc_port_x_axis: Optional[np.ndarray] = None   # mouth/lateral axis (red)
        self._sc_port_quat: Optional[np.ndarray] = None
        self._sc_port_hsv_flip_applied: bool = False
        # Locking state machine: prevents frame-to-frame flip stuttering.
        self._sc_port_hsv_flip_locked: Optional[bool] = None
        self._sc_port_hsv_flip_candidate: Optional[bool] = None
        self._sc_port_hsv_flip_candidate_start: Optional[float] = None
        self._sc_port_hsv_flip_last_update: float = 0.0
        self._sc_port_hsv_flip_required_s: float = 2.0
        self._sc_port_hsv_flip_margin_ratio: float = 1.75
        self._sc_port_hsv_flip_min_pixels: int = 20
        # SC plug yaw candidate lock: prevents 180-deg flip stuttering.
        self._sc_plug_yaw_candidate_locked: Optional[str] = None
        self._sc_plug_yaw_candidate_pending: Optional[str] = None
        self._sc_plug_yaw_candidate_start: Optional[float] = None
        self._sc_plug_yaw_candidate_required_s: float = 1.0
        self._sc_port_orient_last_log_time = 0.0
        self._sc_plug_orient_last_log_time = 0.0
        # Multi-camera SC port orientation consensus state machine.
        # Separate from the per-camera HSV-only lock so both evidence sources
        # (HSV + task-board) from all cameras are jointly considered.
        self._sc_orient_consensus_locked_flip: Optional[bool] = None
        self._sc_orient_consensus_candidate_flip: Optional[bool] = None
        self._sc_orient_consensus_candidate_start: Optional[float] = None
        self._sc_orient_consensus_required_s: float = 1.5
        self._sc_orient_consensus_last_update: float = 0.0

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        if YOLO is None:
            raise ImportError("ultralytics.YOLO could not be imported at runtime")
        self.model = YOLO(self.model_path)
        try:
            # Ensure torch is available when moving model to device
            if torch is None:
                import torch
                torch = globals().get("torch", torch)
            self.model.to(self.device)
        except Exception as exc:
            if self.device != "cpu":
                self.get_logger().warn(f"Falling back to CPU because moving model to {self.device} failed: {exc}")
                self.device = "cpu"
                self.model.to(self.device)
            else:
                raise

        self.subs = {cam: self.create_subscription(Image, f"/{cam}_camera/image", lambda msg, c=cam: self.image_cb(c, msg), 10) for cam in CAMERAS}
        self.info_subs = {cam: self.create_subscription(CameraInfo, f"/{cam}_camera/camera_info", lambda msg, c=cam: self.info_cb(c, msg), 10) for cam in CAMERAS}
        self.annotated_pubs = {cam: self.create_publisher(Image, f"/{cam}_camera/yolo/annotated", 10) for cam in CAMERAS}
        self.json_pubs = {cam: self.create_publisher(String, f"/{cam}_camera/yolo/detections_json", 10) for cam in CAMERAS}
        self.classes_pubs = {cam: self.create_publisher(String, f"/{cam}_camera/yolo/classes", 10) for cam in CAMERAS}
        self.pose_pubs = {cam: self.create_publisher(PoseArray, f"/{cam}_camera/yolo/triangulated_poses_base_link", 10) for cam in CAMERAS}
        self.global_pose_pub = self.create_publisher(PoseArray, "/yolo/triangulated_poses_base_link", 10)
        self.timer = self.create_timer(0.02, self.tick)
        frames = ",".join(f"{cam}:{CAMERA_OPTICAL_FRAMES[cam]}" for cam in CAMERAS)
        self.get_logger().info(
            f"YOLO detector started model={self.model_path} device={self.device} max_hz={self.max_hz:.1f} "
            f"triangulation_tf={BASE_FRAME} exact_camera_frames={frames} per_camera_k_required "
            f"min_cams={TRIANGULATION_MIN_CAMS} max_stamp_spread_s={TRIANGULATION_MAX_STAMP_SPREAD_S:.3f} "
            f"hold_last_valid=True sfp_left_default=sfp_port_{self.sfp_left_default_label} sfp_all_camera_override_s={self.sfp_all_cameras_override_s:.1f} byte_kalman={bool(getattr(self, 'byte_kalman_enable', True))} publish_tf=sfp_port,sc_port,sfp_module,gripper_aliases "
            f"raw_conf={self.raw_conf:.2f} port_conf={self.port_publish_conf:.2f} "
            f"module_conf={self.module_publish_conf:.2f} module_new={self.family_track_new_conf('sfp_module'):.2f} "
            f"module_anchor=center_uv module_stamp_spread_s={TRIANGULATION_MODULE_MAX_STAMP_SPREAD_S:.3f}"
        )

    def resolve_device(self, requested: str) -> str:
        import torch
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

    def family_confirm_hits(self, family: str) -> int:
        if family in {"task_board", "sfp_port", "sc_port", "sfp_module", "sc_plug"}:
            return 1
        return int(self.track_confirm_hits)

    def family_track_low_conf(self, family: str) -> float:
        if family == "task_board":
            return 0.20
        if family in {"sfp_port", "sc_port"}:
            return 0.20
        if family == "sfp_module":
            return 0.15
        if family == "sc_plug":
            return 0.30
        if family == "nic_card":
            return 0.35
        return float(self.track_low_conf)

    def family_track_new_conf(self, family: str) -> float:
        if family == "task_board":
            return 0.25
        if family in {"sfp_port", "sc_port"}:
            return 0.25
        if family == "sfp_module":
            return 0.18
        if family == "sc_plug":
            return 0.45
        if family == "nic_card":
            return 0.50
        return float(self.track_new_conf)

    def feature_enabled(self, family: str) -> bool:
        return self.enable_feature_tracking and self.feature_method not in {"", "none", "off", "false"} and family in self.feature_families

    def family_threshold(self, family: str) -> Tuple[float, float]:
        if family == "task_board":
            return self.task_board_publish_conf, 0.25
        if family in {"sfp_port", "sc_port"}:
            return self.port_publish_conf, 0.25
        if family == "sfp_module":
            return self.module_publish_conf, 0.15
        if family == "sc_plug":
            return self.plug_publish_conf, self.feature_confirmed_publish_conf
        return self.publish_conf, self.track_min_raw_for_publish

    def instance_sort_index(self, name: object) -> int:
        text = norm_name(name)
        if text.endswith("_0"):
            return 0
        if text.endswith("_1"):
            return 1
        try:
            return int(text.rsplit("_", 1)[-1])
        except Exception:
            return 99

    def family_for_name(self, class_name: object) -> Optional[str]:
        name = norm_name(class_name)
        for family, aliases in self.family_aliases.items():
            if name in aliases:
                return family
        return None

    def image_cb(self, camera_name: str, msg: Image) -> None:
        with self.lock:
            self.latest_frames[camera_name] = msg

    def info_cb(self, camera_name: str, msg: CameraInfo) -> None:
        k = np.asarray(list(msg.k), dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        valid = abs(fx) > 1e-9 and abs(fy) > 1e-9
        frame_id = str(getattr(msg.header, "frame_id", "")).strip()
        expected = CAMERA_OPTICAL_FRAMES[camera_name]
        with self.lock:
            self.camera_k[camera_name] = k if valid else None
            self.camera_info_frame_ids[camera_name] = frame_id
            self.camera_info_sizes[camera_name] = (int(msg.width), int(msg.height))
            should_log = not self.camera_info_logged[camera_name]
            if should_log:
                self.camera_info_logged[camera_name] = True
        if should_log:
            self.get_logger().info(
                f"YOLO_CAMERA_INFO cam={camera_name} topic=/{camera_name}_camera/camera_info "
                f"expected_tf_frame={expected} header_frame_id={frame_id or 'EMPTY'} "
                f"K_fx={fx:.6f} K_fy={fy:.6f} K_cx={cx:.6f} K_cy={cy:.6f} valid={valid}"
            )
            if frame_id and frame_id != expected:
                self.get_logger().warn(
                    f"YOLO_CAMERA_INFO_FRAME_MISMATCH cam={camera_name} header_frame_id={frame_id} "
                    f"expected_tf_frame={expected}; using expected_tf_frame only, no fallback"
                )

    def stamp_sec(self, stamp) -> float:
        return float(getattr(stamp, "sec", 0)) + 1e-9 * float(getattr(stamp, "nanosec", 0))

    def ros_time(self) -> float:
        """Return current ROS clock time in seconds (respects simulated clock)."""
        return float(self.get_clock().now().nanoseconds) / 1e9

    def tick(self) -> None:
        now = self.ros_time()
        with self.lock:
            frames = dict(self.latest_frames)
            camera_ks = {cam: (None if self.camera_k[cam] is None else self.camera_k[cam].copy()) for cam in CAMERAS}
            info_frame_ids = dict(self.camera_info_frame_ids)
            info_sizes = dict(self.camera_info_sizes)
        processed = {}
        for cam, msg in frames.items():
            if msg is None or now - self.last_infer_time[cam] < self.min_period:
                continue
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                raw = self.infer(frame, msg, cam)
                dets = [d for d in raw if d.family in FAMILY_ORDER]
                public = self.trackers[cam].update(dets, frame, now)
                # SC ports and SC plugs are intentionally published directly from raw YOLO output.
                # This bypasses tracker confirmation, confidence gates, consensus hold,
                # and feature-quality filtering only for SC detections. All other families
                # keep the existing tracked/publishable path unchanged.
                public = self.replace_sc_detections_with_raw(public, raw)
                processed[cam] = {"msg": msg, "frame": frame, "detections": public, "raw": raw}
                self.last_infer_time[cam] = now
            except Exception as exc:
                self.get_logger().error(f"{cam} inference failed: {exc}")
        if not processed:
            return
        self.latest_outputs.update(processed)
        self.normalize_sfp_consensus(self.latest_outputs)
        self.normalize_sc_consensus(self.latest_outputs)
        self.attach_triangulated_poses(self.latest_outputs, processed, camera_ks, info_frame_ids, info_sizes)
        for cam, data in processed.items():
            detections = data["detections"]
            classes = [str(det.get("instance_name", det.get("class_name", ""))) for det in detections]
            annotated = self.draw_detections(data["frame"], detections)
            if self.draw_feature_debug:
                annotated = self.draw_features(annotated, cam)
            if self.draw_raw_debug:
                annotated = self.draw_raw(annotated, data["raw"])
            text = f"sfp_lr_vote_left={self.sfp_consensus_left_label}" if self.sfp_consensus_signature else "sfp_consensus=none"
            cv2.putText(annotated, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            self.publish_outputs(cam, data["msg"], annotated, detections, classes)

    def infer(self, frame: np.ndarray, msg: Image, camera_name: str) -> List[Detection]:
        results = self.model.predict(source=frame, device=self.device, conf=self.raw_conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        if not results:
            return []
        result = results[0]
        names = getattr(result, "names", getattr(self.model, "names", {}))
        h, w = frame.shape[:2]
        stamp = self.stamp_sec(msg.header.stamp)
        out: List[Detection] = []

        obb = getattr(result, "obb", None)
        if obb is not None and getattr(obb, "data", None) is not None and len(obb) > 0:
            corners_all = to_numpy(getattr(obb, "xyxyxyxy", None))
            xywhr_all = to_numpy(getattr(obb, "xywhr", None))
            xyxy_all = to_numpy(getattr(obb, "xyxy", None))
            confs = to_numpy(getattr(obb, "conf", None))
            clss = to_numpy(getattr(obb, "cls", None))
            if confs is not None and clss is not None:
                for idx, (conf, cls_idx) in enumerate(zip(confs, clss.astype(int))):
                    raw_name = class_name_from_names(names, int(cls_idx))
                    family = self.family_for_name(raw_name)
                    if family is None:
                        continue
                    corners = None
                    if corners_all is not None and idx < len(corners_all):
                        corners = np.asarray(corners_all[idx], dtype=np.float64).reshape(4, 2)
                        bbox = np.asarray([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()], dtype=np.float64)
                    elif xyxy_all is not None and idx < len(xyxy_all):
                        bbox = np.asarray(xyxy_all[idx], dtype=np.float64).reshape(4)
                    else:
                        continue
                    bbox = clip_bbox(bbox, w, h)
                    extra = {}
                    if corners is not None:
                        extra["obb_corners_uv"] = [[float(x), float(y)] for x, y in corners.tolist()]
                    if xywhr_all is not None and idx < len(xywhr_all):
                        cx, cy, rw, rh, angle = [float(v) for v in np.asarray(xywhr_all[idx]).reshape(-1)[:5]]
                        extra["obb_cxcywh_deg"] = [cx, cy, rw, rh, float(np.degrees(angle))]
                    out.append(Detection(family, int(cls_idx), raw_name, float(conf), bbox, bbox_center(bbox), stamp, camera_name, w, h, extra))
                return out

        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return out
        xyxy = to_numpy(boxes.xyxy)
        confs = to_numpy(getattr(boxes, "conf", None))
        clss = to_numpy(getattr(boxes, "cls", None))
        if xyxy is None or confs is None or clss is None:
            return out
        for idx, (bbox_raw, conf, cls_idx) in enumerate(zip(xyxy, confs, clss.astype(int))):
            raw_name = class_name_from_names(names, int(cls_idx))
            family = self.family_for_name(raw_name)
            if family is None:
                continue
            bbox = clip_bbox(np.asarray(bbox_raw, dtype=np.float64).reshape(4), w, h)
            out.append(Detection(family, int(cls_idx), raw_name, float(conf), bbox, bbox_center(bbox), stamp, camera_name, w, h, {}))
        return out

    def det_center_xy(self, det: Dict) -> Tuple[float, float]:
        center = det.get("center_uv")
        if isinstance(center, list) and len(center) >= 2:
            return float(center[0]), float(center[1])
        bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
        return (float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5

    def _magenta_centroid_uv(self, image_bgr: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], int]:
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            return None, 0
        try:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        except Exception:
            return None, 0
        lower1 = np.asarray([140, 40, 40], dtype=np.uint8)
        upper1 = np.asarray([179, 255, 255], dtype=np.uint8)
        lower2 = np.asarray([0, 40, 40], dtype=np.uint8)
        upper2 = np.asarray([10, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) < int(self._sc_port_hsv_flip_min_pixels):
            return None, 0
        pts_np = pts.reshape(-1, 2).astype(np.float64)
        return np.asarray([float(np.mean(pts_np[:, 0])), float(np.mean(pts_np[:, 1]))], dtype=np.float64), int(len(pts_np))

    def _project_gripper_tcp_uv(self, cam: str) -> Optional[np.ndarray]:
        k = self.camera_k.get(cam)
        if k is None:
            return None
        try:
            tf_gripper = self.tf_buffer.lookup_transform(CAMERA_OPTICAL_FRAMES[cam], GRIPPER_TF_SOURCE_FRAMES[0], Time())
        except TransformException:
            return None
        p_cam, _q_cam, _ = transform_to_arrays(tf_gripper)
        if not np.all(np.isfinite(p_cam)) or float(p_cam[2]) <= 1e-4:
            return None
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        return np.asarray([
            fx * float(p_cam[0]) / float(p_cam[2]) + cx,
            fy * float(p_cam[1]) / float(p_cam[2]) + cy,
        ], dtype=np.float64)

    def _assign_sc_pair_labels(self, cam: str, ports: List[Dict], frame: Optional[np.ndarray]) -> Tuple[Dict[int, str], str, Dict]:
        if len(ports) < 2:
            return ({0: "sc_port"} if ports else {}), "single_or_none", {}

        centers = [np.asarray(self.det_center_xy(port), dtype=np.float64) for port in ports[:2]]
        separation = float(np.linalg.norm(centers[0] - centers[1]))
        magenta_uv, magenta_px = self._magenta_centroid_uv(frame)
        if magenta_uv is not None:
            dists = [float(np.linalg.norm(center - magenta_uv)) for center in centers]
            near_pair = min(dists) <= max(60.0, 0.80 * separation)
            distinct_side = abs(dists[0] - dists[1]) >= max(8.0, 0.08 * separation)
            if near_pair and distinct_side:
                idx_sc1 = int(np.argmin(dists))
                idx_sc0 = 1 - idx_sc1
                return (
                    {idx_sc0: "sc_port_0", idx_sc1: "sc_port_1"},
                    "magenta_hsv_nearest_is_sc_port_1",
                    {"magenta_uv": [float(magenta_uv[0]), float(magenta_uv[1])], "magenta_px": magenta_px, "magenta_dists_px": dists},
                )

        tcp_uv = self._project_gripper_tcp_uv(cam)
        if tcp_uv is not None:
            dists = [float(np.linalg.norm(center - tcp_uv)) for center in centers]
            idx_sc0 = int(np.argmin(dists))
            idx_sc1 = 1 - idx_sc0
            return (
                {idx_sc0: "sc_port_0", idx_sc1: "sc_port_1"},
                "gripper_tcp_nearest_is_sc_port_0",
                {"tcp_uv": [float(tcp_uv[0]), float(tcp_uv[1])], "tcp_dists_px": dists},
            )

        ordered = sorted(range(2), key=lambda idx: (centers[idx][0], centers[idx][1]))
        return (
            {ordered[0]: "sc_port_0", ordered[1]: "sc_port_1"},
            "image_order_fallback_no_magenta_no_tcp",
            {"pair_separation_px": separation},
        )

    def raw_sc_detection_to_dict(self, det: Detection, instance_name: str, source: str) -> Dict:
        """Convert a raw YOLO SC-port Detection into the public JSON schema.

        This is deliberately used only for sc_port so SC detections are published
        exactly from the model output after only the global YOLO raw_conf/NMS stage.
        No tracker confirmation, confidence threshold, Kalman/feature gate, or
        publish confidence filtering is applied here.
        """
        bbox = np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(4)
        center = np.asarray(det.center_uv, dtype=np.float64).reshape(2)
        corners = None
        if isinstance(det.extra.get("obb_corners_uv"), list) and len(det.extra["obb_corners_uv"]) == 4:
            try:
                corners = np.asarray(det.extra["obb_corners_uv"], dtype=np.float64).reshape(4, 2)
            except Exception:
                corners = None
        if corners is None:
            corners = rect_corners_from_bbox(bbox)

        edges = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            edges.append((float(np.linalg.norm(b - a)), a, b))
        edges.sort(key=lambda item: item[0])
        long = edges[-1]
        vec = long[2] - long[1]
        length = max(1e-6, float(np.linalg.norm(vec)))
        axis = vec / length
        left = center - 0.5 * length * axis
        right = center + 0.5 * length * axis
        if left[0] > right[0]:
            left, right = right, left
        angle = float(np.arctan2(axis[1], axis[0]))

        data: Dict = {
            "class_id": int(det.class_id),
            "raw_class_name": det.raw_class_name,
            "base_class_name": "sc_port",
            "class_name": instance_name,
            "instance_name": instance_name,
            "confidence": float(det.confidence),
            "raw_confidence": float(det.confidence),
            "bbox_xyxy": [float(v) for v in bbox.tolist()],
            "center_uv": [float(v) for v in center.tolist()],
            "bbox_xyxy_feature": [],
            "center_uv_feature": [],
            "camera_name": det.camera_name,
            "track_id": -1,
            "track_age": 1,
            "track_hit_count": 1,
            "track_miss_count": 0,
            "track_confirmed": True,
            "feature_track_id": -1,
            "feature_quality_score": 0.0,
            "feature_tracked_count": 0,
            "feature_inlier_count": 0,
            "feature_mode": "raw_yolo_no_sc_filter",
            "feature_point_ids_sample": [],
            "stamp_sec": float(det.stamp_sec),
            "image_width": int(det.image_width),
            "image_height": int(det.image_height),
            "servo_anchor_valid": True,
            "servo_anchor_source": "raw_yolo_no_sc_filter",
            "servo_anchor_quality": float(det.confidence),
            "mouth_center_uv": [float(v) for v in center.tolist()],
            "mouth_left_uv": [float(v) for v in left.tolist()],
            "mouth_right_uv": [float(v) for v in right.tolist()],
            "mouth_angle_rad": angle,
            "sc_port_center_uv": [float(v) for v in center.tolist()],
            "sc_port_axis_left_uv": [float(v) for v in left.tolist()],
            "sc_port_axis_right_uv": [float(v) for v in right.tolist()],
            "sc_port_axis_angle_rad": angle,
            "sc_numbering_source": source,
            "sc_raw_publish_no_filter": True,
        }
        for key in ("obb_cxcywh_deg", "obb_corners_uv", "mask_area_px"):
            if key in det.extra:
                data[key] = det.extra[key]
        return data

    def raw_sc_plug_detection_to_dict(self, det: Detection, instance_name: str, source: str) -> Dict:
        """Convert a raw YOLO SC-plug Detection into the public JSON schema.

        This mirrors the existing raw SC-port publishing path, but keeps plug-specific
        fields such as tip/front anchors. It is deliberately used only for sc_plug.
        """
        bbox = np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(4)
        center = np.asarray(det.center_uv, dtype=np.float64).reshape(2)
        corners = None
        if isinstance(det.extra.get("obb_corners_uv"), list) and len(det.extra["obb_corners_uv"]) == 4:
            try:
                corners = np.asarray(det.extra["obb_corners_uv"], dtype=np.float64).reshape(4, 2)
            except Exception:
                corners = None
        if corners is None:
            corners = rect_corners_from_bbox(bbox)

        edges = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            edges.append((float(np.linalg.norm(b - a)), a, b))
        edges.sort(key=lambda item: item[0])
        short_edges = edges[:2]
        mids = [(0.5 * (edge[1] + edge[2]), edge) for edge in short_edges]
        mids.sort(key=lambda item: (float(item[0][0]), float(item[0][1])))
        front = mids[0][1]
        a, b = front[1], front[2]
        if a[0] <= b[0]:
            front_left, front_right = a, b
        else:
            front_left, front_right = b, a
        front_center = 0.5 * (front_left + front_right)
        vec = front_right - front_left
        front_angle = float(np.arctan2(vec[1], vec[0])) if np.linalg.norm(vec) > 1e-6 else 0.0

        data: Dict = {
            "class_id": int(det.class_id),
            "raw_class_name": det.raw_class_name,
            "base_class_name": "sc_plug",
            "class_name": instance_name,
            "instance_name": instance_name,
            "confidence": float(det.confidence),
            "raw_confidence": float(det.confidence),
            "bbox_xyxy": [float(v) for v in bbox.tolist()],
            "center_uv": [float(v) for v in center.tolist()],
            "bbox_xyxy_feature": [],
            "center_uv_feature": [],
            "camera_name": det.camera_name,
            "track_id": -1,
            "track_age": 1,
            "track_hit_count": 1,
            "track_miss_count": 0,
            "track_confirmed": True,
            "feature_track_id": -1,
            "feature_quality_score": 0.0,
            "feature_tracked_count": 0,
            "feature_inlier_count": 0,
            "feature_mode": "raw_yolo_no_sc_plug_filter",
            "feature_point_ids_sample": [],
            "stamp_sec": float(det.stamp_sec),
            "image_width": int(det.image_width),
            "image_height": int(det.image_height),
            "servo_anchor_valid": True,
            "servo_anchor_source": "raw_yolo_no_sc_plug_filter",
            "servo_anchor_quality": float(det.confidence),
            "tip_uv": [float(v) for v in front_center.tolist()],
            "front_center_uv": [float(v) for v in front_center.tolist()],
            "front_left_uv": [float(v) for v in front_left.tolist()],
            "front_right_uv": [float(v) for v in front_right.tolist()],
            "front_angle_rad": front_angle,
            "sc_plug_tip_uv": [float(v) for v in front_center.tolist()],
            "sc_plug_axis_left_uv": [float(v) for v in front_left.tolist()],
            "sc_plug_axis_right_uv": [float(v) for v in front_right.tolist()],
            "sc_plug_axis_angle_rad": front_angle,
            "sc_plug_raw_publish_no_filter": True,
            "sc_plug_numbering_source": source,
        }
        for key in ("obb_cxcywh_deg", "obb_corners_uv", "mask_area_px"):
            if key in det.extra:
                data[key] = det.extra[key]
        return data

    def replace_sc_detections_with_raw(self, public: List[Dict], raw: List[Detection]) -> List[Dict]:
        """Merge SC detections into the public detection list.

        SC port: keep up to two raw YOLO detections so pair numbering can label
                 sc_port_0/sc_port_1 before triangulation.
        SC plug: always use the highest-confidence raw YOLO detection (existing behaviour).
        All other families pass through unchanged.
        """
        other = [d for d in public if not self.is_family(d, "sc_port") and not self.is_family(d, "sc_plug")]
        tracked_sc_ports = [d for d in public if self.is_family(d, "sc_port")]

        raw_sc_ports = sorted(
            [d for d in raw if d.family == "sc_port"],
            key=lambda d: (-float(d.confidence), float(d.center_uv[0]), float(d.center_uv[1])),
        )[: int(self.sc_max_ports)]
        if raw_sc_ports:
            sc_port_out = [
                self.raw_sc_detection_to_dict(d, f"sc_port_{idx}", "raw_yolo_pair_candidate")
                for idx, d in enumerate(raw_sc_ports)
            ]
        else:
            sc_port_out = sorted(
                tracked_sc_ports,
                key=lambda d: (-float(d.get("confidence", 0.0)), self.det_center_xy(d)[0], self.det_center_xy(d)[1]),
            )[: int(self.sc_max_ports)]

        # SC plug: always raw (unchanged behaviour).
        raw_sc_plugs = sorted(
            [d for d in raw if d.family == "sc_plug"],
            key=lambda d: (-float(d.confidence), float(d.center_uv[0]), float(d.center_uv[1])),
        )
        sc_plug_out = [self.raw_sc_plug_detection_to_dict(raw_sc_plugs[0], "sc_plug", "raw_yolo_no_filter")] if raw_sc_plugs else []

        out = other + sc_port_out + sc_plug_out
        return sorted(
            out,
            key=lambda d: (
                FAMILY_ORDER.get(str(d.get("base_class_name", "")), 99),
                self.instance_sort_index(d.get("instance_name", "")),
                -float(d.get("confidence", 0.0)),
            ),
        )

    def replace_sc_ports_with_raw(self, public: List[Dict], raw: List[Detection]) -> List[Dict]:
        return self.replace_sc_detections_with_raw(public, raw)

    def is_family(self, det: Dict, family: str) -> bool:
        return str(det.get("base_class_name", "")) == family or str(det.get("class_name", "")).startswith(family) or str(det.get("instance_name", "")).startswith(family)

    def sfp_numeric_label(self, det: Dict) -> Optional[int]:
        for key in ("instance_name", "class_name", "raw_class_name"):
            text = norm_name(det.get(key, ""))
            if text.startswith("sfp_port_") and text.rsplit("_", 1)[-1].isdigit():
                value = int(text.rsplit("_", 1)[-1])
                if value in (0, 1):
                    return value
        return None

    def normalize_sfp_consensus(self, outputs: Dict[str, Dict]) -> None:
        pairs: Dict[str, Tuple[Dict, Dict]] = {}
        now = self.ros_time()

        for cam, data in outputs.items():
            ports = [d for d in data.get("detections", []) if self.is_family(d, "sfp_port")]
            if len(ports) < 2:
                continue
            ports = sorted(ports, key=lambda d: (-float(d.get("confidence", 0.0)), int(d.get("track_id", 999))))[:2]
            ports = sorted(ports, key=lambda d: (self.det_center_xy(d)[0], self.det_center_xy(d)[1]))
            left, right = ports[0], ports[1]
            lx, ly = self.det_center_xy(left)
            rx, ry = self.det_center_xy(right)
            separation = float(np.hypot(rx - lx, ry - ly))
            if separation < 10.0:
                continue
            pairs[cam] = (left, right)
            for det, side in ((left, "left"), (right, "right")):
                det.update({
                    "sfp_lr_side": side,
                    "sfp_pair_left_minus_right_dx_px": float(lx - rx),
                    "sfp_pair_separation_px": separation,
                    "sfp_cross_camera_vote_available": False,
                })

        if len(pairs) < self.sfp_consensus_min_cams:
            if not (self.sfp_consensus_signature and now - self.sfp_consensus_last_time < self.sfp_consensus_hold_s):
                self.sfp_consensus_signature = None
                self.sfp_override_candidate_label = None
                self.sfp_override_candidate_start = None
            return

        camera_raw_left_labels: Dict[str, Optional[int]] = {}
        for cam, (left, _right) in pairs.items():
            tracker_label = left.get("sfp_tracker_assigned_label")
            if tracker_label is not None:
                camera_raw_left_labels[cam] = self.sfp_numeric_label({"instance_name": tracker_label})
            else:
                camera_raw_left_labels[cam] = self.sfp_numeric_label(left)

        default_left_label = int(getattr(self, "sfp_left_default_label", 1))
        current_left_label = default_left_label if self.sfp_consensus_left_label not in (0, 1) else int(self.sfp_consensus_left_label)
        valid_raw_labels = [label for label in camera_raw_left_labels.values() if label in (0, 1)]
        override_label: Optional[int] = None
        if len(valid_raw_labels) == len(pairs) and len(valid_raw_labels) >= int(self.sfp_consensus_min_cams):
            first_label = int(valid_raw_labels[0])
            if all(int(label) == first_label for label in valid_raw_labels):
                override_label = first_label

        if override_label is not None and override_label != current_left_label:
            if self.sfp_override_candidate_label == override_label:
                if self.sfp_override_candidate_start is not None:
                    elapsed = now - self.sfp_override_candidate_start
                else:
                    self.sfp_override_candidate_start = now
                    elapsed = 0.0
            else:
                self.sfp_override_candidate_label = override_label
                self.sfp_override_candidate_start = now
                elapsed = 0.0
            if elapsed >= float(self.sfp_all_cameras_override_s):
                self.get_logger().info(
                    f"YOLO_SFP_CONSENSUS_OVERRIDE_APPLIED old=sfp_port_{current_left_label} "
                    f"new=sfp_port_{override_label} duration={elapsed:.2f}s "
                    f"required={float(self.sfp_all_cameras_override_s):.2f}s cameras={sorted(pairs.keys())}"
                )
                current_left_label = int(override_label)
                self.sfp_override_candidate_label = None
                self.sfp_override_candidate_start = None
        else:
            self.sfp_override_candidate_label = None
            self.sfp_override_candidate_start = None

        left_label = int(current_left_label)
        right_label = 1 - left_label
        signature = tuple(sorted((cam, left_label) for cam in pairs.keys()))
        stable = self.sfp_consensus_left_label == left_label
        self.sfp_consensus_left_label = left_label
        self.sfp_consensus_signature = signature
        self.sfp_consensus_last_time = now

        for cam, (left, right) in pairs.items():
            raw_left = camera_raw_left_labels.get(cam)
            for det, new_idx, side in ((left, left_label, "left"), (right, right_label, "right")):
                new = f"sfp_port_{new_idx}"
                old = str(det.get("instance_name", det.get("class_name", "")))
                role = "left_camera_frame_is_sfp_port_1_locked_with_all_camera_10s_override"
                det.update({
                    "instance_name": new,
                    "class_name": new,
                    "base_class_name": "sfp_port",
                    "sfp_order_role": role,
                    "sfp_lr_side": side,
                    "sfp_consensus_stable": stable,
                    "sfp_consensus_left_label": f"sfp_port_{left_label}",
                    "sfp_consensus_right_label": f"sfp_port_{right_label}",
                    "sfp_camera_raw_left_label": None if raw_left is None else f"sfp_port_{int(raw_left)}",
                    "sfp_override_candidate_label": None if self.sfp_override_candidate_label is None else f"sfp_port_{int(self.sfp_override_candidate_label)}",
                    "sfp_override_candidate_elapsed_s": 0.0 if self.sfp_override_candidate_start is None else float(now - self.sfp_override_candidate_start),
                    "sfp_override_required_s": float(self.sfp_all_cameras_override_s),
                    "sfp_cross_camera_vote_available": True,
                })
                log_key = (cam, int(det.get("track_id", -1)))
                if old != new and self.sfp_last_log_labels.get(log_key) != new:
                    self.get_logger().info(
                        f"YOLO_SFP_LEFT_LOCK camera={cam} side={side} old={old} new={new} "
                        f"left_label=sfp_port_{left_label} raw_left={raw_left} "
                        f"candidate={self.sfp_override_candidate_label}"
                    )
                self.sfp_last_log_labels[log_key] = new

    def normalize_sc_consensus(self, outputs: Dict[str, Dict]) -> None:
        # SC naming only. With two ports, magenta proximity identifies SC_PORT_1.
        # If no magenta evidence is visible, the port nearest the gripper TCP is SC_PORT_0.
        camera_ports = {}
        for cam, data in outputs.items():
            ports = [d for d in data.get("detections", []) if self.is_family(d, "sc_port")]
            ports = sorted(
                ports,
                key=lambda d: (-float(d.get("confidence", 0.0)), self.det_center_xy(d)[0], self.det_center_xy(d)[1]),
            )[: int(self.sc_max_ports)]
            if ports:
                camera_ports[cam] = ports
        now = self.ros_time()
        if not camera_ports:
            if now - self.sc_consensus_last_time >= self.sc_consensus_hold_s:
                self.sc_consensus_labels = []
                self.sc_consensus_candidate_labels = None
                self.sc_consensus_flip_count = 0
            return

        max_count = max(len(ports) for ports in camera_ports.values())
        labels = ["sc_port"] if max_count == 1 else [f"sc_port_{idx}" for idx in range(min(max_count, int(self.sc_max_ports)))]
        self.sc_consensus_labels = list(labels)
        self.sc_consensus_last_time = now
        self.sc_consensus_candidate_labels = None
        self.sc_consensus_flip_count = 0

        for cam, ports in camera_ports.items():
            frame = (outputs.get(cam) or {}).get("frame")
            label_by_idx, source, dbg = self._assign_sc_pair_labels(cam, ports, frame)
            for idx, port in enumerate(ports):
                if idx not in label_by_idx:
                    continue
                old = str(port.get("instance_name", port.get("class_name", "")))
                new = label_by_idx[idx]
                port.update({
                    "instance_name": new,
                    "class_name": new,
                    "base_class_name": "sc_port",
                    "sc_order_role": f"{source}_candidate_{idx}",
                    "sc_numbering_source": source,
                    "sc_consensus_stable": True,
                    "sc_consensus_labels": list(labels),
                    "sc_pair_count": len(ports),
                })
                if dbg:
                    port["sc_pair_numbering_debug"] = dbg
                if old != new:
                    self.get_logger().info(
                        f"YOLO_SC_RELABEL camera={cam} order={idx} old={old} new={new} "
                        f"source={source} count={len(ports)}"
                    )


    def log_module_tri(self, text: str, min_period_s: float = 1.0) -> None:
        now = self.ros_time()
        if now - self.module_tri_last_log_time >= min_period_s:
            self.module_tri_last_log_time = now
            self.get_logger().info(text)

    def anchor_uv_for_pose(self, det: Dict) -> Tuple[float, float, str]:
        family = norm_name(det.get("base_class_name", det.get("class_name", "")))
        if family == "sfp_module":
            keys = ("center_uv",)
        elif family in {"sfp_port", "sc_port"}:
            keys = ("mouth_center_uv", "center_uv")
        elif family == "sc_plug":
            # SC plug TF position should come from the front/tip mating anchor so that
            # CheatCode receives the plug's functional geometry, not the plastic body center.
            keys = ("tip_uv", "front_center_uv", "center_uv")
        else:
            keys = ("tip_uv", "front_center_uv", "center_uv")
        for key in keys:
            value = det.get(key)
            if isinstance(value, list) and len(value) >= 2:
                return float(value[0]), float(value[1]), key
        x, y = self.det_center_xy(det)
        return x, y, "center_uv"

    def triangulation_key(self, det: Dict) -> Optional[Tuple[str, str]]:
        family = norm_name(det.get("base_class_name", det.get("class_name", "")))
        name = norm_name(det.get("instance_name", det.get("class_name", family)))
        if family not in {"sfp_port", "sc_port", "sfp_module", "sc_plug"}:
            return None
        if family == "sfp_port" and name not in {"sfp_port_0", "sfp_port_1"}:
            return None
        if family == "sc_port" and not name.startswith("sc_port"):
            return None
        if family == "sfp_module":
            return "sfp_module", "sfp_module"
        if family == "sc_plug":
            return "sc_plug", "sc_plug"
        return family, name

    def triangulation_child_frame(self, key: Tuple[str, str]) -> str:
        family, name = key
        return f"yolo_tri/{safe_frame_token(family)}/{safe_frame_token(name)}"

    def calibrated_bearing_ray(self, k: np.ndarray, u: float, v: float) -> Optional[np.ndarray]:
        if k is None:
            return None
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None
        ray = np.asarray([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)
        return ray / max(1e-12, float(np.linalg.norm(ray)))

    def camera_observation_for_triangulation(
        self,
        cam: str,
        image_msg: Image,
        det: Dict,
        k: Optional[np.ndarray],
        info_frame_id: str,
        info_size: Tuple[int, int],
    ) -> Optional[Dict]:
        if k is None:
            return None
        stamp = image_msg.header.stamp
        if int(getattr(stamp, "sec", 0)) == 0 and int(getattr(stamp, "nanosec", 0)) == 0:
            return None
        u, v, uv_source = self.anchor_uv_for_pose(det)
        ray_camera = self.calibrated_bearing_ray(k, u, v)
        if ray_camera is None:
            return None
        camera_frame = CAMERA_OPTICAL_FRAMES[cam]
        try:
            base_from_camera = self.tf_buffer.lookup_transform(BASE_FRAME, camera_frame, Time.from_msg(stamp))
        except TransformException:
            return None
        origin_base, _, base_R_camera = transform_to_arrays(base_from_camera)
        ray_base = base_R_camera @ ray_camera
        ray_base = ray_base / max(1e-12, float(np.linalg.norm(ray_base)))
        confidence = float(det.get("confidence", det.get("raw_confidence", 0.0)))
        feature_quality = float(det.get("feature_quality_score", 0.0))
        weight = max(0.05, confidence * confidence) * max(0.25, 0.50 + feature_quality)
        return {
            "cam": cam,
            "det": det,
            "k": k,
            "info_frame_id": info_frame_id,
            "info_size": info_size,
            "stamp": stamp,
            "stamp_sec": self.stamp_sec(stamp),
            "camera_frame": camera_frame,
            "origin_base": origin_base,
            "base_R_camera": base_R_camera,
            "ray_camera": ray_camera,
            "ray_base": ray_base,
            "u": float(u),
            "v": float(v),
            "uv_source": uv_source,
            "weight": float(weight),
        }

    def triangulate_rays_weighted(self, observations: List[Dict], key: Optional[Tuple[str, str]] = None) -> Optional[Dict]:
        if len(observations) < TRIANGULATION_MIN_CAMS:
            return None
        family = key[0] if key is not None else ""
        max_stamp_spread = TRIANGULATION_MODULE_MAX_STAMP_SPREAD_S if family == "sfp_module" else TRIANGULATION_MAX_STAMP_SPREAD_S
        max_ray_error_allowed = TRIANGULATION_MODULE_MAX_RAY_ERROR_M if family == "sfp_module" else TRIANGULATION_MAX_RAY_ERROR_M
        max_reproj_error_allowed = TRIANGULATION_MODULE_MAX_REPROJ_ERROR_PX if family == "sfp_module" else TRIANGULATION_MAX_REPROJ_ERROR_PX
        stamps = [float(o["stamp_sec"]) for o in observations]
        stamp_spread = max(stamps) - min(stamps)
        if stamp_spread > max_stamp_spread:
            return None
        max_angle = 0.0
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                dot = float(np.clip(np.dot(observations[i]["ray_base"], observations[j]["ray_base"]), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(abs(dot))))
                max_angle = max(max_angle, angle)
        if max_angle < TRIANGULATION_MIN_RAY_ANGLE_DEG:
            return None
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        eye = np.eye(3, dtype=np.float64)
        for obs in observations:
            d = obs["ray_base"].reshape(3, 1)
            c = obs["origin_base"].reshape(3)
            p = eye - d @ d.T
            w = float(obs["weight"])
            A += w * p
            b += w * (p @ c)
        cond = float(np.linalg.cond(A))
        if not np.isfinite(cond) or cond > TRIANGULATION_MAX_CONDITION:
            return None
        try:
            point = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(point)):
            return None
        ray_errors = []
        depths = []
        reproj_errors = []
        for obs in observations:
            c = obs["origin_base"]
            d = obs["ray_base"]
            v = point - c
            depth = float(np.dot(v, d))
            depths.append(depth)
            if depth < TRIANGULATION_MIN_DEPTH_M or depth > TRIANGULATION_MAX_DEPTH_M:
                return None
            closest = c + depth * d
            ray_errors.append(float(np.linalg.norm(point - closest)))
            pc = obs["base_R_camera"].T @ (point - c)
            if float(pc[2]) <= TRIANGULATION_MIN_DEPTH_M:
                return None
            k = obs["k"]
            u_hat = float(k[0, 0] * pc[0] / pc[2] + k[0, 2])
            v_hat = float(k[1, 1] * pc[1] / pc[2] + k[1, 2])
            err = float(np.hypot(u_hat - obs["u"], v_hat - obs["v"]))
            reproj_errors.append(err)
            width, height = obs["info_size"]
            if width > 0 and height > 0 and not (-20.0 <= u_hat <= width + 20.0 and -20.0 <= v_hat <= height + 20.0):
                return None
        rms_ray_error = float(np.sqrt(np.mean(np.square(ray_errors))))
        max_ray_error = float(max(ray_errors))
        rms_reproj_error = float(np.sqrt(np.mean(np.square(reproj_errors))))
        max_reproj_error = float(max(reproj_errors))
        if max_ray_error > max_ray_error_allowed:
            return None
        if max_reproj_error > max_reproj_error_allowed:
            return None
        return {
            "point": point,
            "condition": cond,
            "stamp_spread": stamp_spread,
            "max_ray_angle_deg": max_angle,
            "rms_ray_error_m": rms_ray_error,
            "max_ray_error_m": max_ray_error,
            "rms_reprojection_error_px": rms_reproj_error,
            "max_reprojection_error_px": max_reproj_error,
            "depths_m": depths,
            "reprojection_errors_px": reproj_errors,
        }

    def triangulate_best_observations(self, observations: List[Dict], key: Tuple[str, str]) -> Tuple[Optional[Dict], List[Dict]]:
        if len(observations) < TRIANGULATION_MIN_CAMS:
            return None, []
        observations = sorted(observations, key=lambda o: o["cam"])
        result = self.triangulate_rays_weighted(observations, key)
        if result is not None:
            return result, observations
        if len(observations) <= 2:
            return None, []
        best_result = None
        best_subset: List[Dict] = []
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                subset = [observations[i], observations[j]]
                candidate = self.triangulate_rays_weighted(subset, key)
                if candidate is None:
                    continue
                if best_result is None:
                    best_result, best_subset = candidate, subset
                    continue
                score = candidate["max_reprojection_error_px"] + 100.0 * candidate["max_ray_error_m"]
                best_score = best_result["max_reprojection_error_px"] + 100.0 * best_result["max_ray_error_m"]
                if score < best_score:
                    best_result, best_subset = candidate, subset
        return best_result, best_subset

    def _check_sc_port_hsv_should_flip(
        self,
        image_bgr: Optional[np.ndarray],
        det: Dict,
    ) -> Tuple[bool, bool, int, int]:
        """Return (decision_available, should_flip, pink_plus_count, pink_minus_count).

        Uses two complementary evidence sources:
          1. Local endpoint sampling: count magenta pixels in small patches near each
             OBB mouth axis endpoint.
          2. Global magenta centroid: find the overall magenta centre-of-mass in the
             image and compare its distance to each endpoint.

        Axis mapping (important):
          compute_sc_port_orientation_from_axis() computes
            axis_2d = right_uv - left_uv   (left→right in image)
            x_hint  = -(R @ axis_cam)       (negated → right→left direction)
          Therefore 3D +x axis points toward left_uv, -x points toward right_uv.
            pink_plus  probes near left_uv  (+x end)
            pink_minus probes near right_uv (-x end)
          should_flip=True means pink is on the right_uv/-x side → x must be negated.

        decision_available=True only when:
          max(pink_plus, pink_minus) >= _sc_port_hsv_flip_min_pixels
          AND max/min ratio >= _sc_port_hsv_flip_margin_ratio

        Returns (False, False, 0, 0) when image is absent, UV points are missing,
        or evidence is too weak or ambiguous.
        """
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            return False, False, 0, 0

        left_uv = det.get("sc_port_axis_left_uv")
        right_uv = det.get("sc_port_axis_right_uv")
        if not (isinstance(left_uv, list) and len(left_uv) >= 2
                and isinstance(right_uv, list) and len(right_uv) >= 2):
            return False, False, 0, 0

        try:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        except Exception:
            return False, False, 0, 0

        lower1 = np.asarray([140, 40, 40], dtype=np.uint8)
        upper1 = np.asarray([179, 255, 255], dtype=np.uint8)
        lower2 = np.asarray([0, 40, 40], dtype=np.uint8)
        upper2 = np.asarray([10, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )

        lx, ly = float(left_uv[0]), float(left_uv[1])
        rx, ry = float(right_uv[0]), float(right_uv[1])
        axis_len = float(np.hypot(rx - lx, ry - ly))
        if axis_len < 2.0:
            return False, False, 0, 0

        img_h, img_w = image_bgr.shape[:2]
        sample_r = max(8, int(axis_len * 0.20))

        def _count(cx: float, cy: float) -> int:
            x1 = int(np.clip(cx - sample_r, 0, img_w - 1))
            y1 = int(np.clip(cy - sample_r, 0, img_h - 1))
            x2 = int(np.clip(cx + sample_r, 1, img_w))
            y2 = int(np.clip(cy + sample_r, 1, img_h))
            if x2 <= x1 or y2 <= y1:
                return 0
            return int(np.count_nonzero(mask[y1:y2, x1:x2]))

        # Method 1: local endpoint samples
        pink_plus = _count(lx, ly)    # near left_uv → +x in 3D
        pink_minus = _count(rx, ry)   # near right_uv → -x in 3D

        min_px = int(self._sc_port_hsv_flip_min_pixels)
        ratio_needed = float(self._sc_port_hsv_flip_margin_ratio)
        hi = max(pink_plus, pink_minus)
        lo = min(pink_plus, pink_minus)
        local_strong = hi >= min_px and (lo == 0 or (float(hi) / float(lo)) >= ratio_needed)

        if local_strong:
            return True, pink_minus > pink_plus, pink_plus, pink_minus

        # Method 2: global magenta centroid fallback
        nonzero_pts = cv2.findNonZero(mask)
        if nonzero_pts is not None and len(nonzero_pts) >= min_px:
            pts = nonzero_pts.reshape(-1, 2).astype(np.float64)
            cx_pink = float(np.mean(pts[:, 0]))
            cy_pink = float(np.mean(pts[:, 1]))
            dist_plus = float(np.hypot(cx_pink - lx, cy_pink - ly))
            dist_minus = float(np.hypot(cx_pink - rx, cy_pink - ry))
            # Require a meaningful asymmetry relative to the axis length
            if abs(dist_plus - dist_minus) >= axis_len * 0.15:
                total_px = int(len(pts))
                centroid_on_minus_side = dist_minus < dist_plus
                # Encode result as pseudo endpoint counts for logging
                if centroid_on_minus_side:
                    return True, True, 0, total_px
                else:
                    return True, False, total_px, 0

        return False, False, pink_plus, pink_minus

    def _nearest_task_board_edge_direction_uv(
        self,
        det: Dict,
        outputs: Dict[str, Dict],
        cam: str,
    ) -> Optional[np.ndarray]:
        """Return the normalized 2D image direction of the nearest task-board edge.

        Finds the highest-confidence task_board detection in the given camera, extracts
        its OBB corners (or falls back to bbox corners), and returns the direction of the
        edge that is closest to the query detection centre.

        Returns None if no task_board detection exists in this camera.
        """
        cam_data = outputs.get(cam)
        if cam_data is None:
            return None
        tb_dets = [d for d in cam_data.get("detections", []) if self.is_family(d, "task_board")]
        if not tb_dets:
            return None
        tb = max(tb_dets, key=lambda d: float(d.get("confidence", 0.0)))

        corners_raw = tb.get("obb_corners_uv")
        corners: Optional[np.ndarray] = None
        if isinstance(corners_raw, list) and len(corners_raw) == 4:
            try:
                corners = np.asarray(corners_raw, dtype=np.float64).reshape(4, 2)
            except Exception:
                corners = None
        if corners is None:
            bbox = tb.get("bbox_xyxy")
            if not (isinstance(bbox, list) and len(bbox) >= 4):
                return None
            corners = rect_corners_from_bbox(np.asarray(bbox[:4], dtype=np.float64))

        # Query point: prefer mouth_center_uv / tip_uv / center_uv of the SC detection.
        qx, qy = self.det_center_xy(det)
        for key in ("mouth_center_uv", "tip_uv", "center_uv"):
            val = det.get(key)
            if isinstance(val, list) and len(val) >= 2:
                qx, qy = float(val[0]), float(val[1])
                break

        q_pt = np.asarray([qx, qy], dtype=np.float64)
        best_dir: Optional[np.ndarray] = None
        best_dist = float("inf")
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            edge = p1 - p0
            edge_len = float(np.linalg.norm(edge))
            if edge_len < 1e-9:
                continue
            edge_u = edge / edge_len
            t = float(np.clip(float(np.dot(q_pt - p0, edge_u)), 0.0, edge_len))
            closest = p0 + t * edge_u
            dist = float(np.linalg.norm(q_pt - closest))
            if dist < best_dist:
                best_dist = dist
                best_dir = edge_u.copy()
        return best_dir

    def _project_base_axis_to_image_direction(
        self,
        axis_base: np.ndarray,
        obs: Dict,
        point_base: np.ndarray,
        eps_m: float = 0.02,
    ) -> Optional[np.ndarray]:
        """Return the normalized 2D image direction of axis_base projected from point_base.

        Transforms point_base and point_base + eps_m * axis into the camera optical frame,
        projects both with the camera intrinsics, and returns the normalized 2D direction.
        Returns None when the projection is invalid or the two image points coincide.
        """
        axis = np.asarray(axis_base, dtype=np.float64).reshape(3)
        axis_n = float(np.linalg.norm(axis))
        if axis_n < 1e-12:
            return None
        axis = axis / axis_n

        p0 = np.asarray(point_base, dtype=np.float64).reshape(3)
        p1 = p0 + eps_m * axis

        R_cam_base = np.asarray(obs["base_R_camera"], dtype=np.float64).reshape(3, 3).T
        origin = np.asarray(obs["origin_base"], dtype=np.float64).reshape(3)
        k = np.asarray(obs["k"], dtype=np.float64).reshape(3, 3)
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])

        def _proj(p: np.ndarray) -> Optional[np.ndarray]:
            pc = R_cam_base @ (p - origin)
            if pc[2] < 1e-4 or not np.all(np.isfinite(pc)):
                return None
            u = fx * pc[0] / pc[2] + cx
            v = fy * pc[1] / pc[2] + cy
            if not (np.isfinite(u) and np.isfinite(v)):
                return None
            return np.asarray([u, v], dtype=np.float64)

        uv0 = _proj(p0)
        uv1 = _proj(p1)
        if uv0 is None or uv1 is None:
            return None
        diff = uv1 - uv0
        diff_n = float(np.linalg.norm(diff))
        if diff_n < 1e-9:
            return None
        return diff / diff_n

    def _score_green_opposite_taskboard_edge(
        self,
        q_candidate: np.ndarray,
        obs: Dict,
        point_base: np.ndarray,
        outputs: Dict[str, Dict],
    ) -> Optional[float]:
        """Score a candidate: lower means green/Y is more opposite to the nearest task-board edge.

        Projects the candidate's green/Y base_link axis into the image and computes
        dot(green_image_dir, nearest_task_board_edge_dir).  A lower (more negative) value
        means green is more opposite to the edge, which is the desired physical configuration.
        Returns None when either the task-board or the projection is unavailable.
        """
        edge_dir = self._nearest_task_board_edge_direction_uv(obs["det"], outputs, obs["cam"])
        if edge_dir is None:
            return None
        green_base = quat_to_matrix(q_candidate)[:, 1]
        green_uv = self._project_base_axis_to_image_direction(green_base, obs, point_base)
        if green_uv is None:
            return None
        return float(np.dot(green_uv, edge_dir))

    def compute_sc_port_orientation_consensus(
        self,
        observations: List[Dict],
        point: np.ndarray,
        outputs: Dict[str, Dict],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """Derive SC port orientation from all cameras using HSV + task-board consensus.

        Every observation provides an independent flip vote (via HSV pink-side routing
        as the primary evidence and task-board green-axis anchoring as the fallback).
        The weighted votes decide the 180-degree local-Z flip.  The final X/red axis is
        built from a sign-aligned, weight-averaged combination of all camera x-axes.

        Returns (q_xyzw, x_axis_3d, debug_dict) or (None, None, {}).
        """
        z_down = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        q_flip = rpy_to_quat_xyzw(0.0, 0.0, np.pi)

        # --- Step 1: per-camera orientation candidate and evidence ---
        per_cam: List[Dict] = []
        for obs in observations:
            det = obs["det"]
            left_uv = det.get("sc_port_axis_left_uv")
            right_uv = det.get("sc_port_axis_right_uv")
            if not (isinstance(left_uv, list) and len(left_uv) >= 2
                    and isinstance(right_uv, list) and len(right_uv) >= 2):
                continue
            result_cam = compute_sc_port_orientation_from_axis(
                np.asarray(left_uv, dtype=np.float64),
                np.asarray(right_uv, dtype=np.float64),
                obs["k"],
                obs["base_R_camera"],
            )
            if result_cam is None:
                continue
            q_cam, _x_cam = result_cam
            candidate_a = q_cam
            candidate_b = quat_normalize(quat_multiply_xyzw(q_cam, q_flip))
            cam = obs["cam"]
            weight = float(obs.get("weight", 1.0))
            frame = (outputs.get(cam) or {}).get("frame")
            # HSV: primary flip evidence — which side has the magenta/pink marking?
            hsv_avail, hsv_flip, pink_plus, pink_minus = self._check_sc_port_hsv_should_flip(frame, det)
            # Task-board: fallback evidence — green/Y should be opposite the nearest edge.
            score_a = self._score_green_opposite_taskboard_edge(candidate_a, obs, point, outputs)
            score_b = self._score_green_opposite_taskboard_edge(candidate_b, obs, point, outputs)
            tb_avail = score_a is not None and score_b is not None
            tb_margin = abs(score_a - score_b) if tb_avail else 0.0
            tb_prefers_flip = tb_avail and tb_margin > 0.15 and score_b < score_a
            per_cam.append({
                "cam": cam, "weight": weight, "obs": obs,
                "q_cam": q_cam, "candidate_a": candidate_a, "candidate_b": candidate_b,
                "hsv_avail": hsv_avail, "hsv_flip": hsv_flip,
                "pink_plus": pink_plus, "pink_minus": pink_minus,
                "tb_avail": tb_avail, "score_a": score_a, "score_b": score_b,
                "tb_margin": tb_margin, "tb_prefers_flip": tb_prefers_flip,
            })

        if not per_cam:
            return None, None, {}

        # --- Step 2: weighted voting for flip direction ---
        flip_weight = 0.0
        no_flip_weight = 0.0
        hsv_votes_flip = 0
        hsv_votes_no_flip = 0
        tb_votes_flip = 0
        tb_votes_no_flip = 0
        voting_cams = 0
        for c in per_cam:
            w = c["weight"]
            if c["hsv_avail"]:
                total_px = c["pink_plus"] + c["pink_minus"]
                strength = abs(c["pink_plus"] - c["pink_minus"]) / max(total_px, 1)
                vw = w * (1.0 + strength)
                if c["hsv_flip"]:
                    flip_weight += vw
                    hsv_votes_flip += 1
                else:
                    no_flip_weight += vw
                    hsv_votes_no_flip += 1
                voting_cams += 1
            elif c["tb_avail"] and c["tb_margin"] > 0.15:
                vw = w * c["tb_margin"]
                if c["tb_prefers_flip"]:
                    flip_weight += vw
                    tb_votes_flip += 1
                else:
                    no_flip_weight += vw
                    tb_votes_no_flip += 1
                voting_cams += 1

        # --- Step 3: consensus decision with hysteresis ---
        now_mono = self.ros_time()
        consensus_decided = False
        preferred_flip = False
        if voting_cams >= 1 and (flip_weight + no_flip_weight) > 1e-9:
            winning_flip = flip_weight > no_flip_weight
            winning_w = flip_weight if winning_flip else no_flip_weight
            losing_w = no_flip_weight if winning_flip else flip_weight
            if winning_w >= 1.25 * max(losing_w, 1e-9):
                preferred_flip = winning_flip
                consensus_decided = True
        if consensus_decided:
            if (self._sc_orient_consensus_candidate_flip is None
                    or self._sc_orient_consensus_candidate_flip != preferred_flip):
                self._sc_orient_consensus_candidate_flip = preferred_flip
                self._sc_orient_consensus_candidate_start = now_mono
            elapsed = now_mono - (self._sc_orient_consensus_candidate_start or now_mono)
            if elapsed >= self._sc_orient_consensus_required_s:
                self._sc_orient_consensus_locked_flip = preferred_flip
                self._sc_orient_consensus_last_update = now_mono
        stable_flip = bool(self._sc_orient_consensus_locked_flip) if self._sc_orient_consensus_locked_flip is not None else False

        # --- Step 4: weight-average x-axis across all cameras using stable flip ---
        x_accum = np.zeros(3, dtype=np.float64)
        total_w = 0.0
        for c in per_cam:
            q_chosen = c["candidate_b"] if stable_flip else c["candidate_a"]
            x_base = quat_to_matrix(q_chosen)[:, 0].copy()
            # Align sign against previous stored x-axis to prevent 180-deg vector jumps.
            if self._sc_port_x_axis is not None and float(np.dot(x_base, self._sc_port_x_axis)) < 0.0:
                x_base = -x_base
            x_accum += c["weight"] * x_base
            total_w += c["weight"]

        if total_w < 1e-12:
            return None, None, {}
        x_avg = x_accum / total_w
        # Project perpendicular to z_down and normalize.
        x_avg = x_avg - z_down * float(np.dot(x_avg, z_down))
        x_avg_n = float(np.linalg.norm(x_avg))
        if x_avg_n < 1e-9:
            return None, None, {}
        x_avg = x_avg / x_avg_n

        # --- Step 5: build final quaternion from averaged x-axis ---
        q_final = build_frame_from_z_and_x_hint(
            z_down, x_avg,
            fallbacks=[np.asarray([1.0, 0.0, 0.0], dtype=np.float64)],
        )
        if q_final is None:
            return None, None, {}

        # --- Step 6: green/Y task-board consistency check (post-build safety net) ---
        # Only when consensus is locked: if > 2:1 task-board cameras prefer the
        # flipped orientation, apply the local-Z pi flip as a correction.
        tb_normal_votes_post = 0
        tb_flip_votes_post = 0
        if self._sc_orient_consensus_locked_flip is not None:
            q_final_flipped = quat_normalize(quat_multiply_xyzw(q_final, q_flip))
            for c in per_cam:
                if not c["tb_avail"]:
                    continue
                s_n = self._score_green_opposite_taskboard_edge(q_final, c["obs"], point, outputs)
                s_f = self._score_green_opposite_taskboard_edge(q_final_flipped, c["obs"], point, outputs)
                if s_n is not None and s_f is not None:
                    if s_f < s_n:
                        tb_flip_votes_post += 1
                    else:
                        tb_normal_votes_post += 1
            if tb_flip_votes_post > 0 and tb_flip_votes_post > 2 * tb_normal_votes_post:
                q_final = q_final_flipped
                x_avg = quat_to_matrix(q_final)[:, 0].copy()

        # --- Step 7: quaternion slerp smoothing (applied only after consensus build) ---
        q_result = quat_normalize(q_final)
        if self._sc_port_quat is not None:
            angle_deg = quat_angle_deg_xyzw(q_result, self._sc_port_quat)
            if angle_deg > 150.0 and self._sc_orient_consensus_locked_flip is None:
                # Near-180-deg rotation but consensus not yet locked — hold previous.
                q_result = self._sc_port_quat.copy()
            else:
                q_result = quat_slerp_xyzw(self._sc_port_quat, q_result, 0.15)

        x_result = quat_to_matrix(q_result)[:, 0].copy()
        debug = {
            "per_cam": per_cam,
            "flip_weight": flip_weight,
            "no_flip_weight": no_flip_weight,
            "hsv_votes_flip": hsv_votes_flip,
            "hsv_votes_no_flip": hsv_votes_no_flip,
            "tb_votes_flip": tb_votes_flip,
            "tb_votes_no_flip": tb_votes_no_flip,
            "voting_cams": voting_cams,
            "consensus_decided": consensus_decided,
            "preferred_flip": preferred_flip,
            "stable_flip": stable_flip,
            "x_avg": x_avg,
            "num_cams": len(per_cam),
            "tb_normal_votes_post": tb_normal_votes_post,
            "tb_flip_votes_post": tb_flip_votes_post,
        }
        return q_result, x_result, debug

    def clear_pose_fields(self, det: Dict) -> None:
        prefixes = ("pose_", "bearing_ray_", "ray_pose_", "triangulation_", "triangulated_")
        for key in list(det.keys()):
            if key.startswith(prefixes) or key in {"object_position_metric_valid", "camera_origin_base_link", "camera_name_for_k", "camera_frame_id", "camera_info_header_frame_id", "camera_info_size", "camera_k"}:
                det.pop(key, None)

    def attach_triangulated_poses(
        self,
        outputs: Dict[str, Dict],
        processed: Dict[str, Dict],
        camera_ks: Dict[str, Optional[np.ndarray]],
        info_frame_ids: Dict[str, str],
        info_sizes: Dict[str, Tuple[int, int]],
    ) -> None:
        del processed
        now_msg = self.get_clock().now().to_msg()
        now_sec = self.stamp_sec(now_msg)
        per_cam_arrays = {}
        for cam in CAMERAS:
            arr = PoseArray()
            arr.header.stamp = now_msg
            arr.header.frame_id = BASE_FRAME
            per_cam_arrays[cam] = arr

        groups: Dict[Tuple[str, str], Dict[str, Dict]] = {}
        for cam, data in outputs.items():
            msg = data.get("msg")
            if msg is None:
                continue
            for det in data.get("detections", []):
                self.clear_pose_fields(det)
                key = self.triangulation_key(det)
                if key is None:
                    continue
                current = groups.setdefault(key, {})
                old = current.get(cam)
                if old is None or float(det.get("confidence", 0.0)) > float(old.get("confidence", 0.0)):
                    current[cam] = det

        updated_keys = set()
        for key, by_cam in groups.items():
            observations = []
            for cam, det in by_cam.items():
                data = outputs.get(cam)
                if data is None:
                    continue
                obs = self.camera_observation_for_triangulation(
                    cam,
                    data["msg"],
                    det,
                    camera_ks.get(cam),
                    info_frame_ids.get(cam, ""),
                    info_sizes.get(cam, (0, 0)),
                )
                if obs is not None:
                    observations.append(obs)
            if len(observations) < TRIANGULATION_MIN_CAMS:
                if key == ("sfp_module", "sfp_module"):
                    self.log_module_tri(f"YOLO_MODULE_TRI_WAIT cams={[obs['cam'] for obs in observations]} reason=need_2_cameras")
                continue
            observations.sort(key=lambda o: o["cam"])
            result, used_observations = self.triangulate_best_observations(observations, key)
            if result is None:
                if key == ("sfp_module", "sfp_module"):
                    self.log_module_tri(f"YOLO_MODULE_TRI_REJECT cams={[obs['cam'] for obs in observations]} reason=triangulation_failed")
                continue
            observations = used_observations

            point = result["point"]
            child = self.triangulation_child_frame(key)
            quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            orientation_source = "identity_position_only"
            pose_type = "multi_camera_triangulated_point"
            pose_source = "weighted_least_squares_multi_ray_strict"
            orientation_note = "identity_quaternion_position_only_not_object_orientation"

            if key == ("sfp_module", "sfp_module") and bool(getattr(self, "enable_sfp_module_gripper_orientation", True)):
                try:
                    tf_gripper = self.tf_buffer.lookup_transform(BASE_FRAME, GRIPPER_TF_SOURCE_FRAMES[0], Time())
                    gripper_pos, gripper_quat, _ = transform_to_arrays(tf_gripper)
                    tcp_to_plug = np.asarray(point, dtype=np.float64).reshape(3) - gripper_pos
                    offset_norm = float(np.linalg.norm(tcp_to_plug))
                    min_offset = float(getattr(self, "sfp_module_gripper_orientation_min_offset_m", 0.005))
                    if offset_norm >= min_offset:
                        # Use the frozen SFP port lateral axis as x_hint so both
                        # port and module frames share the same reference axis.
                        # Falls back to gripper local X when ports not yet visible.
                        x_hint_source = "module_y_axis_matches_negative_gripper_tcp_y"
                        q_module = compute_sfp_module_orientation_matching_port(
                            gripper_pos,
                            gripper_quat,
                            point,
                            port_x_axis=self._sfp_port_x_axis,
                        )
                        if q_module is not None:
                            quat = quat_normalize(q_module)
                            Rm = quat_to_matrix(quat)
                            det_m = float(np.linalg.det(Rm))
                            orientation_source = "gripper_tcp_to_plug_geometry_negative_gripper_y_green_axis"
                            pose_type = "triangulated_position_with_geometry_based_orientation"
                            pose_source = "position_weighted_least_squares_orientation_z_tcp_to_plug_y_negative_gripper_tcp"
                            orientation_note = "z_tcp_to_plug_green_axis_negative_gripper_tcp_y_no_other_axis_setup_change"
                            self._object_orientations[child] = quat.copy()
                            now_mono = self.ros_time()
                            if now_mono - self._sfp_module_gripper_orient_last_log_time >= 1.0:
                                self._sfp_module_gripper_orient_last_log_time = now_mono
                                self.get_logger().info(
                                    f"YOLO_SFP_MODULE_AXES "
                                    f"source={x_hint_source} "
                                    f"z_tcp_to_plug=({Rm[0,2]:+.4f},{Rm[1,2]:+.4f},{Rm[2,2]:+.4f}) "
                                    f"x=({Rm[0,0]:+.4f},{Rm[1,0]:+.4f},{Rm[2,0]:+.4f}) "
                                    f"y=({Rm[0,1]:+.4f},{Rm[1,1]:+.4f},{Rm[2,1]:+.4f}) "
                                    f"det={det_m:+.6f} "
                                    f"quat=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f})"
                                )
                                self.get_logger().info(
                                    f"YOLO_SFP_MODULE_ORIENT_MATCHING_GRIPPER_GREEN "
                                    f"green_axis_negative_gripper_tcp_y=True no_hardcoded_axis_correction=True "
                                    f"tcp=({gripper_pos[0]:+.4f},{gripper_pos[1]:+.4f},{gripper_pos[2]:+.4f}) "
                                    f"plug=({point[0]:+.4f},{point[1]:+.4f},{point[2]:+.4f}) "
                                    f"tcp_to_plug=({tcp_to_plug[0]:+.4f},{tcp_to_plug[1]:+.4f},{tcp_to_plug[2]:+.4f}) "
                                    f"norm={offset_norm:.4f}"
                                )
                    else:
                        self.log_module_tri(
                            f"YOLO_SFP_MODULE_GRIPPER_ORIENT_SKIP reason=offset_too_small norm={offset_norm:.4f} min={min_offset:.4f}",
                            min_period_s=1.0,
                        )
                except TransformException as exc:
                    self.log_module_tri(
                        f"YOLO_SFP_MODULE_GRIPPER_ORIENT_WAIT source={GRIPPER_TF_SOURCE_FRAMES[0]} reason={exc}",
                        min_period_s=1.0,
                    )

            if key[0] == "sc_port":
                # SC port: derive orientation from multi-camera HSV + task-board consensus.
                # All observations vote on the 180-deg local-Z flip; the final X/red axis
                # is built from a sign-aligned, weight-averaged combination across cameras.
                # Convention: Z/blue=[0,0,-1], X/red=mouth axis, Y/green=Z×X (right-handed).
                prev_x_axis = self._sc_port_x_axis.copy() if self._sc_port_x_axis is not None else None
                q_sc_port, sc_port_x, dbg = self.compute_sc_port_orientation_consensus(
                    observations, point, outputs
                )
                if q_sc_port is not None:
                    stable_flip = bool(dbg.get("stable_flip", False))
                    self._sc_port_hsv_flip_applied = stable_flip
                    self._sc_port_x_axis = sc_port_x.copy()
                    self._sc_port_quat = q_sc_port.copy()
                    self._object_orientations[child] = q_sc_port.copy()
                    quat = q_sc_port
                    orientation_source = "sc_multi_camera_hsv_taskboard_consensus"
                    pose_type = "triangulated_position_with_geometry_based_orientation"
                    pose_source = "position_weighted_least_squares_orientation_sc_port_multi_cam_consensus"
                    orientation_note = "z_downward_x_sc_port_weighted_avg_multi_cam_hsv_tb_consensus"
                    now_mono = self.ros_time()
                    if now_mono - self._sc_port_orient_last_log_time >= 1.0:
                        self._sc_port_orient_last_log_time = now_mono
                        Rp = quat_to_matrix(quat)
                        det_p = float(np.linalg.det(Rp))
                        per_cam_dbg = dbg.get("per_cam", [])
                        used_cams = [c["cam"] for c in per_cam_dbg]
                        num_cams = dbg.get("num_cams", 0)
                        x_avg = dbg.get("x_avg", sc_port_x)
                        prev_x_str = (
                            f"({prev_x_axis[0]:+.4f},{prev_x_axis[1]:+.4f},{prev_x_axis[2]:+.4f})"
                            if prev_x_axis is not None else "none"
                        )
                        self.get_logger().info(
                            f"YOLO_SC_PORT_AXES "
                            f"orientation_source=sc_multi_camera_hsv_taskboard_consensus "
                            f"num_orientation_cams={num_cams} "
                            f"locked_flip={stable_flip} "
                            f"used_cameras={used_cams} "
                            f"axis_source=weighted_average_all_cameras "
                            f"x=({Rp[0,0]:+.4f},{Rp[1,0]:+.4f},{Rp[2,0]:+.4f}) "
                            f"y=({Rp[0,1]:+.4f},{Rp[1,1]:+.4f},{Rp[2,1]:+.4f}) "
                            f"z=({Rp[0,2]:+.4f},{Rp[1,2]:+.4f},{Rp[2,2]:+.4f}) "
                            f"det={det_p:+.6f} "
                            f"quat=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f})"
                        )
                        self.get_logger().info(
                            f"YOLO_SC_ORIENT_CONSENSUS "
                            f"cameras={used_cams} "
                            f"hsv_votes_flip={dbg.get('hsv_votes_flip', 0)} "
                            f"hsv_votes_no_flip={dbg.get('hsv_votes_no_flip', 0)} "
                            f"tb_votes_flip={dbg.get('tb_votes_flip', 0)} "
                            f"tb_votes_no_flip={dbg.get('tb_votes_no_flip', 0)} "
                            f"final_vote_flip_weight={dbg.get('flip_weight', 0.0):.3f} "
                            f"final_vote_no_flip_weight={dbg.get('no_flip_weight', 0.0):.3f} "
                            f"candidate_flip={dbg.get('preferred_flip', False)} "
                            f"locked_flip={stable_flip} "
                            f"previous_x_axis={prev_x_str} "
                            f"averaged_x_axis=({x_avg[0]:+.4f},{x_avg[1]:+.4f},{x_avg[2]:+.4f}) "
                            f"quat=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f}) "
                            f"source={'multi_camera_consensus' if dbg.get('voting_cams', 0) >= 1 else 'previous_hold'}"
                        )

            if key == ("sc_plug", "sc_plug"):
                # SC plug orientation.
                # Y/green is the PRIMARY axis: exact -gripper TCP X/red.
                #   => sc_plug green and gripper red are antiparallel (180 deg straight line).
                #   => Y is never projected, so it stays perfectly horizontal/aligned.
                # X/red  = normalize(cross(Y, Z_prior)) — right-handed, from RPY prior Z hint.
                # Z/blue = cross(X, Y) — re-orthogonalized from the RPY prior direction,
                #          stays close to the calibrated ~28 deg insertion tilt.
                #
                # WHY Y-primary: build_frame_from_z_and_y_hint uses Z as primary and
                # projects Y perp-Z, which tilts green away from gripper red by the Z tilt
                # angle (~28 deg).  Making Y primary keeps green exactly on the gripper-X
                # axis and lets Z be the secondary (approximately preserved by Gram-Schmidt).
                try:
                    tf_gripper = self.tf_buffer.lookup_transform(BASE_FRAME, GRIPPER_TF_SOURCE_FRAMES[0], Time())
                    gripper_pos, gripper_quat, _ = transform_to_arrays(tf_gripper)
                    tcp_to_plug = np.asarray(point, dtype=np.float64).reshape(3) - gripper_pos
                    offset_norm = float(np.linalg.norm(tcp_to_plug))

                    # Z hint from RPY prior — stable insertion tilt.
                    offset_rpy = np.asarray(
                        getattr(self, "sc_plug_gripper_offset_rpy", (0.4432, -0.4838, 1.3303)),
                        dtype=np.float64,
                    ).reshape(3)
                    q_rpy_offset = rpy_to_quat_xyzw(float(offset_rpy[0]), float(offset_rpy[1]), float(offset_rpy[2]))
                    q_plug_prior = quat_normalize(quat_multiply_xyzw(gripper_quat, q_rpy_offset))
                    z_from_prior = quat_to_matrix(q_plug_prior)[:, 2].copy()

                    # Y = exact -gripper_X (no projection — primary axis).
                    Rg = quat_to_matrix(gripper_quat)
                    y_axis = -Rg[:, 0].copy()
                    y_n = float(np.linalg.norm(y_axis))
                    if y_n < 1e-9:
                        y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
                    else:
                        y_axis = y_axis / y_n

                    # X = cross(Y, Z_hint) — try Z_prior first, fall back to gripper axes.
                    z_hints: List[np.ndarray] = [
                        z_from_prior,
                        Rg[:, 2].copy(),
                        np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
                    ]
                    q_sc_plug: Optional[np.ndarray] = None
                    for z_h in z_hints:
                        x_axis = np.cross(y_axis, z_h)
                        x_n = float(np.linalg.norm(x_axis))
                        if x_n > 1e-9:
                            x_axis = x_axis / x_n
                            z_axis = np.cross(x_axis, y_axis)
                            z_axis = z_axis / max(1e-12, float(np.linalg.norm(z_axis)))
                            R_sc = np.column_stack([x_axis, y_axis, z_axis])
                            if abs(float(np.linalg.det(R_sc)) - 1.0) < 0.01:
                                q_sc_plug = matrix_to_quat(R_sc)
                                break

                    if q_sc_plug is not None:
                        # Fixed 180-degree local-Z yaw flip — always applied, never dynamic.
                        q_flip_local_z = rpy_to_quat_xyzw(0.0, 0.0, np.pi)
                        quat = quat_normalize(quat_multiply_xyzw(q_sc_plug, q_flip_local_z))
                        self._object_orientations[child] = quat.copy()
                        orientation_source = "sc_plug_y_primary_neg_gripper_x_z_rpy_prior"
                        pose_type = "triangulated_position_with_geometry_based_orientation"
                        pose_source = "position_weighted_least_squares_orientation_y_neg_gripper_x_primary"
                        orientation_note = "y_exact_neg_gripper_x_x_cross_y_z_z_cross_x_y_rhs"
                        now_mono = self.ros_time()
                        if now_mono - self._sc_plug_orient_last_log_time >= 1.0:
                            self._sc_plug_orient_last_log_time = now_mono
                            Rm = quat_to_matrix(quat)
                            det_m = float(np.linalg.det(Rm))
                            q_delta_deg = (
                                quat_angle_deg_xyzw(quat, self._sc_port_quat)
                                if self._sc_port_quat is not None else float("nan")
                            )
                            self.get_logger().info(
                                f"YOLO_SC_PLUG_AXES "
                                f"z_src=rpy_prior_secondary y_src=neg_gripper_x_exact "
                                f"x=({Rm[0,0]:+.4f},{Rm[1,0]:+.4f},{Rm[2,0]:+.4f}) "
                                f"y=({Rm[0,1]:+.4f},{Rm[1,1]:+.4f},{Rm[2,1]:+.4f}) "
                                f"z=({Rm[0,2]:+.4f},{Rm[1,2]:+.4f},{Rm[2,2]:+.4f}) "
                                f"det={det_m:+.6f} "
                                f"quat=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f})"
                            )
                            self.get_logger().info(
                                f"YOLO_SC_FRAME_CHECK "
                                f"q_delta_port_plug_deg={q_delta_deg:.2f} "
                                f"tcp_to_plug_norm={offset_norm:.4f} "
                                f"z_prior=({z_from_prior[0]:+.4f},{z_from_prior[1]:+.4f},{z_from_prior[2]:+.4f})"
                            )
                except TransformException as exc:
                    now_mono = self.ros_time()
                    if now_mono - self._sc_plug_orient_last_log_time >= 1.0:
                        self._sc_plug_orient_last_log_time = now_mono
                        self.get_logger().warn(
                            f"YOLO_SC_PLUG_ORIENT_WAIT source={GRIPPER_TF_SOURCE_FRAMES[0]} reason={exc}"
                        )
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, point.tolist())
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])
            cams = [obs["cam"] for obs in observations]
            common = {
                "pose_valid": True,
                "pose_type": pose_type,
                "pose_source": pose_source,
                "pose_frame_id": BASE_FRAME,
                "pose_child_frame_id": child,
                "pose_current_frame_triangulated": True,
                "pose_held_from_previous_triangulation": False,
                "pose_hold_age_s": 0.0,
                "object_position_metric_valid": True,
                "triangulated_position_base_link": {"x": pose.position.x, "y": pose.position.y, "z": pose.position.z},
                "triangulation_cameras": cams,
                "triangulation_camera_count": len(cams),
                "triangulation_stamp_spread_s": float(result["stamp_spread"]),
                "triangulation_max_ray_angle_deg": float(result["max_ray_angle_deg"]),
                "triangulation_rms_ray_error_m": float(result["rms_ray_error_m"]),
                "triangulation_max_ray_error_m": float(result["max_ray_error_m"]),
                "triangulation_rms_reprojection_error_px": float(result["rms_reprojection_error_px"]),
                "triangulation_max_reprojection_error_px": float(result["max_reprojection_error_px"]),
                "triangulation_condition": float(result["condition"]),
                "triangulation_orientation_note": orientation_note,
                "pose_orientation_source": orientation_source,
                "pose_base_link": {
                    "position": {"x": pose.position.x, "y": pose.position.y, "z": pose.position.z},
                    "orientation": {"x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2]), "w": float(quat[3])},
                },
            }
            self.last_triangulated_poses[key] = {
                "point": point.copy(),
                "quat": quat.copy(),
                "child": child,
                "cameras": list(cams),
                "common": dict(common),
                "updated_time_sec": float(now_sec),
            }
            updated_keys.add(key)
            if key == ("sfp_module", "sfp_module"):
                self.log_module_tri(
                    f"YOLO_MODULE_TRI_OK cams={cams} point=({point[0]:+.4f},{point[1]:+.4f},{point[2]:+.4f})",
                    min_period_s=1.0,
                )

            for obs, depth, reproj in zip(observations, result["depths_m"], result["reprojection_errors_px"]):
                det = obs["det"]
                det.update(common)
                det.update({
                    "camera_name_for_k": obs["cam"],
                    "camera_frame_id": obs["camera_frame"],
                    "camera_info_header_frame_id": obs["info_frame_id"],
                    "camera_info_size": [int(obs["info_size"][0]), int(obs["info_size"][1])],
                    "camera_k": {"fx": float(obs["k"][0, 0]), "fy": float(obs["k"][1, 1]), "cx": float(obs["k"][0, 2]), "cy": float(obs["k"][1, 2])},
                    "pose_uv": [float(obs["u"]), float(obs["v"])],
                    "pose_uv_source": obs["uv_source"],
                    "triangulation_depth_from_this_camera_m": float(depth),
                    "triangulation_reprojection_error_this_camera_px": float(reproj),
                    "bearing_ray_camera_optical": {"x": float(obs["ray_camera"][0]), "y": float(obs["ray_camera"][1]), "z": float(obs["ray_camera"][2])},
                    "bearing_ray_base_link": {"x": float(obs["ray_base"][0]), "y": float(obs["ray_base"][1]), "z": float(obs["ray_base"][2])},
                })

        # Compute SFP port pair orientation when both ports are triangulated
        _sfp0_key = ("sfp_port", "sfp_port_0")
        _sfp1_key = ("sfp_port", "sfp_port_1")
        _sfp0_cached = self.last_triangulated_poses.get(_sfp0_key)
        _sfp1_cached = self.last_triangulated_poses.get(_sfp1_key)
        if _sfp0_cached is not None and _sfp1_cached is not None:
            _sfp0_pos = np.asarray(_sfp0_cached["point"], dtype=np.float64)
            _sfp1_pos = np.asarray(_sfp1_cached["point"], dtype=np.float64)
            _sep = float(np.linalg.norm(_sfp1_pos - _sfp0_pos))
            if _sep > 0.005:
                _q_sfp, _x_sfp = compute_sfp_port_orientation_from_pair(
                    _sfp0_pos,
                    _sfp1_pos,
                )
                # Store the lateral axis so the module orientation builder can
                # use it as x_hint to enforce the same frame convention.
                self._sfp_port_x_axis = _x_sfp.copy()
                _Rp = quat_to_matrix(_q_sfp)
                self._sfp_port_z_axis = _Rp[:, 2].copy()
                self._sfp_port_quat = _q_sfp.copy()
                _pair_axis = (_sfp1_pos - _sfp0_pos) / _sep
                _sfp0_child = self.triangulation_child_frame(_sfp0_key)
                _sfp1_child = self.triangulation_child_frame(_sfp1_key)
                self._sfp_port_orientations[_sfp0_child] = _q_sfp.copy()
                self._sfp_port_orientations[_sfp1_child] = _q_sfp.copy()
                _det_p = float(np.linalg.det(_Rp))
                _now_mono = self.ros_time()
                if _now_mono - self._sfp_orient_last_log_time >= 5.0:
                    self._sfp_orient_last_log_time = _now_mono
                    self.get_logger().info(
                        f"YOLO_SFP_PORT_AXES "
                        f"x=({_Rp[0,0]:+.4f},{_Rp[1,0]:+.4f},{_Rp[2,0]:+.4f}) "
                        f"y=({_Rp[0,1]:+.4f},{_Rp[1,1]:+.4f},{_Rp[2,1]:+.4f}) "
                        f"z=({_Rp[0,2]:+.4f},{_Rp[1,2]:+.4f},{_Rp[2,2]:+.4f}) "
                        f"det={_det_p:+.6f} "
                        f"quat=({_q_sfp[0]:+.4f},{_q_sfp[1]:+.4f},{_q_sfp[2]:+.4f},{_q_sfp[3]:+.4f}) "
                        f"pair_axis=({_pair_axis[0]:+.4f},{_pair_axis[1]:+.4f},{_pair_axis[2]:+.4f}) "
                        f"sep={_sep:.4f}m"
                    )
                    self.get_logger().info(
                        f"YOLO_SFP_PORT_PAIR_STABLE "
                        f"sfp0=({_sfp0_pos[0]:+.4f},{_sfp0_pos[1]:+.4f},{_sfp0_pos[2]:+.4f}) "
                        f"sfp1=({_sfp1_pos[0]:+.4f},{_sfp1_pos[1]:+.4f},{_sfp1_pos[2]:+.4f}) "
                        f"sep={_sep:.4f}m"
                    )

        for key, by_cam in groups.items():
            if key in updated_keys:
                continue
            cached = self.last_triangulated_poses.get(key)
            if cached is None:
                continue
            held_common = dict(cached["common"])
            hold_age = float(max(0.0, now_sec - float(cached.get("updated_time_sec", now_sec))))
            if key == ("sfp_module", "sfp_module"):
                self.log_module_tri(f"YOLO_MODULE_TRI_HOLD age={hold_age:.2f}s")
            held_common.update({
                "pose_valid": True,
                "pose_source": "held_previous_weighted_least_squares_multi_ray_strict",
                "pose_current_frame_triangulated": False,
                "pose_held_from_previous_triangulation": True,
                "pose_hold_age_s": hold_age,
            })
            for det in by_cam.values():
                det.update(held_common)

        global_array = PoseArray()
        global_array.header.stamp = now_msg
        global_array.header.frame_id = BASE_FRAME
        transforms: List[TransformStamped] = []
        for key in sorted(self.last_triangulated_poses.keys()):
            cached = self.last_triangulated_poses[key]
            point = np.asarray(cached["point"], dtype=np.float64).reshape(3)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, point.tolist())
            cached_q = quat_normalize(np.asarray(cached.get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64))
            pose.orientation.x = float(cached_q[0])
            pose.orientation.y = float(cached_q[1])
            pose.orientation.z = float(cached_q[2])
            pose.orientation.w = float(cached_q[3])
            global_array.poses.append(pose)
            for cam in cached.get("cameras", CAMERAS):
                if cam in per_cam_arrays:
                    per_cam_arrays[cam].poses.append(pose)
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now_msg
            tf_msg.header.frame_id = BASE_FRAME
            child_id = str(cached["child"])
            tf_msg.child_frame_id = child_id
            tf_msg.transform.translation.x = pose.position.x
            tf_msg.transform.translation.y = pose.position.y
            tf_msg.transform.translation.z = pose.position.z
            stored_q = self._sfp_port_orientations.get(child_id)
            if stored_q is None:
                stored_q = self._object_orientations.get(child_id)
            if stored_q is None:
                stored_q = cached_q
            if stored_q is not None:
                tf_msg.transform.rotation.x = float(stored_q[0])
                tf_msg.transform.rotation.y = float(stored_q[1])
                tf_msg.transform.rotation.z = float(stored_q[2])
                tf_msg.transform.rotation.w = float(stored_q[3])
                if self.ros_time() - self._sfp_orient_last_log_time < 0.1:
                    # Log once per orientation update (throttled by pair computation above)
                    self.get_logger().info(
                        f"YOLO_SFP_PORT_TF child={child_id} "
                        f"pos=({pose.position.x:+.4f},{pose.position.y:+.4f},{pose.position.z:+.4f}) "
                        f"quat=({stored_q[0]:+.4f},{stored_q[1]:+.4f},{stored_q[2]:+.4f},{stored_q[3]:+.4f})"
                    )
            else:
                tf_msg.transform.rotation = pose.orientation
            transforms.append(tf_msg)

        transforms.extend(self.lookup_gripper_alias_transforms(now_msg))

        self.global_pose_pub.publish(global_array)
        for cam in CAMERAS:
            self.pose_pubs[cam].publish(per_cam_arrays[cam])
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def gripper_alias_child_frame(self, source_frame: str) -> str:
        token = safe_frame_token(str(source_frame).split("/")[-1])
        if not token:
            token = safe_frame_token(source_frame)
        return f"{GRIPPER_TF_ALIAS_PREFIX}/{token}"

    def lookup_gripper_alias_transforms(self, stamp_msg) -> List[TransformStamped]:
        """Republish selected gripper TFs as yolo_tri/gripper/* aliases.

        This avoids TF authority conflicts with the simulator/controller while giving the
        perception/control stack a detector-published gripper frame in the same TF tree as
        yolo_tri/sfp_port/* and yolo_tri/sfp_module/*.
        """
        transforms: List[TransformStamped] = []
        now_mono = self.ros_time()
        for source_frame in GRIPPER_TF_SOURCE_FRAMES:
            try:
                tf_src = self.tf_buffer.lookup_transform(BASE_FRAME, source_frame, Time())
            except TransformException as exc:
                if source_frame not in self._gripper_tf_warned_missing:
                    self._gripper_tf_warned_missing.add(source_frame)
                    self.get_logger().warn(
                        f"YOLO_GRIPPER_TF_WAIT source={source_frame} target={BASE_FRAME} reason={exc}"
                    )
                continue

            alias = TransformStamped()
            alias.header.stamp = stamp_msg
            alias.header.frame_id = BASE_FRAME
            alias.child_frame_id = self.gripper_alias_child_frame(source_frame)
            alias.transform.translation = tf_src.transform.translation
            alias.transform.rotation = tf_src.transform.rotation
            transforms.append(alias)

            if now_mono - self._gripper_tf_last_log_time >= GRIPPER_TF_LOG_PERIOD_S:
                t = alias.transform.translation
                q = alias.transform.rotation
                self.get_logger().info(
                    f"YOLO_GRIPPER_TF_ALIAS source={source_frame} child={alias.child_frame_id} "
                    f"pos=({t.x:+.4f},{t.y:+.4f},{t.z:+.4f}) "
                    f"quat=({q.x:+.4f},{q.y:+.4f},{q.z:+.4f},{q.w:+.4f})"
                )
                self._gripper_tf_last_log_time = now_mono
        return transforms

    def publish_outputs(self, cam: str, image_msg: Image, annotated: np.ndarray, detections: List[Dict], classes: List[str]) -> None:
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_msg.header = image_msg.header
        self.annotated_pubs[cam].publish(annotated_msg)
        msg = String()
        msg.data = json.dumps(detections, separators=(",", ":"))
        self.json_pubs[cam].publish(msg)
        cls_msg = String()
        cls_msg.data = ",".join(classes)
        self.classes_pubs[cam].publish(cls_msg)

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        out = image.copy()
        colors = {"task_board": (255, 0, 0), "nic_card": (0, 255, 255), "sc_port": (255, 255, 0), "sfp_port": (0, 165, 255), "sfp_module": (180, 180, 180), "sc_plug": (0, 128, 255)}
        for det in detections:
            family = str(det.get("base_class_name", det.get("class_name", "")))
            color = colors.get(family, (0, 255, 0))
            x1, y1, x2, y2 = [int(round(float(v))) for v in det.get("bbox_xyxy", [0, 0, 0, 0])[:4]]
            corners = det.get("obb_corners_uv")
            if isinstance(corners, list) and len(corners) == 4:
                cv2.polylines(out, [np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))], True, color, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cx, cy = self.det_center_xy(det)
            cv2.circle(out, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, cv2.LINE_AA)
            label = str(det.get("instance_name", det.get("class_name", "")))
            text = f"{label} id={int(det.get('track_id', -1))} conf={float(det.get('confidence', 0.0)):.2f} raw={float(det.get('raw_confidence', 0.0)):.2f} feat={float(det.get('feature_quality_score', 0.0)):.2f} pts={int(det.get('feature_tracked_count', 0))}"
            cv2.putText(out, text, (max(0, x1), max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        return out

    def draw_features(self, image: np.ndarray, cam: str) -> np.ndarray:
        out = image.copy()
        tracker = self.trackers.get(cam)
        if tracker is None:
            return out
        for track in tracker.tracks:
            ft = track.feature
            if ft is None:
                continue
            for old, new in ft.last_motion_pairs[:20]:
                cv2.line(out, tuple(np.round(old).astype(int)), tuple(np.round(new).astype(int)), (0, 180, 255), 1, cv2.LINE_AA)
            for pt in ft.points[:40]:
                cv2.circle(out, tuple(np.round(pt).astype(int)), 2, (0, 255, 0), -1, cv2.LINE_AA)
        return out

    def draw_raw(self, image: np.ndarray, raw: List[Detection]) -> np.ndarray:
        out = image.copy()
        for det in raw:
            x1, y1, x2, y2 = [int(round(float(v))) for v in det.bbox_xyxy[:4]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(out, f"raw {det.family} {det.confidence:.2f}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        return out


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
