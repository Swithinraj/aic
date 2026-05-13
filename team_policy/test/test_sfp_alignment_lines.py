"""Tests for SFP plug/port alignment-line geometry."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from team_policy.planner.combined_yolo_depth_pose_planner import CombinedYoloDepthPosePlanner
    from team_policy.perception.yolov12_detector import YoloV12MultiCameraDetector
except Exception as exc:  # pragma: no cover - ROS/perception deps may be unavailable
    CombinedYoloDepthPosePlanner = None
    YoloV12MultiCameraDetector = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _planner_shell():
    if CombinedYoloDepthPosePlanner is None:
        raise unittest.SkipTest(f"perception imports unavailable: {_IMPORT_ERROR}")
    p = CombinedYoloDepthPosePlanner.__new__(CombinedYoloDepthPosePlanner)
    p._sfp_port_classes = {"sfp_port"}
    p._sfp_module_classes = {"sfp_module", "transceiver"}
    p._sfp_plug_line_classes = {"sfp_plug", "sfp_connector", "sfp_tip", "sfp_module", "transceiver"}
    return p


class TestSfpAlignmentLines(unittest.TestCase):
    def test_set_alignment_line_fields(self):
        if YoloV12MultiCameraDetector is None:
            self.skipTest(f"perception imports unavailable: {_IMPORT_ERROR}")
        det = {}
        node = YoloV12MultiCameraDetector.__new__(YoloV12MultiCameraDetector)
        node._set_alignment_line(
            det,
            "port_mouth",
            np.array([[10.0, 20.0], [30.0, 20.0]], dtype=np.float32),
            "obb_depth",
        )

        self.assertEqual(det["alignment_line_role"], "port_mouth")
        self.assertEqual(det["alignment_line_source"], "obb_depth")
        np.testing.assert_allclose(det["alignment_line_mid_uv"], [20.0, 20.0])
        self.assertAlmostEqual(det["alignment_line_angle_rad"], 0.0)

    def test_sfp_port_uses_centered_long_axis(self):
        planner = _planner_shell()
        det = {
            "class_name": "sfp_port_0",
            "base_class_name": "sfp_port",
            "instance_name": "sfp_port_0",
            "obb_corners_uv": [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]],
        }
        depth = np.full((60, 80), 0.5, dtype=np.float32)
        depth[8:13, :] = 0.2
        depth[28:33, :] = 0.7

        line, source, line_depth, status = planner._select_sfp_alignment_edge(det, depth)

        self.assertEqual(source, "obb_center")
        self.assertEqual(status, "ok_center_axis")
        self.assertAlmostEqual(line_depth, 0.5, places=5)
        self.assertIsNotNone(line)
        np.testing.assert_allclose(line, [[10.0, 20.0], [50.0, 20.0]], atol=1e-6)

    def test_sfp_plug_uses_farthest_short_edge(self):
        planner = _planner_shell()
        det = {
            "class_name": "sfp_module",
            "obb_corners_uv": [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]],
        }
        depth = np.full((60, 80), 0.5, dtype=np.float32)
        depth[:, 8:13] = 0.8
        depth[:, 48:53] = 0.3

        line, source, line_depth, status = planner._select_sfp_alignment_edge(det, depth)

        self.assertEqual(source, "obb_depth")
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(line_depth, 0.8, places=5)
        self.assertIsNotNone(line)
        np.testing.assert_allclose(line, [[10.0, 30.0], [10.0, 10.0]], atol=1e-6)

    def test_edge_depth_sampling_moves_inside_detection(self):
        planner = _planner_shell()
        det = {
            "class_name": "sfp_port_0",
            "base_class_name": "sfp_port",
            "instance_name": "sfp_port_0",
            "obb_corners_uv": [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]],
        }
        depth = np.zeros((60, 80), dtype=np.float32)
        depth[12:15, 12:48] = 0.2
        depth[26:29, 12:48] = 0.7

        line, source, line_depth, status = planner._select_sfp_alignment_edge(det, depth)

        self.assertEqual(source, "obb_center")
        self.assertEqual(status, "ok_center_axis")
        self.assertAlmostEqual(line_depth, 0.2, places=5)
        self.assertIsNotNone(line)
        np.testing.assert_allclose(line, [[10.0, 20.0], [50.0, 20.0]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
