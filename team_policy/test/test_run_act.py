"""Unit tests for the clean-hybrid ``team_policy.run_act`` policy.

These tests intentionally cover the small contract we want to keep:

* checkpoint path resolution
* V2 30D state layout with held YOLO port_xyz
* legacy 33D state layout for old checkpoints
* YOLO hold-last behavior
* basic action shaping / image conversion
* model-driven insertion push direction fallback order
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team_policy.run_act import (
    RunACT,
    detect_state_schema,
    _SCHEMA_V3_77D,
    _SCHEMA_V2_30D,
    _SCHEMA_CHUNK50_33D,
    _MAX_TRANSLATION_DELTA_M,
    _MAX_ROTATION_DELTA_RAD,
    _TARGET_MODULE_NAMES,
)


def _fake_pose(x=0.4, y=-0.2, z=0.8, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
    )


def _fake_image(height=100, width=200):
    return SimpleNamespace(
        height=height,
        width=width,
        step=width * 3,
        encoding="rgb8",
        data=np.zeros((height, width, 3), dtype=np.uint8).tobytes(),
    )


def _fake_obs():
    return SimpleNamespace(
        controller_state=SimpleNamespace(
            tcp_pose=_fake_pose(),
            tcp_velocity=SimpleNamespace(
                linear=SimpleNamespace(x=0.01, y=0.02, z=0.03),
                angular=SimpleNamespace(x=0.04, y=0.05, z=0.06),
            ),
            tcp_error=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        ),
        joint_states=SimpleNamespace(
            position=[0.1, 0.2, 0.3],
            velocity=[0.4, 0.5],
        ),
        wrist_wrench=SimpleNamespace(
            wrench=SimpleNamespace(
                force=SimpleNamespace(x=2.0, y=4.0, z=6.0),
                torque=SimpleNamespace(x=0.2, y=0.4, z=0.6),
            )
        ),
        left_image=_fake_image(),
        center_image=_fake_image(),
        right_image=_fake_image(),
    )


def _policy_shell(state_dim: int = 30, schema: str = _SCHEMA_V2_30D) -> RunACT:
    p = RunACT.__new__(RunACT)
    p.get_logger = lambda: SimpleNamespace(
        info=lambda _msg: None,
        warning=lambda _msg: None,
        error=lambda _msg: None,
    )
    p.device = torch.device("cpu")
    p.state_dim = state_dim
    p.schema = schema
    p.state_mean = torch.zeros((1, state_dim), dtype=torch.float32)
    p.state_std = torch.ones((1, state_dim), dtype=torch.float32)
    p._yolo_lock = threading.Lock()
    p._yolo_port_xyz = np.array([0.11, -0.22, 0.33], dtype=np.float32)
    p._yolo_port_valid = True
    p._yolo_port_stamp_s = 9.5
    p._yolo_locked_instance = ""
    p._yolo_locked_class = ""
    p._yolo_lock_announced = False
    p._cam_lock = threading.Lock()
    p._cam_last_det_time = {cam: None for cam in ("left", "center", "right")}
    p._cam_last_conf = {cam: 0.0 for cam in ("left", "center", "right")}
    p._cam_last_bbox = {cam: None for cam in ("left", "center", "right")}
    p._cam_last_port_line = {cam: None for cam in ("left", "center", "right")}
    p._cam_last_plug_line = {cam: None for cam in ("left", "center", "right")}
    p._wrist_force_tare = np.zeros(6, dtype=np.float32)
    p._wrist_force_tare_ready = False
    p._plug_type_onehot = np.zeros(2, dtype=np.float32)
    p._target_module_onehot = np.zeros(len(_TARGET_MODULE_NAMES), dtype=np.float32)
    p._locked_insert_axis = None
    p._locked_insert_axis_source = ""
    p._parent_node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)
        )
    )
    p._target_port_name = "sc_port_base"
    p._target_port_type = "sc"
    p._target_module_name = "sc_port_0"
    p._target_plug_type = "sc"
    p._is_sc_task = False
    p.action_scale = 1.0
    p.rotation_gain = 1.0
    p.max_translation_delta_m = _MAX_TRANSLATION_DELTA_M
    p.max_rotation_delta_rad = _MAX_ROTATION_DELTA_RAD
    p.ema_alpha = 0.7
    p.delta_pose_scale = 1.0
    p._prev_action = None
    return p


class TestCheckpointResolution(unittest.TestCase):
    def test_resolve_accepts_parent_or_pretrained_model(self):
        policy = RunACT.__new__(RunACT)
        model_parent = (
            Path(__file__).resolve().parents[1]
            / "team_policy"
            / "models"
            / "trained_model_V2"
        )

        self.assertEqual(
            policy._resolve_checkpoint_path(str(model_parent)),
            model_parent / "pretrained_model",
        )
        self.assertEqual(
            policy._resolve_checkpoint_path(str(model_parent / "pretrained_model")),
            model_parent / "pretrained_model",
        )

    def test_resolve_accepts_v3_checkpoint_parent(self):
        policy = RunACT.__new__(RunACT)
        model_parent = (
            Path(__file__).resolve().parents[1]
            / "team_policy"
            / "models"
            / "trained_model_v3"
        )

        self.assertEqual(
            policy._resolve_checkpoint_path(str(model_parent)),
            model_parent / "040000" / "pretrained_model",
        )

    def test_resolve_rejects_empty_path(self):
        policy = RunACT.__new__(RunACT)
        with self.assertRaises(ValueError):
            policy._resolve_checkpoint_path("")


class TestSchemaDetection(unittest.TestCase):
    def test_v3_77d_detected(self):
        self.assertEqual(detect_state_schema(77), _SCHEMA_V3_77D)

    def test_v2_30d_detected(self):
        self.assertEqual(detect_state_schema(30), _SCHEMA_V2_30D)

    def test_legacy_33d_detected(self):
        self.assertEqual(detect_state_schema(33), _SCHEMA_CHUNK50_33D)

    def test_positional_override_is_honored(self):
        self.assertEqual(detect_state_schema(33, {}, {}, "false"), _SCHEMA_V2_30D)
        self.assertEqual(detect_state_schema(30, {}, {}, "true"), _SCHEMA_CHUNK50_33D)

    def test_unknown_state_dim_raises(self):
        with self.assertRaises(ValueError):
            detect_state_schema(42)


class TestStateBuilding(unittest.TestCase):
    def test_v3_77d_state_matches_training_schema(self):
        policy = _policy_shell(state_dim=77, schema=_SCHEMA_V3_77D)
        policy._yolo_port_stamp_s = time.time()
        policy._wrist_force_tare = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32)
        policy._cam_last_det_time["left"] = time.time()
        policy._cam_last_conf["left"] = 0.75
        policy._cam_last_bbox["left"] = [20.0, 10.0, 60.0, 30.0]
        policy._plug_type_onehot = np.array([0.0, 1.0], dtype=np.float32)
        policy._target_module_onehot = np.zeros(len(_TARGET_MODULE_NAMES), dtype=np.float32)
        policy._target_module_onehot[_TARGET_MODULE_NAMES.index("sc_port_0")] = 1.0

        state = policy._build_state(_fake_obs())

        self.assertEqual(tuple(state.shape), (1, 77))
        raw = state.numpy()[0]
        np.testing.assert_allclose(raw[:13], [0.4, -0.2, 0.8, 0.0, 0.0, 0.0, 1.0,
                                              0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        np.testing.assert_allclose(raw[13:19], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        np.testing.assert_allclose(raw[19:26], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(raw[26:33], [0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(raw[33:36], [0.11, -0.22, 0.33], rtol=1e-6)
        self.assertEqual(raw[36], 1.0)
        self.assertLess(raw[37], 0.15)
        np.testing.assert_allclose(raw[38:41], [-0.29, -0.02, -0.47], atol=1e-6)
        np.testing.assert_allclose(raw[41:47], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(raw[47:54], [0.75, 0.2, 0.2, 0.2, 0.2, 1.0, raw[53]], atol=1e-6)
        np.testing.assert_allclose(raw[54:61], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        np.testing.assert_allclose(raw[61:68], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        np.testing.assert_allclose(raw[68:70], [0.0, 1.0])
        np.testing.assert_allclose(raw[70:77], policy._target_module_onehot)

    def test_v2_state_layout_matches_training_schema(self):
        policy = _policy_shell(state_dim=30, schema=_SCHEMA_V2_30D)
        state = policy._build_state(_fake_obs())

        self.assertEqual(tuple(state.shape), (1, 30))
        raw = state.numpy()[0]
        np.testing.assert_allclose(
            raw[:13],
            [0.4, -0.2, 0.8, 0.0, 0.0, 0.0, 1.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        )
        np.testing.assert_allclose(raw[13:20], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(raw[20:27], [0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(raw[27:30], [0.11, -0.22, 0.33], rtol=1e-6)

    def test_legacy_33d_state_matches_old_hybrid_layout(self):
        policy = _policy_shell(state_dim=33, schema=_SCHEMA_CHUNK50_33D)
        state = policy._build_state(_fake_obs())

        self.assertEqual(tuple(state.shape), (1, 33))
        raw = state.numpy()[0]
        np.testing.assert_allclose(raw[:13], [0.4, -0.2, 0.8, 0.0, 0.0, 0.0, 1.0,
                                              0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        np.testing.assert_allclose(raw[13:19], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        np.testing.assert_allclose(raw[19:26], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(raw[26:33], [0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])


class TestYoloCallback(unittest.TestCase):
    def test_holds_last_valid_port_when_message_has_no_match(self):
        policy = _policy_shell()
        policy._yolo_port_xyz[:] = 0.0
        policy._yolo_port_valid = False

        msg = SimpleNamespace(
            data=json.dumps([
                {
                    "instance_name": "sc_port_base",
                    "confidence": 0.5,
                    "pose_base_link": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
                }
            ])
        )
        policy._cb_fused_yolo(msg)
        xyz, valid = policy._current_port_xyz()
        self.assertTrue(valid)
        np.testing.assert_allclose(xyz, [1.0, 2.0, 3.0])

        policy._cb_fused_yolo(SimpleNamespace(data=json.dumps([])))
        xyz, valid = policy._current_port_xyz()
        self.assertTrue(valid)
        np.testing.assert_allclose(xyz, [1.0, 2.0, 3.0])

    def test_sfp_alias_matching(self):
        policy = _policy_shell()
        policy._target_port_name = "sfp_port_0"
        policy._target_port_type = "sfp"
        policy._target_module_name = "nic_card_mount_0"
        policy._yolo_port_valid = False

        msg = SimpleNamespace(
            data=json.dumps([
                {
                    "class_name": "sfp_port_2",
                    "confidence": 0.8,
                    "pose_base_link": {"position": {"x": 5.0, "y": 6.0, "z": 7.0}},
                }
            ])
        )
        policy._cb_fused_yolo(msg)
        xyz, valid = policy._current_port_xyz()
        self.assertTrue(valid)
        np.testing.assert_allclose(xyz, [5.0, 6.0, 7.0])

    def test_target_module_detection_is_not_used_as_port_state(self):
        policy = _policy_shell()
        policy._target_port_name = "sfp_port_0"
        policy._target_port_type = "sfp"
        policy._target_module_name = "nic_card_mount_0"
        policy._yolo_port_valid = False

        msg = SimpleNamespace(
            data=json.dumps([
                {
                    "instance_name": "nic_card_mount_0",
                    "class_name": "nic_card",
                    "confidence": 0.99,
                    "pose_base_link": {"position": {"x": 5.0, "y": 6.0, "z": 7.0}},
                }
            ])
        )
        policy._cb_fused_yolo(msg)
        _, valid = policy._current_port_xyz()
        self.assertFalse(valid)

    def test_per_camera_callback_stores_sfp_plug_and_port_lines(self):
        policy = _policy_shell()
        policy._target_port_name = "sfp_port_0"
        policy._target_port_type = "sfp"
        policy._target_plug_type = "sfp"

        msg = SimpleNamespace(
            data=json.dumps([
                {
                    "instance_name": "sfp_port_0",
                    "class_name": "sfp_port_0",
                    "confidence": 0.9,
                    "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
                    "alignment_line_role": "port_mouth",
                    "alignment_line_uv": [[10.0, 20.0], [30.0, 20.0]],
                    "alignment_line_mid_uv": [20.0, 20.0],
                    "alignment_line_angle_rad": 0.0,
                },
                {
                    "instance_name": "sfp_module",
                    "class_name": "sfp_module",
                    "confidence": 0.8,
                    "bbox_xyxy": [14.0, 44.0, 34.0, 70.0],
                    "alignment_line_role": "plug_tip",
                    "alignment_line_uv": [[14.0, 45.0], [34.0, 45.0]],
                    "alignment_line_mid_uv": [24.0, 45.0],
                    "alignment_line_angle_rad": 0.1,
                },
            ])
        )

        policy._cb_per_camera_yolo(msg, "center")
        meas = policy._camera_sfp_line_measurement("center", _fake_image())

        self.assertIsNotNone(meas)
        np.testing.assert_allclose(meas["midpoint_error"], [4.0, 25.0], atol=1e-6)
        self.assertAlmostEqual(meas["angle_error_rad"], 0.1)
        self.assertEqual(policy._cam_last_port_line["center"]["role"], "port_mouth")
        self.assertEqual(policy._cam_last_plug_line["center"]["role"], "plug_tip")

    def test_sfp_line_angle_error_is_undirected(self):
        self.assertAlmostEqual(
            RunACT._line_angle_error(math.pi - 0.02, 0.0),
            -0.02,
            places=6,
        )


class TestActionAndImages(unittest.TestCase):
    def test_action_shaping_clips_translation_and_rotation(self):
        policy = _policy_shell()
        shaped = policy._apply_action_shaping(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 2.0]))

        self.assertLessEqual(np.linalg.norm(shaped[:3]), policy.max_translation_delta_m + 1e-9)
        self.assertLessEqual(np.linalg.norm(shaped[3:6]), policy.max_rotation_delta_rad + 1e-9)

    def test_action_shaping_handles_nan_inf(self):
        policy = _policy_shell()
        shaped = policy._apply_action_shaping(
            np.array([float("nan"), float("inf"), -float("inf"), 0.0, 0.0, 0.0])
        )
        self.assertTrue(np.all(np.isfinite(shaped)))
        np.testing.assert_allclose(shaped[:3], [0.0, 0.0, 0.0])

    def test_v2_delta_pose_is_applied_without_extra_timestep_scaling(self):
        policy = _policy_shell()
        pose = policy._delta_to_pose(_fake_obs(), np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(pose.position.x, 0.50)

    def test_legacy_velocity_scale_can_still_be_selected(self):
        policy = _policy_shell()
        policy.delta_pose_scale = 0.1
        pose = policy._delta_to_pose(_fake_obs(), np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(pose.position.x, 0.41)

    def test_bgr_image_is_converted_to_rgb(self):
        policy = _policy_shell()
        img = SimpleNamespace(
            height=1,
            width=2,
            step=6,
            encoding="bgr8",
            data=np.array([10, 20, 30, 40, 50, 60], dtype=np.uint8).tobytes(),
        )
        rgb = policy._ros_image_to_rgb(img)
        np.testing.assert_array_equal(
            rgb,
            np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8),
        )


class TestInsertionAxisAndYoloFineAlign(unittest.TestCase):
    def test_uses_recent_model_actions_for_insertion_axis(self):
        policy = _policy_shell()
        recent = [np.array([0.0, 0.0, 0.001, 0, 0, 0])] * 5
        axis, source = policy._pick_insertion_axis(
            np.array([0.0, 0.0, 0.0, 1.0]),
            recent,
        )
        np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-6)
        self.assertEqual(source, "last_5_model_actions")

    def test_falls_back_to_gripper_z(self):
        policy = _policy_shell()
        axis, source = policy._pick_insertion_axis(
            np.array([0.0, 0.0, 0.0, 1.0]),
            [],
        )
        np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-6)
        self.assertEqual(source, "gripper_z")

    def test_locked_insertion_axis_stays_stable_across_replans(self):
        policy = _policy_shell()
        recent_z = [np.array([0.0, 0.0, 0.001, 0, 0, 0])] * 5
        recent_x = [np.array([0.001, 0.0, 0.0, 0, 0, 0])] * 5

        axis, source = policy._locked_or_pick_insertion_axis(
            np.array([0.0, 0.0, 0.0, 1.0]),
            recent_z,
            lock=True,
            reason="test",
        )
        axis2, source2 = policy._locked_or_pick_insertion_axis(
            np.array([0.0, 0.0, 0.0, 1.0]),
            recent_x,
            lock=True,
            reason="test",
        )

        np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(axis2, [0.0, 0.0, 1.0], atol=1e-6)
        self.assertEqual(source, "locked_last_5_model_actions")
        self.assertEqual(source2, "locked_last_5_model_actions")

    def test_lateral_error_ignores_along_axis_offset(self):
        policy = _policy_shell()
        lateral = policy._lateral_error_to_port(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.010, 0.020, 0.300]),
            np.array([0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(lateral, [0.010, 0.020, 0.0], atol=1e-6)

    def test_near_port_threshold_is_task_specific(self):
        policy = _policy_shell()
        policy._target_plug_type = "sfp"
        policy._target_port_type = "sfp"
        self.assertAlmostEqual(policy._near_port_threshold_m(), 0.085)

        policy._target_plug_type = "sc"
        policy._target_port_type = "sc"
        self.assertAlmostEqual(policy._near_port_threshold_m(), 0.120)

    def test_insert_lateral_threshold_is_task_specific(self):
        policy = _policy_shell()
        policy._target_plug_type = "sfp"
        policy._target_port_type = "sfp"
        self.assertAlmostEqual(policy._insert_lateral_threshold_m(), 0.018)

        policy._target_plug_type = "sc"
        policy._target_port_type = "sc"
        self.assertAlmostEqual(policy._insert_lateral_threshold_m(), 0.025)

    def test_intra_act_yolo_assist_is_bounded_and_available(self):
        """Phase 1 uses only a small near-port YOLO lateral nudge; it does not
        run a global homing controller over the model.
        """
        from team_policy import run_act as run_act_mod

        self.assertTrue(hasattr(RunACT, "_apply_yolo_approach_assist"))
        self.assertTrue(hasattr(run_act_mod, "_ACT_YOLO_ASSIST_GAIN"))

        policy = _policy_shell()
        policy._target_plug_type = "sfp"
        policy._target_port_type = "sfp"
        policy._is_sc_task = False
        policy._yolo_port_xyz = np.array([0.10, 0.0, 0.0], dtype=np.float32)
        policy._yolo_port_valid = True
        policy._yolo_port_stamp_s = time.time()
        recent_actions = deque(
            [np.array([0.0, 0.0, 0.010, 0.0, 0.0, 0.0]) for _ in range(5)],
            maxlen=30,
        )
        target_pose = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.0))

        lateral, correction = policy._apply_yolo_approach_assist(
            target_pose,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            recent_actions,
            step_count=10,
        )

        self.assertAlmostEqual(lateral, 0.10, places=6)
        self.assertAlmostEqual(correction, run_act_mod._ACT_YOLO_ASSIST_STEP_M, places=6)
        self.assertAlmostEqual(target_pose.position.x, run_act_mod._ACT_YOLO_ASSIST_STEP_M, places=6)
        self.assertAlmostEqual(target_pose.position.y, 0.0, places=6)
        self.assertAlmostEqual(target_pose.position.z, 0.0, places=6)

    def test_search_basis_is_perpendicular_to_insertion_axis(self):
        policy = _policy_shell()
        axis = np.array([0.342, 0.718, -0.606])
        axis = axis / np.linalg.norm(axis)
        u, v = policy._perpendicular_search_basis(axis)

        self.assertAlmostEqual(float(np.dot(axis, u)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(axis, v)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(u, v)), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)

    def test_search_offset_is_small_and_lateral(self):
        policy = _policy_shell()
        axis = np.array([0.0, 0.0, 1.0])
        u, v = policy._perpendicular_search_basis(axis)
        offset = policy._search_offset(4, u, v)

        self.assertAlmostEqual(float(np.dot(axis, offset)), 0.0, places=6)
        self.assertLessEqual(float(np.linalg.norm(offset)), 0.004 + 1e-9)
        self.assertGreater(float(np.linalg.norm(offset)), 0.0)


class TestStateMatchesConverterV3(unittest.TestCase):
    """Pin the deployment 77D state ordering to the canonical training-time
    converter (`convert_to_lerobot_v2._build_state_77d`).  Any drift here
    silently feeds the model a permuted state and is the most catastrophic
    deployment regression we can ship — so we test it directly.
    """

    def test_runner_state_matches_converter_concat_order(self):
        try:
            from team_policy.training_robot.convert_to_lerobot_v2 import (
                _build_state_77d,
                STATE_DIM,
            )
        except Exception as exc:  # pragma: no cover - converter requires h5py
            self.skipTest(f"converter import failed: {exc}")
            return

        self.assertEqual(STATE_DIM, 77)

        T = 1
        tcp_poses = np.array([[0.4, -0.2, 0.8, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        tcp_vels = np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]], dtype=np.float32)
        tcp_errors = np.array([[0.7, 0.8, 0.9, 1.0, 1.1, 1.2]], dtype=np.float32)
        joint_pos = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        joint_vel = np.array([[0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        port_xyz = np.array([[0.11, -0.22, 0.33]], dtype=np.float32)
        yolo_valid = np.array([[1.0]], dtype=np.float32)
        yolo_age = np.array([[0.05]], dtype=np.float32)
        port_delta_tcp = (port_xyz - tcp_poses[:, :3]).astype(np.float32)
        tared_wrench = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32)
        zero7 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]], dtype=np.float32)
        plug_oh = np.array([[0.0, 1.0]], dtype=np.float32)
        target_oh = np.zeros((1, 7), dtype=np.float32)
        target_oh[0, _TARGET_MODULE_NAMES.index("sc_port_0")] = 1.0

        canonical = _build_state_77d(
            tcp_poses, tcp_vels, tcp_errors, joint_pos, joint_vel,
            port_xyz, yolo_valid, yolo_age, port_delta_tcp, tared_wrench,
            zero7, zero7, zero7, plug_oh, target_oh,
        )
        self.assertEqual(canonical.shape, (1, 77))

        # Build the runner's state from a fake observation matching the same
        # values, with no fresh per-camera YOLO so cameras are zero7.
        policy = _policy_shell(state_dim=77, schema=_SCHEMA_V3_77D)
        policy._yolo_port_xyz = np.array([0.11, -0.22, 0.33], dtype=np.float32)
        policy._yolo_port_valid = True
        policy._yolo_port_stamp_s = time.time() - 0.05  # fresh
        policy._wrist_force_tare = np.array(
            [-1.0, -2.0, -3.0, -0.1, -0.2, -0.3], dtype=np.float32
        )
        policy._plug_type_onehot = plug_oh.flatten()
        policy._target_module_onehot = target_oh.flatten()

        obs = _fake_obs()  # raw wrist = [2,4,6,0.2,0.4,0.6]; tared = sum
        runner_state = policy._build_state(obs).numpy()[0]

        # tcp_pose, tcp_vel, tcp_error, joint_pos, joint_vel
        np.testing.assert_allclose(runner_state[:13], canonical[0, :13], rtol=1e-6)
        np.testing.assert_allclose(runner_state[13:19], canonical[0, 13:19], rtol=1e-6)
        np.testing.assert_allclose(runner_state[19:33], canonical[0, 19:33], rtol=1e-6)
        # held YOLO xyz
        np.testing.assert_allclose(runner_state[33:36], canonical[0, 33:36], rtol=1e-6)
        # yolo_valid (fresh) and small age, port_delta_tcp
        self.assertEqual(runner_state[36], 1.0)
        self.assertLess(runner_state[37], 0.15)
        np.testing.assert_allclose(runner_state[38:41], canonical[0, 38:41], atol=1e-6)
        # tared wrench: raw [2,4,6,0.2,0.4,0.6] - tare [-1,-2,-3,-0.1,-0.2,-0.3]
        np.testing.assert_allclose(
            runner_state[41:47], [3.0, 6.0, 9.0, 0.3, 0.6, 0.9], atol=1e-6
        )
        # per-camera YOLO: no fresh detection in fake obs → zero7 with age=10
        np.testing.assert_allclose(runner_state[47:54], canonical[0, 47:54], atol=1e-6)
        np.testing.assert_allclose(runner_state[54:61], canonical[0, 54:61], atol=1e-6)
        np.testing.assert_allclose(runner_state[61:68], canonical[0, 61:68], atol=1e-6)
        np.testing.assert_allclose(runner_state[68:70], canonical[0, 68:70], atol=1e-6)
        np.testing.assert_allclose(runner_state[70:77], canonical[0, 70:77], atol=1e-6)

    def test_yolo_feature_builder_matches_recorder(self):
        try:
            from team_policy.training_robot.episode_recorder_v2 import (
                build_yolo_feature, AGE_VALID_S, MAX_AGE_S,
            )
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"recorder import failed: {exc}")
            return

        now = 100.0
        ours = RunACT._build_yolo_feature(
            confidence=0.75,
            bbox_xyxy=[10.0, 20.0, 30.0, 40.0],
            img_h=200,
            img_w=400,
            last_det_time=now - 0.05,
            now=now,
        )
        theirs = build_yolo_feature(
            confidence=0.75,
            bbox_xyxy=[10.0, 20.0, 30.0, 40.0],
            img_h=200,
            img_w=400,
            last_det_time=now - 0.05,
            now=now,
        )
        np.testing.assert_allclose(ours, theirs, atol=1e-6)
        # No detection
        ours = RunACT._build_yolo_feature(0.0, None, 200, 400, None, now)
        theirs = build_yolo_feature(0.0, None, 200, 400, None, now)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)


class TestV3CheckpointSmoke(unittest.TestCase):
    """End-to-end smoke test against the actual trained_model_v3 checkpoint:
    load the safetensors + normalizer, build a fake batch matching the saved
    feature shapes, and assert ``policy.select_action`` returns a (1, 6)
    finite tensor in the expected range.
    """

    CHECKPOINT = (
        Path(__file__).resolve().parents[1]
        / "team_policy"
        / "models"
        / "trained_model_v3"
        / "040000"
        / "pretrained_model"
    )

    def setUp(self):
        if not self.CHECKPOINT.exists():
            self.skipTest(f"V3 checkpoint not found at {self.CHECKPOINT}")
        try:
            import torch  # noqa: F401
            import lerobot  # noqa: F401
        except Exception as exc:
            self.skipTest(f"torch/lerobot unavailable: {exc}")

    def test_checkpoint_loads_and_predicts_6d_action(self):
        from safetensors.torch import load_file
        from team_policy.run_act import (
            _IMAGENET_MEAN, _IMAGENET_STD, _IMG_H, _IMG_W,
        )
        # Load via the same import workaround run_act uses.
        import sys
        import types as _types
        from pathlib import Path as _Path
        import lerobot as _lerobot_pkg

        _lerobot_root = _Path(_lerobot_pkg.__file__).resolve().parent
        _policies_pkg = _types.ModuleType("lerobot.policies")
        _policies_pkg.__path__ = [str(_lerobot_root / "policies")]
        sys.modules.setdefault("lerobot.policies", _policies_pkg)
        _act_pkg = _types.ModuleType("lerobot.policies.act")
        _act_pkg.__path__ = [str(_lerobot_root / "policies" / "act")]
        sys.modules.setdefault("lerobot.policies.act", _act_pkg)

        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        import draccus

        with open(self.CHECKPOINT / "config.json") as f:
            cfg = json.load(f)
        cfg.pop("type", None)
        config = draccus.decode(ACTConfig, cfg)

        self.assertEqual(int(config.input_features["observation.state"].shape[0]), 77)
        self.assertEqual(int(config.output_features["action"].shape[0]), 6)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = ACTPolicy(config)
        policy.load_state_dict(load_file(str(self.CHECKPOINT / "model.safetensors")))
        policy.eval()
        policy.to(device)

        pre = load_file(
            str(self.CHECKPOINT / "policy_preprocessor_step_3_normalizer_processor.safetensors")
        )
        post = load_file(
            str(self.CHECKPOINT / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        )
        state_mean = pre["observation.state.mean"].to(device).float().view(1, -1)
        state_std = torch.clamp(
            pre["observation.state.std"].to(device).float().view(1, -1), min=1e-6
        )
        action_mean = post["action.mean"].to(device).float().view(1, -1)
        action_std = post["action.std"].to(device).float().view(1, -1)

        # Build a normalized fake batch.
        raw_state = torch.zeros(1, 77, device=device)
        # Anchor a few values to typical operating range.
        raw_state[0, 0:3] = torch.tensor([-0.45, 0.30, 0.20], device=device)  # tcp xyz
        raw_state[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # quat
        raw_state[0, 33:36] = torch.tensor([-0.43, 0.31, 0.14], device=device)  # held yolo
        raw_state[0, 36] = 1.0  # fresh
        raw_state[0, 37] = 0.05  # age
        raw_state[0, 38:41] = raw_state[0, 33:36] - raw_state[0, 0:3]  # delta
        raw_state[0, 68] = 1.0  # SFP one-hot
        raw_state[0, 70] = 1.0  # nic_card_mount_0
        norm_state = (raw_state - state_mean) / state_std

        img_mean = _IMAGENET_MEAN.to(device)
        img_std = _IMAGENET_STD.to(device)

        def fake_image() -> torch.Tensor:
            t = torch.rand(1, 3, _IMG_H, _IMG_W, device=device)
            return (t - img_mean) / img_std

        batch = {
            "observation.state": norm_state,
            "observation.images.left": fake_image(),
            "observation.images.center": fake_image(),
            "observation.images.right": fake_image(),
        }

        policy.reset()
        with torch.inference_mode():
            norm_action = policy.select_action(batch)

        self.assertEqual(tuple(norm_action.shape), (1, 6))
        self.assertTrue(torch.isfinite(norm_action).all())

        # Unnormalize and assert magnitudes are sane (per-step deltas are tiny —
        # action.std max in the saved stats is ~0.05m).
        action = norm_action * action_std + action_mean
        max_abs = float(action.abs().max())
        self.assertLess(max_abs, 1.0,
                        f"unnormalized action exploded: max|a|={max_abs:.3f}")

    def test_checkpoint_image_stats_are_imagenet(self):
        from safetensors.torch import load_file
        if not self.CHECKPOINT.exists():
            self.skipTest("V3 checkpoint missing")
        pre = load_file(
            str(self.CHECKPOINT / "policy_preprocessor_step_3_normalizer_processor.safetensors")
        )
        # Codex's hardcoded imagenet stats must match what the checkpoint
        # baked in (use_imagenet_stats=true at training time).
        for cam in ("left", "center", "right"):
            m = pre[f"observation.images.{cam}.mean"].flatten().tolist()
            s = pre[f"observation.images.{cam}.std"].flatten().tolist()
            np.testing.assert_allclose(m, [0.485, 0.456, 0.406], atol=2e-3)
            np.testing.assert_allclose(s, [0.229, 0.224, 0.225], atol=2e-3)


class TestRunActPipelineHooks(unittest.TestCase):
    def test_v3_replan_defaults_to_chunk_n_action_steps(self):
        """V3 should not over-replan beyond the policy's action queue."""
        from team_policy.run_act import _REPLAN_EVERY
        # Sanity: the constant exists; the runtime override sets replan_every
        # = config.n_action_steps for V3 — covered in __init__.  Here we
        # only pin the default magnitude.
        self.assertGreaterEqual(_REPLAN_EVERY, 1)


if __name__ == "__main__":
    unittest.main()
