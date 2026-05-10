"""Smoke-test the accessible ACT checkpoints used by ``team_policy.run_act``.

This intentionally loads the real checkpoints instead of a mock policy. It
catches the failure modes that pure helper tests cannot:

* checkpoint folder shape / required files
* LeRobot ACT config decoding
* model weight loading
* state/image tensor assembly against the deployed training schema
* one inference call producing a finite 6D Cartesian delta action

Run with:
    pixi run python -m unittest test.test_trained_model_v2_smoke -v
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from team_policy.run_act import RunACT, _SCHEMA_V2_30D, _SCHEMA_V3_77D


class _Logger:
    def info(self, _msg):
        pass

    def warning(self, _msg):
        pass

    def error(self, _msg):
        pass


class _Clock:
    def now(self):
        return SimpleNamespace(
            nanoseconds=0,
            to_msg=lambda: SimpleNamespace(),
        )


class _FakeNode:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

    def declare_parameter(self, name: str, default):
        if name == "checkpoint_path":
            return SimpleNamespace(value=str(self.checkpoint_path))
        return SimpleNamespace(value=default)

    def create_subscription(self, *_args, **_kwargs):
        return None

    def get_logger(self):
        return _Logger()

    def get_clock(self):
        return _Clock()


class _Image:
    height = 480
    width = 640
    step = 640 * 3
    encoding = "rgb8"
    data = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()


def _synthetic_observation():
    pose = SimpleNamespace(
        position=SimpleNamespace(x=-0.44, y=0.30, z=0.20),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    velocity = SimpleNamespace(
        linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    wrench = SimpleNamespace(
        force=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        torque=SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    return SimpleNamespace(
        controller_state=SimpleNamespace(
            tcp_pose=pose,
            tcp_velocity=velocity,
            tcp_error=[0.0] * 6,
        ),
        joint_states=SimpleNamespace(position=[0.0] * 7, velocity=[0.0] * 7),
        wrist_wrench=SimpleNamespace(wrench=wrench),
        left_image=_Image(),
        center_image=_Image(),
        right_image=_Image(),
    )


class TestTrainedModelV2Smoke(unittest.TestCase):
    def test_checkpoints_load_and_select_finite_action(self):
        model_root = Path(__file__).resolve().parents[1] / "team_policy" / "models"
        checkpoints = [
            (model_root / "trained_model_v3", _SCHEMA_V3_77D, 77),
            (model_root / "trained_model_V2" / "pretrained_model", _SCHEMA_V2_30D, 30),
            (model_root / "trained_model_V2_1" / "pretrained_model", _SCHEMA_V2_30D, 30),
        ]

        for checkpoint, expected_schema, expected_dim in checkpoints:
            with self.subTest(checkpoint=checkpoint.name):
                if not checkpoint.exists():
                    self.skipTest(f"checkpoint not present: {checkpoint}")

                policy = RunACT(_FakeNode(checkpoint))
                self.assertEqual(policy.schema, expected_schema)
                self.assertEqual(policy.state_dim, expected_dim)
                self.assertEqual(policy.action_dim, 6)

                policy._yolo_port_xyz = np.array([-0.42, 0.32, 0.18], dtype=np.float32)
                policy._yolo_port_valid = True
                policy._yolo_port_stamp_s = __import__("time").time()

                batch = policy._to_batch(_synthetic_observation())
                self.assertEqual(tuple(batch["observation.state"].shape), (1, expected_dim))
                self.assertEqual(tuple(batch["observation.images.left"].shape), (1, 3, 480, 640))
                self.assertEqual(tuple(batch["observation.images.center"].shape), (1, 3, 480, 640))
                self.assertEqual(tuple(batch["observation.images.right"].shape), (1, 3, 480, 640))

                with torch.inference_mode():
                    norm_action = policy.policy.select_action(batch)
                raw_action = (norm_action * policy.action_std + policy.action_mean)[0].cpu().numpy()
                shaped = policy._apply_action_shaping(raw_action)

                self.assertEqual(shaped.shape, (6,))
                self.assertTrue(np.isfinite(shaped).all())
                self.assertLessEqual(
                    np.linalg.norm(shaped[:3]),
                    policy.max_translation_delta_m + 1e-7,
                )
                self.assertLessEqual(
                    np.linalg.norm(shaped[3:6]),
                    policy.max_rotation_delta_rad + 1e-7,
                )


if __name__ == "__main__":
    unittest.main()
