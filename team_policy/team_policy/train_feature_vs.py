from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CAMERAS = ("left", "center", "right")
CAMERA_TO_ID = {name: i for i, name in enumerate(CAMERAS)}
ID_TO_CAMERA = {i: name for name, i in CAMERA_TO_ID.items()}

CAMERA_CORE_DIM = 24
CAMERA_EXTRA_DIM = 4
PER_CAMERA_DIM = CAMERA_CORE_DIM + CAMERA_EXTRA_DIM
MULTICAM_VISUAL_DIM = len(CAMERAS) * PER_CAMERA_DIM


def log(msg: str) -> None:
    print(msg, flush=True)


def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def as_float_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list) or isinstance(x, tuple):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        try:
            return np.asarray(json.loads(x), dtype=np.float32)
        except Exception:
            return np.fromstring(x.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


@dataclass
class Detection:
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        return 0.5 * (self.xyxy[0] + self.xyxy[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.xyxy[1] + self.xyxy[3])

    @property
    def w(self) -> float:
        return max(1.0, self.xyxy[2] - self.xyxy[0])

    @property
    def h(self) -> float:
        return max(1.0, self.xyxy[3] - self.xyxy[1])


@dataclass
class CameraMeasurement:
    camera: str
    camera_id: int
    ok: bool
    reason: str
    port_label: str = ""
    module_label: str = ""
    port_conf: float = 0.0
    module_conf: float = 0.0
    port_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    module_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    port_uv: tuple[float, float] = (0.0, 0.0)
    module_center_uv: tuple[float, float] = (0.0, 0.0)
    module_tip_uv: tuple[float, float] = (0.0, 0.0)
    err_uv: tuple[float, float] = (0.0, 0.0)
    err_px: float = 0.0
    angle_rad: float = 0.0
    feature_quality: float = 0.0
    port_edge_quality: float = 0.0
    module_edge_quality: float = 0.0
    port_corner_quality: float = 0.0
    module_corner_quality: float = 0.0
    score: float = 0.0


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FeatureVSNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
        mlp_blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        blocks = []
        for _ in range(mlp_blocks):
            blocks.append(ResidualBlock(hidden_size, dropout))
        self.head = nn.Sequential(
            *blocks,
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input(x)
        z, _ = self.gru(z)
        z = z[:, -1]
        return self.head(z)


class FeatureSequenceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        state_dim: int,
        augment: bool,
        state_noise_std: float,
        visual_noise_std: float,
        sequence_drop_prob: float,
    ):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.w = torch.from_numpy(w.astype(np.float32))
        self.state_dim = int(state_dim)
        self.augment = bool(augment)
        self.state_noise_std = float(state_noise_std)
        self.visual_noise_std = float(visual_noise_std)
        self.sequence_drop_prob = float(sequence_drop_prob)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x[idx].clone()

        if self.augment:
            if self.state_noise_std > 0:
                x[:, : self.state_dim] += torch.randn_like(x[:, : self.state_dim]) * self.state_noise_std

            if self.visual_noise_std > 0:
                for cam_i in range(len(CAMERAS)):
                    start = self.state_dim + cam_i * PER_CAMERA_DIM
                    end = start + CAMERA_CORE_DIM
                    x[:, start:end] += torch.randn_like(x[:, start:end]) * self.visual_noise_std

            if self.sequence_drop_prob > 0 and x.shape[0] > 1:
                for t in range(1, x.shape[0]):
                    if random.random() < self.sequence_drop_prob:
                        x[t] = x[t - 1]

        return x, self.y[idx], self.w[idx]


class Relabeler:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.root = Path(cfg.dataset_root)
        self.output_dir = Path(cfg.output_dir)
        self.port_labels = {x.strip() for x in cfg.port_labels.split(",") if x.strip()}
        self.module_labels = {x.strip() for x in cfg.module_labels.split(",") if x.strip()}
        self.target_dims = parse_int_list(cfg.target_dims)
        self.target_names = [self.action_dim_name(d) for d in self.target_dims]

        self.seq_buffers: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.cfg.seq_len))

        self.samples_x: list[np.ndarray] = []
        self.samples_y: list[np.ndarray] = []
        self.samples_w: list[float] = []

        self.meta_episode: list[int] = []
        self.meta_frame: list[int] = []
        self.meta_local_index: list[int] = []
        self.meta_best_camera: list[int] = []
        self.meta_valid_mask: list[list[int]] = []
        self.meta_valid_count: list[int] = []
        self.meta_best_err_px: list[float] = []
        self.meta_best_port_conf: list[float] = []
        self.meta_best_module_conf: list[float] = []
        self.meta_best_feature_quality: list[float] = []

        self.rejects = defaultdict(int)
        self.camera_valids = defaultdict(int)
        self.best_camera_accepts = defaultdict(int)
        self.valid_count_accepts = defaultdict(int)
        self.episode_stats: dict[int, dict[str, Any]] = {}

        self.yolo = None
        self.yolo_names = {}

    @staticmethod
    def action_dim_name(dim: int) -> str:
        names = {
            0: "dx",
            1: "dy",
            2: "dz",
            3: "roll",
            4: "pitch",
            5: "yaw",
        }
        return names.get(dim, f"action_{dim}")

    def load_yolo(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("ultralytics is not importable. Install ultralytics in this environment.") from e

        weights = Path(self.cfg.yolo_weights)
        if not weights.exists():
            fallback = self.root / "best.pt"
            if fallback.exists():
                weights = fallback
            else:
                raise RuntimeError(
                    f"YOLO weights not found: {self.cfg.yolo_weights}. "
                    f"Put best.pt under {self.root} or pass --yolo-weights."
                )

        log(f"[{now_s()}] YOLO_LOAD weights={weights}")
        self.yolo = YOLO(str(weights))
        self.yolo_names = dict(self.yolo.names)
        log(f"[{now_s()}] YOLO_NAMES {self.yolo_names}")

    def find_parquets(self) -> list[Path]:
        paths = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not paths:
            paths = sorted(self.root.glob("**/*.parquet"))
        return paths

    def video_path_for(self, parquet_path: Path, camera: str) -> Path:
        chunk = parquet_path.parent.name
        file_name = parquet_path.with_suffix(".mp4").name
        return self.root / "videos" / f"observation.images.{camera}" / chunk / file_name

    def read_frame(self, cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def yolo_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        assert self.yolo is not None
        results = self.yolo.predict(
            source=frames,
            imgsz=self.cfg.yolo_imgsz,
            conf=self.cfg.yolo_conf,
            iou=self.cfg.yolo_iou,
            verbose=False,
            device=self.cfg.yolo_device,
            batch=self.cfg.yolo_batch,
        )

        out: list[list[Detection]] = []
        for r in results:
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    cls = int(b.cls[0].item())
                    label = str(self.yolo_names.get(cls, cls))
                    conf = float(b.conf[0].item())
                    xyxy = tuple(float(v) for v in b.xyxy[0].detach().cpu().numpy().tolist())
                    dets.append(Detection(label=label, conf=conf, xyxy=xyxy))
            out.append(dets)
        return out

    def crop_quality(self, image: np.ndarray, xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
        h, w = image.shape[:2]
        x1 = int(clamp(xyxy[0], 0, w - 1))
        y1 = int(clamp(xyxy[1], 0, h - 1))
        x2 = int(clamp(xyxy[2], x1 + 1, w))
        y2 = int(clamp(xyxy[3], y1 + 1, h))
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return 0.0, 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx * sx + sy * sy)
        edge_q = float(np.clip(np.percentile(mag, 90) / 80.0, 0.0, 1.0))

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=80,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=3,
        )

        if corners is None:
            corner_q = 0.0
        else:
            area = max(1.0, float((x2 - x1) * (y2 - y1)))
            corner_q = float(np.clip(len(corners) / 40.0, 0.0, 1.0) * np.clip(area / 3000.0, 0.2, 1.0))

        return edge_q, corner_q

    def choose_measurement(self, image: np.ndarray, camera: str, dets: list[Detection]) -> CameraMeasurement:
        ports = [d for d in dets if d.label in self.port_labels and d.conf >= self.cfg.min_port_conf]
        modules = [d for d in dets if d.label in self.module_labels and d.conf >= self.cfg.min_module_conf]

        if not ports:
            return CameraMeasurement(camera=camera, camera_id=CAMERA_TO_ID[camera], ok=False, reason="missing_port")

        if not modules:
            return CameraMeasurement(camera=camera, camera_id=CAMERA_TO_ID[camera], ok=False, reason="missing_module")

        module = max(modules, key=lambda d: d.conf)
        mc = np.array([module.cx, module.cy], dtype=np.float32)

        port = min(
            ports,
            key=lambda d: np.linalg.norm(np.array([d.cx, d.cy], dtype=np.float32) - mc) - 40.0 * d.conf,
        )

        port_uv = np.array([port.cx, port.cy], dtype=np.float32)
        module_center = np.array([module.cx, module.cy], dtype=np.float32)

        if module.h >= module.w:
            candidates = [
                np.array([module.cx, module.xyxy[1]], dtype=np.float32),
                np.array([module.cx, module.xyxy[3]], dtype=np.float32),
            ]
        else:
            candidates = [
                np.array([module.xyxy[0], module.cy], dtype=np.float32),
                np.array([module.xyxy[2], module.cy], dtype=np.float32),
            ]

        module_tip = min(candidates, key=lambda p: np.linalg.norm(p - port_uv))
        err = port_uv - module_tip
        err_px = float(np.linalg.norm(err))

        if err_px > self.cfg.max_visual_error_px:
            return CameraMeasurement(camera=camera, camera_id=CAMERA_TO_ID[camera], ok=False, reason="visual_error_too_large")

        port_edge_q, port_corner_q = self.crop_quality(image, port.xyxy)
        module_edge_q, module_corner_q = self.crop_quality(image, module.xyxy)

        feature_quality = float(
            np.clip(
                0.30 * port_edge_q
                + 0.20 * port_corner_q
                + 0.30 * module_edge_q
                + 0.20 * module_corner_q,
                0.0,
                1.0,
            )
        )

        if feature_quality < self.cfg.min_feature_quality:
            return CameraMeasurement(camera=camera, camera_id=CAMERA_TO_ID[camera], ok=False, reason="feature_quality_low")

        angle_rad = float(math.atan2(float(err[1]), float(err[0])))

        score = (
            1.6 * float(port.conf)
            + 1.6 * float(module.conf)
            + 1.1 * feature_quality
            - 0.0035 * err_px
        )

        return CameraMeasurement(
            camera=camera,
            camera_id=CAMERA_TO_ID[camera],
            ok=True,
            reason="ok",
            port_label=port.label,
            module_label=module.label,
            port_conf=float(port.conf),
            module_conf=float(module.conf),
            port_xyxy=port.xyxy,
            module_xyxy=module.xyxy,
            port_uv=(float(port_uv[0]), float(port_uv[1])),
            module_center_uv=(float(module_center[0]), float(module_center[1])),
            module_tip_uv=(float(module_tip[0]), float(module_tip[1])),
            err_uv=(float(err[0]), float(err[1])),
            err_px=err_px,
            angle_rad=angle_rad,
            feature_quality=feature_quality,
            port_edge_quality=port_edge_q,
            module_edge_quality=module_edge_q,
            port_corner_quality=port_corner_q,
            module_corner_quality=module_corner_q,
            score=float(score),
        )

    def measurement_to_core(self, m: CameraMeasurement, image_hw: tuple[int, int]) -> np.ndarray:
        h, w = image_hw

        px1, py1, px2, py2 = m.port_xyxy
        mx1, my1, mx2, my2 = m.module_xyxy

        err_x = m.err_uv[0] / float(w)
        err_y = m.err_uv[1] / float(h)
        err_norm = m.err_px / float(math.sqrt(w * w + h * h))

        port_x = m.port_uv[0] / float(w)
        port_y = m.port_uv[1] / float(h)
        tip_x = m.module_tip_uv[0] / float(w)
        tip_y = m.module_tip_uv[1] / float(h)
        module_cx = m.module_center_uv[0] / float(w)
        module_cy = m.module_center_uv[1] / float(h)

        port_w = max(1.0, px2 - px1) / float(w)
        port_h = max(1.0, py2 - py1) / float(h)
        module_w = max(1.0, mx2 - mx1) / float(w)
        module_h = max(1.0, my2 - my1) / float(h)

        return np.asarray(
            [
                err_x,
                err_y,
                err_norm,
                port_x,
                port_y,
                tip_x,
                tip_y,
                module_cx,
                module_cy,
                port_w,
                port_h,
                module_w,
                module_h,
                m.port_conf,
                m.module_conf,
                m.feature_quality,
                m.port_edge_quality,
                m.module_edge_quality,
                m.port_corner_quality,
                m.module_corner_quality,
                math.sin(m.angle_rad),
                math.cos(m.angle_rad),
                port_w / max(port_h, 1e-6),
                module_w / max(module_h, 1e-6),
            ],
            dtype=np.float32,
        )

    def invalid_camera_block(self, camera: str) -> np.ndarray:
        cam_id = CAMERA_TO_ID[camera]
        cam_id_norm = float(cam_id) / float(max(1, len(CAMERAS) - 1))
        core = np.zeros(CAMERA_CORE_DIM, dtype=np.float32)
        extra = np.asarray([0.0, 0.0, self.cfg.invalid_err_norm, cam_id_norm], dtype=np.float32)
        return np.concatenate([core, extra], axis=0)

    def valid_camera_block(self, m: CameraMeasurement, image_hw: tuple[int, int]) -> np.ndarray:
        cam_id_norm = float(m.camera_id) / float(max(1, len(CAMERAS) - 1))
        score_norm = clamp(m.score / self.cfg.score_scale, 0.0, 1.0)
        err_px_norm = clamp(m.err_px / self.cfg.max_visual_error_px, 0.0, 1.0)
        core = self.measurement_to_core(m, image_hw)
        extra = np.asarray([1.0, score_norm, err_px_norm, cam_id_norm], dtype=np.float32)
        return np.concatenate([core, extra], axis=0)

    def build_multicam_feature(
        self,
        state: np.ndarray,
        measurements: dict[str, CameraMeasurement],
        detections_by_camera: dict[str, tuple[np.ndarray, list[Detection]]],
    ) -> np.ndarray:
        blocks = []

        for cam in CAMERAS:
            m = measurements.get(cam)
            if m is not None and m.ok:
                image_hw = detections_by_camera[cam][0].shape[:2]
                blocks.append(self.valid_camera_block(m, image_hw))
            else:
                blocks.append(self.invalid_camera_block(cam))

        return np.concatenate([state.astype(np.float32), *blocks], axis=0).astype(np.float32)

    def make_label(self, actions: np.ndarray, episodes: np.ndarray, local_idx: int) -> np.ndarray | None:
        ep = episodes[local_idx]
        end = min(len(actions), local_idx + max(1, self.cfg.label_horizon))
        valid = []

        for j in range(local_idx, end):
            if episodes[j] != ep:
                break
            valid.append(actions[j])

        if not valid:
            return None

        a = np.mean(np.stack(valid, axis=0), axis=0).astype(np.float32)

        if max(self.target_dims) >= len(a):
            raise RuntimeError(f"target dim {max(self.target_dims)} exceeds action dimension {len(a)}")

        y = a[self.target_dims].astype(np.float32)

        if self.cfg.label_clip > 0:
            y = np.clip(y, -self.cfg.label_clip, self.cfg.label_clip)

        if not np.all(np.isfinite(y)):
            return None

        return y

    def sample_weight(self, valid_measurements: list[CameraMeasurement], y: np.ndarray) -> float:
        best = max(valid_measurements, key=lambda m: m.score)
        conf = 0.5 * (best.port_conf + best.module_conf)
        q = best.feature_quality
        err_factor = 1.0 - clamp(best.err_px / self.cfg.max_visual_error_px, 0.0, 0.85)

        valid_count = len(valid_measurements)
        valid_bonus = 1.0 + 0.15 * max(0, valid_count - 1)

        action_norm = float(np.linalg.norm(y))
        action_factor = 1.0

        if self.cfg.downweight_large_actions > 0:
            action_factor = 1.0 / (1.0 + action_norm / self.cfg.downweight_large_actions)

        weight = self.cfg.min_sample_weight + 2.0 * conf * q * err_factor * action_factor * valid_bonus
        return float(clamp(weight, self.cfg.min_sample_weight, self.cfg.max_sample_weight))

    def process_one_selected_frame(
        self,
        row: pd.Series,
        local_idx: int,
        detections_by_camera: dict[str, tuple[np.ndarray, list[Detection]]],
        states: np.ndarray,
        actions: np.ndarray,
        episodes: np.ndarray,
    ) -> None:
        ep = int(episodes[local_idx])
        frame_index = int(row.get("frame_index", local_idx))

        measurements: dict[str, CameraMeasurement] = {}
        valid_measurements: list[CameraMeasurement] = []

        for cam in CAMERAS:
            img, dets = detections_by_camera[cam]
            m = self.choose_measurement(img, cam, dets)
            measurements[cam] = m

            if m.ok:
                valid_measurements.append(m)
                self.camera_valids[cam] += 1
            else:
                self.rejects[f"{cam}:{m.reason}"] += 1

        valid_count = len(valid_measurements)

        if valid_count < self.cfg.min_valid_cameras:
            self.rejects["sample:not_enough_valid_cameras"] += 1
            return

        if valid_count == 0:
            self.rejects["sample:no_valid_camera"] += 1
            return

        y = self.make_label(actions, episodes, local_idx)

        if y is None:
            self.rejects["sample:bad_label"] += 1
            return

        if self.cfg.min_abs_label > 0 and float(np.linalg.norm(y)) < self.cfg.min_abs_label:
            self.rejects["sample:label_too_small"] += 1
            return

        x = self.build_multicam_feature(states[local_idx], measurements, detections_by_camera)

        if not np.all(np.isfinite(x)):
            self.rejects["sample:bad_feature"] += 1
            return

        self.seq_buffers[ep].append(x)

        if len(self.seq_buffers[ep]) < self.cfg.seq_len:
            self.rejects["sample:seq_warmup"] += 1
            return

        seq = np.stack(list(self.seq_buffers[ep]), axis=0).astype(np.float32)
        weight = self.sample_weight(valid_measurements, y)

        best = max(valid_measurements, key=lambda m: m.score)
        valid_mask = [1 if measurements[cam].ok else 0 for cam in CAMERAS]

        self.samples_x.append(seq)
        self.samples_y.append(y.astype(np.float32))
        self.samples_w.append(weight)

        self.meta_episode.append(ep)
        self.meta_frame.append(frame_index)
        self.meta_local_index.append(local_idx)
        self.meta_best_camera.append(best.camera_id)
        self.meta_valid_mask.append(valid_mask)
        self.meta_valid_count.append(valid_count)
        self.meta_best_err_px.append(best.err_px)
        self.meta_best_port_conf.append(best.port_conf)
        self.meta_best_module_conf.append(best.module_conf)
        self.meta_best_feature_quality.append(best.feature_quality)

        self.best_camera_accepts[best.camera] += 1
        self.valid_count_accepts[str(valid_count)] += 1

        st = self.episode_stats.setdefault(
            ep,
            {
                "accepted": 0,
                "left_valid": 0,
                "center_valid": 0,
                "right_valid": 0,
                "best_left": 0,
                "best_center": 0,
                "best_right": 0,
                "first_frame": frame_index,
                "last_frame": frame_index,
            },
        )

        st["accepted"] += 1
        st["last_frame"] = frame_index
        st[f"best_{best.camera}"] += 1

        for cam in CAMERAS:
            if measurements[cam].ok:
                st[f"{cam}_valid"] += 1

        n = len(self.samples_x)
        if n % self.cfg.accept_log_every == 0:
            valid_text = ",".join([cam for cam in CAMERAS if measurements[cam].ok])
            log(
                f"[{now_s()}] RELABEL_ACCEPTED n={n} ep={ep} frame={frame_index} "
                f"valid={valid_text} best={best.camera} err={best.err_px:.1f}px "
                f"port_conf={best.port_conf:.2f} module_conf={best.module_conf:.2f} "
                f"feat={best.feature_quality:.2f} y={np.round(y, 5).tolist()}"
            )

    def relabel_parquet(self, parquet_path: Path, parquet_id: int, total_parquets: int) -> None:
        log(f"[{now_s()}] PARQUET_START {parquet_id + 1}/{total_parquets} path={parquet_path}")

        df = pd.read_parquet(parquet_path)
        log(f"[{now_s()}] PARQUET_INFO rows={len(df)} columns={list(df.columns)}")

        required = ["observation.state", "action", "episode_index"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise RuntimeError(f"Missing columns in {parquet_path}: {missing}")

        states = np.stack([as_float_array(v) for v in df["observation.state"].values]).astype(np.float32)
        actions = np.stack([as_float_array(v) for v in df["action"].values]).astype(np.float32)
        episodes = df["episode_index"].to_numpy(dtype=np.int64)

        log(
            f"[{now_s()}] PARQUET_ARRAYS state_shape={states.shape} action_shape={actions.shape} "
            f"episodes={len(np.unique(episodes))}"
        )

        videos = {}

        for cam in CAMERAS:
            vp = self.video_path_for(parquet_path, cam)

            if not vp.exists():
                raise RuntimeError(f"Missing video for camera={cam}: {vp}")

            cap = cv2.VideoCapture(str(vp))

            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for camera={cam}: {vp}")

            videos[cam] = cap

            log(
                f"[{now_s()}] VIDEO camera={cam} path={vp} "
                f"frames={int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} "
                f"size={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )

        unique_eps = np.unique(episodes)
        selected_indices: list[int] = []

        for ep_i, ep in enumerate(unique_eps):
            ep_indices = np.where(episodes == ep)[0]

            if len(ep_indices) == 0:
                continue

            if self.cfg.max_frames_per_episode > 0:
                ep_indices = ep_indices[: self.cfg.max_frames_per_episode]

            ep_selected = ep_indices[:: self.cfg.frame_stride]
            selected_indices.extend(ep_selected.tolist())

            if ep_i % self.cfg.episode_log_every == 0:
                log(
                    f"[{now_s()}] EPISODE_QUEUE parquet={parquet_id + 1}/{total_parquets} "
                    f"ep={int(ep)} ep_idx={ep_i + 1}/{len(unique_eps)} "
                    f"frames={len(ep_indices)} selected={len(ep_selected)} "
                    f"range=({int(ep_indices[0])},{int(ep_indices[-1])})"
                )

        if self.cfg.max_frames_per_parquet > 0:
            selected_indices = selected_indices[: self.cfg.max_frames_per_parquet]

        selected_indices = sorted(set(int(i) for i in selected_indices))

        log(
            f"[{now_s()}] PARQUET_SELECTED count={len(selected_indices)} "
            f"frame_stride={self.cfg.frame_stride} mode=multicam_fused"
        )

        batch_frames: list[np.ndarray] = []
        batch_context: list[tuple[int, str]] = []

        def flush_batch() -> None:
            nonlocal batch_frames, batch_context

            if not batch_frames:
                return

            det_batches = self.yolo_batch(batch_frames)
            grouped: dict[int, dict[str, tuple[np.ndarray, list[Detection]]]] = defaultdict(dict)

            for (idx, cam), img, dets in zip(batch_context, batch_frames, det_batches):
                grouped[idx][cam] = (img, dets)

            for idx in sorted(grouped.keys()):
                if all(cam in grouped[idx] for cam in CAMERAS):
                    self.process_one_selected_frame(
                        row=df.iloc[idx],
                        local_idx=idx,
                        detections_by_camera=grouped[idx],
                        states=states,
                        actions=actions,
                        episodes=episodes,
                    )
                else:
                    self.rejects["sample:missing_camera_frame"] += 1

            batch_frames = []
            batch_context = []

        last_ep = None

        for count_i, idx in enumerate(selected_indices):
            ep = int(episodes[idx])

            if ep != last_ep:
                last_ep = ep
                log(
                    f"[{now_s()}] EPISODE_PROCESS ep={ep} local_start={idx} "
                    f"selected_progress={count_i + 1}/{len(selected_indices)}"
                )

            for cam in CAMERAS:
                frame = self.read_frame(videos[cam], idx)

                if frame is None:
                    self.rejects[f"{cam}:frame_read_failed"] += 1
                    continue

                batch_frames.append(frame)
                batch_context.append((idx, cam))

            if len(batch_frames) >= self.cfg.yolo_batch_frames:
                flush_batch()

            if self.cfg.max_samples > 0 and len(self.samples_x) >= self.cfg.max_samples:
                break

        flush_batch()

        for cap in videos.values():
            cap.release()

        log(
            f"[{now_s()}] PARQUET_DONE path={parquet_path} "
            f"accepted_total={len(self.samples_x)} rejects={dict(list(self.rejects.items())[:16])}"
        )

    def relabel(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        self.load_yolo()
        parquets = self.find_parquets()

        if not parquets:
            raise RuntimeError(f"No parquet files found under {self.root}")

        log(f"[{now_s()}] MULTICAM_SAMPLE enabled camera_order={CAMERAS}")
        log(f"[{now_s()}] DATASET_ROOT {self.root}")
        log(f"[{now_s()}] PARQUETS_FOUND count={len(parquets)}")

        for p in parquets[:10]:
            log(f"[{now_s()}] PARQUET {p}")

        if len(parquets) > 10:
            log(f"[{now_s()}] PARQUET ... remaining={len(parquets) - 10}")

        for pi, parquet_path in enumerate(parquets):
            self.relabel_parquet(parquet_path, pi, len(parquets))

            if self.cfg.max_samples > 0 and len(self.samples_x) >= self.cfg.max_samples:
                log(f"[{now_s()}] RELABEL_STOP reason=max_samples n={len(self.samples_x)}")
                break

        if len(self.samples_x) == 0:
            raise RuntimeError(
                "Relabeling produced zero samples. "
                "Check video decoding, YOLO weights, class names, thresholds, and --frame-stride."
            )

        X = np.stack(self.samples_x, axis=0).astype(np.float32)
        Y = np.stack(self.samples_y, axis=0).astype(np.float32)
        W = np.asarray(self.samples_w, dtype=np.float32)

        meta = {
            "episode": np.asarray(self.meta_episode, dtype=np.int64),
            "frame": np.asarray(self.meta_frame, dtype=np.int64),
            "local_index": np.asarray(self.meta_local_index, dtype=np.int64),
            "best_camera": np.asarray(self.meta_best_camera, dtype=np.int64),
            "valid_mask": np.asarray(self.meta_valid_mask, dtype=np.int64),
            "valid_count": np.asarray(self.meta_valid_count, dtype=np.int64),
            "best_err_px": np.asarray(self.meta_best_err_px, dtype=np.float32),
            "best_port_conf": np.asarray(self.meta_best_port_conf, dtype=np.float32),
            "best_module_conf": np.asarray(self.meta_best_module_conf, dtype=np.float32),
            "best_feature_quality": np.asarray(self.meta_best_feature_quality, dtype=np.float32),
        }

        state_dim = int(X.shape[-1] - MULTICAM_VISUAL_DIM)

        info = {
            "dataset_root": str(self.root),
            "sample_mode": "multicam_fused",
            "camera_order": list(CAMERAS),
            "target_dims": self.target_dims,
            "target_names": self.target_names,
            "seq_len": self.cfg.seq_len,
            "input_dim": int(X.shape[-1]),
            "state_dim": state_dim,
            "visual_dim": MULTICAM_VISUAL_DIM,
            "camera_core_dim": CAMERA_CORE_DIM,
            "camera_extra_dim": CAMERA_EXTRA_DIM,
            "per_camera_dim": PER_CAMERA_DIM,
            "num_samples": int(len(X)),
            "camera_valids": dict(self.camera_valids),
            "best_camera_accepts": dict(self.best_camera_accepts),
            "valid_count_accepts": dict(self.valid_count_accepts),
            "rejects": dict(self.rejects),
            "episode_stats": self.episode_stats,
            "yolo_names": self.yolo_names,
            "port_labels": sorted(self.port_labels),
            "module_labels": sorted(self.module_labels),
            "feature_names": self.feature_names(state_dim),
        }

        log(f"[{now_s()}] RELABEL_DONE X={X.shape} Y={Y.shape} W={W.shape}")
        log(f"[{now_s()}] INPUT_DIMS state_dim={state_dim} per_camera_dim={PER_CAMERA_DIM} visual_dim={MULTICAM_VISUAL_DIM} input_dim={X.shape[-1]}")
        log(f"[{now_s()}] CAMERA_VALIDS {dict(self.camera_valids)}")
        log(f"[{now_s()}] BEST_CAMERA_ACCEPTS {dict(self.best_camera_accepts)}")
        log(f"[{now_s()}] VALID_COUNT_ACCEPTS {dict(self.valid_count_accepts)}")
        log(f"[{now_s()}] TARGET_DIMS {self.target_dims} TARGET_NAMES {self.target_names}")
        log(f"[{now_s()}] REJECTS_TOP {dict(sorted(self.rejects.items(), key=lambda kv: kv[1], reverse=True)[:40])}")

        return X, Y, W, meta, info

    @staticmethod
    def feature_names(state_dim: int) -> list[str]:
        core_names = [
            "err_x_norm",
            "err_y_norm",
            "err_norm",
            "port_x_norm",
            "port_y_norm",
            "module_tip_x_norm",
            "module_tip_y_norm",
            "module_center_x_norm",
            "module_center_y_norm",
            "port_w_norm",
            "port_h_norm",
            "module_w_norm",
            "module_h_norm",
            "port_conf",
            "module_conf",
            "feature_quality",
            "port_edge_quality",
            "module_edge_quality",
            "port_corner_quality",
            "module_corner_quality",
            "err_angle_sin",
            "err_angle_cos",
            "port_aspect",
            "module_aspect",
        ]

        extra_names = [
            "valid",
            "score_norm",
            "err_px_norm",
            "camera_id_norm",
        ]

        names = [f"state_{i}" for i in range(state_dim)]

        for cam in CAMERAS:
            names.extend([f"{cam}_{name}" for name in core_names + extra_names])

        return names


def build_or_load_cache(cfg: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / cfg.cache_name
    info_path = output_dir / "feature_vs_cache.debug.json"

    if cache_path.exists() and not cfg.rebuild_cache:
        log(f"[{now_s()}] CACHE_LOAD {cache_path}")
        z = np.load(cache_path, allow_pickle=True)

        X = z["X"].astype(np.float32)
        Y = z["Y"].astype(np.float32)
        W = z["W"].astype(np.float32)

        meta = {
            "episode": z["meta_episode"].astype(np.int64),
            "frame": z["meta_frame"].astype(np.int64),
            "local_index": z["meta_local_index"].astype(np.int64),
            "best_camera": z["meta_best_camera"].astype(np.int64),
            "valid_mask": z["meta_valid_mask"].astype(np.int64),
            "valid_count": z["meta_valid_count"].astype(np.int64),
            "best_err_px": z["meta_best_err_px"].astype(np.float32),
            "best_port_conf": z["meta_best_port_conf"].astype(np.float32),
            "best_module_conf": z["meta_best_module_conf"].astype(np.float32),
            "best_feature_quality": z["meta_best_feature_quality"].astype(np.float32),
        }

        if info_path.exists():
            info = json.loads(info_path.read_text())
        else:
            info = {}

        if info.get("sample_mode") != "multicam_fused":
            log(f"[{now_s()}] WARN cache sample_mode is not multicam_fused. Use --rebuild-cache if this is an old cache.")

        log(f"[{now_s()}] CACHE_LOADED X={X.shape} Y={Y.shape} W={W.shape}")
        return X, Y, W, meta, info

    relabeler = Relabeler(cfg)
    X, Y, W, meta, info = relabeler.relabel()

    np.savez_compressed(
        cache_path,
        X=X,
        Y=Y,
        W=W,
        meta_episode=meta["episode"],
        meta_frame=meta["frame"],
        meta_local_index=meta["local_index"],
        meta_best_camera=meta["best_camera"],
        meta_valid_mask=meta["valid_mask"],
        meta_valid_count=meta["valid_count"],
        meta_best_err_px=meta["best_err_px"],
        meta_best_port_conf=meta["best_port_conf"],
        meta_best_module_conf=meta["best_module_conf"],
        meta_best_feature_quality=meta["best_feature_quality"],
    )

    info_path.write_text(json.dumps(to_jsonable(info), indent=2))

    log(f"[{now_s()}] CACHE_SAVED {cache_path}")
    log(f"[{now_s()}] CACHE_DEBUG_SAVED {info_path}")

    return X, Y, W, meta, info


def split_by_episode(meta: dict[str, np.ndarray], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    episodes = np.unique(meta["episode"])
    rng.shuffle(episodes)

    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val_eps = set(int(x) for x in episodes[:n_val])

    is_val = np.asarray([int(ep) in val_eps for ep in meta["episode"]], dtype=bool)
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        n = len(meta["episode"])
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(round(n * val_fraction)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    return train_idx, val_idx


def compute_norm(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std


def compute_y_norm(y: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = y.mean(axis=0, keepdims=True).astype(np.float32)
    std = y.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std


def camera_distribution_text(camera_ids: np.ndarray) -> str:
    parts = []
    total = max(1, int(len(camera_ids)))

    for cam_id, name in ID_TO_CAMERA.items():
        n = int(np.sum(camera_ids == cam_id))
        parts.append(f"{name}={n}({100.0 * n / total:.1f}%)")

    return " ".join(parts)


def valid_count_distribution_text(valid_count: np.ndarray) -> str:
    parts = []
    total = max(1, int(len(valid_count)))

    for n in sorted(np.unique(valid_count).tolist()):
        c = int(np.sum(valid_count == n))
        parts.append(f"{int(n)}cam={c}({100.0 * c / total:.1f}%)")

    return " ".join(parts)


def valid_mask_distribution(meta: dict[str, np.ndarray]) -> dict[str, int]:
    masks = meta["valid_mask"]
    out = defaultdict(int)

    for row in masks:
        key = "".join([CAMERAS[i][0].upper() if int(v) == 1 else "-" for i, v in enumerate(row)])
        out[key] += 1

    return dict(out)


def parse_target_loss_weights(cfg: argparse.Namespace, output_dim: int, device: torch.device) -> torch.Tensor:
    vals = parse_float_list(cfg.target_loss_weights)

    if not vals:
        vals = [1.0] * output_dim

    if len(vals) != output_dim:
        raise RuntimeError(f"--target-loss-weights has {len(vals)} values but output_dim={output_dim}")

    arr = np.asarray(vals, dtype=np.float32)
    arr = arr / max(1e-6, float(arr.mean()))

    return torch.from_numpy(arr).to(device)


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float,
    target_weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    loss = loss * target_weight.reshape(1, -1)
    loss = loss.mean(dim=1)
    return (loss * weight).sum() / (weight.sum() + 1e-6)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_names: list[str],
    target_loss_weights: torch.Tensor,
    huber_beta: float,
) -> dict[str, Any]:
    model.eval()

    preds = []
    tgts = []
    losses = []

    for xb, yb, wb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)

        pred = model(xb)
        loss = weighted_huber_loss(
            pred=pred,
            target=yb,
            weight=wb,
            beta=huber_beta,
            target_weight=target_loss_weights,
        )

        losses.append(float(loss.item()))
        preds.append(pred.detach().cpu().numpy())
        tgts.append(yb.detach().cpu().numpy())

    pred_n = np.concatenate(preds, axis=0)
    tgt_n = np.concatenate(tgts, axis=0)

    pred = pred_n * y_std.reshape(1, -1) + y_mean.reshape(1, -1)
    tgt = tgt_n * y_std.reshape(1, -1) + y_mean.reshape(1, -1)

    mae = np.mean(np.abs(pred - tgt), axis=0)
    rmse = np.sqrt(np.mean((pred - tgt) ** 2, axis=0))

    return {
        "loss": float(np.mean(losses)),
        "mae": {name: float(v) for name, v in zip(target_names, mae)},
        "rmse": {name: float(v) for name, v in zip(target_names, rmse)},
    }


@torch.no_grad()
def evaluate_subset(
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    idx: np.ndarray,
    state_dim: int,
    cfg: argparse.Namespace,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_names: list[str],
    target_loss_weights: torch.Tensor,
) -> dict[str, Any] | None:
    if len(idx) == 0:
        return None

    ds = FeatureSequenceDataset(
        x=x_val[idx],
        y=y_val[idx],
        w=w_val[idx],
        state_dim=state_dim,
        augment=False,
        state_noise_std=0.0,
        visual_noise_std=0.0,
        sequence_drop_prob=0.0,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    m = evaluate(
        model=model,
        loader=loader,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        target_names=target_names,
        target_loss_weights=target_loss_weights,
        huber_beta=cfg.huber_beta,
    )

    m["count"] = int(len(idx))
    return m


@torch.no_grad()
def evaluate_multicam_groups(
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    meta_val: dict[str, np.ndarray],
    state_dim: int,
    cfg: argparse.Namespace,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_names: list[str],
    target_loss_weights: torch.Tensor,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    best_cam = meta_val["best_camera"]
    valid_count = meta_val["valid_count"]

    by_best = {}
    for cam_id, name in ID_TO_CAMERA.items():
        idx = np.where(best_cam == cam_id)[0]
        m = evaluate_subset(
            model=model,
            x_val=x_val,
            y_val=y_val,
            w_val=w_val,
            idx=idx,
            state_dim=state_dim,
            cfg=cfg,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            target_names=target_names,
            target_loss_weights=target_loss_weights,
        )
        if m is not None:
            by_best[name] = m

    by_count = {}
    for n in sorted(np.unique(valid_count).tolist()):
        idx = np.where(valid_count == n)[0]
        m = evaluate_subset(
            model=model,
            x_val=x_val,
            y_val=y_val,
            w_val=w_val,
            idx=idx,
            state_dim=state_dim,
            cfg=cfg,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            target_names=target_names,
            target_loss_weights=target_loss_weights,
        )
        if m is not None:
            by_count[f"{int(n)}cam"] = m

    out["by_best_camera"] = by_best
    out["by_valid_count"] = by_count
    return out


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: argparse.Namespace,
    info: dict[str, Any],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    target_dims = parse_int_list(cfg.target_dims)

    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": vars(cfg),
        "info": info,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "target_dims": target_dims,
        "target_names": [Relabeler.action_dim_name(d) for d in target_dims],
        "input_dim": int(info.get("input_dim", 0)),
        "output_dim": len(target_dims),
        "model_type": "FeatureVSNet",
        "sample_mode": "multicam_fused",
        "action_order_assumption": "dx,dy,dz,roll,pitch,yaw",
        "camera_order": list(CAMERAS),
        "camera_core_dim": CAMERA_CORE_DIM,
        "camera_extra_dim": CAMERA_EXTRA_DIM,
        "per_camera_dim": PER_CAMERA_DIM,
        "visual_dim": MULTICAM_VISUAL_DIM,
        "state_dim": int(info.get("state_dim", 0)),
        "feature_names": info.get("feature_names", []),
    }

    torch.save(payload, path)


def train(cfg: argparse.Namespace) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[{now_s()}] TRAIN_START")
    log(f"[{now_s()}] CONFIG {json.dumps(to_jsonable(vars(cfg)), indent=2)}")

    target_dims = parse_int_list(cfg.target_dims)
    target_names = [Relabeler.action_dim_name(d) for d in target_dims]

    log(f"[{now_s()}] MULTICAM_FUSED_TRAINING enabled")
    log(f"[{now_s()}] CAMERA_ORDER {CAMERAS}")

    if 4 not in target_dims:
        log(f"[{now_s()}] WARN pitch dim 4 is not in target_dims={target_dims}")
    else:
        log(f"[{now_s()}] PITCH_ENABLED target_dims={target_dims} target_names={target_names}")

    X, Y, W, meta, info = build_or_load_cache(cfg)

    if Y.shape[1] != len(target_dims):
        raise RuntimeError(f"Y dim mismatch: Y.shape={Y.shape}, target_dims={target_dims}")

    state_dim = int(info.get("state_dim", X.shape[-1] - MULTICAM_VISUAL_DIM))

    if X.shape[-1] != state_dim + MULTICAM_VISUAL_DIM:
        raise RuntimeError(
            f"Input dim mismatch: input_dim={X.shape[-1]} state_dim={state_dim} "
            f"expected={state_dim + MULTICAM_VISUAL_DIM}"
        )

    train_idx, val_idx = split_by_episode(meta, cfg.val_fraction, cfg.seed)

    log(f"[{now_s()}] SPLIT train={len(train_idx)} val={len(val_idx)} total={len(X)}")
    log(f"[{now_s()}] BEST_CAMERA_DIST_ALL {camera_distribution_text(meta['best_camera'])}")
    log(f"[{now_s()}] BEST_CAMERA_DIST_TRAIN {camera_distribution_text(meta['best_camera'][train_idx])}")
    log(f"[{now_s()}] BEST_CAMERA_DIST_VAL {camera_distribution_text(meta['best_camera'][val_idx])}")
    log(f"[{now_s()}] VALID_COUNT_DIST_ALL {valid_count_distribution_text(meta['valid_count'])}")
    log(f"[{now_s()}] VALID_MASK_DIST_ALL {valid_mask_distribution(meta)}")

    x_mean, x_std = compute_norm(X[train_idx])
    y_mean, y_std = compute_y_norm(Y[train_idx])

    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    X_train_np = Xn[train_idx].astype(np.float32)
    Y_train_np = Yn[train_idx].astype(np.float32)
    W_train_np = W[train_idx].astype(np.float32)

    X_val_np = Xn[val_idx].astype(np.float32)
    Y_val_np = Yn[val_idx].astype(np.float32)
    W_val_np = W[val_idx].astype(np.float32)

    meta_val = {
        "best_camera": meta["best_camera"][val_idx],
        "valid_count": meta["valid_count"][val_idx],
        "valid_mask": meta["valid_mask"][val_idx],
    }

    train_ds = FeatureSequenceDataset(
        x=X_train_np,
        y=Y_train_np,
        w=W_train_np,
        state_dim=state_dim,
        augment=cfg.augment,
        state_noise_std=cfg.state_noise_std,
        visual_noise_std=cfg.visual_noise_std,
        sequence_drop_prob=cfg.sequence_drop_prob,
    )

    val_ds = FeatureSequenceDataset(
        x=X_val_np,
        y=Y_val_np,
        w=W_val_np,
        state_dim=state_dim,
        augment=False,
        state_noise_std=0.0,
        visual_noise_std=0.0,
        sequence_drop_prob=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    log(f"[{now_s()}] DEVICE {device}")

    model = FeatureVSNet(
        input_dim=X.shape[-1],
        output_dim=Y.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        mlp_blocks=cfg.mlp_blocks,
        dropout=cfg.dropout,
    ).to(device)

    target_loss_weights = parse_target_loss_weights(cfg, Y.shape[-1], device)

    log(f"[{now_s()}] MODEL input_dim={X.shape[-1]} output_dim={Y.shape[-1]} targets={target_names}")
    log(f"[{now_s()}] INPUT_DIMS state_dim={state_dim} per_camera_dim={PER_CAMERA_DIM} visual_dim={MULTICAM_VISUAL_DIM}")
    log(f"[{now_s()}] TARGET_LOSS_WEIGHTS {target_loss_weights.detach().cpu().numpy().round(4).tolist()}")
    log(f"[{now_s()}] NORMALIZATION y_mean={np.round(y_mean.reshape(-1), 6).tolist()} y_std={np.round(y_std.reshape(-1), 6).tolist()}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, cfg.epochs),
        eta_min=cfg.min_lr,
    )

    best_score = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    train_info = {
        "sample_mode": "multicam_fused",
        "camera_order": list(CAMERAS),
        "target_dims": target_dims,
        "target_names": target_names,
        "input_dim": int(X.shape[-1]),
        "output_dim": int(Y.shape[-1]),
        "seq_len": int(cfg.seq_len),
        "num_samples": int(len(X)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "state_dim": state_dim,
        "visual_dim": MULTICAM_VISUAL_DIM,
        "camera_core_dim": CAMERA_CORE_DIM,
        "camera_extra_dim": CAMERA_EXTRA_DIM,
        "per_camera_dim": PER_CAMERA_DIM,
        "x_mean_shape": list(x_mean.shape),
        "x_std_shape": list(x_std.shape),
        "y_mean": y_mean.reshape(-1).tolist(),
        "y_std": y_std.reshape(-1).tolist(),
        "cache_info": info,
        "best_camera_dist_all": {ID_TO_CAMERA[i]: int(np.sum(meta["best_camera"] == i)) for i in range(len(CAMERAS))},
        "best_camera_dist_train": {ID_TO_CAMERA[i]: int(np.sum(meta["best_camera"][train_idx] == i)) for i in range(len(CAMERAS))},
        "best_camera_dist_val": {ID_TO_CAMERA[i]: int(np.sum(meta["best_camera"][val_idx] == i)) for i in range(len(CAMERAS))},
        "valid_count_dist_all": {str(i): int(np.sum(meta["valid_count"] == i)) for i in sorted(np.unique(meta["valid_count"]).tolist())},
        "valid_mask_dist_all": valid_mask_distribution(meta),
        "target_loss_weights": target_loss_weights.detach().cpu().numpy().tolist(),
        "action_order_assumption": "dx,dy,dz,roll,pitch,yaw",
    }

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(to_jsonable({"config": vars(cfg), "train_info": train_info}), indent=2))

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for xb, yb, wb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            pred = model(xb)

            loss = weighted_huber_loss(
                pred=pred,
                target=yb,
                weight=wb,
                beta=cfg.huber_beta,
                target_weight=target_loss_weights,
            )

            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()
            losses.append(float(loss.item()))

        scheduler.step()

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            target_names=target_names,
            target_loss_weights=target_loss_weights,
            huber_beta=cfg.huber_beta,
        )

        train_loss = float(np.mean(losses))
        val_loss = float(val_metrics["loss"])

        mae_text = " ".join([f"{k}={v:.6f}" for k, v in val_metrics["mae"].items()])
        rmse_text = " ".join([f"{k}={v:.6f}" for k, v in val_metrics["rmse"].items()])
        lr = opt.param_groups[0]["lr"]

        log(
            f"[{now_s()}] epoch={epoch:03d} lr={lr:.3e} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"mae({mae_text}) rmse({rmse_text}) time={time.time() - t0:.1f}s"
        )

        if cfg.log_camera_metrics and (epoch == 1 or epoch % cfg.camera_metrics_every == 0):
            group_metrics = evaluate_multicam_groups(
                model=model,
                x_val=X_val_np,
                y_val=Y_val_np,
                w_val=W_val_np,
                meta_val=meta_val,
                state_dim=state_dim,
                cfg=cfg,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
                target_names=target_names,
                target_loss_weights=target_loss_weights,
            )

            for group_name, group_data in group_metrics.items():
                for name, m in group_data.items():
                    cam_mae = " ".join([f"{k}={v:.6f}" for k, v in m["mae"].items()])
                    log(
                        f"[{now_s()}] VAL_GROUP group={group_name} name={name} "
                        f"count={m['count']} loss={m['loss']:.6f} mae({cam_mae})"
                    )

        score = val_loss

        save_checkpoint(
            last_path,
            model,
            cfg,
            train_info,
            x_mean,
            x_std,
            y_mean,
            y_std,
            epoch,
            val_metrics,
        )

        if score < best_score:
            best_score = score

            save_checkpoint(
                best_path,
                model,
                cfg,
                train_info,
                x_mean,
                x_std,
                y_mean,
                y_std,
                epoch,
                val_metrics,
            )

            log(f"[{now_s()}] BEST_SAVED epoch={epoch} path={best_path} score={best_score:.6f}")

    log(f"[{now_s()}] TRAIN_DONE best={best_path} last={last_path}")
    log(
        f"[{now_s()}] INTEGRATION use checkpoint={best_path} "
        f"target_dims={target_dims} target_names={target_names} sample_mode=multicam_fused"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layout", default="v3_63d")

    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-device", default=None)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--yolo-conf", type=float, default=0.15)
    p.add_argument("--yolo-iou", type=float, default=0.50)
    p.add_argument("--yolo-batch", type=int, default=32)
    p.add_argument("--yolo-batch-frames", type=int, default=96)

    p.add_argument("--port-labels", default="sfp_port")
    p.add_argument("--module-labels", default="sfp_module,sc_plug")
    p.add_argument("--min-port-conf", type=float, default=0.35)
    p.add_argument("--min-module-conf", type=float, default=0.35)
    p.add_argument("--min-feature-quality", type=float, default=0.05)
    p.add_argument("--max-visual-error-px", type=float, default=520.0)
    p.add_argument("--min-valid-cameras", type=int, default=1)
    p.add_argument("--score-scale", type=float, default=5.0)
    p.add_argument("--invalid-err-norm", type=float, default=1.0)

    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--max-frames-per-episode", type=int, default=0)
    p.add_argument("--max-frames-per-parquet", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)

    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--label-horizon", type=int, default=3)
    p.add_argument("--target-dims", default="0,1,4,5")
    p.add_argument("--label-clip", type=float, default=0.15)
    p.add_argument("--min-abs-label", type=float, default=0.0)

    p.add_argument("--downweight-large-actions", type=float, default=0.08)
    p.add_argument("--min-sample-weight", type=float, default=0.15)
    p.add_argument("--max-sample-weight", type=float, default=2.5)

    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--mlp-blocks", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--min-lr", type=float, default=8e-7)
    p.add_argument("--weight-decay", type=float, default=1.5e-4)
    p.add_argument("--huber-beta", type=float, default=0.35)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--target-loss-weights", default="1.35,1.35,0.75,0.75")

    p.add_argument("--augment", action="store_true")
    p.add_argument("--state-noise-std", type=float, default=0.002)
    p.add_argument("--visual-noise-std", type=float, default=0.025)
    p.add_argument("--sequence-drop-prob", type=float, default=0.03)

    p.add_argument("--cache-name", default="feature_vs_cache_pitch_v4_multicam.npz")
    p.add_argument("--rebuild-cache", action="store_true")

    p.add_argument("--episode-log-every", type=int, default=1)
    p.add_argument("--accept-log-every", type=int, default=1000)

    p.add_argument("--log-camera-metrics", action="store_true")
    p.add_argument("--camera-metrics-every", type=int, default=5)

    args = p.parse_args()

    if args.frame_stride < 1:
        args.frame_stride = 1

    if args.min_valid_cameras < 1:
        args.min_valid_cameras = 1

    if args.min_valid_cameras > len(CAMERAS):
        args.min_valid_cameras = len(CAMERAS)

    return args


if __name__ == "__main__":
    train(parse_args())
