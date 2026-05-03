# SmolVLA Training Setup Guide
## From Scratch to Training on a New Machine

---

## 1. System Requirements

- **OS**: Ubuntu 20.04 / 22.04 / 24.04
- **GPU**: 16GB+ VRAM minimum (RTX 3090, RTX 4090, A100, etc.)
- **CUDA**: 12.1+
- **Python**: 3.10 – 3.12
- **Disk**: ~50GB free (dataset ~20GB + checkpoints ~10GB)
- **RAM**: 16GB+ recommended

Check GPU before starting:
```bash
nvidia-smi
```

---

## 2. Clone LeRobot and Create Environment

```bash
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[smolvla]"
pip install av  # PyAV for video decoding
```

---

## 3. Copy the Fixed Dataset

Transfer `aic_act_yolo_mansi` (the fixed dataset — 357 episodes, 282,791 frames) to the new machine.

**From another machine via scp:**
```bash
scp -r /path/to/aic_act_yolo_mansi user@NEW_MACHINE_IP:~/lerobot/
```

**Or copy from Google Drive / USB to:**
```
~/lerobot/aic_act_yolo_mansi/
```

Verify the dataset is correct (should show 357 episodes):
```bash
python3 -c "
import json
info = json.load(open('aic_act_yolo_mansi/meta/info.json'))
print('Episodes:', info['total_episodes'])
print('Frames:', info['total_frames'])
print('Tasks:')
import pandas as pd
print(pd.read_parquet('aic_act_yolo_mansi/meta/tasks.parquet'))
"
```

Expected output:
```
Episodes: 357
Frames: 282791
Tasks:
   task_index                                               task
0           0  Insert the SFP module into the target SFP port...
1           1         Insert the SC plug into the target SC port
```

---

## 4. Apply All Required Patches

These patches fix 3 bugs in LeRobot that prevent training on a local dataset.
Run all of them from inside `~/lerobot/` with the venv active.

### Patch 1 — Force PyAV video backend (video_utils.py)

torchcodec is incompatible with PyTorch < 2.5. Force PyAV instead.

**Change 1a — default backend:**
```bash
sed -i 's/backend = get_safe_default_codec()/backend = "pyav"  # torchcodec incompatible with PyTorch 2.4/' \
  src/lerobot/datasets/video_utils.py
```

**Change 1b — torchcodec call → torchvision pyav call:**
```bash
sed -i 's/return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s, return_uint8=return_uint8)/return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, "pyav", return_uint8=return_uint8)/' \
  src/lerobot/datasets/video_utils.py
```

Verify:
```bash
grep -n "backend = \|return decode_video_frames" src/lerobot/datasets/video_utils.py | head -10
```
Expected lines:
```
145:        backend = "pyav"  # torchcodec incompatible with PyTorch 2.4
147:        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, "pyav", ...
149:        return decode_video_frames_torchvision(
```

---

### Patch 2 — Bypass HuggingFace Hub version check (utils.py)

LeRobot tries to verify the dataset on HuggingFace Hub even for local datasets. This patches it to skip Hub lookup silently.

Open the file:
```bash
nano src/lerobot/datasets/utils.py
```

Find the function `get_repo_versions` (search with Ctrl+W). Replace its entire body with a try/except:

**Before:**
```python
def get_repo_versions(repo_id: str) -> list[packaging.version.Version]:
    api = HfApi()
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))
    return repo_versions
```

**After:**
```python
def get_repo_versions(repo_id: str) -> list[packaging.version.Version]:
    try:
        api = HfApi()
        repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
        repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
        repo_versions = []
        for ref in repo_refs:
            with contextlib.suppress(packaging.version.InvalidVersion):
                repo_versions.append(packaging.version.parse(ref))
        return repo_versions
    except Exception:
        return []
```

Then find `get_safe_version` and locate the block that says `if not hub_versions:`. Change the `raise RevisionNotFoundError(...)` line to a return:

**Before:**
```python
if not hub_versions:
    raise RevisionNotFoundError(...)
```

**After:**
```python
if not hub_versions:
    return f"v{target_version}"
```

Save and exit (Ctrl+O, Enter, Ctrl+X).

Verify:
```bash
grep -n "return \[\]\|return f\"v{target" src/lerobot/datasets/utils.py
```
Expected:
```
332:        return []
366:        return f"v{target_version}"
```

---

### Patch 3 — Fix task string lookup (dataset_reader.py)

LeRobot uses `.name` to get the task string from the DataFrame row, but `.name` returns the integer row index, not the task text.

```bash
sed -i 's/item\["task"\] = self._meta\.tasks\.iloc\[task_idx\]\.name/item["task"] = self._meta.tasks.iloc[task_idx]["task"]/' \
  src/lerobot/datasets/dataset_reader.py
```

Verify:
```bash
grep -n "task" src/lerobot/datasets/dataset_reader.py | grep "item\["
```
Expected:
```
296:        item["task"] = self._meta.tasks.iloc[task_idx]["task"]
```

---

## 5. Verify All Patches Applied

```bash
echo "=== video_utils.py ===" && \
grep -n "backend = \|return decode_video_frames" src/lerobot/datasets/video_utils.py | head -6

echo "=== utils.py ===" && \
grep -n "return \[\]\|return f.v{target" src/lerobot/datasets/utils.py

echo "=== dataset_reader.py ===" && \
grep -n "item\[.task.\]" src/lerobot/datasets/dataset_reader.py
```

---

## 6. Run Training

```bash
cd ~/lerobot
source .venv/bin/activate

export LEROBOT_VIDEO_BACKEND=pyav
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --dataset.repo_id=local/aic_act_yolo_mansi \
  --dataset.root=/HOME/USER/lerobot/aic_act_yolo_mansi \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/aic_smolvla_v1 \
  --job_name=aic_smolvla_v1 \
  --policy.device=cuda \
  --batch_size=16 \
  --wandb.enable=false \
  --steps=100000 \
  --save_freq=20000
```

> Replace `/HOME/USER/lerobot/aic_act_yolo_mansi` with the actual path on the new machine.

**Healthy startup looks like:**
```
INFO Creating dataset
INFO Creating policy
Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
Loading weights: 100%|...| 489/489
INFO num_learnable_params=99880992 (100M)
INFO num_total_params=450046176 (450M)
INFO Start offline training on a fixed dataset
Training:   0%| 0/100000 [00:00]
INFO step:1 loss:2.3xx ...
```

---

## 7. Resume from Checkpoint (if session interrupted)

```bash
lerobot-train \
  --dataset.repo_id=local/aic_act_yolo_mansi \
  --dataset.root=/HOME/USER/lerobot/aic_act_yolo_mansi \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/aic_smolvla_v1 \
  --job_name=aic_smolvla_v1 \
  --policy.device=cuda \
  --batch_size=16 \
  --wandb.enable=false \
  --steps=100000 \
  --save_freq=20000 \
  --resume=true
```

Checkpoints are saved at steps: **20000, 40000, 60000, 80000, 100000**
Location: `outputs/train/aic_smolvla_v1/checkpoints/`

---

## 8. Expected Training Time

| GPU | ~Steps/sec | 100K steps |
|---|---|---|
| RTX 3090 (24GB) | ~2 steps/s | ~14 hours |
| RTX 4090 (24GB) | ~3 steps/s | ~9 hours |
| A100 (40GB) | ~5 steps/s | ~6 hours |
| Colab T4 (15GB) | ~1.5 steps/s | ~18 hours |

---

## Troubleshooting Quick Reference

| Error | Fix |
|---|---|
| `No module named lerobot.scripts.train` | Use `lerobot-train` CLI, not `python -m lerobot.scripts.train` |
| `torchcodec libavutil not found` | Patch 1 (video_utils.py) not applied |
| `RepositoryNotFoundError 401` | Patch 2 (utils.py) not applied |
| `Task cannot be None` | Patch 3 (dataset_reader.py) not applied, OR dataset not fixed (run fix_dataset_for_smolvla.py) |
| `CUDA out of memory` | GPU < 16GB VRAM — reduce `--batch_size` or use Colab |
| `FileNotFoundError info.json` | Wrong `--dataset.root` path — check with `find ~ -name "info.json" -path "*aic*"` |
