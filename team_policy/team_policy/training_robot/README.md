# CheatCode Training Data Pipeline

Full imitation-learning pipeline: collect expert demos → validate → convert → train ACT → deploy.

```
aic_engine runs N trials
  → aic_model loads cheatcode_collector
    → CheatCode (ground-truth oracle) executes each trial
      → EpisodeRecorder saves observations + expert actions → episode_XXXXX.hdf5
        → validate_episode.py checks every file
          → convert_to_lerobot.py converts HDF5 → LeRobot v3.0 dataset
            → lerobot-train trains an ACT neural network policy
              → team_policy.run_act deploys the checkpoint
```

> **Important:** `CheatCode` requires `ground_truth:=true`.
> The final deployed policy (`run_act.py`) uses only cameras + robot state — no hidden TF frames.

---

## Files

| File | Purpose |
|---|---|
| `cheatcode_collector.py` | Policy wrapper — records expert demos via CheatCode |
| `episode_recorder.py` | Buffers and saves one episode as HDF5 (schema v4) |
| `validate_episode.py` | Validates one HDF5 episode file |
| `convert_to_lerobot.py` | Converts HDF5 episodes → LeRobot v3.0 parquet + MP4 videos |
| `configs/orientation_sweep_50_trials.yaml` | **50-trial engine config** — fully populated board |
| `configs/test_5_trials.yaml` | 5-trial smoke test config |
| `configs/generate_50_trials.py` | Regenerates the 50-trial YAML |
| `../run_act.py` | **Deployment policy** — loads local ACT checkpoint, runs at 20 Hz |

Generated data is git-ignored:
```
training_robot/episodes/
training_robot/lerobot_datasets/
training_robot/dataset/
outputs/
```

---

## Scene Configuration — Fully Populated Board

Every trial in `orientation_sweep_50_trials.yaml` spawns **all components simultaneously**:

| Component | Count | Varied per trial |
|---|---|---|
| NIC cards (nic_rail_0 … nic_rail_4) | **5** (all present) | Each card's rail translation |
| SC mounts (sc_rail_0, sc_rail_1) | **2** (both present) | Each SC rail's translation |
| Task board pose | 10 different (x, y, yaw) | Cycles through positions A–J |
| Background mounts | 3 arrangements | Varied with mount_set 1–3 |

The **task** still targets one specific port per trial (40 SFP + 10 SC), so the robot learns to find the correct port on a realistically crowded board.

To regenerate the YAML after editing parameters in the generator:
```bash
cd ~/ros2_ws/src/aic
.pixi/envs/default/bin/python team_policy/team_policy/training_robot/configs/generate_50_trials.py
```

---

## HDF5 Schema v4 — What Gets Recorded

Each episode is saved at 20 Hz:

| Field | Shape | Description |
|---|---|---|
| `observations/images/{left,center,right}` | `(T, H, W, 3)` | Camera images, uint8 RGB, gzip-compressed |
| `observations/tcp_pose` | `(T, 7)` | Tool position + quaternion in base frame |
| `observations/tcp_velocity` | `(T, 6)` | Cartesian tool velocity (linear + angular) |
| `observations/tcp_error` | `(T, 6)` | Pose error to current target |
| `observations/joint_positions` | `(T, 7)` | Joint angles (rad) |
| `observations/joint_velocity` | `(T, 7)` | Per-joint velocity (rad/s) |
| `observations/wrist_force` | `(T, 6)` | F/T sensor readings |
| `observations/relative_pose` | `(T, 7)` | Target port pose in plug-tip frame |
| `observations/privileged_tf/transforms` | `(T, 5, 7)` | TF snapshots (debug only) |
| `actions/commanded_pose` | `(T, 7)` | Absolute TCP target commanded by CheatCode |
| `actions/delta_pose` | `(T, 6)` | Position delta + axis-angle rotation delta |
| `actions/velocity` | `(T, 6)` | Finite-difference velocity of commanded pose |

**Robot state vector for training (33D):**
```
tcp_pose (7) + tcp_velocity (6) + tcp_error (6) + joint_positions (7) + joint_velocity (7) = 33
```

---

## Complete Workflow

### Step 0 — Shell Environment (run this in every terminal)

```bash
cd ~/ros2_ws/src/aic
export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export CONFIG="$TRAIN_ROOT/configs/orientation_sweep_50_trials.yaml"
export EPISODES="$TRAIN_ROOT/episodes/orientation_sweep"
export LEROBOT="$TRAIN_ROOT/lerobot_datasets"
```

After any code change to `episode_recorder.py`, `cheatcode_collector.py`, or `run_act.py`:
```bash
pixi reinstall ros-kilted-team-policy
```

---

### Step 1 — Collect Episodes

Each run of the 50-trial config produces up to 50 episodes.
You already have `run_001` (6 episodes). **Start from `run_002`.**

#### Terminal 1 — Start Simulation + Engine

```bash
cd ~/ros2_ws/src/aic
export TRAIN_ROOT="$(pwd)/team_policy/team_policy/training_robot"
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  gazebo_gui:=false \
  launch_rviz:=false \
  aic_engine_config_file:="$TRAIN_ROOT/configs/orientation_sweep_50_trials.yaml"
```

Wait until you see `[aic_engine] Waiting for model...` before starting Terminal 2.

#### Terminal 2 — Start Collector

```bash
cd ~/ros2_ws/src/aic
export TRAIN_ROOT="$(pwd)/team_policy/team_policy/training_robot"
export RUN_ID="run_002"                                   # increment each new run
export OUTPUT_DIR="$TRAIN_ROOT/episodes/orientation_sweep/$RUN_ID"

pixi reinstall ros-kilted-team-policy

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.training_robot.cheatcode_collector \
  -p output_dir:="$OUTPUT_DIR" \
  -p num_episodes:=50 \
  -p success_only:=true
```

Expected output per episode:
```
DataCollectionPolicy ready — target=50 episodes, output=.../run_002, success_only=True
collector/episode=0 port=sfp/sfp_port_0
[1/50] Saved .../episode_00000.hdf5
...
[50/50] Saved .../episode_00049.hdf5
on_shutdown(...)
```

Both terminals exit automatically when all 50 trials finish.

#### Collecting more batches

Use a new `RUN_ID` each time — **never reuse**, the recorder overwrites from `episode_00000.hdf5`:
```bash
export RUN_ID="run_003"
export OUTPUT_DIR="$TRAIN_ROOT/episodes/orientation_sweep/$RUN_ID"
# then run Terminal 1 and Terminal 2 again
```

---

### Step 2 — Count and Validate

Count all collected episodes across all runs:
```bash
find $TRAIN_ROOT/episodes/orientation_sweep -name 'episode_*.hdf5' | wc -l
```

Validate all episodes (stops on first failure):
```bash
for f in $TRAIN_ROOT/episodes/orientation_sweep/run_*/episode_*.hdf5; do
  echo "--- $f ---"
  pixi run python -m team_policy.training_robot.validate_episode --file "$f" || break
done
```

Validate a single episode:
```bash
pixi run python -m team_policy.training_robot.validate_episode \
  --file $TRAIN_ROOT/episodes/orientation_sweep/run_002/episode_00000.hdf5
```

A good episode ends with:
```
PASS — episode looks valid (N frames, Xs, success=True)
```

#### Episode quality thresholds

| Quality | Rule |
|---|---|
| Excellent | `success == 1` and `final_error <= 0.005` m |
| Good | `success == 1` and `final_error <= 0.02` m |
| Skip | `success == 0` or `final_error > 0.02` m |

---

### Step 3 — Convert to LeRobot Format

Merge all runs into one folder, then convert:

```bash
cd ~/ros2_ws/src/aic
export TRAIN_ROOT="$(pwd)/team_policy/team_policy/training_robot"
export LEROBOT="$TRAIN_ROOT/lerobot_datasets"

# Merge all runs into one folder (renumber episode files)
MERGED="$TRAIN_ROOT/episodes/orientation_sweep/merged"
rm -rf "$MERGED" && mkdir -p "$MERGED"
idx=0
for f in $TRAIN_ROOT/episodes/orientation_sweep/run_*/episode_*.hdf5; do
  cp "$f" "$MERGED/episode_$(printf '%05d' $idx).hdf5"
  idx=$((idx + 1))
done
echo "Merged $idx episodes"

# Convert
pixi run python -m team_policy.training_robot.convert_to_lerobot \
  --input  "$MERGED" \
  --output "$LEROBOT/orientation_sweep_all" \
  --success_only \
  --max_final_error 0.02
```

Output structure:
```
lerobot_datasets/orientation_sweep_all/
  meta/
    info.json           (33D state, 6D action, video feature schema)
    stats.json          (mean/std/min/max for normalisation)
    tasks.parquet
    episodes/chunk-000/file-000.parquet
  data/chunk-000/
    file-000.parquet
  videos/
    observation.images.left/chunk-000/file-000.mp4
    observation.images.center/chunk-000/file-000.mp4
    observation.images.right/chunk-000/file-000.mp4
```

---

### Step 4 — Train ACT Policy

```bash
cd ~/ros2_ws/src/aic
export TRAIN_ROOT="$(pwd)/team_policy/team_policy/training_robot"
export LEROBOT="$TRAIN_ROOT/lerobot_datasets"

pixi run lerobot-train \
  --dataset.repo_id=local/orientation_sweep_all \
  --dataset.root="$LEROBOT/orientation_sweep_all" \
  --policy.type=act \
  --output_dir=outputs/train/aic_act_run_001 \
  --job_name=aic_act_run_001 \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=100000 \
  --save_freq=20000
```

For CPU-only machines replace `--policy.device=cuda` with `--policy.device=cpu`.

Training writes checkpoints every 20 000 steps:
```
outputs/train/aic_act_run_001/checkpoints/
  020000/pretrained_model/   ← deploy this folder
    config.json
    model.safetensors
    policy_preprocessor_step_3_normalizer_processor.safetensors
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
  020000/training_state/     ← resume training only
```

Resume training from a checkpoint:
```bash
pixi run lerobot-train \
  --config_path=outputs/train/aic_act_run_001/checkpoints/020000/pretrained_model/train_config.json \
  --resume=true \
  --steps=100000
```

---

### Step 5 — Deploy and Test the Trained Policy

#### Terminal 1 — Start Simulation (no ground_truth needed)

```bash
cd ~/ros2_ws/src/aic
export TRAIN_ROOT="$(pwd)/team_policy/team_policy/training_robot"
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=false \
  start_aic_engine:=true \
  gazebo_gui:=true \
  launch_rviz:=true \
  aic_engine_config_file:="$TRAIN_ROOT/configs/orientation_sweep_50_trials.yaml"
```

#### Terminal 2 — Run the Trained Policy

```bash
cd ~/ros2_ws/src/aic
export CKPT="$(pwd)/outputs/train/aic_act_run_001/checkpoints/100000/pretrained_model"

pixi reinstall ros-kilted-team-policy

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.run_act \
  -p checkpoint_path:="$CKPT"
```

The robot will attempt cable insertion using only cameras and robot state.
Watch Gazebo/RViz to see whether it succeeds.

---

## How Many Episodes You Need

| Episodes | Expected outcome |
|---|---|
| 5 | Smoke test only — overfits, does not generalise |
| 50 | Minimum viable — works on seen board poses |
| 200–500 | Recommended for robust generalisation |
| 1000+ | Research-grade, covers full pose distribution |

The fully-populated board config gives richer visual diversity per episode (robot sees all 5 NIC cards + both SC ports simultaneously) so fewer episodes are needed compared to a sparse board.

---

## Architecture Summary (ACT)

```
Inputs at each 20 Hz step:
  ├─ 3× camera images (left, center, right)  → ResNet-18 encoder → image tokens
  └─ 33D robot state                         → linear projection → state token

Transformer encoder-decoder:
  Attends over tokens → predicts chunk of K=100 future 6D actions

Output per step:
  100 × [dx, dy, dz, drx, dry, drz]   applied at 20 Hz = 5 s of planned motion
```

Action chunking reduces compounding errors: instead of 100 independent predictions,
one coherent 100-step plan is generated. The policy replans every 5 seconds to correct drift.

---

## Complete Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  TERMINAL 1 — aic_eval container                                     │
│                                                                      │
│  Gazebo + aic_engine                                                 │
│    • Spawns: robot + task board (all 5 NIC cards + 2 SC ports)       │
│    • Config: orientation_sweep_50_trials.yaml                        │
│    • ground_truth:=true → publishes hidden TF frames                 │
│    • Calls insert_cable() once per trial                             │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ ROS 2 topics + services
┌─────────────────────────────▼────────────────────────────────────────┐
│  TERMINAL 2 — cheatcode_collector (aic_model)                        │
│                                                                      │
│  DataCollectionPolicy.insert_cable():                                │
│    • Wraps move_robot() to capture commanded poses                   │
│    • Records at 20 Hz in background thread:                          │
│        3× camera images, tcp_pose, tcp_vel, tcp_err                  │
│        joint_positions, joint_velocity, wrist_force                  │
│        relative_pose (plug→port), privileged_tf snapshots            │
│    • Runs CheatCode as the expert (uses ground-truth TF)             │
│    • On success → episode_XXXXX.hdf5                                 │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  validate_episode.py  — checks shapes, quaternions, timing, etc.     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  convert_to_lerobot.py                                               │
│    • HDF5 images → MP4 videos                                        │
│    • Builds 33D state + 6D delta action per frame                    │
│    • Writes parquet + stats.json + info.json                         │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  lerobot-train (ACT)                                                 │
│    Input:  3× images + 33D state                                     │
│    Output: 100 × 6D delta TCP actions                                │
│    Result: pretrained_model/ checkpoint                              │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  team_policy.run_act  (deployment)                                   │
│    • Loads local checkpoint via checkpoint_path ROS param            │
│    • Runs at 20 Hz: image + state → 6D delta → MODE_POSITION command │
│    • No ground_truth required                                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Common Issues

### Fewer than 50 episodes saved after a run

Some trials failed (CheatCode returned `False`). Normal for difficult poses.
Start a new run with an incremented `RUN_ID` — the engine replays all 50 trials fresh.

### `Collector cannot find ground truth TF`

Terminal 1 must have `ground_truth:=true`. Without it, `relative_pose` and `privileged_tf`
save as zeros and validation will flag the episode.

### `joint_velocity` is all zeros after validation

Episode was recorded before schema v4 or the controller wasn't publishing joint velocities.
Re-collect after running:
```bash
pixi reinstall ros-kilted-team-policy
```

### `success=1` but `final_error` is large (e.g. 0.045 m)

Episode is suspicious. Keep `--max_final_error 0.02` in the converter to skip it.
Only raise the threshold after visually inspecting that the demo is genuinely useful.

### Reusing the same `RUN_ID`

Don't. The recorder always starts from `episode_00000.hdf5` and will silently overwrite
existing files. Always increment `RUN_ID` for each new collection run.

### `checkpoint_path` parameter not set

`team_policy.run_act` requires the parameter:
```bash
-p checkpoint_path:=/absolute/path/to/pretrained_model
```
The path must point to the `pretrained_model/` subfolder inside a checkpoint directory,
not the checkpoint root or the `training_state/` folder.
