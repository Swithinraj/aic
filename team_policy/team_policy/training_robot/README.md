# Training Robot — Imitation Learning Pipeline

Collect expert demos → validate → convert → train ACT → deploy.

---

## What We're Building

### The goal

Train a robot arm to autonomously insert cables into a server task board — specifically:
- **NIC cards** (SFP plugs) into 5 rails on the board
- **SC fiber mounts** into 2 rails on the board

The trained policy must work using only what the robot can see at competition time: **3 cameras + robot joint/force state**. No ground-truth positions, no hidden TF frames.

### Why imitation learning?

Writing a hand-coded controller for precise cable insertion is brittle — small pose errors or visual variation cause failures. Instead we:
1. Use a "cheating" expert (**CheatCode**) that has access to hidden ground-truth TF frames in simulation to execute perfect insertions
2. Record everything the expert does as demonstrations
3. Train a neural network to reproduce that behavior from camera + state alone

This is **behaviour cloning** — the network learns to imitate the expert.

### The expert: CheatCode

CheatCode is a privileged policy that only works in simulation with `ground_truth:=true`. It reads hidden TF frames that tell it the exact position of every port. It never misses an insertion. We use it purely as a data source — it is **never deployed** on the real robot or in competition.

### How sessions are generated (getting diverse training data)

A single scene (fixed board pose, one cable type, one rail) is not enough to train a policy that generalises. We need diversity across board positions, which rails are targeted, and visual clutter. At the same time we can't randomise everything every trial — Gazebo leaks RAM when entities respawn too often, and CheatCode's TF lookups get stale if the board pose jumps mid-session.

The solution is **competition-format sessions**: one Gazebo launch = one fixed board pose, 3 trials back-to-back. Board pose varies *between* sessions instead of within them.

**Each session runs 3 trials:**

| Trial | Task | What spawns |
|-------|------|-------------|
| 1 | NIC insertion (SFP plug) | 1 NIC card on target rail + background mounts |
| 2 | NIC insertion (SFP plug, different rail) | 1 NIC card on different rail + background mounts |
| 3 | SC insertion | 1 SC mount on target rail + background mounts |

(Sessions 11–20 reverse the order to `SC → NIC → NIC` so the policy also sees SC insertions as trial 1.)

**What varies across the 50 sessions:**

| Variable | Values | Purpose |
|----------|--------|---------|
| Board pose | 10 table poses A–J (x/y/yaw vary, z/roll/pitch fixed at `1.14 / 0 / 0`) | Force the policy to handle different board positions |
| NIC rail target | cycles through all 5 rails | Cover every insertion site on the board |
| SC rail target | alternates between rails 0 and 1 | Cover both SC sites |
| Background mounts | 3 clutter arrangements | Visual variety — policy learns to ignore distractors |
| NIC/SC translation | small offsets (±15–42 mm) | Sub-rail position variation |

**Why only `x`, `y`, and `yaw` vary — not `z`, `roll`, `pitch`:**
The board must stay flat on the table. Changing `z` would float or sink it through the table, and `roll`/`pitch` would tilt it off — both cause physically invalid scenes that CheatCode cannot solve.


**Setup the enviroment variables**
```bash
export AIC_ROOT=$(git rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml
```
**Generating the session YAMLs:**

The 50 YAMLs in `configs/sessions/` are auto-generated — don't edit them by hand:

```bash
cd $AIC_ROOT
python3 team_policy/team_policy/training_robot/configs/generate_competition_sessions.py --sessions 50
```

This writes `session_01.yaml … session_50.yaml`. See `generate_competition_sessions.py` for the pose list (`POSES`), rail pattern (`SESSION_PATTERNS`), and mount arrangements (`MOUNT_SETS`) — edit those constants to change the diversity.

**Alternate mode — one trial per Gazebo launch:**
If the eval container leaves the gripper in a bad state between trials (e.g. stuck open), use `--trials-per-session 1`. This generates 150 single-trial files (`session_001.yaml … session_150.yaml`), one for each trial in the original 50×3 plan. Slower because Gazebo restarts every episode, but every episode starts from a fully reset robot, gripper, cable, and board:

```bash
cd $AIC_ROOT
rm -f team_policy/team_policy/training_robot/configs/sessions/session_*.yaml
python3 team_policy/team_policy/training_robot/configs/generate_competition_sessions.py \
  --sessions 50 \
  --trials-per-session 1
```

### What gets recorded per episode

At 10 Hz, for every timestep:

| Data | Shape | What it is |
|------|-------|------------|
| Left / center / right camera | (T, H, W, 3) | RGB images from 3 wrist/head cameras |
| TCP pose | (T, 7) | Tool position + quaternion in robot base frame |
| TCP velocity | (T, 6) | Cartesian linear + angular velocity |
| TCP error | (T, 6) | Pose error to current CheatCode target |
| Joint positions | (T, 7) | All 7 joint angles in radians |
| Joint velocity | (T, 7) | Per-joint velocity |
| Wrist force/torque | (T, 6) | F/T sensor readings |
| Delta action | (T, 6) | What CheatCode commanded — this is the training label |

**33D robot state vector used for training:**
```
tcp_pose(7) + tcp_velocity(6) + tcp_error(6) + joint_positions(7) + joint_velocity(7) = 33
```

### The policy: ACT (Action Chunking with Transformers)

```
At each 10 Hz step:

  Inputs
  ------
  3x camera images  -->  ResNet-18 encoder  -->  image feature tokens
  33D robot state   -->  linear projection  -->  state token

  Transformer encoder-decoder
  ---------------------------
  Attends over all tokens
  Predicts a CHUNK of 100 future actions at once

  Output
  ------
  100 x [dx, dy, dz, drx, dry, drz]   (6D delta TCP pose)
  Applied at 10 Hz = 10 seconds of planned motion
```

**Why action chunking?** Predicting one action at a time compounds errors — small mistakes at step 1 corrupt step 2. Predicting 100 steps as one coherent plan is much more stable. The policy replans every 10 seconds to correct any drift.

**At deployment:** the policy receives a camera frame + robot state, outputs the 100-step chunk, executes it, then replans. No ground truth, no TF frames — only what a real competition robot would have.

### End-to-end data flow

```
Gazebo simulation (ground_truth:=true)
  |
  |  CheatCode executes insertions using hidden TF frames
  |  EpisodeRecorder captures images + state + actions at 10 Hz
  v
episodes/run_001/ ... run_050/
  150 HDF5 files  (50 sessions x 3 trials)
  |
  |  validate_episode.py  -- checks shapes, timing, success flag
  v
episodes/merged/
  150 renumbered HDF5 files
  |
  |  convert_to_lerobot.py  -- HDF5 -> parquet + MP4, builds stats
  v
lerobot_datasets/aic_dataset_v1/
  |
  |  lerobot-train (ACT)
  |  Input:  3x images + 33D state
  |  Output: 100 x 6D delta actions
  |  100k gradient steps, checkpoints every 20k
  v
outputs/train/aic_act_run_001/checkpoints/100000/pretrained_model/
  |
  |  team_policy.run_act  (deployed in competition)
  |  Runs at 10 Hz using only cameras + robot state
  v
Robot inserts cable autonomously
```

---

## Quick Start (first-time setup)

### Step A — Set environment variables (run in every terminal you open)

```bash
export AIC_ROOT=$(git rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml
```

`git rev-parse --show-toplevel` finds the repo root automatically — works for any username and any clone location. The FastDDS profile disables shared memory to prevent htop from showing inflated real RAM usage for aic_model on 16GB machines. **Paste these three lines at the top of every terminal you open for this pipeline.**

### Step B — Build after any code change

If you edit `cheatcode_collector.py`, `episode_recorder.py`, or `run_act.py`, rebuild before running:

```bash
cd $AIC_ROOT && pixi reinstall ros-kilted-team-policy
```

### Step C — Pipeline overview

| Step | What happens |
|------|-------------|
| [1. Collect](#step-1----collect-episodes-50-sessions--3-trials--150-episodes) | Run 50 Gazebo sessions × 3 trials → 150 HDF5 episode files |
| [2. Validate](#step-2----count-and-validate) | Check every HDF5 file passes quality thresholds |
| [3. Convert](#step-3----convert-to-lerobot-format) | Merge + convert HDF5 (auto-downsamples ~20 Hz to 10 Hz) → LeRobot parquet + MP4 dataset |
| [4. Train](#step-4----train-act-policy) | Train ACT neural network on the dataset |
| [5. Deploy](#step-5----deploy-and-test-the-trained-policy) | Load checkpoint, run policy in simulation |

> **Important:** Steps 1–2 require `ground_truth:=true` in Terminal 1 (Gazebo side).
> The deployed policy (Step 5) uses only cameras + robot state — no ground truth needed.

---

## Step 1 — Collect Episodes (50 sessions × 3 trials = 150 episodes)

Run one session at a time. Each session takes ~10–15 minutes.

#### Terminal 1 — Start Simulation + Engine

```bash
export AIC_ROOT=$(git rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=$TRAIN_ROOT/configs/sessions/session_01.yaml
```



Wait until Gazebo opens and you see:
```
No node with name 'aic_model' found. Retrying...
```




#### Terminal 2 — Start CheatCode Collector

```bash
export AIC_ROOT=$(git rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml
export RUN_ID=run_001
export OUTPUT_DIR="$TRAIN_ROOT/episodes/$RUN_ID"

cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.training_robot.cheatcode_collector \
  -p output_dir:="$OUTPUT_DIR" \
  -p num_episodes:=3 \
  -p success_only:=true
```

```bash

from yolo

export FASTRTPS_DEFAULT_PROFILES_FILE=~/ros2_ws/src/aic/team_policy/fastdds_no_shm.xml
cd ~/ros2_ws/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector \
    -p output_dir:=/tmp/aic_dataset_fresh \
    -p num_episodes:=200 \
    -p success_only:=true
```




Expected output per episode:
```
[INFO] DataCollectionPolicy ready — target=3 episodes
[INFO] collector/episode=0 port=sfp/sfp_port_0
[INFO] [1/3] Saved .../run_001/episode_00000.hdf5
[INFO] collector/episode=1 port=sfp/sfp_port_0
[INFO] [2/3] Saved .../run_001/episode_00001.hdf5
[INFO] collector/episode=2 port=sc/sc_port_base
[INFO] [3/3] Saved .../run_001/episode_00002.hdf5
```

Terminal 2 exits on its own after 3 episodes. Press `Ctrl+C` in Terminal 1.

#### Next session — only two things change

**Terminal 1** — increment session number:
```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=$TRAIN_ROOT/configs/sessions/session_02.yaml
```

**Terminal 2** — increment RUN_ID (`AIC_ROOT` and `TRAIN_ROOT` stay set from before):
```bash
export RUN_ID=run_002
export OUTPUT_DIR="$TRAIN_ROOT/episodes/$RUN_ID"

cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.training_robot.cheatcode_collector \
  -p output_dir:="$OUTPUT_DIR" \
  -p num_episodes:=3 \
  -p success_only:=true
```

#### Session reference

| Session | Terminal 1 config | Terminal 2 RUN_ID |
|---------|-------------------|-------------------|
| 1 | `session_01.yaml` | `run_001` |
| 2 | `session_02.yaml` | `run_002` |
| 3 | `session_03.yaml` | `run_003` |
| … | … | … |
| 50 | `session_50.yaml` | `run_050` |

#### Expected folder layout after all 50 sessions

```
training_robot/episodes/
  run_001/
    episode_00000.hdf5   <- NIC0, pose A
    episode_00001.hdf5   <- NIC2, pose A
    episode_00002.hdf5   <- SC0,  pose A
  run_002/
    episode_00000.hdf5   <- NIC1, pose B
    episode_00001.hdf5   <- NIC3, pose B
    episode_00002.hdf5   <- SC1,  pose B
  ...
  run_050/
    episode_00000.hdf5
    episode_00001.hdf5
    episode_00002.hdf5
```

**50 sessions × 3 trials = 150 HDF5 files total.**

If a trial fails (CheatCode returns False), `success_only:=true` discards it — no file saved.
Just rerun that session with a new `RUN_ID`.

---

## Step 2 — Count and Validate

Count all collected episodes across all runs:
```bash
find $TRAIN_ROOT/episodes -name 'episode_*.hdf5' | wc -l
```

Validate all episodes (stops on first failure):
```bash
cd $AIC_ROOT
for f in $TRAIN_ROOT/episodes/run_*/episode_*.hdf5; do
  echo "--- $f ---"
  pixi run python -m team_policy.training_robot.validate_episode --file "$f" || break
done
```

Validate a single episode:
```bash
cd $AIC_ROOT
pixi run python -m team_policy.training_robot.validate_episode \
  --file $TRAIN_ROOT/episodes/run_001/episode_00000.hdf5
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

## Step 3 — Convert to LeRobot Format

Merge all runs into one renumbered folder, then convert:

```bash
export LEROBOT=$TRAIN_ROOT/lerobot_datasets

# Merge all runs into one folder using symlinks (avoids duplicating ~60GB of HDF5 data)
# Using symlinks instead of cp saves disk space — convert reads through them identically
MERGED="$TRAIN_ROOT/episodes/merged"
rm -rf "$MERGED" && mkdir -p "$MERGED"
idx=0
for f in $TRAIN_ROOT/episodes/run_*/episode_*.hdf5; do
  ln -s "$f" "$MERGED/episode_$(printf '%05d' $idx).hdf5"
  idx=$((idx + 1))
done
echo "Merged $idx episodes"

# Convert to LeRobot format
cd $AIC_ROOT
pixi run python -m team_policy.training_robot.convert_to_lerobot \
  --input  "$MERGED" \
  --output "$LEROBOT/aic_dataset_v1" \
  --success_only \
  --max_final_error 0.02 \
  --target_hz 10
```

Output structure:
```
lerobot_datasets/aic_dataset_v1/
  meta/
    info.json           <- 33D state, 6D action, video feature schema
    stats.json          <- mean/std/min/max for normalisation
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

## Step 4 — Train ACT Policy

### Required fix before first run (lerobot 0.5.1 bug)

lerobot 0.5.1 ships with a broken GR00T policy file that crashes on import even though
GR00T is never used for ACT training. Apply this one-time patch before running training:

```bash
sed -i \
  's/backbone_cfg: dict = field(init=False,/backbone_cfg: dict = field(default=None, init=False,/g;
   s/action_head_cfg: dict = field(init=False,/action_head_cfg: dict = field(default=None, init=False,/g;
   s/action_horizon: int = field(init=False,/action_horizon: int = field(default=None, init=False,/g;
   s/action_dim: int = field(init=False,/action_dim: int = field(default=None, init=False,/g' \
  $AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/lerobot/policies/groot/groot_n1.py
```

Verify it worked:
```bash
grep "default=None, init=False" \
  $AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/lerobot/policies/groot/groot_n1.py | head -1
# Should print a line — if empty, re-run the sed command above
```

This patch is applied to the pixi environment only — it is NOT part of the source tree and
will revert if pixi reinstalls the environment. Re-apply whenever you see:
`TypeError: non-default argument 'backbone_cfg' follows default argument`

### Run training (inside tmux to survive terminal close)

```bash
tmux new -s training   # create session — detach with Ctrl+B then D, reattach with: tmux attach -t training
```

```bash
export LEROBOT=$TRAIN_ROOT/lerobot_datasets

cd $AIC_ROOT
pixi run lerobot-train \
  --dataset.repo_id=local/aic_dataset_v1 \
  --dataset.root="$LEROBOT/aic_dataset_v1" \
  --policy.type=act \
  --policy.push_to_hub=false \
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
  020000/pretrained_model/   <- deploy this folder
    config.json
    model.safetensors
    policy_preprocessor_step_3_normalizer_processor.safetensors
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
  020000/training_state/     <- resume training only
```

Resume training from a checkpoint:
```bash
cd $AIC_ROOT
pixi run lerobot-train \
  --config_path=outputs/train/aic_act_run_001/checkpoints/020000/pretrained_model/train_config.json \
  --resume=true \
  --steps=100000
```

---

## Step 5 — Deploy and Test the Trained Policy

#### Terminal 1 — Start Simulation (no ground_truth needed)

```bash
export AIC_ROOT=$(git rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=false \
  start_aic_engine:=true \
  aic_engine_config_file:=$TRAIN_ROOT/configs/sessions/session_01.yaml
```

#### Terminal 2 — Run the Trained Policy

```bash
export AIC_ROOT=$(git rev-parse --show-toplevel)
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml
export CKPT=$AIC_ROOT/outputs/train/aic_act_run_001/checkpoints/100000/pretrained_model

cd $AIC_ROOT
pixi reinstall ros-kilted-team-policy

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.run_act \
  -p checkpoint_path:="$CKPT"
```

The robot will attempt cable insertion using only cameras and robot state — no ground truth.
Watch Gazebo to see whether it succeeds.

---

## How Many Episodes You Need

| Episodes | Expected outcome |
|---|---|
| 5–10 | Smoke test only — overfits, does not generalise |
| 50 | Minimum viable — works on seen board poses |
| 150 | Good starting point — covers all 10 poses + all rails |
| 300–500 | Recommended for robust generalisation |
| 1000+ | Research-grade, full pose and rail distribution |

---

## Run log — `aic_act_run_001` (current checkpoint)

| Field | Value |
|---|---|
| Sessions collected | 67 (`run_001` … `run_067`) |
| Raw HDF5 episodes | ~402 |
| Episodes after filter (`success_only` + `max_final_error 0.02`) | **157** |
| Frames in dataset | 70,764 @ 10 Hz |
| Training steps | 100,000 / 100,000 |
| Final loss | 0.033 (grad norm ≈ 2.85) |
| Duration | ~8 h 56 min on CUDA |
| Checkpoints | every 20k → `020000`, `040000`, `060000`, `080000`, `100000`, `last` |
| Deployed checkpoint | `outputs/train/aic_act_run_001/checkpoints/100000/pretrained_model/` |

This sits between the "Good starting point" and "Recommended" rows above — expect occasional collisions on unseen poses; more sessions or a force-stop guard in `run_act.py` are the next levers.

---

## Common Issues

### `AIC_ROOT` or `TRAIN_ROOT` is empty / path not found

You opened a new terminal without running the exports. Paste these at the top of every terminal:
```bash
export AIC_ROOT=$(git rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
```

### RAM fills up / PC freezes mid-run

You are using the old `orientation_sweep_50_trials.yaml` which spawns all 5 NICs + 2 SC mounts
for 50 trials without restarting Gazebo. Switch to the session YAMLs in `configs/sessions/`.
Each session is only 3 trials with 1 entity spawned per trial.

### Score drops to ~30 after trial 1

Board pose was jumping between trials, causing stale TF lookups.
The session YAMLs fix this by keeping the board pose fixed for all 3 trials.

### Fewer than 3 episodes saved after a session

Some trials failed (CheatCode returned False). Normal for difficult poses.
Start a new session with an incremented `RUN_ID` and the same session YAML to retry,
or skip to the next session YAML — the missed episodes are not critical.

### `Collector cannot find ground truth TF`

Terminal 1 must have `ground_truth:=true`. Without it, `relative_pose` and `privileged_tf`
save as zeros and validation will flag the episode.

### `joint_velocity` is all zeros after validation

Recorded before schema v4 or controller was not publishing joint velocities. Re-collect after:
```bash
cd $AIC_ROOT && pixi reinstall ros-kilted-team-policy
```

### `success=1` but `final_error` is large (e.g. 0.045 m)

Episode is suspicious. Keep `--max_final_error 0.02` in the converter to skip it.

### Reusing the same `RUN_ID`

Do not. The recorder always starts from `episode_00000.hdf5` and silently overwrites existing
files. Always increment `RUN_ID` for each new collection run.

### `checkpoint_path` parameter not set

`team_policy.run_act` requires:
```bash
-p checkpoint_path:=/absolute/path/to/pretrained_model
```
The path must point to the `pretrained_model/` subfolder, not the checkpoint root or
`training_state/` folder.

---

## Historical note — the old 50-trial config

Earlier iterations used `orientation_sweep_50_trials.yaml`, which ran all 50 trials in a single Gazebo session with all 5 NICs + 2 SC mounts spawned simultaneously every trial. That caused RAM leaks (Gazebo accumulates entities across respawns) and score drops (CheatCode's TF lookups went stale when the board pose jumped mid-session). The competition-session YAMLs described above replace it.

If you find that config, delete it — it is not used anywhere in the current pipeline.

---

## Files

| File | Purpose |
|---|---|
| `cheatcode_collector.py` | Policy wrapper — records expert demos via CheatCode |
| `episode_recorder.py` | Buffers and saves one episode as HDF5 (schema v4) |
| `validate_episode.py` | Validates one HDF5 episode file |
| `convert_to_lerobot.py` | Converts HDF5 episodes to LeRobot v3.0 parquet + MP4 videos |
| `configs/generate_competition_sessions.py` | Generates competition session YAMLs |
| `configs/sessions/session_01.yaml ... session_NN.yaml` | Session configs (3 trials each) |
| `../run_act.py` | Deployment policy — loads local ACT checkpoint, runs at 10 Hz |

Generated data is git-ignored:
```
training_robot/episodes/
training_robot/lerobot_datasets/
```
