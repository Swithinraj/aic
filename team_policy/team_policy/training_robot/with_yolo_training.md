# YOLO-Guided Data Collection & Training Pipeline

## What this pipeline does

CheatCode (the official expert controller) performs perfect cable insertions while this pipeline records every frame: camera images, robot state, wrist forces, ground-truth TF poses, and — critically — YOLO-detected port positions. The result is a dataset where the policy can learn to insert cables using only what the real robot can see, without needing ground-truth information at inference time.

### Why YOLO for port position?

Ground-truth TF gives perfect port position during data collection but is unavailable on the real robot. YOLO gives the same imperfect-but-consistent signal at both training and inference time. The policy therefore learns to use a signal it will actually have, with no distribution shift between training and deployment.

### Training state (30D)

```
tcp_pose(7)  +  tcp_velocity(6)  +  joint_positions(7)  +  joint_velocity(7)  +  port_xyz_in_base(3)
     7               6                      7                      7                      3         = 30
```

`port_xyz_in_base` is filled from YOLO detections during both collection and inference. The converter (`convert_to_lerobot.py`) reproduces the exact inference behaviour so there is no distribution shift:

| Frame | Training value | Inference value |
|-------|---------------|-----------------|
| Before first YOLO detection | `[0, 0, 0]` (cold-start) | `[0, 0, 0]` (reset at trial start) |
| YOLO valid | real YOLO position | real YOLO position |
| YOLO temporarily lost mid-episode | last-known YOLO position | last-known YOLO position (callback holds last value) |

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────┐
│  Terminal 1: Gazebo + aic_engine (restarts every run)   │
│  → Runs 3 trials, then exits automatically              │
│  → Uses session_XX.yaml to vary board pose each run     │
└─────────────────────────┬───────────────────────────────┘
                          │ ROS 2 topics
┌─────────────────────────▼───────────────────────────────┐
│  Terminal 2: cheatcode_collector + embedded YOLO         │
│  → Starts YOLO + metric depth + CAD registration         │
│  → Publishes /fused_yolo/detections_json internally     │
│  → Wraps CheatCode with observation recording           │
│  → Injects wrench compliance gains                      │
│  → Applies quality gates, saves episode_NNNNN.hdf5      │
└─────────────────────────────────────────────────────────┘
```

---

## Terminology

| Term | Meaning |
|------|---------|
| **Session** | One launch of `aic_engine` with one `session_XX.yaml`. Runs 3 trials automatically then exits. |
| **Run** | One folder (`run_001`, `run_002`, …). One run = one session = up to 3 saved episodes. |
| **Trial** | One cable insertion attempt by CheatCode. A trial may be saved or rejected by the quality gate. |
| **Episode** | One saved `.hdf5` file = one passing trial. |
| **Quality gate** | Automatic checks that reject bad episodes before they enter the dataset. |

### Trial order per session

Sessions 1–10 and 21–100 use **NIC → NIC → SC** order:

| Trial | Port type | Cable | Saved as |
|-------|-----------|-------|----------|
| 1 | SFP → NIC slot A | `sfp_sc_cable` | `episode_00000.hdf5` |
| 2 | SFP → NIC slot B | `sfp_sc_cable` | `episode_00001.hdf5` |
| 3 | SC fiber port | `sfp_sc_cable_reversed` | `episode_00002.hdf5` |

Sessions 11–20 deliberately use **SC → NIC → NIC** order (see `generate_competition_sessions.py`) to add training diversity:

| Trial | Port type | Cable | Saved as |
|-------|-----------|-------|----------|
| 1 | SC fiber port | `sfp_sc_cable_reversed` | `episode_00000.hdf5` |
| 2 | SFP → NIC slot A | `sfp_sc_cable` | `episode_00001.hdf5` |
| 3 | SFP → NIC slot B | `sfp_sc_cable` | `episode_00002.hdf5` |

Because the episode filename matches the trial index (0-based), a rejected trial leaves a gap. For example if sessions 11-20 SC trial (trial 0) is rejected, you will see `episode_00001.hdf5` and `episode_00002.hdf5` with no `episode_00000.hdf5` — this is correct behaviour, not a bug.

SFP trials almost always pass (final error ~1 mm). SC trials sometimes pass, sometimes are rejected — this is normal. The quality gate is working correctly either way.

---

## Prerequisites

Before starting, verify these are in place:

```bash
# 1. pixi is installed
pixi --version

# 2. distrobox container is built
distrobox list | grep aic_eval

# 3. YOLO model exists
ls $(git rev-parse --show-toplevel)/.pixi/envs/default/lib/python3.12/site-packages/team_policy/models/yolov12.pt

# 4. Session YAML files exist (should show 100 files)
ls $(git rev-parse --show-toplevel)/team_policy/team_policy/training_robot/configs/sessions/ | wc -l
```

---

## One-time setup

### 1. Set environment variables

Add these to your `~/.bashrc` so they are always available:

```bash
export AIC_ROOT=$(git -C ~/ros2_ws/src/aic rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml
```

Then reload: `source ~/.bashrc`

> **Important**: These variables must be set in the terminal where you run Terminals 2 and 3.
> Terminal 1 (distrobox) uses **hardcoded absolute paths only** — `$TRAIN_ROOT` does not expand inside distrobox.

### 2. Sync source files to pixi site-packages

The collector runs from the pixi-installed copy, not the source tree. Sync after any edit or after `pixi reinstall`:

```bash
SITE=$AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/team_policy
cp $TRAIN_ROOT/episode_recorder.py    $SITE/training_robot/episode_recorder.py
cp $TRAIN_ROOT/cheatcode_collector.py $SITE/training_robot/cheatcode_collector.py
cp $TRAIN_ROOT/convert_to_lerobot.py  $SITE/training_robot/convert_to_lerobot.py
cp $AIC_ROOT/team_policy/team_policy/run_act.py $SITE/run_act.py
```

Verify the sync worked:
```bash
python3 -c "from team_policy.training_robot.episode_recorder import SCHEMA_VERSION; print('schema:', SCHEMA_VERSION)"
# Expected: schema: 5
```

---

## Step 1 — Collect episodes

Open **2 terminals**. Set env vars in Terminal 2 before starting.

---

### Terminal 1 — Simulation

**Restart at the beginning of every run. Use a hardcoded absolute path for the session YAML.**

```bash
# Replace session_01 with session_02, session_03, ... each run
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    gazebo_gui:=false \
    launch_rviz:=false \
    aic_engine_config_file:=/home/YOUR_USER/ros2_ws/src/aic/team_policy/team_policy/training_robot/configs/sessions/session_01.yaml
```

> **Why absolute path?** Environment variables like `$TRAIN_ROOT` are expanded by your shell *before* being passed to distrobox. If `$TRAIN_ROOT` is not set in the current shell (common when opening a new terminal), the path silently becomes `/configs/sessions/session_01.yaml` which does not exist, and `aic_engine` crashes without printing any useful error. Always use the full absolute path in Terminal 1.

**Wait for this line before starting Terminal 2:**
```
[aic_engine] No node with name 'aic_model' found. Retrying...
```

What you will see before it (normal, ignore):
```
[joint_state_broadcaster_spawner]: Waiting for controller_manager ...
[gz_sim]: Unable to create renderer ...
[spawner_joint_state_broadcaster]: Controller spawner couldn't activate controllers ...
```

These errors always appear at startup. They are harmless. Only the `Retrying...` line matters.

---

### Terminal 2 — Collector + YOLO

**Restart at the beginning of every run. Increment the run number each time.**

```bash
# Run 1
cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector \
    -p output_dir:=$TRAIN_ROOT/episodes/run_001
```

Change `run_001` → `run_002` → `run_003` … for each new session.

You do not need to pass `num_episodes` or `success_only` here. The collector defaults to 3 episodes per run, and `success_only` defaults to `true`.
The collector starts the YOLO planner inside the same process by default, so do not run `combined_yolo_depth_pose_planner` separately during collection.

**Expected output on startup:**
```
[INFO] Combined planner node started: YOLO + metric depth + CAD registration...
[INFO] Embedded YOLO planner started
[INFO] DataCollectionPolicy ready — target=3 episodes, output=.../episodes/run_001, success_only=True
```

**Expected output during a run:**
```
[INFO] collector/episode=0 port=sfp/sfp_port_0
[INFO] collector/episode=0 force_baseline=20.8N          ← resting F/T (gripper weight)
[INFO] pfrac: 0.01 xy_error: 0.071 0.111  integrators: ...  ← CheatCode approaching
[INFO] [1/3] Saved .../run_001/episode_00000.hdf5        ← episode passed all gates

[INFO] collector/episode=1 port=sfp/sfp_port_0
...
[INFO] [2/3] Saved .../run_001/episode_00001.hdf5

[INFO] collector/episode=2 port=sc/sc_port_base
...
[INFO] [3/3] Saved .../run_001/episode_00002.hdf5        ← SC also saved (if quality gate passes)
```

**If an episode is rejected:**
```
collector/episode=N rejected by quality gate — discarded (port=sc_port_base max_err=0.010m ...)
collector/episode=N FAILED — discarded
collector/episode=N too short — discarded
```

Rejected episodes are never written to disk. The collector continues to the next trial automatically.

---

### Quality gates — what gets rejected and why

Every episode must pass all three checks before it is saved:

| Gate | SFP threshold | SC threshold | What triggers rejection |
|------|--------------|-------------|------------------------|
| **Minimum frames** | 10 frames | 10 frames | CheatCode crashed or timed out immediately |
| **Final insertion error** | < 3 mm | < 10 mm | Plug tip was not at port at end of trial |
| **Sustained contact force** | < 20 N for < 0.5 s | < 20 N for < 0.5 s | Robot pushed hard against something for too long |

Force measurements are **baseline-subtracted**: the resting F/T reading (~20–21 N from gripper + cable weight) is measured at the start of each episode and subtracted from all force values. "Contact force 0.5 N" means 0.5 N above the resting baseline, not 0.5 N absolute.

YOLO coverage (`yolo_valid_fraction`) is recorded but does **not** gate episode saving. An episode can be saved even if YOLO had 0% coverage — the converter will use zeros for all frames (matching inference cold-start), and the policy learns to handle that state.

---

### After 3 trials — end-of-run checklist

The engine exits automatically after trial 3. You will see Gazebo cleanup messages in Terminal 1.

1. Wait for Terminal 2 to finish the session or stop printing new collector messages after trial 3
2. `Ctrl+C` Terminal 1
3. Verify what was saved:
   ```bash
   ls -lh $TRAIN_ROOT/episodes/run_001/
   ```
4. Restart Terminal 1 with the **next** session YAML (`session_02.yaml`)
5. Wait for `Retrying...`
6. Restart Terminal 2 with the **next** run folder (`run_002`)

---

### Check episode quality at any time

Run this command to inspect all saved episodes across all runs:

```bash
cd $AIC_ROOT && pixi run python3 -c "
import h5py, numpy as np, pathlib

base = pathlib.Path('team_policy/team_policy/training_robot/episodes')
runs = sorted(base.glob('run_*'))
total = 0
for run in runs:
    eps = sorted(run.glob('episode_*.hdf5'))
    if not eps: continue
    print(f'\n=== {run.name} ({len(eps)} eps) ===')
    for ep in eps:
        with h5py.File(ep) as f:
            m = f['metadata']
            port_t = m.attrs.get('port_type','?')
            port   = m.attrs.get('port_name','?')
            frames = f['observations/tcp_pose'].shape[0]
            err    = m.attrs.get('final_error', float('nan'))
            force  = m.attrs.get('max_force', float('nan'))
            sust   = m.attrs.get('sustained_penalty_duration_s', float('nan'))
            yolo   = m.attrs.get('yolo_valid_fraction', float('nan'))
            ok     = '✓' if m.attrs.get('success',0) else '✗'
            print(f'  {ok} {ep.name}  {port_t}/{port:<14} frames={frames:3d}  err={err*1000:4.1f}mm  force={force:.2f}N  sust={sust:.2f}s  yolo={yolo:.0%}')
            total += 1
print(f'\nTotal saved: {total} episodes across {len(runs)} runs')
"
```

**What good episodes look like:**
```
✓ episode_00000.hdf5  sfp/sfp_port_0   frames=447  err= 1.1mm  force=0.38N  sust=0.00s  yolo=100%
✓ episode_00001.hdf5  sfp/sfp_port_0   frames=492  err= 1.1mm  force=0.28N  sust=0.00s  yolo=100%
✓ episode_00002.hdf5  sc/sc_port_base  frames=496  err= 0.6mm  force=1.88N  sust=0.00s  yolo=100%
```

---

### Session YAML files — why they exist

The 100 YAML files in `configs/sessions/` vary the task board's x, y, and yaw position across sessions. This forces the policy to learn to use YOLO (since port position changes every session), not just memorize a fixed insertion trajectory.

Each session file:
- Fixes the board at one pose for all 3 trials within that session (so CheatCode always sees a consistent scene)
- Uses a different pose than every other session (so the training dataset covers a wide range of board positions)

Session 1 through 50 are sufficient for a first training run (~100 quality episodes). Sessions 51–100 add diversity for a stronger model.

---

### Normal visual artifacts in Gazebo

**Cable "in the air"**
The SFP-SC cable has one end attached to the gripper (the plug being inserted) and one free end. The free end hangs from the gripper in a loop or sometimes gets flung to the side during physics initialization. This is normal and does not affect data quality — CheatCode uses TF poses, not cable positions, to control the arm.

**No NIC card visible**
The SC trial has no NIC cards in the scene — this is correct. For sessions 1–10 and 21–100 (NIC→NIC→SC order) this is trial 3. For sessions 11–20 (SC→NIC→NIC order) this is trial 1. If you see no NIC card at the start of what you expect to be a NIC trial, check whether the prior trials already completed while you were watching.

**Pink/magenta bounding box on the board**
A Gazebo visualization panel is open. Close it with the X in the Gazebo side panel — it uses GPU bandwidth but does not affect the simulation.

**Board at a steep angle**
Each session YAML sets a different yaw for the board. Some sessions have the board nearly sideways from the camera perspective. This is intentional — it provides training diversity.

---

## Step 2 — Merge all runs into one folder

Before converting, merge all run folders into a single folder using symlinks (no data is copied):

```bash
MERGED=$TRAIN_ROOT/episodes/all
rm -rf "$MERGED" && mkdir -p "$MERGED"
N=0
for f in $(ls $TRAIN_ROOT/episodes/run_*/*.hdf5 | sort); do
    ln -s "$f" "$MERGED/episode_$(printf '%05d' $N).hdf5"
    N=$((N+1))
done
echo "Total episodes merged: $N"
```

---

## Step 3 — Convert to LeRobot dataset

```bash
cd $AIC_ROOT
pixi run python -m team_policy.training_robot.convert_to_lerobot \
    --input  $TRAIN_ROOT/episodes/all \
    --output $TRAIN_ROOT/lerobot_datasets/aic_act_yolo_v2 \
    --success_only
```

The converter:
- Builds a 30D state vector: `tcp_pose(7) + tcp_vel(6) + joint_pos(7) + joint_vel(7) + port_xyz(3)`
- Fills `port_xyz` using the same logic as inference: zeros before first YOLO detection, real YOLO when valid, hold-last-known when YOLO is temporarily lost
- Writes LeRobot-format Parquet files and video-encoded camera observations

> **Dataset version**: Use `aic_act_yolo_v2` for any dataset converted after the port_xyz fix (2026-04-29). The old `aic_act_yolo_v1` used GT+noise for missing YOLO frames and should not be used for new training runs.

---

## Step 4 — Train

Run inside `tmux` so training survives closing the terminal:

```bash
tmux new -s training
# Detach any time: Ctrl+B then D
# Reattach:        tmux attach -t training
```

```bash
cd $AIC_ROOT
pixi run lerobot-train \
    --dataset.repo_id=local/aic_act_yolo_v2 \
    --dataset.root=$TRAIN_ROOT/lerobot_datasets/aic_act_yolo_v2 \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/aic_act_yolo_v2 \
    --job_name=aic_act_yolo_v2 \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=100000 \
    --save_freq=20000
```

Checkpoints are saved every 20k steps at `outputs/train/aic_act_yolo_v2/checkpoints/XXXXXX/pretrained_model/`.

**When to stop**: Monitor training loss. For 100 episodes, loss typically plateaus around 80–100k steps. For 300 episodes, train to 150–200k steps.

---

## Step 5 — Run inference

Open 3 terminals. Set env vars before starting.

**Terminal 1 — Simulation (no ground truth at inference)**
```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true \
    gazebo_gui:=false \
    launch_rviz:=false
```

**Terminal 2 — YOLO planner**
```bash
cd $AIC_ROOT && pixi run ros2 run team_policy combined_yolo_depth_pose_planner
```

**Terminal 3 — ACT policy**
```bash
cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act \
    -p checkpoint_path:=$AIC_ROOT/outputs/train/aic_act_yolo_v2/checkpoints/100000/pretrained_model
```

> **After any edit to `run_act.py`**, sync it before running inference:
> ```bash
> cp $AIC_ROOT/team_policy/team_policy/run_act.py \
>    $AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/team_policy/run_act.py
> ```

---

## Folder layout

```
<repo_root>/
  team_policy/team_policy/training_robot/
    configs/
      sessions/
        session_01.yaml   ← board pose variant 1 of 100
        session_02.yaml
        ...
        session_100.yaml
    episodes/
      run_001/            ← up to 3 episodes from session 1
        episode_00000.hdf5
        episode_00001.hdf5
        episode_00002.hdf5   (SC, if quality gate passed)
      run_002/
      ...
      all/                ← symlinks to all episodes, created before Step 3
    lerobot_datasets/
      aic_act_yolo_v2/    ← LeRobot format, created by Step 3
    episode_recorder.py    ← records observations, applies quality gates
    cheatcode_collector.py ← wraps CheatCode, drives the collection loop
    convert_to_lerobot.py  ← HDF5 → LeRobot dataset
    analyze_yolo_sessions.py ← inspects per-run YOLO coverage & quality

  outputs/train/
    aic_act_yolo_v2/
      checkpoints/
        020000/pretrained_model/
        040000/pretrained_model/
        ...
        100000/pretrained_model/   ← use this for inference
```

---

## HDF5 episode structure

Each `.hdf5` file contains:

```
observations/
  tcp_pose          (T, 7)   — end-effector pose [x,y,z,qx,qy,qz,qw] in base_link
  tcp_velocity      (T, 6)   — end-effector velocity [vx,vy,vz,wx,wy,wz]
  joint_positions   (T, 7)   — joint angles [rad]
  joint_velocity    (T, 7)   — joint velocities [rad/s]
  wrist_force       (T, 6)   — F/T sensor [fx,fy,fz,tx,ty,tz]
  tcp_error         (T, 6)   — controller tracking error (near-zero; NOT GT distance)
  relative_pose     (T, 7)   — port pose in plug-tip frame (GT privileged, for analysis)
  yolo_port_xyz     (T, 3)   — YOLO-detected port position in base_link [x,y,z]
  yolo_port_valid   (T,)     — bool: did YOLO have a detection this frame?
  privileged_tf/
    transforms      (T,5,7)  — selected TF pairs for debugging
    valid           (T,5)    — bool: was each TF available?
    frame_pairs     (5,)     — string labels for the 5 TF pairs
  images/
    center          (T,H,W,3) — center camera RGB
    left            (T,H,W,3) — left camera RGB
    right           (T,H,W,3) — right camera RGB

actions/
  commanded_pose    (T, 7)   — CheatCode's target pose each step
  delta_pose        (T, 6)   — finite difference of commanded poses (velocity-like)
  velocity          (T, 6)   — same as delta_pose, last step = zeros

metadata/
  schema_version              — "5"
  port_type / port_name       — "sfp" / "sfp_port_0" etc.
  success                     — 1 if CheatCode reported success
  final_error                 — plug-to-port distance [m] at end of trial
  max_force                   — peak baseline-subtracted contact force [N]
  sustained_penalty_duration_s — seconds above 20 N contact force
  force_baseline_n            — resting F/T reading subtracted from all force metrics
  yolo_valid_fraction         — fraction of frames with real YOLO detections (1.0 = perfect)
```

---

## Troubleshooting

**`aic_engine` starts but never prints `Retrying...`**
The session YAML path is wrong. Most common cause: `$TRAIN_ROOT` was not set in the terminal where you ran the distrobox command, so the path became empty. Always use a hardcoded absolute path in Terminal 1.

**Collector exits immediately with `Reached target of N episodes`**
The `output_dir` folder already has `.hdf5` files from a previous run. The collector counts existing files and adds to them. Either use a new `run_NNN` folder or delete the existing episodes.

**`TF_OLD_DATA` spam in Terminal 2 — during restart AND during active trials**
Normal in both cases, for different reasons:
- **At sim restart**: The sim clock resets to 0. The YOLO node's TF buffer still has cached robot joint transforms from the previous session (e.g. timestamped at t=180s). These appear "from the future" relative to the new clock and are flushed.
- **During active trials**: The cable physics plugin (`cable_0/link_9`, `cable_0/link_10`, `cable_0/cable_connection_1`, etc.) publishes TF slightly late relative to the main sim clock. These frames are never needed for YOLO port detection — YOLO only uses robot joint TFs to transform image detections to base_link. The cable TF lag is a Gazebo physics plugin timing issue and is completely harmless.

In both cases: do not restart just because of this warning. YOLO port detection is unaffected when saved episodes show high `yolo_valid_fraction`.

**All SC episodes rejected by quality gate**
CheatCode descends a fixed 15 mm below the SC port frame origin (`sc_port_base_link`). This frame is at the housing chassis, not the optical fiber connection point, so the final insertion error is sometimes > 10 mm. Whether SC passes depends on the board pose in the session YAML. Sessions where CheatCode's fixed descent aligns well with the SC port geometry pass; others do not. This is expected. SFP episodes (which are the majority of training data) are unaffected.

**Controller spawner errors at startup**
Normal. Gazebo brings up joint controllers in stages. The spawner retries until they are ready. These errors always appear and are always harmless.

**Episode count stuck — files not appearing**
Run `echo $TRAIN_ROOT` to verify the variable is set. If empty, set it and relaunch Terminal 2.

**`/fused_yolo/detections_json` looks like it contains only `task_board`**
This is often just output truncation. That topic publishes one long JSON list and `task_board` is intentionally sorted first, so a shortened console line can hide the later `fix`, `nic_card`, `sc_port`, or `sfp_port_*` entries. Expand/copy the full JSON or check the saved episode's `yolo_valid_fraction` before assuming the other detections are missing.

**Force gate rejects every episode**
The baseline subtraction may have failed. Check the log for `force_baseline=XX.XN` — it should read 18–22 N. If it reads 0 N, the observation was not available at episode start (sim still loading). Try increasing the `time.sleep` in the baseline measurement loop in `cheatcode_collector.py`.

**Gazebo performance is slow (< 50% real-time)**
Close the Gazebo side panels (Entity tree, Component inspector, Visualize contacts). Each panel runs an additional render pass. Set `gazebo_gui:=false` in Terminal 1 if you do not need the 3D view at all — this gives the fastest collection speed.
