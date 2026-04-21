# CheatCode Training Data Pipeline

This folder contains the imitation-learning data pipeline for collecting expert
demonstrations from `aic_example_policies.ros.CheatCode`.

The high-level flow is:

```text
aic_engine trial
  -> aic_model loads team_policy.training_robot.cheatcode_collector
  -> collector runs CheatCode as the expert
  -> EpisodeRecorder records observations + expert actions
  -> episode_XXXXX.hdf5 files are written
  -> validate_episode.py checks the files
  -> convert_to_lerobot.py converts HDF5 episodes to a LeRobot-style dataset
```

Important: `CheatCode` requires `ground_truth:=true`. Ground truth is allowed for
training data generation, but the final learned policy should not use hidden
ground-truth inputs during evaluation.

## Files

- `cheatcode_collector.py`: policy wrapper used by `aic_model`.
- `episode_recorder.py`: writes HDF5 episodes.
- `validate_episode.py`: validates one HDF5 episode.
- `convert_to_lerobot.py`: converts collected HDF5 files to parquet/metadata.
- `configs/orientation_sweep_3_trials.yaml`: example engine config with varied
  task-board pose/orientation.

Generated data is ignored by Git:

```text
training_robot/episodes/
training_robot/lerobot_datasets/
training_robot/dataset/
```

## One-Time Setup

From the AIC repo root:

```bash
cd ~/ros2_ws/src/aic

export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
```

Install/reinstall the package after changing training code:

```bash
pixi install
pixi reinstall ros-kilted-team-policy
```

## Collect One 3-Trial Batch

The provided config has 3 trials. One engine run should save up to 3 successful
episodes.

Use two terminals.

### Terminal 1: Simulation + Engine

```bash
cd ~/ros2_ws/src/aic

export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  gazebo_gui:=false \
  launch_rviz:=false \
  aic_engine_config_file:="$TRAIN_ROOT/configs/orientation_sweep_3_trials.yaml"
```

Wait until the engine is up and waiting/retrying for the model.

### Terminal 2: Collector Model

```bash
cd ~/ros2_ws/src/aic

export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export RUN_ID="run_001"
export OUTPUT_DIR="$TRAIN_ROOT/episodes/orientation_sweep/$RUN_ID"

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.training_robot.cheatcode_collector \
  -p output_dir:="$OUTPUT_DIR" \
  -p num_episodes:=3 \
  -p success_only:=true
```

Expected collector logs:

```text
DataCollectionPolicy ready
collector/episode=0 ...
[1/3] Saved .../episode_00000.hdf5
[2/3] Saved .../episode_00001.hdf5
[3/3] Saved .../episode_00002.hdf5
on_shutdown(...)
```

After the batch finishes, stop Terminal 1 if it did not exit on its own.

## Collect Multiple Batches

Because the config has 3 trials, collect more data by rerunning the two-terminal
sequence with a new `RUN_ID` each time:

```bash
export RUN_ID="run_002"
export OUTPUT_DIR="$TRAIN_ROOT/episodes/orientation_sweep/$RUN_ID"
```

For 9 episodes, run:

```text
run_001 -> about 3 episodes
run_002 -> about 3 episodes
run_003 -> about 3 episodes
```

For 150 episodes, you need about 50 of these 3-trial runs, or a larger engine
YAML with more trials.

## Count Episodes

```bash
cd ~/ros2_ws/src/aic

find team_policy/team_policy/training_robot/episodes/orientation_sweep \
  -name 'episode_*.hdf5' | sort

find team_policy/team_policy/training_robot/episodes/orientation_sweep \
  -name 'episode_*.hdf5' | wc -l
```

## Validate Episodes

Validate one episode:

```bash
cd ~/ros2_ws/src/aic

pixi run python -m team_policy.training_robot.validate_episode \
  --file team_policy/team_policy/training_robot/episodes/orientation_sweep/run_001/episode_00000.hdf5
```

Validate all episodes under one dataset folder:

```bash
cd ~/ros2_ws/src/aic

for f in team_policy/team_policy/training_robot/episodes/orientation_sweep/run_*/episode_*.hdf5; do
  echo "Validating $f"
  pixi run python -m team_policy.training_robot.validate_episode --file "$f" || break
done
```

Expected ending:

```text
PASS - episode looks valid
```

For new schema-v2 episodes, validation should show:

```text
Schema v2 keys
Delta pose actions
Relative pose (plug -> target)
Quality metrics
```

## Convert To LeRobot-Style Dataset

Convert one run:

```bash
cd ~/ros2_ws/src/aic

pixi run python -m team_policy.training_robot.convert_to_lerobot \
  --input team_policy/team_policy/training_robot/episodes/orientation_sweep/run_001 \
  --output team_policy/team_policy/training_robot/lerobot_datasets/orientation_sweep_run_001 \
  --success_only
```

Expected output structure:

```text
lerobot_datasets/orientation_sweep_run_001/
  meta/
    info.json
    stats.json
    episodes.jsonl
    tasks.jsonl
  data/chunk-000/
    episode_000000.parquet
    episode_000001.parquet
    episode_000002.parquet
```

Note: the current converter writes parquet state/action metadata. It does not yet
write image videos under `videos/`, so full image-based ACT training compatible
with `RunACT.py` still needs image/video export support.

## Training Command Template

Only run this after the converted dataset format matches the LeRobot policy you
want to train. For image-based ACT, first complete video/image export in
`convert_to_lerobot.py`.

Template:

```bash
cd ~/ros2_ws/src/aic

pixi run lerobot-train \
  --dataset.repo_id=local/aic_orientation_sweep \
  --policy.type=act \
  --output_dir=outputs/train/aic_act_orientation_sweep \
  --job_name=aic_act_orientation_sweep \
  --policy.device=cuda \
  --wandb.enable=false
```

If training on a CPU-only machine, change:

```bash
--policy.device=cpu
```

## Useful Paths

```bash
export AIC_ROOT=~/ros2_ws/src/aic
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export CONFIG="$TRAIN_ROOT/configs/orientation_sweep_3_trials.yaml"
export EPISODES="$TRAIN_ROOT/episodes/orientation_sweep"
export LEROBOT="$TRAIN_ROOT/lerobot_datasets"
```

## Common Issues

### Only 3 Episodes Are Saved

This is expected with `orientation_sweep_3_trials.yaml`. The collector records
trials; it does not create trials. The number of trials comes from the engine
YAML.

### Reusing The Same Output Folder

Do not reuse the same `output_dir` unless you want to overwrite
`episode_00000.hdf5`, `episode_00001.hdf5`, etc. Use a new `RUN_ID`.

### Collector Cannot Find Ground Truth TF

Make sure Terminal 1 uses:

```bash
ground_truth:=true
```

### Headless Collection

It is safe to disable GUI tools:

```bash
gazebo_gui:=false
launch_rviz:=false
```

The Gazebo server/simulation still runs in the background and publishes the data
needed for collection.
