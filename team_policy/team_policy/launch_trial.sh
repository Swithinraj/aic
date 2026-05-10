#!/usr/bin/env bash
# ============================================================================
# launch_trial.sh — One-command AIC trial launcher
#
# Usage:
#   ./launch_trial.sh                     # defaults: session_01, GUI on
#   ./launch_trial.sh --session 03        # use session_03.yaml
#   ./launch_trial.sh --no-gui            # headless (faster)
#   ./launch_trial.sh --session 05 --no-gui
#
# This script coordinates the 5 processes in the correct startup order:
#   1. Zenoh router  (inside aic_eval)
#   2. Gazebo + sim  (inside aic_eval)
#   3. YOLO planner  (pixi env on host)
#   4. ACT model     (pixi env on host)
#   5. AIC engine    (inside aic_eval)
#
# Press Ctrl-C to shut everything down cleanly.
# ============================================================================
set -euo pipefail

# ---- Defaults ----
SESSION_NUM="01"
GAZEBO_GUI="true"
LAUNCH_RVIZ="true"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --session) SESSION_NUM="$2"; shift 2 ;;
        --no-gui)  GAZEBO_GUI="false"; LAUNCH_RVIZ="false"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SESSION_YAML="/home/zaid/ws_aic/src/aic/team_policy/team_policy/training_robot/configs/sessions/session_${SESSION_NUM}.yaml"
CHECKPOINT="/home/zaid/ws_aic/src/aic/team_policy/team_policy/models/trained_model_v3/040000/pretrained_model"
PIXI="/home/zaid/.pixi/bin/pixi"
PIXI_MANIFEST="/home/zaid/ws_aic/src/aic/pixi.toml"
CONTAINER="aic_eval"

# Log files
LOG_DIR="/tmp/aic_trial_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "╔════════════════════════════════════════╗"
echo "║     AIC Trial Launcher                 ║"
echo "╠════════════════════════════════════════╣"
echo "║  Session : ${SESSION_NUM}                          ║"
echo "║  GUI     : ${GAZEBO_GUI}                        ║"
echo "║  Logs    : ${LOG_DIR}"
echo "╚════════════════════════════════════════╝"
echo ""

if [ ! -f "$SESSION_YAML" ]; then
    echo "ERROR: Session file not found: $SESSION_YAML"
    exit 1
fi

# ---- Cleanup function ----
PIDS=()
cleanup() {
    echo ""
    echo ">>> Shutting down all processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Kill processes inside container
    docker exec "$CONTAINER" bash -c \
        "pkill -9 -f 'rmw_zenohd|aic_engine|combined_yolo_depth_pose_planner|component_container|robot_state_pub|gz sim|rviz2|aic_adapter|ros_gz' 2>/dev/null" \
        || true
    echo ">>> Cleanup done. Logs in $LOG_DIR"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ---- Helper: run command inside aic_eval container ----
in_container() {
    docker exec "$CONTAINER" bash -c "
        source /ws_aic/install/setup.bash
        export RMW_IMPLEMENTATION=rmw_zenoh_cpp
        export ZENOH_ROUTER_CHECK_ATTEMPTS=0
        $1
    "
}

in_container_bg() {
    docker exec "$CONTAINER" bash -c "
        source /ws_aic/install/setup.bash
        export RMW_IMPLEMENTATION=rmw_zenoh_cpp
        export ZENOH_ROUTER_CHECK_ATTEMPTS=0
        $1
    " > "$2" 2>&1 &
    PIDS+=($!)
}

# If using distrobox for GUI (X11 forwarding)
in_distrobox_bg() {
    distrobox enter "$CONTAINER" -- bash -c "
        source /ws_aic/install/setup.bash
        export RMW_IMPLEMENTATION=rmw_zenoh_cpp
        export ZENOH_ROUTER_CHECK_ATTEMPTS=0
        $1
    " > "$2" 2>&1 &
    PIDS+=($!)
}

# ============================================================================
# STEP 0: Kill any leftover processes
# ============================================================================
echo ">>> [0/5] Cleaning up old processes..."
docker exec "$CONTAINER" bash -c \
    "pkill -9 -f 'rmw_zenohd|aic_engine|aic_model|combined_yolo_depth_pose_planner|component_container|robot_state_pub|gz sim|rviz2|aic_adapter|ros_gz' 2>/dev/null" \
    || true
# Also kill any host-side model
pkill -9 -f "aic_model" 2>/dev/null || true
pkill -9 -f "combined_yolo_depth_pose_planner" 2>/dev/null || true
sleep 2

# ============================================================================
# STEP 1: Start Zenoh router
# ============================================================================
echo ">>> [1/5] Starting Zenoh router..."
in_container_bg "ros2 run rmw_zenoh_cpp rmw_zenohd" "$LOG_DIR/zenoh_router.log"
sleep 3

# Verify router is running
if ! docker exec "$CONTAINER" pgrep -f rmw_zenohd > /dev/null 2>&1; then
    echo "ERROR: Zenoh router failed to start. Check $LOG_DIR/zenoh_router.log"
    exit 1
fi
echo "    ✓ Zenoh router running"

# ============================================================================
# STEP 2: Start Gazebo + Sim
# ============================================================================
echo ">>> [2/5] Starting Gazebo simulation..."
if [ "$GAZEBO_GUI" = "true" ]; then
    # Use distrobox for X11 forwarding (GUI)
    in_distrobox_bg "
        ros2 launch aic_bringup aic_gz_bringup.launch.py \
            start_aic_engine:=false \
            gazebo_gui:=${GAZEBO_GUI} \
            launch_rviz:=${LAUNCH_RVIZ}
    " "$LOG_DIR/sim.log"
else
    # Headless — docker exec is fine
    in_container_bg "
        ros2 launch aic_bringup aic_gz_bringup.launch.py \
            start_aic_engine:=false \
            gazebo_gui:=false \
            launch_rviz:=false
    " "$LOG_DIR/sim.log"
fi

# Wait for simulation to be ready (controllers activated)
echo "    Waiting for simulation to be ready..."
for i in $(seq 1 90); do
    if grep -q "Successfully switched controllers" "$LOG_DIR/sim.log" 2>/dev/null; then
        echo "    ✓ Simulation ready (${i}s)"
        break
    fi
    if [ "$i" -eq 90 ]; then
        echo "ERROR: Simulation didn't start in 90s. Check $LOG_DIR/sim.log"
        tail -20 "$LOG_DIR/sim.log"
        exit 1
    fi
    sleep 1
done

# ============================================================================
# STEP 3: Start YOLO planner (pixi env)
# ============================================================================
echo ">>> [3/5] Starting YOLO port planner..."
"$PIXI" run --manifest-path "$PIXI_MANIFEST" ros2 run team_policy combined_yolo_depth_pose_planner \
    > "$LOG_DIR/yolo.log" 2>&1 &
PIDS+=($!)
sleep 2

# ============================================================================
# STEP 4: Start ACT Model (pixi env)
# ============================================================================
echo ">>> [4/5] Starting ACT model..."
"$PIXI" run --manifest-path "$PIXI_MANIFEST" ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act \
    -p checkpoint_path:="$CHECKPOINT" \
    > "$LOG_DIR/model.log" 2>&1 &
PIDS+=($!)

# Wait for model to fully load
echo "    Waiting for model to load..."
for i in $(seq 1 60); do
    if grep -q "Using policy: run_act" "$LOG_DIR/model.log" 2>/dev/null; then
        echo "    ✓ Model loaded (${i}s)"
        break
    fi
    if grep -q "Error\|FATAL\|Traceback" "$LOG_DIR/model.log" 2>/dev/null; then
        echo "ERROR: Model failed to load. Check $LOG_DIR/model.log"
        tail -20 "$LOG_DIR/model.log"
        exit 1
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Model didn't load in 60s. Check $LOG_DIR/model.log"
        tail -20 "$LOG_DIR/model.log"
        exit 1
    fi
    sleep 1
done

# Extra wait for the model to start spinning and respond to lifecycle queries
sleep 3

# ============================================================================
# STEP 5: Start Engine
# ============================================================================
echo ">>> [5/5] Starting AIC engine (session_${SESSION_NUM})..."
echo ""
echo "════════════════════════════════════════"
echo "  Engine output (live):"
echo "════════════════════════════════════════"

# Run engine in foreground so user sees the output
in_container "
    ros2 run aic_engine aic_engine --ros-args \
        -p config_file_path:=${SESSION_YAML} \
        -p use_sim_time:=true
" 2>&1 | tee "$LOG_DIR/engine.log"

echo ""
echo "════════════════════════════════════════"
echo "  Engine finished. Logs: $LOG_DIR"
echo "════════════════════════════════════════"

# Copy scoring results if available
if [ -d /home/zaid/aic_results ]; then
    cp -r /home/zaid/aic_results "$LOG_DIR/results" 2>/dev/null || true
fi
