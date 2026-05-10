#!/bin/bash
#
# launch_act_trial.sh — One-shot script to deploy ACT model and run a full trial session.
#
# Usage:
#   bash launch_act_trial.sh [session_config.yaml]
#
# Default session: session_01.yaml (3 trials: NIC0, NIC2, SC0)
#
set -euo pipefail

AIC_ROOT=/home/zaid/ws_aic/src/aic
SESSION_CONFIG="${1:-$AIC_ROOT/team_policy/team_policy/training_robot/configs/sessions/session_01.yaml}"
CKPT="$AIC_ROOT/team_policy/team_policy/models/trained_model_v3/040000/pretrained_model"

echo "============================================"
echo "  ACT Model Deployment — Trial Launch"
echo "============================================"
echo "Session config: $SESSION_CONFIG"
echo "Checkpoint:     $CKPT"
echo ""

# --- Step 1: Kill existing processes ---
echo "[1/5] Cleaning up existing processes..."
docker exec aic_eval bash -c "
  ps aux | grep -E 'ros2|gz|aic_|zenoh|component|robot_state|create|spawner|pixi|combined_yolo' \
    | grep -v grep | grep -v gnome | awk '{print \\\$2}' \
    | xargs -r kill -9 2>/dev/null || true
" 2>/dev/null
sleep 2
echo "       Done."

# --- Step 2: Start Zenoh Router ---
echo "[2/5] Starting Zenoh router..."
docker exec -d aic_eval bash -c '
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ZENOH_CONFIG_OVERRIDE="mode=\"router\";listen/endpoints=[\"tcp/[::]:7447\"];connect/endpoints=[];routing/router/peers_failover_brokering=true;transport/shared_memory/enabled=false"
export ZENOH_CONFIG_OVERRIDE
export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5
ros2 run rmw_zenoh_cpp rmw_zenohd > /tmp/zenoh_router.log 2>&1
'
sleep 2
echo "       Zenoh router running on :7447"

# --- Step 3: Start YOLO planner (V2 policy state uses held port_xyz) ---
echo "[3/5] Starting YOLO port planner..."
docker exec -d aic_eval bash -c "
cd $AIC_ROOT
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE=';transport/shared_memory/enabled=false'
/home/zaid/.pixi/bin/pixi run ros2 run team_policy combined_yolo_depth_pose_planner \
    > /tmp/aic_yolo_planner.log 2>&1
"
sleep 2
echo "       YOLO planner starting; logs: /tmp/aic_yolo_planner.log"

# --- Step 4: Start ACT Model (pre-load) ---
echo "[4/5] Starting ACT model (pre-loading PyTorch weights)..."
docker exec -d aic_eval bash -c "
cd $AIC_ROOT
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE=';transport/shared_memory/enabled=false'
/home/zaid/.pixi/bin/pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act \
    -p checkpoint_path:=$CKPT \
    > /tmp/aic_model_inside.log 2>&1
"

# Wait for model to finish loading
for i in $(seq 1 60); do
    if docker exec aic_eval grep -q "Using policy: run_act" /tmp/aic_model_inside.log 2>/dev/null; then
        echo "       Model loaded in ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Model failed to load within 60s"
        docker exec aic_eval cat /tmp/aic_model_inside.log 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Verify model is discoverable
STATE=$(docker exec aic_eval bash -c "source /ws_aic/install/setup.bash && export RMW_IMPLEMENTATION=rmw_zenoh_cpp && ros2 lifecycle get /aic_model 2>&1" || echo "FAIL")
if echo "$STATE" | grep -q "unconfigured"; then
    echo "       Model responsive: $STATE"
else
    echo "WARNING: Model state check returned: $STATE"
    echo "         Proceeding anyway..."
fi

# --- Step 5: Launch Simulation + Engine ---
echo "[5/5] Launching simulation + engine..."
docker exec -d aic_eval bash -c "
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export DISPLAY=:0
export ZENOH_CONFIG_OVERRIDE=';transport/shared_memory/enabled=false'
> /tmp/aic_sim_launch.log
ros2 launch aic_bringup aic_gz_bringup.launch.py \
    ground_truth:=false \
    start_aic_engine:=true \
    aic_engine_config_file:=$SESSION_CONFIG \
    model_discovery_timeout_seconds:=180 \
    model_configure_timeout_seconds:=120 \
    gazebo_gui:=false \
    launch_rviz:=false \
    > /tmp/aic_sim_launch.log 2>&1
"

echo ""
echo "============================================"
echo "  RUNNING — Monitoring trial progress..."
echo "============================================"
echo ""

# Monitor until completion
STARTED=$(date +%s)
while true; do
    ELAPSED=$(( $(date +%s) - STARTED ))

    # Check for completion
    if docker exec aic_eval grep -q "Complete Scoring Results" /tmp/aic_sim_launch.log 2>/dev/null; then
        echo ""
        echo "============================================"
        echo "  ALL TRIALS COMPLETE (${ELAPSED}s)"
        echo "============================================"
        echo ""
        cat /home/zaid/aic_results/scoring.yaml 2>/dev/null || docker exec aic_eval grep "aic_engine" /tmp/aic_sim_launch.log 2>/dev/null | grep -E "total:|score:|message:" | head -20
        break
    fi

    # Check for engine death
    if docker exec aic_eval grep -q "Engine Stopped with Errors" /tmp/aic_sim_launch.log 2>/dev/null; then
        echo ""
        echo "ERROR: Engine stopped with errors at ${ELAPSED}s"
        docker exec aic_eval grep "aic_engine" /tmp/aic_sim_launch.log 2>/dev/null | grep -E "ERROR|score|message" | tail -20
        exit 1
    fi

    # Progress output every 15s
    if [ $((ELAPSED % 15)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
        LAST=$(docker exec aic_eval grep "aic_engine" /tmp/aic_sim_launch.log 2>/dev/null | tail -1)
        MODEL=$(docker exec aic_eval tail -1 /tmp/aic_model_inside.log 2>/dev/null)
        echo "[${ELAPSED}s] Engine: $(echo "$LAST" | sed 's/.*aic_engine]: //')"
        echo "        Model:  $(echo "$MODEL" | sed 's/.*aic_model]: //')"
    fi

    sleep 1

    # Safety timeout (15 min)
    if [ $ELAPSED -gt 900 ]; then
        echo "TIMEOUT: No completion after 900s"
        exit 1
    fi
done
