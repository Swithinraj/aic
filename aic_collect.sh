#!/usr/bin/env bash
# =============================================================================
# aic_collect.sh — Automated multi-trial episode collection for AIC
# =============================================================================
#
# HOW IT WORKS
# ------------
# This script drives two tmux panes in parallel:
#   Pane 0 (left)  — distrobox / aic_engine   (Terminal 1)
#   Pane 1 (right) — pixi / aic_model         (Terminal 2)
#
# For each trial it:
#   1. Launches aic_engine with the correct session_XX.yaml in pane 0
#   2. Tails pane 0's log file until "Retrying..." appears
#   3. Launches the collector with the correct run_XXX folder in pane 1
#   4. Tails pane 0's log file until "process has finished cleanly" appears
#   5. Sends tmux send-keys C-c to BOTH panes in order (T2 first, then T1)
#      → this delivers SIGINT to the foreground process group of each pane,
#        exactly like pressing Ctrl+C at the keyboard
#   6. Waits for processes to exit, then starts the next trial
#
# WHY tmux send-keys C-c WORKS WHERE OTHER APPROACHES FAIL
# ---------------------------------------------------------
# kill -SIGINT <pid> targets a single PID. Distrobox pipelines and pixi
# launchers spawn process trees. tmux send-keys C-c sends SIGINT to the
# entire foreground process group of the terminal, which propagates into
# every child — including processes inside the distrobox container.
#
# PREREQUISITES
# -------------
#   sudo apt install tmux         (or dnf install tmux on Fedora)
#   $AIC_ROOT and $EPISODES_DIR exported in ~/.bashrc
#   distrobox container 'aic_eval' running
#
# USAGE
# -----
#   bash aic_collect.sh              # run all sessions (1 to N)
#   bash aic_collect.sh 3            # start from session 03
#   bash aic_collect.sh 3 10         # run sessions 03 through 10
#
# Monitor progress in a second terminal:
#   tmux attach -t aic_collect
#
# =============================================================================

set -euo pipefail

# =============================================================================
# ── CONFIG — edit these if your paths change ─────────────────────────────────
# =============================================================================

# Absolute path to sessions directory (used in distrobox, no env vars allowed)
SESSIONS_DIR="/home/zaid/ws_aic/src/aic/team_policy/team_policy/training_robot/configs/sessions"

DISTROBOX_CONTAINER="aic_eval"
TMUX_SESSION="aic_collect"

# Patterns to watch for in Terminal 1 log
TRIGGER_READY="aic_engine-[0-9]+.*No node with name 'aic_model' found. Retrying"
TRIGGER_DONE="aic_engine-[0-9]+\]: process has finished cleanly"

# Tuning
READY_TIMEOUT=180    # seconds to wait for aic_engine ready signal
DONE_TIMEOUT=900     # seconds to wait for session completion (3 trials × ~3 min each + margin)
READY_POLL=2         # polling interval while waiting for ready
DONE_POLL=3          # polling interval while waiting for done
POST_CTRLC_WAIT=20   # seconds to wait after Ctrl+C before starting next trial (Zenoh needs ~15s to release port 7447)

# Temp directories — safe to delete after a run
# STATUS_DIR holds only the matched trigger lines (~10 lines per trial, negligible RAM on tmpfs)
# Full output stays in the tmux panes and is never written to disk
STATUS_DIR="/tmp/aic_collect_status"
SCRIPT_DIR="/tmp/aic_collect_scripts"

# Score log — appended across all trials, kept after the run
SCORES_CSV="${HOME}/aic_scores.csv"

# =============================================================================
# ── PREFLIGHT CHECKS ─────────────────────────────────────────────────────────
# =============================================================================

if ! command -v tmux &>/dev/null; then
    echo "[ERROR] tmux is not installed. Run: sudo apt install tmux"
    exit 1
fi

if [[ ! -d "$SESSIONS_DIR" ]]; then
    echo "[ERROR] Sessions directory not found: $SESSIONS_DIR"
    exit 1
fi

TOTAL=$(ls "$SESSIONS_DIR"/*.yaml 2>/dev/null | wc -l)
if (( TOTAL == 0 )); then
    echo "[ERROR] No .yaml files found in: $SESSIONS_DIR"
    exit 1
fi

# =============================================================================
# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────
# =============================================================================

START=${1:-28}
END=${2:-$TOTAL}

if (( START < 1 || END > TOTAL || START > END )); then
    echo "[ERROR] Invalid range: $START–$END (total sessions: $TOTAL)"
    exit 1
fi

mkdir -p "$STATUS_DIR" "$SCRIPT_DIR"

# Write CSV header only if the file does not exist yet
if [[ ! -f "$SCORES_CSV" ]]; then
    echo "run_id,trial_number,score" > "$SCORES_CSV"
    echo "Score log created: $SCORES_CSV"
else
    echo "Appending scores to existing log: $SCORES_CSV"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AIC Episode Collector"
echo "  Sessions dir : $SESSIONS_DIR"
echo "  Total sessions: $TOTAL"
echo "  Running range : $START → $END"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# =============================================================================
# ── TMUX SETUP ───────────────────────────────────────────────────────────────
# =============================================================================

# Kill any leftover session from a previous run
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

# New session: one window, split into two panes side by side
#   Pane 0 = left  (Terminal 1)
#   Pane 1 = right (Terminal 2)
tmux new-session  -d -s "$TMUX_SESSION" -x 240 -y 55
tmux split-window -h -t "${TMUX_SESSION}:0"
tmux select-pane  -t "${TMUX_SESSION}:0.0"

echo "tmux session created. Monitor with:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""

# =============================================================================
# ── HELPERS ──────────────────────────────────────────────────────────────────
# =============================================================================

# wait_for FILE PATTERN TIMEOUT POLL_INTERVAL
# Returns 0 if pattern found, 1 on timeout.
wait_for() {
    local file="$1" pattern="$2" timeout="$3" poll="$4"
    local elapsed=0
    while ! grep -qE "$pattern" "$file" 2>/dev/null; do
        sleep "$poll"
        elapsed=$(( elapsed + poll ))
        if (( elapsed >= timeout )); then
            return 1
        fi
    done
    return 0
}

# ts — timestamp prefix
ts() { date "+[%H:%M:%S]"; }

# kill_lingering — force-kill ROS2/Gazebo processes that survived Ctrl+C
kill_lingering() {
    local patterns="component_container|aic_engine|ros_gz|aic_adapter|robot_state_publisher|zenoh"
    local pids
    pids=$(pgrep -f "$patterns" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "$(ts) Sending SIGINT to lingering processes: $(echo $pids | tr '\n' ' ')"
        kill -SIGINT $pids 2>/dev/null || true
        sleep 3
        # Anything still alive gets SIGKILL
        pids=$(pgrep -f "$patterns" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "$(ts) Force-killing remaining processes: $(echo $pids | tr '\n' ' ')"
            kill -SIGKILL $pids 2>/dev/null || true
        fi
    fi
    # T2: aic_model collector
    pids=$(pgrep -f "aic_model" 2>/dev/null || true)
    [[ -n "$pids" ]] && kill -SIGKILL $pids 2>/dev/null || true
}

# wait_for_port — block until port 7447 is no longer bound, with timeout
wait_for_port() {
    local port=7447
    local timeout=40
    local elapsed=0
    echo "$(ts) Waiting for Zenoh port ${port} to clear..."
    while ss -tlnp 2>/dev/null | grep -q ":${port}"; do
        sleep 2
        elapsed=$(( elapsed + 2 ))
        if (( elapsed >= timeout )); then
            echo "$(ts) [WARN] Port ${port} still occupied after ${timeout}s. Proceeding anyway."
            return 0
        fi
    done
    echo "$(ts) Port ${port} clear."
}

# log_scores STATUS_FILE RUN_ID
# Parses score lines from the status file and appends to SCORES_CSV.
# CSV format: run_id,trial_number,score
# Example line matched:
#   [aic_engine-8] ... ✓ Trial 'trial_2' completed successfully! Score: 92.913012
log_scores() {
    local status_file="$1" run="$2"
    local found=0
    while IFS= read -r line; do
        local trial score
        trial=$(echo "$line" | grep -oE "trial_[0-9]+" | grep -oE "[0-9]+")
        score=$(echo "$line" | grep -oE "Score: [0-9.]+" | grep -oE "[0-9.]+")
        if [[ -n "$trial" && -n "$score" ]]; then
            echo "${run},${trial},${score}" >> "$SCORES_CSV"
            echo "$(ts) Score logged: run=${run} trial=${trial} score=${score}"
            found=$(( found + 1 ))
        fi
    done < <(grep -E "Score:" "$status_file" 2>/dev/null)
    if (( found == 0 )); then
        echo "$(ts) [WARN] No score lines found in status file for ${run}."
    fi
}

# =============================================================================
# ── MAIN LOOP ────────────────────────────────────────────────────────────────
# =============================================================================

FAILED_TRIALS=()

for i in $(seq "$START" "$END"); do

    SESSION=$(printf "session_%02d.yaml" "$i")
    RUN=$(printf "run_%03d" "$i")
    T1_STATUS="$STATUS_DIR/t1_$(printf '%03d' "$i").status"
    T1_SCRIPT="$SCRIPT_DIR/t1_$(printf '%03d' "$i").sh"
    T2_SCRIPT="$SCRIPT_DIR/t2_$(printf '%03d' "$i").sh"

    echo "══════════════════════════════════════════════════"
    echo "  Trial $i / $END   |   $SESSION  →  $RUN"
    echo "══════════════════════════════════════════════════"

    # Fresh status file for this trial
    > "$T1_STATUS"

    # ── Write T1 runner script ──────────────────────────────────────────────
    # Absolute paths only — distrobox does not inherit caller's env vars.
    # tee splits the stream: full output goes to the pane (stdout), while
    # grep filters only the two trigger lines into the tiny status file.
    # No full log is written to disk, so /tmp (tmpfs) RAM usage is negligible.
    cat > "$T1_SCRIPT" << EOF
#!/usr/bin/env bash
distrobox enter -r ${DISTROBOX_CONTAINER} -- /entrypoint.sh \\
    ground_truth:=true \\
    start_aic_engine:=true \\
    gazebo_gui:=true \\
    launch_rviz:=false \\
    aic_engine_config_file:=${SESSIONS_DIR}/${SESSION} \\
    2>&1 | tee >(grep --line-buffered -E "aic_engine-[0-9]+.*Retrying|aic_engine-[0-9]+\]: process has finished cleanly|Score:" >> "${T1_STATUS}")
EOF
    chmod +x "$T1_SCRIPT"

    # ── Write T2 runner script ──────────────────────────────────────────────
    # \${AIC_ROOT} and \${EPISODES_DIR} expand in the PANE shell (from ~/.bashrc)
    # ${RUN} expands NOW from this outer script.
    # No log file — full output goes to the pane only.
    cat > "$T2_SCRIPT" << EOF
#!/usr/bin/env bash
if [[ -z "\${AIC_ROOT:-}" ]]; then
    echo "[ERROR] AIC_ROOT is not set. Export it in ~/.bashrc and restart the terminal."
    exit 1
fi
if [[ -z "\${EPISODES_DIR:-}" ]]; then
    echo "[ERROR] EPISODES_DIR is not set. Export it in ~/.bashrc and restart the terminal."
    exit 1
fi
cd "\${AIC_ROOT}" && pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=team_policy.training_robot.cheatcode_collector -p output_dir:="\${EPISODES_DIR}/${RUN}"
EOF
    chmod +x "$T2_SCRIPT"

    # ── Terminal 1: Launch aic_engine ───────────────────────────────────────
    tmux send-keys -t "${TMUX_SESSION}:0.0" "bash ${T1_SCRIPT}" Enter
    echo "$(ts) T1 started. Waiting for aic_engine ready signal (timeout: ${READY_TIMEOUT}s)..."

    if ! wait_for "$T1_STATUS" "$TRIGGER_READY" "$READY_TIMEOUT" "$READY_POLL"; then
        echo "$(ts) [ERROR] Trial $i: aic_engine never became ready after ${READY_TIMEOUT}s."
        echo "$(ts)         Check status file: $T1_STATUS"
        echo "$(ts)         Skipping this trial."
        FAILED_TRIALS+=("$i (aic_engine never ready)")
        tmux send-keys -t "${TMUX_SESSION}:0.0" C-c
        sleep 5
        continue
    fi
    echo "$(ts) aic_engine ready."

    # ── Terminal 2: Launch collector ────────────────────────────────────────
    tmux send-keys -t "${TMUX_SESSION}:0.1" "bash ${T2_SCRIPT}" Enter
    echo "$(ts) T2 started. Collecting... (timeout: ${DONE_TIMEOUT}s)"

    if ! wait_for "$T1_STATUS" "$TRIGGER_DONE" "$DONE_TIMEOUT" "$DONE_POLL"; then
        echo "$(ts) [WARN] Trial $i: did not see clean exit signal within ${DONE_TIMEOUT}s."
        echo "$(ts)        Sending Ctrl+C anyway."
        FAILED_TRIALS+=("$i (timeout — no clean exit signal)")
    else
        echo "$(ts) Clean exit detected."
        log_scores "$T1_STATUS" "$RUN"
    fi

    # ── Shutdown: Ctrl+C → kill lingering → port check ──────────────────────────────
    # Step 1: polite Ctrl+C to both panes
    echo "$(ts) Sending Ctrl+C to T2 (collector)..."
    tmux send-keys -t "${TMUX_SESSION}:0.1" C-c
    sleep 2
    echo "$(ts) Sending Ctrl+C to T1 (aic_engine)..."
    tmux send-keys -t "${TMUX_SESSION}:0.0" C-c
    sleep 3

    # Step 2: kill anything that survived Ctrl+C (distrobox container processes
    # run in a separate namespace and often ignore terminal SIGINT)
    echo "$(ts) Cleaning up lingering processes..."
    kill_lingering

    # Step 3: block until Zenoh releases port 7447, then add a fixed buffer
    wait_for_port
    sleep "$POST_CTRLC_WAIT"

    echo "$(ts) Trial $i done."
    echo ""

done

# =============================================================================
# ── SUMMARY ──────────────────────────────────────────────────────────────────
# =============================================================================

echo "══════════════════════════════════════════════════"
echo "  Collection complete. Trials $START–$END."
echo "  Status files : $STATUS_DIR  (trigger lines only, safe to delete)"

if (( ${#FAILED_TRIALS[@]} > 0 )); then
    echo ""
    echo "  Failed / skipped trials:"
    for entry in "${FAILED_TRIALS[@]}"; do
        echo "    - Trial $entry"
    done
else
    echo "  All trials completed without errors."
fi

echo "══════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  Inspect episodes : ls -lh \$EPISODES_DIR/run_*/"
echo "  Count episodes   : find \$EPISODES_DIR -name 'episode_*.hdf5' | wc -l"
echo "  Merge for training: see README Step 4"
