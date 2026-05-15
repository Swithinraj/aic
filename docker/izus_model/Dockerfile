FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi and supervisor (supervisor runs YOLO + aic_model as separate
# processes inside this container, mirroring the 3-terminal local workflow).
RUN apt update && apt install -y git supervisor && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# 1. Copy only configuration files to cache external dependencies
COPY pixi.toml robostack_mapping.json /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

# 2. Strip local paths and install heavy ML/ROS libraries (This layer caches the 10GB payload!)
SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache \
    sed -i '/{ path = /d' /ws_aic/src/aic/pixi.toml && \
    cd /ws_aic/src/aic && pixi install

# 3. Copy the local packages (If code changes, the build resumes from here)
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY izus_policy /ws_aic/src/aic/izus_policy

# Fail the build immediately if the YOLO weights are missing. The v1 image
# shipped on 2026-05-14 contained 'yolov12.pt' but the detector only looks
# for 'yolov12_obb.pt' / 'yolo_v26_obb.pt'; the server-side eval crashed at
# YOLO node init. Better to break the build than to discover this in eval.
RUN test -f /ws_aic/src/aic/izus_policy/izus_policy/models/yolov12_obb.pt \
 || test -f /ws_aic/src/aic/izus_policy/izus_policy/models/yolo_v26_obb.pt \
 || (echo "FATAL: no YOLO weights (yolov12_obb.pt or yolo_v26_obb.pt) under izus_policy/izus_policy/models/" \
     && ls -la /ws_aic/src/aic/izus_policy/izus_policy/models/ \
     && exit 1)

# 4. Restore the original pixi.toml and install only the local code
COPY pixi.toml /ws_aic/src/aic/
RUN --mount=type=cache,target=/root/.cache/rattler/cache \
    cd /ws_aic/src/aic && pixi install \
 && rm -rf /ws_aic/src/aic/.pixi/bld /ws_aic/src/aic/.pixi/artifacts-v0 \
 && find /ws_aic/src/aic/.pixi/envs -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + \
 && /ws_aic/src/aic/.pixi/envs/default/bin/python -m compileall -q -j 0 \
      /ws_aic/src/aic/.pixi/envs/default/lib \
      /ws_aic/src/aic/izus_policy 2>/dev/null || true

# 5. Pre-bake heavy imports & verify YOLO weights at build time.
# Per the forum guidance: heavy imports + weight parsing happen on the build
# host (no GPU here) so any broken dependency or corrupted weight file fails
# the build, never the eval. The .pyc cache produced here also speeds up the
# first runtime import. GPU warmup itself still has to happen at runtime.
RUN --mount=type=cache,target=/root/.cache/rattler/cache \
    cd /ws_aic/src/aic && pixi run --as-is python - <<'PY'
import importlib, sys, pathlib
print("[prebake] importing heavy modules ...", flush=True)
for mod in ("numpy", "cv2", "torch", "ultralytics", "transformers",
            "transforms3d", "scipy", "rclpy"):
    importlib.import_module(mod)
    print(f"[prebake]   ok: {mod}", flush=True)

# Find and CPU-load the YOLO weights to verify they parse cleanly.
import torch
from ultralytics import YOLO
candidates = [
    pathlib.Path("/ws_aic/src/aic/izus_policy/izus_policy/models/yolov12_obb.pt"),
    pathlib.Path("/ws_aic/src/aic/izus_policy/izus_policy/models/yolo_v26_obb.pt"),
]
weight = next((p for p in candidates if p.is_file()), None)
if weight is None:
    print("[prebake] FATAL: no yolov12_obb.pt or yolo_v26_obb.pt found", flush=True)
    sys.exit(1)
print(f"[prebake] loading YOLO weights on CPU: {weight}", flush=True)
model = YOLO(str(weight))
model.to("cpu")
n_params = sum(p.numel() for p in model.model.parameters())
print(f"[prebake] YOLO loaded, params={n_params}", flush=True)
print("[prebake] done", flush=True)
PY

WORKDIR /ws_aic/src/aic
COPY docker/izus_model/supervisord.conf /etc/aic/supervisord.conf
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

# prepare the credentials file
if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# start the model
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

exec /usr/bin/supervisord -c /etc/aic/supervisord.conf
EOF

ENTRYPOINT ["/entrypoint.sh"]
