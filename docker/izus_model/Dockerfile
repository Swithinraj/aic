FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | bash
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

# 4. Restore the original pixi.toml and install only the local code
COPY pixi.toml /ws_aic/src/aic/
RUN --mount=type=cache,target=/root/.cache/rattler/cache \
    cd /ws_aic/src/aic && pixi install \
 && rm -rf /ws_aic/src/aic/.pixi/bld /ws_aic/src/aic/.pixi/artifacts-v0 \
 && find /ws_aic/src/aic/.pixi/envs -type d \( -name __pycache__ -o -name tests -o -name test \) -prune -exec rm -rf {} + \
 && find /ws_aic/src/aic/.pixi/envs -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

WORKDIR /ws_aic/src/aic
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

exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=izus_policy.run_hybrid", "-p", "use_sim_time:=true"]
