### To test the end-to-end pipeline

- Open one terminal for simulation:

```bash
    distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true
```

- Open another terminal for perception and policy:

```bash
    pixi run ros2 run aic_model aic_model --ros-args   -p policy:=izus_policy.run_hybrid
```
