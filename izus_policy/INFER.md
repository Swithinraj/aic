### To test the end-to-end pipeline

- Open one terminal for simulation:

```bash
    distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true
```

- Open another terminal for perception:

```bash
    pixi run ros2 run izus_policy test_yolov12_perception
```

- Open a third terminal for the controller:

```bash
    pixi run ros2 run aic_model aic_model --ros-args   -p policy:=izus_policy.run_hybrid
```
