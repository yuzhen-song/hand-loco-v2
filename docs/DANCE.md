# Four fingers / one dance

The CLI now uses the faster profile described in [SPEED.md](SPEED.md). The recorded
results below describe the original baseline; pass `--baseline` to reproduce it.

A 43-second continuous choreography in MuJoCo: a small body dip, backward shuffle,
left slide, right slide, forward shuffle, another dip, and a standing finish.
The robot uses one free-base dynamics rollout. There are **no root resets, imposed
body motion, or stitched independent rollouts** between moves.

The simulation completed the full sequence with a maximum body tilt of 5.66 degrees.
The recorded section displacements were:

| Move | Measured movement |
| --- | ---: |
| Backward shuffle | 64.9 mm backward |
| Slide left | 45.0 mm left |
| Slide right | 45.6 mm right |
| Forward shuffle | 65.4 mm forward |
| Opening / closing dip | approximately 3.8 mm COM height range each |

The final horizontal offset from the initial position was approximately 5.9 mm.
This near-return happened in this deterministic trial; it is not an implemented
closed-loop return-to-start controller. These are small, slow physical movements,
not a jumping or acrobatic demonstration.

## Control and evidence

- The established steering gait handles forward/backward motion. Cartesian
  fingertip IK handles sideways motion and the small dip.
- Walking segments fade to neutral before changing direction. One continuous
  position governor retains its state across all transitions.
- Commands run at 50 Hz with 2 rad/s velocity and 8 rad/s² acceleration limits;
  physics and the simulated local servo run at 500 Hz.
- The assumed torque ceiling is 0.30 Nm per motor, with the same uncalibrated
  simulated PD response as the earlier locomotion experiments.
- The maximum command speed was 0.990 rad/s, maximum tracking error 3.59 degrees,
  and maximum IK residual 0.099 mm. No non-tip floor contact or tilt stop occurred.
- [`validation/dance_v1.json`](../validation/dance_v1.json) records each section,
  command limits, physical metrics, source hashes and limitations.

The dance screen checks full sequence completion, positive progress in the requested
axis during every walking section, bounded commands, IK residual below 2 mm,
maximum tilt below 20 degrees, and absence of the contact stop. It does not certify
precise choreography, repeatability under perturbation or real-robot deployment.

Self-collision remains disabled, and terrain contacts use CAD convex hull geometry.
Supply voltage, encoder calibration, mechanical range and actuator dynamics still
need physical measurements. This adds no hardware motion executor.

## Run and render on CPU

Use the existing project environment; see [SIM2REAL.md](SIM2REAL.md).
Choose a new output directory for each run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/dance.py --baseline --output outputs/experiments/dance/my_baseline_run

.venv/bin/python -B tools/render_dance.py --run outputs/experiments/dance/my_baseline_run
```

The first command writes `metrics.json`, `interface.json`, `trace.json`, and
`states.json`. The second uses OSMesa/llvmpipe on the CPU, verifies the recorded
source hashes, and renders `dance.mp4` at **1x speed**, 10 fps, 576 x 384 pixels.
It labels each move and draws the measured free-root XY path. The checkerboard
floor and colored fingertips are visual styling only. No music is included.

The final pose is held for one video frame, making the full clip approximately
43.1 seconds. `video_validation.json` records decoded frame count and input hashes.
The original completed development run is in `outputs/experiments/dance/baseline_v1/`; generated videos
and raw logs remain local and are ignored by Git.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B -m unittest discover -s tests -v
```

At the baseline revision, all 26 tests passed. The current suite has 30 tests. The new tests check segment boundaries, continuous commands
at every transition, and that planning does not modify the rollout's state.
