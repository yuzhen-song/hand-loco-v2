# Faster forward, backward and lateral movement

Fourteen sequential CPU simulations compared 4, 3 and 2-second gait periods.
Sideways trials also compared 16 and 24 mm Cartesian strides. The selected profile
uses a **2-second gait period in all four directions** and a **24 mm lateral stride**.
The forward/backward ABD sweep remains unchanged.

## Measured speed

These speeds use actual free-root displacement over the recorded interval from
3.002 to 11.982 simulated seconds, excluding the initial settling/ramp. Each case
ran for 12 seconds. Playback speed is unrelated to these measurements.

| Direction | Original | Selected | Increase |
| --- | ---: | ---: | ---: |
| Forward | 9.36 mm/s | 18.80 mm/s | 2.01x |
| Backward | 9.89 mm/s | 19.00 mm/s | 1.92x |
| Left | 6.55 mm/s | 21.96 mm/s | 3.35x |
| Right | 8.21 mm/s | 21.29 mm/s | 2.59x |

The assumed torque ceiling remains **0.30 Nm**; joint command limits remain
**2 rad/s and 8 rad/s²**. Command/physics rates, actuator gains and contact settings
were not raised. The selected forward/backward cases have approximately 6.4-degree
maximum tracking error, versus approximately 3.2 degrees in the slower baseline.
This is a measurable cost of the increased speed, not an exact-tracking result.

Selection requires the existing task screen to pass, maximum tracking error below
8 degrees, maximum tilt below 10 degrees, and post-ramp cross speed below 35% of
requested-axis speed. The fastest eligible candidate is selected per direction.
All 14 task trials passed their original narrow task screen; the original leftward
baseline misses the additional 35% cross-speed criterion. All trials, including
that selection failure, are preserved in
[`validation/directional_speed_v1.json`](../validation/directional_speed_v1.json).

The selected configuration is
[`configs/directional_speed.json`](../configs/directional_speed.json). Its measured
speeds are evidence from these trials, not guaranteed performance targets.

## Continuous sequence check

The selected profile also completed the full 43-second dance without any root
reset, non-tip contact stop or tilt stop:

| Eight-second walking section, including transitions | Original | Faster |
| --- | ---: | ---: |
| Backward | 64.9 mm | 106.5 mm |
| Left | 45.0 mm | 120.0 mm |
| Right | 45.6 mm | 118.9 mm |
| Forward | 65.4 mm | 103.0 mm |

These section ratios differ from post-ramp speeds because the section includes
settling and fade-out. Maximum tilt was 5.62 degrees; maximum tracking error was
6.49 degrees. Peak command velocity was 1.83 rad/s, with acceleration bounded at
8 rad/s². The final horizontal offset was approximately 23.9 mm, versus 5.9 mm in
the original dance. Faster movement therefore does not imply better return accuracy.
See [`validation/dance_fast_v1.json`](../validation/dance_fast_v1.json).

This remains a deterministic simulation with uncalibrated actuator response and
self-collision disabled. It is not a validated hardware speed setting.

## Run

Run all commands from the repository root in the existing environment.

```bash
# Reproduce the complete sequential comparison; choose a fresh output directory.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/speed_sweep.py \
  --output outputs/experiments/speed/my_sweep

# The dance CLI now loads configs/directional_speed.json by default.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/dance.py \
  --output outputs/experiments/dance/my_fast_run
.venv/bin/python -B tools/render_dance.py \
  --run outputs/experiments/dance/my_fast_run

# Retain the original slower version explicitly.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/dance.py --baseline \
  --output outputs/experiments/dance/my_slow_run
```

A sweep writes `summary.json` and `selected_profile.json`, but never overwrites the
committed configuration automatically. To test another sweep's selection, pass
`--profile outputs/experiments/speed/my_sweep/selected_profile.json` to `dance.py`.
The standalone `locomotion_tasks.py` defaults remain unchanged so previous cases
retain their parameters. `sim2real_candidate.json` also remains the slow hardware
preparation candidate; the faster dance is a separate simulation profile.

The completed faster video is
`outputs/experiments/dance/fast_v1/dance.mp4`: normal-speed CPU replay, with the
measured XY path. Rendering uses OSMesa/llvmpipe and verifies full MP4 decoding.

## Validation

All 30 tests passed, including speed measurement after ramp-up, rejection of faster
but ineligible candidates, application of each directional setting, unchanged torque
assumption, and continuity of both baseline and fast dance transitions.
