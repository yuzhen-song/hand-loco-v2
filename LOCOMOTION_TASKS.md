# Direction, terrain and jump experiments

This CPU-only extension tests the four-finger CAD robot in **free-base MuJoCo dynamics**.
It does not use Wuji Hand, G1, a learned policy, or a live hardware interface.
Every result below is one deterministic simulation trial, not a real-robot capability claim.

## Results

Fifteen trials were completed in order: the 12-case initial suite, then three targeted
follow-ups. All trials reached their requested duration without a non-fingertip terrain
contact or the 45-degree emergency tilt stop. Completing a run does **not** imply task success.
Machine-readable parameters and results, including failures, are in
[`validation/locomotion_tasks_v1.json`](validation/locomotion_tasks_v1.json).

| Task | Time | Progress along requested direction | Result of narrow screen |
| --- | ---: | ---: | --- |
| Backward, -X | 12 s | 98.7 mm | Pass; 3.1 mm cross drift |
| Left, +Y | 12 s | 67.5 mm | Pass; 22.8 mm cross drift |
| Right, -Y | 12 s | 81.4 mm | Pass; 21.6 mm cross drift |
| Uphill 3 degrees, 4 s gait period | 12 s | 77.4 mm | Pass |
| Uphill 5 degrees, 4 s gait period | 12 s | 60.0 mm | Pass |
| Uphill 10 degrees, 4 s gait period | 12 s | 13.1 mm | Fail: insufficient progress |
| Single 5 mm step, Cartesian planner | 20 s | 43.8 mm | Fail: not all four feet crossed; IK error |
| Single 10 mm step, Cartesian planner | 20 s | 46.6 mm | Fail: not all four feet crossed; IK error |
| Single 20 mm step, Cartesian planner | 20 s | 49.6 mm | Fail: not all four feet crossed; IK error |
| Jump, assumed 0.30 Nm | 6 s | — | Fail: no qualifying flight |
| Jump, assumed 0.50 Nm, deeper crouch | 6 s | — | Fail: no qualifying flight; IK error |
| Jump, assumed 0.95 Nm, deeper crouch | 6 s | — | Fail: no qualifying flight; IK error |
| Single 5 mm step, steering-gait height correction | 24 s | 47.6 mm | Fail: not all four feet crossed; IK error |
| Uphill 10 degrees, 3 s gait period | 12 s | 27.1 mm | Pass, but slow and marginal |
| Jump, 0.50 Nm, 0.10 s requested push | 6 s | — | Fail: no qualifying flight |

Walking and step cases use an assumed 0.30 Nm torque limit. Jump cases vary the
assumed limit; **0.95 Nm is a historical simulation ceiling, not a verified XC330
continuous rating**. No hardware current-to-torque conversion is inferred.
Left and right are world-frame directions relative to the initial pose. The existing
forward gait remains available in `tools/sim2real.py`.

### What worked, and what remains unresolved

- Reversing the established steering gait produced backward travel. Cartesian
  fingertip trajectories produced sideways travel, with substantial perpendicular
  drift still to correct. These are timed gaits, not velocity-tracking controllers.
- The slope tests **start already on a uniformly inclined plane**, aligned with it.
  They do not test entering a ramp from level ground. Increasing gait frequency
  improved the tested 10-degree case, without increasing its torque ceiling.
- The step tests are **single low risers**, not staircases. The initial planner
  uses perfect simulated root pose and a known height map. Ground-anchored targets
  and the changing body pose led to large IK residuals (about 92–101 mm), so a
  completed run cannot be presented as a successful climb. The follow-up retains
  the steering gait and solves height corrections in a nominal body frame; its
  maximum residual fell to about 25.5 mm, but it still did not climb the step.
  Body-height/support planning and reachable footholds need further work; increasing
  the simulation duration alone is not a demonstrated fix.
- The synchronized crouch/extension commands did not create a qualifying jump.
  The longest no-contact interval was only 2–4 ms in the initial jump tests.
  This is contact flicker, not convincing flight. Larger requested extension also
  encountered about 3.7 mm IK error. Shortening the requested push did not overcome
  the bounded servo response. This does **not** establish that the hardware cannot
  jump: torque-speed behavior, reflected inertia, supply voltage and real limits
  have not been measured. Do not raise limits based solely on these tests.

## Implementation

`tools/locomotion_tasks.py` provides three planners:

- `abd`: the existing bounded steering gait, with sign reversal for backward travel.
- `cartesian`: bounded four-joint Jacobian IK for fingertip trajectories. Flat
  sideways walking interpolates a precomputed phase table. Step trials use the
  actual simulated root pose and the terrain height map as oracle inputs.
- `terrain_abd`: an experimental step follow-up that retains the steering gait
  and adds IK height corrections relative to a nominal body pose. The terrain
  lookup still uses perfect simulated forward position. It is not a successful
  step controller yet.

Jump trials use synchronized fingertip height targets for crouch, extension and
recovery. The same position governor remains active for every task: 50 Hz commands,
2 rad/s velocity limit, 8 rad/s² acceleration limit, and a 2 ms simulated servo/physics
step. Command limits describe this experiment, not measured actuator maxima.

The root is initialized once, then moves through physics alone. IK uses a separate
MuJoCo data object; it never teleports the simulated robot. Terrain contacts include
all assembly collision parts, with visual duplicates disabled before compilation.
Robot self-collision remains disabled, and contact meshes are CAD convex hulls.
The original baseline model XML is not overwritten.

### Screening criteria

All passing tasks require finite dynamics, completion, bounded emitted commands,
no non-tip terrain contacts, and no IK solve residual above 2 mm.
Walking additionally requires more than 20 mm directional progress, less than
`max(20 mm, 0.5 * progress)` cross drift, and less than 20 degrees maximum tilt
relative to the terrain. This is a deliberately narrow exploratory screen; it does
not certify precise tracking, disturbance rejection, or repeatability.

Steps additionally require every fingertip mesh to be fully beyond the edge by
2 mm, above the top within 2 mm, at least one fingertip contacting the step, and
that condition maintained for 0.2 s. Forward body displacement alone is insufficient.

A jump event needs positive takeoff COM vertical velocity above 0.02 m/s, at least
40 ms without terrain contact, at least 3 mm COM rise after takeoff, at least 2 mm
simultaneous minimum fingertip clearance, and subsequent landing. Initial falling
and brief contact loss do not count. Recovery checks final-second tilt below
20 degrees, not a fully qualified landing controller.

## Reproduce on CPU

Use the environment described in [REPRODUCTION.md](REPRODUCTION.md) and
[SIM2REAL.md](SIM2REAL.md). No additional simulation dependencies or GPU are needed.
Run commands from the repository root. Every output directory must be new.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/locomotion_tasks.py \
  --suite --output outputs/locomotion_tasks_my_suite
```

The suite runs backward, left, right, 3/5/10-degree slopes, 5/10/20-mm steps, then
three jump hypotheses **sequentially**. Each case writes:

- `metrics.json`: task parameters, source hashes, success checks and failure evidence;
- `interface.json`: emitted commands and simulated feedback;
- `trace.json`: body motion, support and contact observations;
- `states.json`: recorded joint and free-root poses for visualization.

`summary.json` is written when the suite completes. Full local logs and videos
remain in ignored `outputs/`; the compact tracked validation includes all 15 cases
and hashes of their complete metrics. Historical source hashes distinguish the
initial implementation from the follow-up planner revision.

To reproduce a particular recorded case, extract its parameters into a new JSON
file. For example, the successful faster 10-degree slope follow-up:

```bash
.venv/bin/python -B - <<'PY'
import json
from pathlib import Path
results = json.loads(Path('validation/locomotion_tasks_v1.json').read_text())['results']
case = next(r['task'] for r in results if r['task']['name'] == 'slope_10deg_faster')
Path('outputs').mkdir(exist_ok=True)
Path('outputs/slope_case.json').write_text(json.dumps(case, indent=2) + '\n')
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/locomotion_tasks.py \
  --case outputs/slope_case.json --output outputs/slope_my_retry
```

Use `step_5mm_terrain_abd` or `jump_short_push` in the extraction command to
reproduce the other follow-ups. Preserving these failures is intentional.

### CPU video replay

```bash
.venv/bin/python -B tools/replay_locomotion_tasks.py \
  --runs outputs/locomotion_tasks_my_suite/01_reverse \
         outputs/locomotion_tasks_my_suite/02_left \
         outputs/locomotion_tasks_my_suite/03_right \
         outputs/slope_my_retry \
  --output outputs/locomotion_review.mp4
```

The renderer forces OSMesa/llvmpipe, checks the renderer identity, limits CPU threads,
and decodes the completed MP4 to verify the frame count. It replays actual recorded
states at approximately **2x speed**, with task name, simulation time, torque
assumption and final screen result. Loading recorded root poses is visualization
only; the original rollouts used free dynamics. A JSON manifest records input
hashes and clip frame counts. Source changes to the shared model builders are
rejected to avoid silently replaying with a different geometry pipeline.

The development run produced `outputs/locomotion_tasks_review/all_attempts.mp4`,
including every initial and follow-up attempt, with successful and failed cases
clearly labeled. Videos are generated artifacts and are not committed to Git.

### Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B -m unittest discover -s tests -v
```

All 24 tests passed, including slope geometry, step dimensions, sideways trajectory
direction, IK isolation from actual dynamics, jump false-positive rejection, existing
command limits, calibration checks and collision-mask regression checks.

## Before sim-to-real

The current hand still needs measured supply voltage, encoder zeros/directions,
mechanical limits and actuator response. Hardware motor IDs/order and the existing
read-only audit remain unchanged. These new task logs are **not** accepted as ready
hardware plans by the existing exporter: they use their own task-specific screen,
and no live walking or jumping executor has been added.

Prioritize closed-loop directional control and gentle slopes after flat walking is
physically validated. Step climbing needs reachable footholds and body/support
planning. Jumping additionally needs experimentally established actuator dynamics
and landing behavior. See [SIM2REAL.md](SIM2REAL.md) for the existing handoff workflow.
