# Sim-to-real preparation

This project now shares bounded, timestamped position commands between simulation
and offline encoder conversion. **Physical walking has not been validated, and
there is no live motion executor.** The only physical-device utility is an
explicitly selected, read-only audit. Supply voltage, encoder calibration,
mechanical ROM and usable actuator torque remain unmeasured on this assembly.

## What changed

- Preserve `tools/walk_x.py` as the historical baseline.
- `tools/sim2real.py` uses a 4-second gait cycle, half the old ABD sweep
  (±0.6 rad), a 0.08 rad lift, one second of settling and a two-second ramp.
- `tools/trajectory.py` bounds emitted host commands to 2 rad/s and 8 rad/s²
  at 50 Hz. This is host trajectory shaping, not undocumented servo firmware
  emulation. Position feedback remains 20 Hz, diagnostics 5 Hz.
- Simulation PD gains are 60 and 3, replacing the old benchmark's 400 and 14
  for the new experiments only. These are not the motor PID register values.
- The candidate assumes 0.30 Nm available torque; sensitivity runs use 0.18,
  0.30 and 0.50 Nm, plus a 0.95 Nm comparison. None is a conversion from 350 mA.
- All CAD collision components can contact the floor. Visual duplicates are
  disabled. Masks are written **before model compilation**, so MuJoCo rebuilds
  body collision masks and bounding-volume structures. A test lowers the assembly
  into the floor and verifies non-tip contacts are actually detected.
- Friction is tested at 0.4, 0.8 and 1.2; contact time constants at 0.004 and
  0.008 s. These are exploratory assumptions, not measured surface properties.
- Stop and record a simulation if sampled tilt exceeds 45 degrees or a non-tip
  component touches the floor. Screening samples contact/tilt at 50 Hz.

The original mass model (0.785912524 kg) and the user's accepted omission of
16 light adaptors and one screw are retained. Self-collision is **not** certified:
CAD convex hulls are unsuitable for claiming exact assembly clearance, and
robot-to-robot collisions remain disabled. Only whole-assembly versus floor
contact is enabled in the new scene.

## CPU reproduction

Use the project environment from `REPRODUCTION.md` and `requirements.txt`.
No GPU, training or GPT API is required. Choose fresh output directories.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/sim2real.py --sweep \
  --output outputs/sim2real/screen_new

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/sim2real.py \
  --case configs/sim2real_candidate.json --render \
  --output outputs/sim2real/video_new

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover -s tests -v
```

Rendering defaults to OSMesa/llvmpipe with two rendering threads. MuJoCo runs at
500 Hz. The native position command is reordered by motor ID without changing
its coordinate signs; only a measured encoder calibration can do that conversion.

Each run saves `metrics.json`, `interface.json`, `body_trace.json`, and optionally
`walk.mp4`. Metrics capture input/source hashes and explicitly assumed parameters.
The interface log records **every emitted 50 Hz command**, not just 20 Hz feedback.
Reports include rate bounds, tracking error, torque saturation, sampled tilt,
floor penetration and the observed number of contacting fingertips.

## Observed results

The compact evidence is committed in `validation/sim2real_screen_v2.json`.
All 11 runs completed 10 simulated seconds and passed the narrow exploratory
screen. The screen requires forward displacement >10 mm, sampled tilt <20°,
no sampled non-tip floor contact and command-rate compliance. It is not a
physical deployment or quasi-static stability criterion.

| Configuration | Travel in 10 s | Final yaw | Max tracking error | Joint-step torque saturation |
|---|---:|---:|---:|---:|
| Original position baseline, 0.95 Nm | 307.1 mm | −8.97° | 10.63° | 65.54% |
| Slow trot, assumed 0.18 Nm | 76.8 mm | −0.73° | 3.35° | 13.09% |
| Selected slow trot, assumed 0.30 Nm | 75.9 mm | −0.60° | 3.24° | 4.61% |
| Slow trot, assumed 0.50 Nm | 76.3 mm | −0.56° | 3.17° | 2.10% |
| Crawl timing, assumed 0.30 Nm | 56.2 mm | −3.91° | 6.14° | 4.52% |

Several parameters changed together; this is a candidate comparison, not a
causal attribution of the improvement to one parameter. The selected candidate's
peak emitted speed is 0.888 rad/s and peak acceleration is 8 rad/s². Its largest
sampled tilt is 3.51°. The contact solver's deepest sampled contact distance is
about −0.049 mm; this is not the same metric as the old video's minimum mesh
vertex height and must not be compared directly to that value.

The crawl scheduler requests three stance phases, but measured contacts vary:
three fingers do not remain continuously loaded. Neither gait is established as
quasi-static. The selected trot is an improved simulation candidate, not a claim
that a real 350 mA servo setting will reproduce it. Armature, damping, servo
response, contact behavior and payload distribution still need physical matching.

## Obtain the missing hardware data

The user reports unchanged control boards/power setup after replacing motors,
but does not know the supply voltage. Do not infer 5 V from that statement.
The reference XL driver uses model 1200; this project expects XC330-M288-T / 1240.

On the **computer physically connected to the hand**, optionally install the
hardware audit dependencies in its own environment. They are separate from
simulation requirements and have not been installed on the simulation server:

```bash
python -m pip install -r requirements-hardware.txt
python tools/audit_hardware.py --port YOUR_SERIAL_DEVICE \
  --output outputs/sim2real/hardware_audit.json
```

Select the actual device explicitly (for example a Windows COM port or a Linux
serial-by-id path), with other serial clients stopped. The tool only reads
registers: model/IDs, voltage, mode, Drive Mode, current limits, gains, profile
settings, position, current, temperature and device faults. It never enables
motors, sends target positions or changes device registers. Wrong model or
communication/device errors stop the audit. The snapshot is not a live readiness
monitor and cannot authorize later movement. Mock-device tests cover this path;
actual SDK/device integration remains untested.

Use `configs/hardware_calibration.example.json` as a measurement worksheet. Save
machine-specific results as `configs/hardware_calibration.local.json` (ignored).
For each motor ID, measure:

- Which native model joint it actuates and the direction mapping (+1 or −1).
- Absolute encoder angle at that joint's **native model zero**, not an assumed
  generic straight-finger zero.
- Lower and upper mechanical angles in that same native coordinate system.
- The Drive Mode register used during calibration (reversing it invalidates signs).

Also record measured supply voltage and a measurement reference. Old reference
zero offsets and limits are not imported. The current gait exceeds some native
CAD-export limits; the historical simulator overrides are not proof that the
physical hand permits those poses. Resolve this with actual mechanical evidence.

## Offline encoder conversion

```bash
python tools/prepare_hardware.py \
  --run outputs/sim2real/screen_new/trot_torque_0.3 \
  --calibration configs/hardware_calibration.local.json \
  --output outputs/sim2real/encoder_candidate.json
```

The shipped unmeasured template is deliberately rejected. With complete measured
calibration, the utility checks names/order, command timing, velocity/acceleration
and mechanical ROM, then converts native radians to signed encoder pulses. It
rejects quantization that crosses a mechanical limit and reports the rate bounds
of the quantized sequence separately. Quantization may add discrete acceleration;
if its rate check fails, reshape and replay the quantized sequence before hardware
use. It never marks an offline file hardware-ready and never opens a serial port.

Before a live motion backend is added, remaining work includes measured actuator
response/current/torque and supply behavior, reliable collision geometry, a
supported initial-pose procedure, fresh device-state checks, and verified stopping
behavior on stale feedback, tracking error or device faults. Those depend on the
actual assembly and cannot be supplied honestly by inventing values in JSON.

Official references:
- https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/
- https://github.com/ROBOTIS-GIT/DynamixelSDK
