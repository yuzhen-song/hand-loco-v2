# XC330 position interface

The user confirmed XC motors, original IDs 0–15, original per-finger ordering,
and the original current-based position mode and host update rates. The model
variant is XC330-M288-T, matching the current CAD mass description.

## Confirmed interface convention

| Finger | DIP | PIP | MCP-ABD | MCP-FLX |
|---|---|---|---|---|
| A | 0 | 1 | 2 | 3 |
| B | 4 | 5 | 6 | 7 |
| C | 8 | 9 | 10 | 11 |
| D | 12 | 13 | 14 | 15 |

`configs/xc330_hand.json` records model number 1240, DYNAMIXEL Protocol 2,
1 Mbps reference baud, operating mode 5, commands every 20 ms, positions every
50 ms, and diagnostic feedback every 200 ms. These are requested configuration
values, not values read back from a connected device.

The communication layout and timing were inspected in the separately extracted
`wimsu/msc-in-hand-assembly` reference repository (`hardware/platform/joint_names.py`,
`hardware/platform/platform_io.py`, `calibration/control_timing.yaml`). Its driver
checks model 1200 / XL330. That model check is not applicable to this XC assembly.
The reference repository has not been modified, imported, or installed as a
runtime dependency. The new simulation interface is implemented locally.

The original reference Goal Current setting is 350 mA; PID register defaults
are P=600, I=0, D=200, with ABD P/D multiplied by 0.75. Profile settings are
2 rad/s and 8 rad/s². They are recorded as reference defaults, not readbacks or
validated electrical/thermal settings for this assembly. They are not used as
MuJoCo PD gains or as a claim to reproduce the servo firmware's profile behavior.

Official model, mode and register units:
https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/

## CPU simulation

The historical baseline remains available as the default. Select the new
position interface explicitly:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tools/walk_x.py 10 \
  --controller position --sim-torque-limit-nm 0.95 \
  --report outputs/sim2real/interface_check_new/metrics.json
```

This example deliberately uses the old model's 0.95 Nm simulation ceiling for
interface comparison. It does NOT infer that 350 mA yields 0.95 Nm, that this is a
continuous rating, or that the hardware can execute the gait. The CLI requires
an explicit simulation ceiling instead of inventing a current-to-torque conversion.

The planner's joint-position request is reordered into motor-ID order and held
between 50 Hz updates. The simulated local position servo observes physics at
500 Hz; host position feedback is sampled at 20 Hz and diagnostics at 5 Hz.
The local plant still uses the baseline approximate PD gains (400, 14).
The host's 20 Hz telemetry does not slow the motor's local feedback loop.
The original gait timing is retained; no gait-speed or profile emulation is added.

The report records sampled requested/actual joint positions, simulated velocity
and torque, tracking error and saturation. All joint angles remain in native
model coordinates even when reordered by motor ID. They are NOT absolute
encoder commands. The diagnostic torque is not presented as measured current.

## Validation and remaining work

Seven CPU unit checks cover individual motor mapping, command holding, all
three sample rates, local servo response between host samples, and invalid
inputs. A 10-second CPU dynamics run completed without rendering or GPU use.
The run produced 500 command updates, 200 position samples and 50 diagnostic
samples. It travelled approximately +307.1 mm under the baseline 0.95 Nm plant
assumption. This validates the software integration, not physical deployment.
The maximum requested-position tracking error was 0.1856 rad. Approximately
65.5% of joint/physics-step samples hit the assumed torque ceiling, which is
a material limitation of the baseline plant and requires actuator/load validation.
Raw evidence is in `outputs/sim2real/interface_check/metrics.json` (ignored by Git).

Before adding a physical output backend, measure/confirm native-model-to-encoder
axis directions, zero offsets, actual supply voltage and usable torque/current
behavior, and validate the physical ROM against the gait. Old calibration values
are not copied because the current CAD coordinate conventions differ. The config
leaves these calibration fields unset. No serial opening, motor enable, firmware
flash, or physical motion exists in this interface.

The baseline still uses fingertip-only collision and unusually high friction.
Whole-body collision, realistic contact, motor response and load/thermal limits
remain to be validated. The modeled mass remains 0.785912524 kg; omission of the
16 small adaptors and one screw is the user's accepted approximation.

## Subsequent sim-to-real workflow

[SIM2REAL.md](SIM2REAL.md) documents the newer rate-limited trajectory generator,
whole-assembly floor collision, lower-torque screening, read-only hardware audit
and offline encoder conversion. The results above remain the historical
position-interface baseline; they are not relabeled as results of the new scene.
