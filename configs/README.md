# Configuration index

| File | Purpose |
| --- | --- |
| `directional_speed.json` | Selected faster simulation-only profile for four directions; used by the dance CLI |
| `sim2real_candidate.json` | Original slower candidate for sim-to-real preparation; unchanged |
| `xc330_hand.json` | XC330 IDs, joint ordering, timing and uncalibrated hardware metadata |
| `hardware_calibration.example.json` | Incomplete measurement worksheet; intentionally rejected as live calibration |

Store machine-specific measurements in `*.local.json`, which Git ignores. Faster
simulation parameters are not validated motor settings. See
[Speed experiments](../docs/SPEED.md) and [Sim-to-real](../docs/SIM2REAL.md).
