# Tool index

Run these from the repository root with the project Python environment. Code paths
remain stable to preserve imports and model/config resolution.

| Role | Entry points / modules |
| --- | --- |
| Faster directional movement | `speed_sweep.py` |
| Continuous choreography | `dance.py` (fast by default; `--baseline` for slow) |
| Direction/terrain/jump experiments | `locomotion_tasks.py` |
| CPU video rendering | `render_dance.py`, `replay_locomotion_tasks.py` |
| Sim-to-real screening | `sim2real.py` |
| Read-only physical device audit | `audit_hardware.py` (explicit port required) |
| Offline encoder conversion | `prepare_hardware.py` |
| Shared sampled control and trajectory shaping | `hand_interface.py`, `trajectory.py` |
| CAD model compilation / standing poses | `build_model.py`, `stance.py` |
| Historical walking reference | `walk_x.py`, `walk.py` |

The simulation scripts do not send hardware commands. There is no live motion
executor. See the [documentation index](../docs/README.md) for exact reproduction
commands, evidence and limitations.
