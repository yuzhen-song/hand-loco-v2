# Camera-free final assembly: partial mass assignment

`robot.urdf` is a new model derived from the CameraV2 export on the local
`locomotion` branch at `4013393`. The original CAD export is preserved.
The camera and camera mount were removed from both visual and collision geometry.
All retained joint frames, limits and geometry follow the native CameraV2 export.

This model currently assigns **0.785912524 kg**, excluding the 17 unspecified
component occurrences (16 servo adaptors and one mounting screw). These retain
their geometry and contribute zero mass until their masses are supplied.
This is a partial mass model, not a measured whole-assembly mass.

Mass assumptions are recorded in `../final_assembly_masses.json`:

- Each of the 16 XC330-M288-T servos contributes 23 g over its complete CAD
  assembly, including the dummy case, caps, horns and connector parts.
- Solid hinge mass and steel rail/block masses use the supplied values in full.
  Two guide blocks are represented by five STL components each; their supplied
  mass is assigned once per complete block, not once per STL.
- Other supplied printed masses use 15% of the supplied solid mass. This is a
  uniform-density approximation to 15% infill; shell/perimeter mass is not added.
- The complete Actuonix assembly contributes 40 g, shared by volume across
  `clamshell_bot_10417`, `clamshell_top_10416`, `extension_10404_50mm`,
  `guide_10429`, `plug_10428`, and `shaft_10409_50mm`. This includes both the
  stationary housing and moving actuator components.

Centers of mass and full inertia tensors are integrated from closed STL solids
assuming uniform volume density, transformed into each URDF link frame, and
combined using the parallel-axis theorem. Collision meshes are not counted as
additional mass. Uniform density across each assembly is an approximation to
the actual internal distribution of motors and purchased parts.

`mass_report.json` contains the source hashes, per-occurrence assignments,
unassigned parts, assembly grouping and per-link inertial properties.

Regenerate from the unmodified CameraV2 export:

```powershell
uv run --with numpy --with scipy --with trimesh python sim/assign_final_assembly_masses.py --source outputs/camera-v2/robot.urdf
uv run --with numpy --with scipy --with trimesh python -m unittest discover -s sim/tests -p test_final_assembly_masses.py
```

For simulation integration, retain the CameraV2 canonical-coordinate adapter:
the native rail endpoint is -0.05 m and several native joint axes differ from
the calibrated hardware convention. Use the URDF inertials; do not copy the old
LEAP approximate inertials or globally rescale to 1.2 kg, as the existing
locomotion loader does. That loader has not been migrated to this new model.
No physical motion or sim-to-real qualification is implied by these edits.
