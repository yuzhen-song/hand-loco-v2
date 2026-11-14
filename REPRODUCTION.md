# Local walking reproduction

Source: `https://github.com/yuzhen-song/hand-loco-v2`, commit
`3cbb26e7a17fcdecae4099440b62c15a7983db8e`.

The local run uses the original `walk_x.py` gait, motor gains, torque limits,
IK tables, friction settings and physics timestep. No policy training or GPT
API calls are involved. The robot advances through MuJoCo dynamics; its root
trajectory is not prescribed.

## Verified result

The completed run advanced **304.1 mm in 10 seconds** (30.4 mm/s), with
-2.35 mm lateral drift and a final trunk height of 204.7 mm. Corrected trunk
yaw was -8.34 degrees; final minimum fingertip height was -3.85 mm relative
to the floor. The source README reports 302 mm travel, so forward displacement
was reproduced closely. Its yaw number used the different calculation noted below.

The new video is `outputs/reproduction/hand_walk_10s_complete.mp4`:
640x360, 30 fps, 300 decodable frames, exactly 10 seconds. Numeric results are
in `metrics.json`; container/frame checks are in `video_validation.json`.

## Environment

Python 3.11.17 is installed under `.runtime/python` and the isolated environment
is `.venv`. Exact package versions are saved in
`outputs/reproduction/environment.txt`. This run uses MuJoCo 3.17.0.

Both shared NVIDIA GPUs were busy at startup. Rendering therefore uses the
system OSMesa library and CPU llvmpipe, with two rendering threads. CUDA devices
are hidden from the process. No system packages or drivers were changed.

## Run

From this repository, after dependencies have been installed:

```bash
export MUJOCO_GL=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export LP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''

.venv/bin/python -B tools/build_model.py --check
.venv/bin/python -B tools/walk_x.py 10 --render --width 640 --height 360 \
  --output outputs/reproduction/hand_walk_10s_complete.mp4 \
  --report outputs/reproduction/metrics.json
```

Choose a new output filename for another recording: the recorder rejects an
existing video path. The repository's original videos in `figures/` are kept.
The build command regenerates the tracked generated XML from the CAD URDF.

## Local adaptations

- Intermediate compiled XML stays in `.runtime/tmp`, rather than a shared
  `/tmp/_built_raw.xml` filename.
- Video output and JSON report paths are explicit CLI options.
- Frames are streamed to a single-threaded encoder at 30 fps, rather than held
  together in memory. Sampling follows simulated time.
- The completed recording uses 640x360 frames, a 512-pixel shadow map and no
  offscreen multisampling to reduce CPU rendering cost. Physics is unchanged.
- Yaw is computed from the trunk rotation matrix. The original printed
  `degrees(qpos[5])`, which is a quaternion component, not a yaw angle; its
  README yaw figure is therefore not directly comparable to the corrected metric.
- First, middle and final video snapshots are retained for visual inspection.

## Interpretation

The model intentionally enables collision only for the fingertips and floor.
It has high fingertip friction and compliant contact, so some ground penetration
is expected and is reported. This is reproduction of the repository's simplified
simulation, not evidence of hardware feasibility or full-body collision safety.

`stand_check.log` contains the five-second standing checks for both keyframes.
`walk_complete.log`, `metrics.json`, and `hand_walk_10s_complete.mp4` describe the
new walking trial. The first recording process was terminated with exit code
143 before completion; `walk.log` and `hand_walk_10s.mp4` belong to that incomplete
attempt and must not be used as the final result. The videos already shipped in
`figures/` are reference artifacts.
