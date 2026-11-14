# Workspace organization

The reorganization applies only to **hand-loco-v2**. Other projects and external
reference repositories were not moved or inspected. No experiment data was deleted.

```text
hand-loco-v2/
  README.md                  project entry point
  docs/                      guides, experiment reports and this directory map
  tools/                     runnable tools and shared control/model modules
    README.md                tools grouped by role
  configs/                   simulation profiles and calibration templates
    README.md                configuration purpose and scope
  model/                     CAD, mass data and the historical generated model
  tests/                     regression and unit tests
  validation/                compact committed experimental evidence
  figures/                   original repository images and reference videos
  outputs/                   generated local artifacts (ignored by Git)
    baselines/               original reproduction and regression records
    sim2real/                current interface and sim-to-real screening records
    experiments/
      terrain/               initial direction, slope, step and jump attempts
      dance/                 baseline_v1 and fast_v1 continuous sequences
      speed/                 four-direction speed comparisons
    videos/terrain_review/   combined terrain/direction/jump video and viewer
    handoff/                 unchanged hardware handoff bundle and verification
    archive/                 superseded v1 sim-to-real records
  .venv/                     existing environment, unchanged
  .runtime/                  existing local interpreter/build intermediates
  .bootstrap/                existing bootstrap files, unchanged
```

## Useful current paths

| Artifact | Current location |
| --- | --- |
| Fast dance video | `outputs/experiments/dance/fast_v1/dance.mp4` |
| Original dance video | `outputs/experiments/dance/baseline_v1/dance.mp4` |
| All terrain/jump attempt video | `outputs/videos/terrain_review/all_attempts.mp4` |
| Terrain/jump viewer | `outputs/videos/terrain_review/index.html` |
| Speed comparison data | `outputs/experiments/speed/sweep_v1/` |
| Valid slow sim-to-real video | `outputs/sim2real/candidate_video_v2/walk.mp4` |
| Valid sim-to-real sensitivity data | `outputs/sim2real/screen_v2/` |
| Original reproduction | `outputs/baselines/reproduction/` |
| Previously delivered ZIP | `outputs/handoff/hand-loco-sim2real-82366b0.zip` |

The exact explicit old-to-new mapping is in
[`workspace_moves.json`](workspace_moves.json). Old chat links to moved local files
may need their path replaced using this mapping. The handoff ZIP and its SHA256
file remain at their original locations; its contents were not repackaged.

Historical raw metrics, command logs and recorded states are preserved without
rewriting their content. Consequently, historical manifests can still contain their
original paths and source hashes. Resolve old paths through the mapping; do not
reinterpret those source hashes as the current revision. Replaying old recordings
with source verification may require their recorded Git revision.

Current document links and example artifact locations have been updated. Command
examples are always executed from the repository root, even when a document is
under `docs/`. Model/config/tool locations remain stable so imports, calibration
paths and the existing environment continue to work.
