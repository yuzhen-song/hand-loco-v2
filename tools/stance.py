"""Render both standing poses side by side, one row per stance, plus a fingertip close-up."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_model as B  # noqa: E402

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(B.ROOT, "figures", "stances.png")
PANELS = {
    "stance1": [(90, -10, 0.40), (30, -18, 0.40), (90, 78, 0.40)],
    "stance2": [(90, -10, 0.40), (30, -18, 0.40), (90, 78, 0.40)],
}


def main():
    m, keys = B.build()
    state = {}
    for nm, q in keys.items():
        state[nm] = B.report(m, q, nm)
        print()

    r = mujoco.Renderer(m, height=520, width=650)
    rows = []
    for nm, q in keys.items():
        d = state[nm]
        c = d.subtree_com[0]
        frames = []
        for az, el, dist in PANELS[nm]:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [c[0], c[1], 0.06]
            cam.distance = dist
            cam.azimuth, cam.elevation = az, el
            r.update_scene(d, camera=cam)
            frames.append(r.render().copy())
        # fingertip close-up
        f = d.body(B.TIP_BODY["A"]).xpos
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [f[0], f[1], 0.02]
        cam.distance = 0.11
        cam.azimuth, cam.elevation = 115, -5
        r.update_scene(d, camera=cam)
        frames.append(r.render().copy())
        rows.append(np.hstack(frames))

    grid = np.vstack(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.imwrite(OUT, grid)
    print("saved", OUT, grid.shape)


if __name__ == "__main__":
    main()
