"""2&2 trot gait for the hand, recorded to figures/walk.mp4.

Which way can this hand walk?  Every finger joint (MCP-FLX / PIP / DIP) rotates about the
world X axis, so the fingertips can only swing inside a YZ plane; DIP does not move the
foot at all (its axis is collinear with the tip bone). Measured travel of one foot with
the tip kept on the floor:

    joint      axis        foot travel
    MCP-FLX    X (1,0,0)   +-18 mm in z          (lift only)
    PIP        X (1,0,0)   +-37 mm in y          (the walking stroke)
    MCP-ABD    Y (0,1,0)   +23 mm in x, but only on legs A/B - legs C/D sit under the
                           ABD axis and get 4 mm, so an x step cannot be balanced
    DIP        collinear    0 mm

So the walkable direction is +y, and a step is PIP-driven: the tip stays on the floor
(FLX compensates, table below) while PIP sweeps it back 37 mm, then the leg swings
forward with the tip lifted.  Legs move 2&2: A+B together, C+D together, half a cycle
apart, with mirrored PIP so both sides push the trunk the same way.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_model as B  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PERIOD = 1.80          # s per trot cycle (the fastest that stays upright)
DUTY = 0.70            # fraction of the cycle the foot is on the floor
LIFT = 0.30            # extra MCP-FLX during swing, rad
KP, KD = 150.0, 6.0
TORQUE = B.TORQUE
PIP_FRONT, PIP_BACK = 0.10, -2.00      # PIP endpoints of the stroke
GROUPS = {"A": 0, "B": 0, "C": 1, "D": 1}


def _foot_min_z(m, d, base, qadr, leg, pip, flx):
    q = np.array(base, dtype=float)
    q[qadr[f"unit{leg}-PIP"]] = pip
    q[qadr[f"unit{leg}-MCP-FLX"]] = flx
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    return float(B.tip_points(m, d, leg)[:, 2].min())


def build_table(m, d, base, qadr):
    """PIP -> MCP-FLX lookup that keeps this leg's tip exactly on the floor."""
    pips = np.arange(PIP_BACK, PIP_FRONT + 0.05, 0.15)
    tab = {}
    for leg in ("A", "C"):
        rows = []
        for pip in pips:
            lo, hi = 1.25, 2.60
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if _foot_min_z(m, d, base, qadr, leg, pip, mid) > 0.0:
                    hi = mid
                else:
                    lo = mid
            rows.append((pip, 0.5 * (lo + hi)))
        tab[leg] = np.array(rows)
    return tab


def main():
    render = "--render" in sys.argv
    seconds = 12.0
    for a in sys.argv[1:]:
        if a[0].isdigit():
            seconds = float(a)

    m, keys = B.build()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stance1"))
    base = keys["stance1"]
    qadr = {f"unit{lg}-{jn}": m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, f"unit{lg}-{jn}")] for lg in B.LEGS for jn in B.JOINTS}
    dadr = {f"unit{lg}-{jn}": m.jnt_dofadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, f"unit{lg}-{jn}")] for lg in B.LEGS for jn in B.JOINTS}
    idx = np.array([qadr[k] for k in qadr])
    didx = np.array([dadr[k] for k in dadr])
    tab = build_table(m, d, base, qadr)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stance1"))

    if render:
        import imageio
        r = mujoco.Renderer(m, height=486, width=864)
        cam = mujoco.MjvCamera()
        cam.distance = 0.52
        cam.azimuth, cam.elevation = 125, -20
        frames = []

    y0 = float(d.qpos[1])
    dt = m.opt.timestep
    every = max(1, int(round(1.0 / 30.0 / dt)))
    for i in range(int(seconds / dt)):
        t = i * dt
        tgt = np.array(base, dtype=float)
        for lg in B.LEGS:
            grp = GROUPS[lg]
            p = (t / PERIOD + 0.5 * grp) % 1.0
            s = p / DUTY if p < DUTY else 1.0 - (p - DUTY) / (1.0 - DUTY)
            sw = (p - DUTY) / (1.0 - DUTY) if p >= DUTY else 0.0
            pip = PIP_FRONT + (PIP_BACK - PIP_FRONT) * s if grp == 0 \
                else PIP_BACK + (PIP_FRONT - PIP_BACK) * s
            tb = tab["A" if grp == 0 else "C"]
            flx = float(np.interp(pip, tb[:, 0], tb[:, 1]))
            tgt[qadr[f"unit{lg}-PIP"]] = pip
            tgt[qadr[f"unit{lg}-MCP-FLX"]] = flx + LIFT * np.sin(np.pi * sw)
        d.ctrl[:] = np.clip(KP * (tgt[idx] - d.qpos[idx]) - KD * d.qvel[didx],
                            -TORQUE, TORQUE)
        mujoco.mj_step(m, d)
        if render and i % every == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.085]
            r.update_scene(d, camera=cam)
            frames.append(r.render().copy())

    dy = d.qpos[1] - y0
    print(f"travelled {dy*1000:+.1f} mm in {seconds:.0f} s  ({dy/seconds*1000:+.1f} mm/s)")
    print(f"  x drift={d.qpos[0]*1000:+.1f} mm  yaw={np.degrees(d.qpos[5]):+.1f} deg  "
          f"trunk z={d.qpos[2]*1000:.1f} mm")
    R = d.body("platform_mount").xmat.reshape(3, 3)
    up = -R[:, 2]
    print(f"  pitch={np.degrees(np.arctan2(-up[0],up[2])):+.1f} deg "
          f"roll={np.degrees(np.arctan2(up[1],up[2])):+.1f} deg")
    if render:
        out = os.path.join(B.ROOT, "figures", "walk.mp4")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        imageio.mimsave(out, frames, fps=30, quality=7, macro_block_size=8)
        print("saved", out, len(frames), "frames")


if __name__ == "__main__":
    main()
