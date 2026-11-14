"""2&2 trot toward +x on stance1 (FLX=90deg: MCP-ABD axis vertical, upright).

Drive ONLY the MCP-ABD servo (now standing upright, axis=(0,0,1)); rotating it
sweeps the leg in the horizontal plane.  PIP/DIP come from a precomputed IK
table that locks the fingertip at a fixed world (y,z) during stance, so the
foot does not slide on the floor - the body translates forward instead.

2&2 trot = diagonal pairs: A+D stance while B+C swing, then swap (duty 0.5).
"""
import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_model as B  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PERIOD = 2.00
RAMP = 0.80
DUTY = 0.50
LIFT = 0.16          # extra MCP-FLX during swing only (lifts the foot)
KP, KD = 400.0, 14.0
TORQUE = B.TORQUE

ABD_HI, ABD_LO = +1.20, -1.20     # tip-front -> tip-back (~33 mm x stroke)
GROUPS = {"A": 0, "D": 0, "B": 1, "C": 1}   # diagonal pairs (2&2)
FLIP = {"A": False, "B": False, "C": True, "D": True}   # measured push direction


def setup():
    m, keys = B.build()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stance1"))
    qadr = {f"unit{lg}-{jn}": m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, f"unit{lg}-{jn}")] for lg in B.LEGS for jn in B.JOINTS}
    dadr = {f"unit{lg}-{jn}": m.jnt_dofadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, f"unit{lg}-{jn}")] for lg in B.LEGS for jn in B.JOINTS}
    return m, d, keys["stance1"], qadr, dadr


def tip(m, d, base, qadr, leg, abd, pip, dip, flx):
    q = np.array(base, dtype=float)
    q[qadr[f"unit{leg}-MCP-FLX"]] = flx
    q[qadr[f"unit{leg}-MCP-ABD"]] = abd
    q[qadr[f"unit{leg}-PIP"]] = pip
    q[qadr[f"unit{leg}-DIP"]] = dip
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    P = B.tip_points(m, d, leg)
    i = int(np.argmin(P[:, 2]))
    return P[i].copy()


def build_tables(m, d, base, qadr, flx):
    """per leg: rows of (abd, PIP, DIP) keeping fingertip y,z at its stance value,
       tracking continuously from the stance1 seed so the solution never jumps."""
    d.qpos[:] = base
    mujoco.mj_forward(m, d)
    tabs = {}
    for leg in B.LEGS:
        P = B.tip_points(m, d, leg)
        i = int(np.argmin(P[:, 2]))
        y0 = float(P[i, 1]); z0 = float(P[i, 2])
        pip = float(base[qadr[f"unit{leg}-PIP"]])
        dip = float(base[qadr[f"unit{leg}-DIP"]])
        rows = []
        for abd in np.arange(ABD_LO - 0.05, ABD_HI + 0.06, 0.05):
            for _ in range(60):
                p = tip(m, d, base, qadr, leg, abd, pip, dip, flx)
                err = np.array([p[1] - y0, p[2] - z0])
                if np.linalg.norm(err) < 2e-5:
                    break
                h = 1e-4
                p1 = tip(m, d, base, qadr, leg, abd, pip + h, dip, flx)
                p2 = tip(m, d, base, qadr, leg, abd, pip, dip + h, flx)
                J = np.array([[(p1[1] - p[1]) / h, (p2[1] - p[1]) / h],
                              [(p1[2] - p[2]) / h, (p2[2] - p[2]) / h]])
                try:
                    step = np.linalg.solve(J, -err)
                except np.linalg.LinAlgError:
                    break
                pip = float(np.clip(pip + np.clip(step[0], -0.1, 0.1), -2.0, 0.489))
                dip = float(np.clip(dip + np.clip(step[1], -0.1, 0.1), -0.5, 1.958))
            rows.append((abd, pip, dip))
        tabs[leg] = np.array(rows)
    return tabs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seconds", type=float, nargs="?", default=10.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--controller", choices=("baseline", "position"), default="baseline")
    parser.add_argument("--sim-torque-limit-nm", type=float,
                        help="Explicit assumed plant ceiling for position mode; not derived from current")
    parser.add_argument("--output", type=Path, default=Path(B.ROOT)/"figures/walk_x.mp4")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=486)
    args = parser.parse_args()
    render, seconds = args.render, args.seconds
    if not np.isfinite(seconds) or seconds <= 0:
        parser.error("seconds must be positive and finite")
    if args.controller == "position":
        if (args.sim_torque_limit_nm is None or not np.isfinite(args.sim_torque_limit_nm)
                or not 0 < args.sim_torque_limit_nm <= B.TORQUE):
            parser.error(f"position mode requires --sim-torque-limit-nm in (0, {B.TORQUE}]")
    elif args.sim_torque_limit_nm is not None:
        parser.error("--sim-torque-limit-nm applies only to position mode")
    if seconds < 0.002:
        parser.error("seconds must include at least one physics step (0.002 s)")
    if args.report and args.report.exists():
        parser.error("report already exists; choose a new path")
    if render and args.output.exists():
        parser.error("output already exists; choose a new path")

    m, d, base, qadr, dadr = setup()
    idx = np.array([qadr[k] for k in qadr])
    didx = np.array([dadr[k] for k in dadr])
    flx_stance = float(base[qadr["unitA-MCP-FLX"]])
    tabs = build_tables(m, d, base, qadr, flx_stance)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stance1"))

    if render:
        import imageio
        m.vis.quality.shadowsize = 512
        m.vis.quality.offsamples = 0
        r = mujoco.Renderer(m, height=args.height, width=args.width)
        cam = mujoco.MjvCamera()
        cam.distance = 0.75
        cam.azimuth, cam.elevation = 125, -22
        cam.lookat[:] = [0.11, -0.05, 0.09]   # fixed camera - travel is visible
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.output), fps=30, quality=7,
                                    macro_block_size=1, ffmpeg_params=["-threads", "1"])
        from OpenGL import GL
        renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
        print("renderer:", renderer_name, flush=True)
        frame_count = 0

    interface = None
    if args.controller == "position":
        from hand_interface import MODEL_NAMES, SampledPositionInterface
        if tuple(qadr) != MODEL_NAMES:
            raise RuntimeError("model joint order does not match the position interface")
        interface = SampledPositionInterface(d.qpos[idx], m.opt.timestep,
                                              args.sim_torque_limit_nm, KP, KD)

    x0 = float(d.qpos[0])
    y0 = float(d.qpos[1])
    dt = m.opt.timestep
    next_frame = 1.0 / 30.0
    for i in range(int(seconds / dt)):
        t = i * dt
        blend = min(1.0, t / RAMP) if RAMP > 0 else 1.0
        blend = blend * blend * (3.0 - 2.0 * blend)
        tgt = np.array(base, dtype=float)
        for lg in B.LEGS:
            p = (t / PERIOD + 0.5 * GROUPS[lg]) % 1.0
            if p < DUTY:                        # stance: foot clamped (y,z fixed by table)
                s = p / DUTY
                sw = 0.0
                if FLIP[lg]:
                    abd = ABD_HI + (ABD_LO - ABD_HI) * s
                else:
                    abd = ABD_LO + (ABD_HI - ABD_LO) * s
                flx = flx_stance
            else:                               # swing: lift and carry forward
                s = (p - DUTY) / (1.0 - DUTY)
                sw = s
                u = s * s * (3.0 - 2.0 * s)
                if FLIP[lg]:
                    abd = ABD_LO + (ABD_HI - ABD_LO) * u
                else:
                    abd = ABD_HI + (ABD_LO - ABD_HI) * u
                flx = flx_stance + LIFT * np.sin(np.pi * sw)
            tb = tabs[lg]
            pip = float(np.interp(abd, tb[:, 0], tb[:, 1]))
            dip = float(np.interp(abd, tb[:, 0], tb[:, 2]))
            k = f"unit{lg}-"
            tgt[qadr[k + "MCP-ABD"]] = base[qadr[k + "MCP-ABD"]] + blend * (abd - base[qadr[k + "MCP-ABD"]])
            tgt[qadr[k + "MCP-FLX"]] = base[qadr[k + "MCP-FLX"]] + blend * (flx - base[qadr[k + "MCP-FLX"]])
            tgt[qadr[k + "PIP"]] = base[qadr[k + "PIP"]] + blend * (pip - base[qadr[k + "PIP"]])
            tgt[qadr[k + "DIP"]] = base[qadr[k + "DIP"]] + blend * (dip - base[qadr[k + "DIP"]])
        if interface is None:
            d.ctrl[:] = np.clip(KP * (tgt[idx] - d.qpos[idx]) - KD * d.qvel[didx],
                                -TORQUE, TORQUE)
        else:
            d.ctrl[:] = interface.step(tgt[idx], d.qpos[idx], d.qvel[didx])
        mujoco.mj_step(m, d)
        if render and d.time + 1e-9 >= next_frame:
            r.update_scene(d, camera=cam)
            frame = r.render().copy()
            writer.append_data(frame)
            frame_count += 1
            next_frame = (frame_count + 1) / 30.0
            if frame_count in (1, 150, 300):
                imageio.imwrite(str(args.output.with_name(f"frame_{frame_count:04d}.png")), frame)

    mujoco.mj_forward(m, d)
    dx = d.qpos[0] - x0
    dy = d.qpos[1] - y0
    print(f"travelled +x: {dx*1000:+.1f} mm in {seconds:.0f} s  ({dx/seconds*1000:+.1f} mm/s)")
    R = d.body("platform_mount").xmat.reshape(3, 3)
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    print(f"  lateral y drift = {dy*1000:+.1f} mm  yaw={yaw:+.1f} deg  "
          f"trunk z={d.qpos[2]*1000:.1f} mm")
    R = d.body("platform_mount").xmat.reshape(3, 3)
    up = -R[:, 2]
    print(f"  pitch={np.degrees(np.arctan2(-up[0],up[2])):+.1f} deg "
          f"roll={np.degrees(np.arctan2(up[1],up[2])):+.1f} deg")
    if render:
        writer.close()
        r.close()
        print("saved", args.output, frame_count, "frames", flush=True)
    if args.report:
        metrics = dict(simulation_seconds=float(d.time), travel_x_mm=float(dx*1000),
            lateral_drift_mm=float(dy*1000), speed_mm_s=float(dx/d.time*1000),
            yaw_deg=yaw, trunk_height_mm=float(d.qpos[2]*1000),
            deepest_final_tip_z_mm=min(float(B.tip_points(m,d,lg)[:,2].min()) for lg in B.LEGS)*1000,
            final_contacts=int(d.ncon), mujoco_version=mujoco.__version__,
            rendering_backend=os.environ.get("MUJOCO_GL"),
            renderer=renderer_name if render else None, frames=frame_count if render else 0,
            fps=30 if render else None, video=str(args.output) if render else None,
            controller=args.controller,
            position_interface=interface.report() if interface else None,
            source="new MuJoCo dynamics execution; original gait parameters unchanged",
            yaw_note="Computed from trunk rotation matrix; original script printed a quaternion component as degrees.")
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(metrics,indent=2)+"\n")


if __name__ == "__main__":
    main()
