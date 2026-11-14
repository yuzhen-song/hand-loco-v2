"""Compile model/full_hand_final/robot.urdf into a free-base MuJoCo model for locomotion.

Beyond a plain URDF compile:
  * wraps the URDF root link in `platform_mount` + freejoint and restores the root
    link's inertial (MuJoCo drops it silently: 0.1595 kg of 0.7859 kg)
  * welds the linear actuator shut at -0.05 m: the prismatic joint is deleted and the
    offset baked into `guide_10429`, so it has no degree of freedom and cannot drift
  * keeps collision only on the four fingertips. MuJoCo inflates concave CAD meshes to
    their convex hull (fingertip volume ratio 0.71), so 38 live collision copies per
    link gave ~600 spurious contacts and the palm, not the fingertips, rested on the floor
  * adds 16 finger motors, a friction floor, and the standing keyframe

Usage:
  python tools/build_model.py            # -> model/quadruped_sim_v2.xml
  python tools/build_model.py --check    # also hold the keyframe for 3 s
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "model", "full_hand_final")
URDF = os.path.join(SRC_DIR, "robot.urdf")
OUT_XML = os.path.join(ROOT, "model", "quadruped_sim_v2.xml")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

LA_LOCK = -0.05
LEGS = "ABCD"
JOINTS = ("MCP-FLX", "MCP-ABD", "PIP", "DIP")
TIP_BODY = {"A": "finger_tip_bone_2", "B": "finger_tip_bone_3",
            "C": "finger_tip_bone", "D": "finger_tip_bone_4"}

# Historical simulation standing poses; physical ROM is not yet calibrated.
# stance1: MCP-FLX is about pi/2, placing the steering axis approximately vertical.
# PIP/DIP are folded to keep the fingertip near vertical.
# stance2: keeps the leg longer but the tip bone tilts ~5 deg off vertical.
STANCES = {
    "stance1": {"MCP-FLX": 1.5708, "MCP-ABD": 0.00, "PIP": -1.40, "DIP": -0.20},
    "stance2": {"MCP-FLX": 1.30, "MCP-ABD": 0.00, "PIP": -1.60, "DIP": 0.40},
}
DEFAULT_STANCE = "stance1"
STAND_HEIGHT = 0.2130    # placeholder; each keyframe gets its own measured height

# Historical simulation overrides. Their claimed collision/ROM verification is
# not reproduced here; do not treat these ranges as measured hardware limits.
JOINT_LIMITS = {
    "MCP-FLX": (-0.2268, 2.2000),
    "MCP-ABD": (-1.4580, 1.4580),
    "PIP": (-2.0000, 0.4890),
    "DIP": (-0.5000, 1.9580),
}
TORQUE = 0.95            # Historical simulated ceiling, NOT a continuous rating
ARM_DAMP = 0.10
TIP_FRICTION = "10.0 0.05 0.01"

ROOT_INERTIAL = (
    '<inertial pos="0.00603964183665823 0.0452026246748193 0.0952415038898925" '
    'mass="0.159544859630194" '
    'fullinertia="0.000242816313 5.31142042e-05 0.0002416894 '
    '4.67336127e-07 2.86916936e-06 1.29626652e-07"/>'
)


def _rot(q, v):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ np.asarray(v, dtype=float)


def weld_linear_actuator(xml):
    mt = re.search(r'<joint name="linearActuator"[^>]*/>', xml)
    if mt is None:
        raise RuntimeError("linearActuator joint not found in the compiled model")
    tag = mt.group(0)
    am = re.search(r'axis="([^"]+)"', tag)
    pm = re.search(r'pos="([^"]+)"', tag)
    axis = np.array([float(v) for v in am.group(1).split()]) if am else np.array([0.0, 0.0, 1.0])
    off = np.array([float(v) for v in pm.group(1).split()]) if pm else np.zeros(3)

    stack = []
    for m2 in re.finditer(r"<body\b[^>]*>|</body>", xml[:mt.start()]):
        if m2.group(0).startswith("</"):
            stack.pop()
        else:
            stack.append(m2)
    body_tag = stack[-1].group(0)
    bpos = np.array([float(v) for v in re.search(r'pos="([^"]+)"', body_tag).group(1).split()])
    bquat = np.array([float(v) for v in re.search(r'quat="([^"]+)"', body_tag).group(1).split()])

    delta = _rot(bquat, off) + _rot(bquat, axis * LA_LOCK)
    new_pos = " ".join("{:.9g}".format(v) for v in (bpos + delta))
    xml = xml.replace(body_tag, re.sub(r'pos="[^"]+"', f'pos="{new_pos}"', body_tag, count=1), 1)
    return xml.replace(tag, f"<!-- linearActuator welded at {LA_LOCK:.3f} m -->", 1)


def apply_joint_limits(xml):
    """Apply historical simulation ranges, not hardware-calibrated ROM."""

    def fix(mt):
        tag = mt.group(0)
        nm = re.search(r'name="([^"]+)"', tag)
        if nm is None:
            return tag
        for jn, (lo, hi) in JOINT_LIMITS.items():
            if not nm.group(1).endswith("-" + jn):
                continue
            rg = re.search(r'range="([^"]+)"', tag)
            if rg is None:
                return tag
            return tag.replace(rg.group(0), 'range="{:.6g} {:.6g}"'.format(lo, hi))
        return tag

    return re.sub(r"<joint\b[^>]*/>", fix, xml)


def keep_collision_only_at_tips(xml):
    def kill(chunk):
        def repl(m2):
            t = m2.group(0)
            if 'name="floor"' in t:
                return t
            t = re.sub(r'\bcontype="[^"]*"', 'contype="0"', t)
            t = re.sub(r'\bconaffinity="[^"]*"', 'conaffinity="0"', t)
            if "contype=" not in t:
                t = t.replace("<geom ", '<geom contype="0" conaffinity="0" ', 1)
            return t
        return re.sub(r"<geom\b[^>]*/>", repl, chunk)

    spans = []
    for m2 in re.finditer(r'<body name="finger_tip_bone[^"]*"[^>]*>', xml):
        spans.append((m2.start(), xml.find("</body>", m2.start()) + len("</body>")))
    out, last = [], 0
    for s, e in spans:
        out.append(kill(xml[last:s]))
        out.append(xml[s:e])
        last = e
    out.append(kill(xml[last:]))
    return "".join(out)


def compile_xml():
    txt = Path(URDF).read_text(encoding="utf-8").replace("package://", "")
    txt = txt.replace(
        "</robot>",
        '  <mujoco><compiler discardvisual="false" balanceinertia="true" '
        'boundmass="1e-9" boundinertia="1e-12"/></mujoco>\n</robot>',
    )
    tmp = os.path.join(SRC_DIR, "_build_tmp.urdf")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(txt)
    try:
        m0 = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    raw_dir = os.path.join(ROOT, ".runtime", "tmp")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "built_raw.xml")
    mujoco.mj_saveLastXML(raw_path, m0)
    xml = Path(raw_path).read_text(encoding="utf-8")

    xml = weld_linear_actuator(xml)

    xml = re.sub(r"<worldbody>(.*)</worldbody>",
                 lambda mm: ('<worldbody>\n'
                             '    <light name="top" pos="0.3 -0.3 1.2" diffuse="1 1 1"/>\n'
                             '    <geom name="floor" type="plane" size="3 3 0.02" '
                             'solref="0.008 1" friction="1.8 0.05 0.01" rgba="0.85 0.85 0.85 1"/>\n'
                             '    <body name="platform_mount" pos="0 0 {h}" quat="0 1 0 0">\n'
                             '      <freejoint/>\n'
                             '      ' + ROOT_INERTIAL + '\n'
                             '{inner}</body>\n  </worldbody>').format(h=STAND_HEIGHT,
                                                                     inner=mm.group(1)),
                 xml, count=1, flags=re.DOTALL)

    xml = re.sub(r"<mujoco\b[^>]*>",
                 lambda mm: mm.group(0) + '\n'
                 '  <compiler meshdir="full_hand_final"/>\n'
                 '  <visual><global offwidth="1280" offheight="960"/></visual>\n'
                 '  <option timestep="0.002" integrator="implicitfast"/>',
                 xml, count=1)

    def patch_joint(mt):
        tag = mt.group(0)

        def put(t, attr, val):
            if re.search(rf'\b{attr}="[^"]*"', t):
                return re.sub(rf'\b{attr}="[^"]*"', f'{attr}="{val}"', t)
            return t.replace("<joint ", f'<joint {attr}="{val}" ', 1)

        tag = put(tag, "armature", "0.01")
        tag = put(tag, "damping", str(ARM_DAMP))
        if re.search(r'name="unit[A-D]-', tag):
            tag = put(tag, "actuatorfrcrange", f"{-TORQUE} {TORQUE}")
        return tag

    xml = re.sub(r"<joint\b[^>]*/>", patch_joint, xml)
    xml = re.sub(r'(<geom [^>]*mesh="finger_tip_bone[^"]*"[^>]*?)/>',
                 rf'\1 friction="{TIP_FRICTION}" condim="4" solref="0.0005 1" solimp="0.999 0.9999 0.0001"/>', xml)
    xml = apply_joint_limits(xml)
    xml = keep_collision_only_at_tips(xml)

    acts = ["  <actuator>"]
    for lg in LEGS:
        for jn in JOINTS:
            acts.append(f'    <motor name="m_unit{lg}-{jn}" joint="unit{lg}-{jn}" gear="1"/>')
    acts.append("  </actuator>")
    xml = xml.replace("</mujoco>", "\n".join(acts) + "\n</mujoco>")
    return xml


def stance_qpos(m, stance, height=STAND_HEIGHT):
    """Full qpos for one named stance (freejoint first, then the 16 finger joints)."""
    q = np.array(m.qpos0, dtype=float)
    q[0:7] = [0.0, 0.0, height, 0.0, 1.0, 0.0, 0.0]
    for lg in LEGS:
        for jn in JOINTS:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"unit{lg}-{jn}")
            q[m.jnt_qposadr[j]] = stance[jn]
    return q


def tip_points(m, d, lg):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TIP_BODY[lg])
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        R = d.geom_xmat[g].reshape(3, 3)
        pts.append((R @ m.mesh_vert[a:a + n].T).T + d.geom_xpos[g])
    return np.vstack(pts)


def measure(m, d):
    """Everything that defines the initial state, in the same terms as the old model."""
    out = {}
    for lg in LEGS:
        P = tip_points(m, d, lg)
        i = int(np.argmin(P[:, 2]))
        dip = d.body(TIP_BODY[lg]).xpos
        # finger_tip_bone runs along the body's -Y axis, so this column tells where it points
        down = float(-d.body(TIP_BODY[lg]).xmat.reshape(3, 3)[2, 1])
        out[lg] = dict(contact=P[i], tip_z=float(P[i, 2]), dip_z=float(dip[2]),
                       dip_above=float(dip[2] - P[i, 2]),
                       tip_down=down,
                       tip_tilt=float(np.degrees(np.arccos(np.clip(-down, -1, 1)))))
    return out


def build(write=True, clearance=0.0016, xml_transform=None):
    """Compile, then bake one keyframe per stance with its own measured standing height."""
    xml = compile_xml()
    if xml_transform is not None:
        xml = xml_transform(xml)
    def load(text):
        # Absolute meshdir permits an in-memory build without rewriting baseline XML.
        return mujoco.MjModel.from_xml_string(text.replace(
            'meshdir="full_hand_final"', f'meshdir="{SRC_DIR}"'))
    m = load(xml)

    keyframes = {}
    for name, st in STANCES.items():
        d = mujoco.MjData(m)
        q = stance_qpos(m, st)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        drop = min(float(tip_points(m, d, lg)[:, 2].min()) for lg in LEGS)
        q[2] -= drop - clearance
        keyframes[name] = q

    xml = xml.replace(f'<body name="platform_mount" pos="0 0 {STAND_HEIGHT}"',
                      f'<body name="platform_mount" pos="0 0 {keyframes[DEFAULT_STANCE][2]:.6f}"', 1)
    kf = ["  <keyframe>"]
    for name, q in keyframes.items():
        kf.append(f'    <key name="{name}" qpos="{" ".join(f"{v:.6f}" for v in q)}"/>')
    kf.append("  </keyframe>")
    xml = xml.replace("</mujoco>", "\n".join(kf) + "\n</mujoco>")
    if write:
        with open(OUT_XML, "w", encoding="utf-8") as fh:
            fh.write(xml)
    return load(xml), keyframes


def keyframe_qpos(m, name):
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, name)
    if k < 0:
        raise KeyError(f"no keyframe named {name!r}")
    return np.array(m.key_qpos[k], dtype=float)


def report(m, q, name="stance"):
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    feet = measure(m, d)
    print(f"[{name}]  nq={m.nq} nu={m.nu} ngeom={m.ngeom} mass={m.body_subtreemass[0]:.4f} kg")
    print(f"stand: mount z={q[2]:.4f}  leg={q[2]-np.mean([feet[l]['tip_z'] for l in LEGS]):.4f}")
    for lg in LEGS:
        f = feet[lg]
        print(f"  {lg}: contact=({f['contact'][0]:+.4f},{f['contact'][1]:+.4f},{f['tip_z']:+.5f}) "
              f"tip_tilt={f['tip_tilt']:5.2f} deg  tip_down={f['tip_down']:+.4f}  "
              f"dip_above_tip={f['dip_above']*1000:6.1f} mm")
    names = [f"unit{lg}-{jn}" for lg in LEGS for jn in JOINTS]
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    dadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    d.qvel[:] = 0
    for _ in range(2500):          # 5 s hold
        d.ctrl[:] = np.clip(60.0 * (q[qadr] - d.qpos[qadr]) - 3.0 * d.qvel[dadr], -TORQUE, TORQUE)
        mujoco.mj_step(m, d)
    R = d.body("platform_mount").xmat.reshape(3, 3)
    up = -R[:, 2]                       # trunk "up" axis is local -z
    pitch = np.degrees(np.arctan2(-up[0], up[2]))
    roll = np.degrees(np.arctan2(up[1], up[2]))
    depth = min(float(tip_points(m, d, lg)[:, 2].min()) for lg in LEGS)
    print(f"hold 5 s: mount z {q[2]:.4f} -> {d.qpos[2]:.4f}  drift={np.linalg.norm(d.qpos[:2])*1000:.1f} mm")
    print(f"  trunk pitch={pitch:+.2f} deg  roll={roll:+.2f} deg  upright={up[2]:+.4f}")
    print(f"  deepest tip penetration = {depth*1000:+.3f} mm   ncon={d.ncon}")
    print(f"  CoM z = {d.subtree_com[0][2]:.4f} (was mount {d.qpos[2]:.4f})")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    mm, keys = build()
    print(f"wrote {OUT_XML}")
    print(f"  keyframes: {list(keys)}")
    if a.check:
        for nm in keys:
            report(mm, keys[nm], nm)
            print()
