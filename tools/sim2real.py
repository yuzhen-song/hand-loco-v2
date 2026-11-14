"""CPU-only sim-to-real screening with shared bounded position commands.

No physical device I/O. Torque, friction and servo response are hypotheses,
not estimates inferred from the configured electrical current limit.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('GALLIUM_DRIVER','llvmpipe')
os.environ.setdefault('LP_NUM_THREADS','2')

import mujoco
import numpy as np
import build_model as B
from hand_interface import MODEL_NAMES, MOTOR_NAMES, SampledPositionInterface
from trajectory import check_trajectory
from walk_x import build_tables, FLIP


@dataclass(frozen=True)
class Case:
    name: str = 'crawl_candidate'
    seconds: float = 10.0
    stance: str = 'stance1'
    gait: str = 'crawl'
    period_s: float = 4.0
    settle_s: float = 1.0
    ramp_s: float = 2.0
    abd_amplitude_rad: float = .6
    lift_rad: float = .08
    torque_limit_nm: float = .3
    kp: float = 60.0
    kd: float = 3.0
    friction: float = .8
    contact_time_s: float = .004
    velocity_limit_rad_s: float = 2.0
    acceleration_limit_rad_s2: float = 8.0

    def validate(self):
        if self.stance not in B.STANCES or self.gait not in ('trot','crawl'):
            raise ValueError('unsupported stance/gait')
        for key,value in asdict(self).items():
            if isinstance(value,(int,float)) and (not np.isfinite(value) or value<=0):
                raise ValueError('positive finite parameter required: '+key)
        if self.seconds < self.settle_s+self.period_s or self.torque_limit_nm>B.TORQUE:
            raise ValueError('duration must include a gait cycle; torque cannot exceed baseline ceiling')
        if self.abd_amplitude_rad>1.2 or self.lift_rad>.16 or self.contact_time_s<.004:
            raise ValueError('unsupported amplitude/lift/contact stiffness')
        if Path(self.name).name != self.name or self.name in ('.','..'):
            raise ValueError('invalid case name')


def contact_xml(xml, case):
    """Set masks BEFORE compilation so body masks and collision BVHs are rebuilt.

    Keep one CAD collision copy per component, versus the floor only.
    Self-collision remains unqualified: concave components need better geometry.
    """
    tree=ET.fromstring(xml)
    for geom in tree.find('worldbody').iter('geom'):
        if geom.get('name')=='floor':
            contype,affinity=2,1
        elif int(geom.get('group','0'))==0:
            contype,affinity=1,2
        else:
            contype,affinity=0,0
        geom.set('contype',str(contype));geom.set('conaffinity',str(affinity))
        geom.set('friction',f'{case.friction} 0.005 0.0001')
        geom.set('condim','3');geom.set('solref',f'{case.contact_time_s} 1')
        geom.set('solimp','0.95 0.99 0.001 0.5 2')
    return ET.tostring(tree,encoding='unicode')


def planner(t,base,qadr,tables,case):
    target=base.copy()
    if t<case.settle_s:return target
    t-=case.settle_s
    blend=min(1.,t/case.ramp_s);blend=blend*blend*(3-2*blend)
    phases={'A':0,'D':0,'B':.5,'C':.5}
    duty=.5
    if case.gait=='crawl':
        phases={'A':0,'D':.25,'B':.5,'C':.75};duty=.75
    amp=case.abd_amplitude_rad
    for leg in B.LEGS:
        p=(t/case.period_s+phases[leg])%1
        if p<duty:
            u=p/duty; lift=0.
        else:
            swing=(p-duty)/(1-duty)
            u=1-swing*swing*(3-2*swing)
            lift=case.lift_rad*np.sin(np.pi*swing)
        abd=(-amp+2*amp*u)*(-1 if FLIP[leg] else 1)
        tb=tables[leg];stem=f'unit{leg}-'
        values={'MCP-ABD':abd,'MCP-FLX':base[qadr[stem+'MCP-FLX']]+lift,
                'PIP':np.interp(abd,tb[:,0],tb[:,1]),'DIP':np.interp(abd,tb[:,0],tb[:,2])}
        for joint,value in values.items():
            a=qadr[stem+joint];target[a]=base[a]+blend*(value-base[a])
    return target


def run(case, destination, render=False):
    case.validate()
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    source_files=['tools/sim2real.py','tools/build_model.py','tools/hand_interface.py',
                  'tools/trajectory.py','tools/walk_x.py','configs/xc330_hand.json']
    source_hashes={p:hashlib.sha256((Path(B.ROOT)/p).read_bytes()).hexdigest() for p in source_files}
    m,keys=B.build(write=False,xml_transform=lambda xml:contact_xml(xml,case))
    floor=m.geom('floor').id
    base=keys[case.stance].copy();d=mujoco.MjData(m)
    qadr={n:int(m.jnt_qposadr[m.joint(n).id]) for n in MODEL_NAMES}
    idx=np.array(list(qadr.values()))
    didx=np.array([m.jnt_dofadr[m.joint(n).id] for n in MODEL_NAMES])
    ranges=np.array([m.jnt_range[m.joint(n).id] for n in MODEL_NAMES])
    if not np.array_equal(m.actuator_trnid[:,0],[m.joint(n).id for n in MODEL_NAMES]):
        raise RuntimeError('unexpected actuator ordering')
    if np.any(base[idx]<ranges[:,0]) or np.any(base[idx]>ranges[:,1]):
        raise ValueError('initial stance outside model limits')
    tables=build_tables(m,d,base,qadr,float(base[qadr['unitA-MCP-FLX']]),
                        -case.abd_amplitude_rad,case.abd_amplitude_rad)
    d=mujoco.MjData(m);d.qpos[:]=base;mujoco.mj_forward(m,d)
    servo=SampledPositionInterface(base[idx],m.opt.timestep,case.torque_limit_nm,
        case.kp,case.kd,rate_limits=(case.velocity_limit_rad_s,case.acceleration_limit_rad_s2),
        joint_limits=ranges)
    tip_ids={m.body(B.TIP_BODY[leg]).id for leg in B.LEGS}
    dt=float(m.opt.timestep); nsteps=round(case.seconds/dt)
    x0,y0=float(d.qpos[0]),float(d.qpos[1])
    min_height=float(d.qpos[2]);max_tilt=0.;min_contact_distance=0.
    non_tip_samples=0;monitor_samples=0;stop_reason='duration_completed'; body_trace=[]
    command_validation=None; frames=0;renderer=None;writer=None
    renderer_name=None
    if render:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw,ImageFont
        m.vis.quality.shadowsize=512;m.vis.quality.offsamples=0
        renderer=mujoco.Renderer(m,height=360,width=640)
        camera=mujoco.MjvCamera();camera.distance=.8;camera.azimuth=125;camera.elevation=-22
        camera.lookat[:]=[.13,-.05,.09]
        writer=imageio.get_writer(str(destination/'walk.mp4'),fps=30,quality=8,
                                 macro_block_size=1,ffmpeg_params=['-threads','1'])
        font=ImageFont.load_default(size=13)
        from OpenGL import GL
        renderer_name=GL.glGetString(GL.GL_RENDERER).decode()
    try:
        for k in range(nsteps):
            target=planner(k*dt,base,qadr,tables,case)
            d.ctrl[:]=servo.step(target[idx],d.qpos[idx],d.qvel[didx])
            mujoco.mj_step(m,d)
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                stop_reason='nonfinite_state';break
            if k%10==0:
                mujoco.mj_forward(m,d)
                up=-d.body('platform_mount').xmat.reshape(3,3)[:,2]
                tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))))
                max_tilt=max(max_tilt,tilt);min_height=min(min_height,float(d.qpos[2]))
                nontip=0;support=set()
                for contact in d.contact:
                    if floor not in (contact.geom1,contact.geom2):continue
                    other=contact.geom2 if contact.geom1==floor else contact.geom1
                    body=int(m.geom_bodyid[other]);min_contact_distance=min(min_contact_distance,float(contact.dist))
                    if body not in tip_ids:nontip+=1
                    else:support.add(body)
                monitor_samples+=1;non_tip_samples+=int(nontip>0)
                body_trace.append({'time_s':float(d.time),'xyz_m':d.qpos[:3].tolist(),
                                   'tilt_deg':tilt,'supporting_tip_bodies':len(support),
                                   'non_tip_floor_contacts':nontip})
                if tilt>45:
                    stop_reason='tilt_exceeded_45_deg';break
                if nontip:
                    stop_reason='non_tip_floor_contact';break
            if render and d.time+1e-9>=(frames+1)/30:
                mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera)
                picture=Image.new('RGB',(640,432),'#152230')
                picture.paste(Image.fromarray(renderer.render()),(0,36));draw=ImageDraw.Draw(picture)
                draw.text((8,4),f'SIMULATION | {case.gait} | {case.period_s:g}s cycle | actuator uncalibrated',font=font,fill='white')
                draw.text((8,20),f't={d.time:.2f}s | 50 Hz commands / 20 Hz feedback | CPU',font=font,fill='#b9cede')
                draw.text((8,399),f'Assumed torque {case.torque_limit_nm:g} Nm | friction {case.friction:g} | all parts vs floor',font=font,fill='#ffd285')
                draw.text((8,415),'Host limits: 2 rad/s, 8 rad/s^2 | NOT a hardware qualification',font=font,fill='white')
                writer.append_data(np.asarray(picture));frames+=1
                if frames in (1,150,300):picture.save(destination/f'frame_{frames:04d}.png')
    finally:
        if writer:writer.close()
        if renderer:renderer.close()
    mujoco.mj_forward(m,d)
    R=d.body('platform_mount').xmat.reshape(3,3)
    iface=servo.report()
    command_valid=True
    try:
        command_validation=check_trajectory([s['time_s'] for s in servo.command_samples],
            [s['command_native_rad_by_motor_id'] for s in servo.command_samples],
            case.velocity_limit_rad_s,case.acceleration_limit_rad_s2)
    except ValueError as exc:
        command_valid=False;command_validation={'command_validation_error':str(exc)}
    dx=float(d.qpos[0]-x0)
    # Explicit exploratory screen, not an electrical/thermal or mechanical qualification.
    screen_pass=(command_valid and stop_reason=='duration_completed' and max_tilt<20 and non_tip_samples==0 and dx>.01)
    result={'case':asdict(case),'source_sha256':source_hashes,'simulation_seconds':float(d.time),'stop_reason':stop_reason,
            'screen_pass':screen_pass,'travel_x_mm':dx*1000,'drift_y_mm':float(d.qpos[1]-y0)*1000,
            'yaw_deg':float(np.degrees(np.arctan2(R[1,0],R[0,0]))),'max_tilt_deg':max_tilt,
            'minimum_trunk_height_mm':min_height*1000,'max_sampled_contact_penetration_mm':-min_contact_distance*1000,
            'non_tip_floor_contact_samples':non_tip_samples,'monitor_samples':monitor_samples,
            'sampled_support_counts':{str(n):sum(s['supporting_tip_bodies']==n for s in body_trace) for n in range(5)},
            'max_tracking_error_deg':float(np.degrees(servo.max_tracking_error_rad)),
            'torque_saturation_fraction':iface['torque_saturation_fraction'],
            **command_validation,'frames':frames,'renderer':renderer_name,
            'qualification':'exploratory_simulation_not_hardware_ready',
            'limitations':['unmeasured actuator torque/response/supply','unmeasured encoder calibration and mechanical ROM',
                           'self collision not certified: only whole assembly vs floor enabled',
                           'friction and contact compliance are sensitivity assumptions'],
            'source_urdf_sha256':hashlib.sha256(Path(B.URDF).read_bytes()).hexdigest()}
    (destination/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (destination/'interface.json').write_text(json.dumps(iface,indent=2)+'\n')
    (destination/'body_trace.json').write_text(json.dumps(body_trace,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def screening_cases():
    base=Case()
    cases=[replace(base,name='slow_trot_095',gait='trot',torque_limit_nm=.95)]
    for gait in ('trot','crawl'):
        for torque in (.18,.3,.5):
            cases.append(replace(base,name=f'{gait}_torque_{torque:g}',gait=gait,torque_limit_nm=torque))
    cases.extend([replace(base,name='crawl_mu_04',friction=.4),
                  replace(base,name='crawl_mu_12',friction=1.2),
                  replace(base,name='crawl_stance2',stance='stance2'),
                  replace(base,name='crawl_soft_contact',contact_time_s=.008)])
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sweep',action='store_true')
    p.add_argument('--case',type=Path,help='JSON containing Case fields, or prior metrics.json')
    p.add_argument('--render',action='store_true')
    args=p.parse_args()
    if args.sweep:
        if args.case or args.render:p.error('sweep is CPU dynamics only; render a selected case separately')
        args.output.mkdir(parents=True,exist_ok=False)
        results=[run(c,args.output/c.name) for c in screening_cases()]
        (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    else:
        data=json.loads(args.case.read_text()) if args.case else {}
        case=Case(**data.get('case',data))
        run(case,args.output,args.render)


if __name__=='__main__':main()
