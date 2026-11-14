"""A continuous CPU-simulated four-finger dance, with no root resets or hardware I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MUJOCO_GL','disable')
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import mujoco
import numpy as np
import build_model as B
from locomotion_tasks import Task,Planner,FeetIK,scene_xml,smooth
from hand_interface import SampledPositionInterface
from trajectory import check_trajectory

# Durations are physical simulation seconds, not edited video time.
ROUTINE=[('Ready','hold',1.),('Hello dip','dip',4.),
         ('Backward shuffle','backward',8.),('Slide left','left',8.),
         ('Slide right','right',8.),('Forward shuffle','forward',8.),
         ('Final dip','dip',4.),('Finish','hold',2.)]
COLORS=[(.32,.78,1.,1.),(1.,.65,.26,1.),(.72,.48,1.,1.),(.36,.95,.65,1.)]


def segment_at(t):
    start=0.
    for i,(label,kind,duration) in enumerate(ROUTINE):
        if t<start+duration or i==len(ROUTINE)-1:
            return i,label,kind,max(0.,min(t-start,duration)),duration
        start+=duration
    raise AssertionError('empty routine')


def dance_scene(xml):
    root=ET.fromstring(scene_xml(xml,Task()))
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',{'name':'dance_grid','type':'2d','builtin':'checker',
                                 'rgb1':'.08 .12 .17','rgb2':'.16 .22 .29','width':'256','height':'256'})
    ET.SubElement(asset,'material',{'name':'dance_floor','texture':'dance_grid',
                                  'texrepeat':'16 16','texuniform':'true','reflectance':'.05'})
    floor=root.find(".//geom[@name='floor']");floor.set('material','dance_floor');floor.set('rgba','1 1 1 1')
    for leg,color in zip(B.LEGS,COLORS):
        body=root.find(".//body[@name='"+B.TIP_BODY[leg]+"']")
        for geom in body.findall('geom'):geom.set('rgba',' '.join(map(str,color)))
    return ET.tostring(root,encoding='unicode')


class Choreography:
    def __init__(self,m,base,profile=None):
        self.ik=FeetIK(m,base);self.neutral=base[self.ik.idx].copy();self.seed=self.neutral.copy()
        self.walkers={}
        if profile is not None:
            if profile.get('qualification')!='simulation_only' or profile.get('assumed_torque_nm')!=.3:
                raise ValueError('expected simulation-only profile with unchanged 0.3 Nm assumption')
            if set(profile.get('directions',{}))!={'backward','left','right','forward'}:
                raise ValueError('profile must define all four directions')
        for direction in ('backward','left','right','forward'):
            params={'planner':'cartesian' if direction in ('left','right') else 'abd',
                    'stride_m':.016,'period_s':4.}
            if profile is not None:
                params={k:profile['directions'][direction][k] for k in params}
            task=Task(direction=direction,**params);task.validate()
            self.walkers[direction]=Planner(m,base,task)

    def target(self,t,data):
        _,label,kind,local,duration=segment_at(t)
        q=self.neutral.copy()
        if kind=='dip':
            # A slow 6 mm relative fingertip lift produces a physical body dip.
            # The final second holds neutral for recovery before the next section.
            phase=min(local/3.,1.)
            lift=.006*np.sin(np.pi*phase)**2
            for i,leg in enumerate(B.LEGS):
                sl=slice(i*4,i*4+4)
                q[sl],_=self.ik.solve(leg,self.ik.points[leg]+[0,0,lift],self.seed[sl])
            self.seed=q.copy()
        elif kind!='hold':
            raw=self.walkers[kind].target(local,data)
            # Fade into neutral during the last second; the same rate governor
            # continues across every segment boundary without resetting velocity.
            blend=smooth((duration-local)/1.)
            q=self.neutral+blend*(raw-self.neutral)
        return q

    @property
    def max_ik_residual(self):
        return max([self.ik.max_residual]+[p.ik.max_residual for p in self.walkers.values()])


def run(output,profile=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m,keys=B.build(write=False,xml_transform=dance_scene);base=keys['stance1'].copy()
    dance=Choreography(m,base,profile);ik=dance.ik
    d=mujoco.MjData(m);d.qpos[:]=base;mujoco.mj_forward(m,d)
    assert np.array_equal(m.actuator_trnid[:,0],[m.joint(n).id for n in ik.qadr])
    servo=SampledPositionInterface(base[ik.idx],m.opt.timestep,.3,60,3,
                                  rate_limits=(2,8),joint_limits=ik.bounds)
    floor=m.geom('floor').id;tips=set(ik.ids.values());dt=float(m.opt.timestep)
    seconds=sum(r[2] for r in ROUTINE);stop='duration_completed';max_tilt=0.;max_pen=0.
    states=[{'time_s':0.,'qpos':d.qpos.tolist(),'segment':0}];trace=[];sections=[]
    old_segment=-1;start_xyz=d.qpos[:3].copy();start_com=d.subtree_com[0].copy()
    section_min_z=float(start_com[2]);section_max_z=float(start_com[2]);requested=base[ik.idx].copy()
    for k in range(round(seconds/dt)):
        i,label,kind,local,duration=segment_at(k*dt)
        if i!=old_segment:
            if old_segment>=0:
                sections[-1].update(displacement_xyz_mm=((d.qpos[:3]-start_xyz)*1000).tolist(),
                    com_height_range_mm=(section_max_z-section_min_z)*1000)
                print(json.dumps(sections[-1]),flush=True)
            sections.append({'index':i,'label':label,'kind':kind,'start_s':float(d.time),'requested_duration_s':duration})
            start_xyz=d.qpos[:3].copy();section_min_z=section_max_z=float(d.subtree_com[0,2]);old_segment=i
        if k%10==0:requested=dance.target(k*dt,d)
        try:d.ctrl[:]=servo.step(requested,d.qpos[ik.idx],d.qvel[ik.didx])
        except ValueError as exc:stop='command_rejected: '+str(exc);break
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            stop='nonfinite_state';break
        up=-d.body('platform_mount').xmat.reshape(3,3)[:,2]
        tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))));max_tilt=max(max_tilt,tilt)
        support=set();bad=0
        for contact in d.contact:
            if floor not in (contact.geom1,contact.geom2):continue
            other=contact.geom2 if contact.geom1==floor else contact.geom1
            body=int(m.geom_bodyid[other]);max_pen=max(max_pen,-float(contact.dist))
            if body in tips:support.add(body)
            else:bad+=1
        z=float(d.subtree_com[0,2]);section_min_z=min(section_min_z,z);section_max_z=max(section_max_z,z)
        if k%10==9:
            trace.append({'time_s':float(d.time),'segment':i,'root_xyz_m':d.qpos[:3].tolist(),
                          'com_height_m':z,'tilt_deg':tilt,'supporting_tips':len(support),'non_tip_contacts':bad})
        if k%50==49:states.append({'time_s':float(d.time),'qpos':d.qpos.tolist(),'segment':i})
        if bad:stop='non_tip_terrain_contact';break
        if tilt>45:stop='tilt_exceeded_45_deg';break
    sections[-1].update(displacement_xyz_mm=((d.qpos[:3]-start_xyz)*1000).tolist(),
                        com_height_range_mm=(section_max_z-section_min_z)*1000)
    if states[-1]['time_s']<d.time-1e-8:states.append({'time_s':float(d.time),'qpos':d.qpos.tolist(),'segment':old_segment})
    rate_ok=True
    try:rates=check_trajectory([r['time_s'] for r in servo.command_samples],
        [r['command_native_rad_by_motor_id'] for r in servo.command_samples],2,8)
    except ValueError as exc:rates={'error':str(exc)};rate_ok=False
    direction_axis={'backward':(0,-1),'forward':(0,1),'left':(1,1),'right':(1,-1)}
    for s in sections:
        if s['kind'] in direction_axis:
            axis,sign=direction_axis[s['kind']]
            s['requested_direction_progress_mm']=s['displacement_xyz_mm'][axis]*sign
    directions_ok=all(s.get('requested_direction_progress_mm',1)>0 for s in sections)
    # This checks sequence completion and directional signs, not precise choreography.
    passed=bool(stop=='duration_completed' and rate_ok and max_tilt<20 and
                dance.max_ik_residual<.002 and directions_ok and len(sections)==len(ROUTINE))
    sources=['tools/dance.py','tools/locomotion_tasks.py','tools/build_model.py','tools/sim2real.py',
             'tools/hand_interface.py','tools/trajectory.py','tools/walk_x.py','configs/xc330_hand.json',
             'model/full_hand_final/robot.urdf']
    result={'speed_profile':profile,'screen_pass':passed,'stop_reason':stop,'simulation_seconds':float(d.time),'segments':sections,
            'max_tilt_deg':max_tilt,'max_ik_residual_mm':dance.max_ik_residual*1000,
            'max_tracking_error_deg':float(np.degrees(servo.max_tracking_error_rad)),
            'torque_saturation_fraction':servo.report()['torque_saturation_fraction'],
            'max_contact_penetration_mm':max_pen*1000,'command_bounds':rates,
            'root_displacement_xyz_mm':((d.qpos[:3]-base[:3])*1000).tolist(),
            'assumed_torque_limit_nm':.3,'physics_timestep_s':dt,'root_resets_during_sequence':0,
            'source_sha256':{p:hashlib.sha256((Path(B.ROOT)/p).read_bytes()).hexdigest() for p in sources},
            'qualification':'continuous_free_dynamics_simulation_only_not_hardware_validated',
            'limitations':['self-collision disabled; CAD convex hull terrain contacts',
                          'unmeasured servo response, supply voltage and physical calibration',
                          'timed open-loop choreography, not precise tracking or return-to-start control']}
    for name,obj in [('metrics',result),('states',states),('trace',trace),('interface',servo.report())]:
        (output/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--profile',type=Path,default=Path(B.ROOT)/'configs/directional_speed.json')
    group.add_argument('--baseline',action='store_true',help='use the original 4-second gait and 16 mm lateral stride')
    args=p.parse_args()
    profile=None if args.baseline else json.loads(args.profile.read_text())
    run(args.output,profile)


if __name__=='__main__':main()
