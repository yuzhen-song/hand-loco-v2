"""Sequential CPU exploration: reverse, sidestep, slope, step and jump.

Root pose is initialized once and then evolves only through MuJoCo dynamics.
Step planning uses the known terrain and perfect simulated base pose; this is
explicitly not a deployable perception/state-estimation implementation.
"""
import argparse
from dataclasses import dataclass,asdict,replace
import hashlib,json,os
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MUJOCO_GL','disable')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import mujoco
import numpy as np
import build_model as B
from hand_interface import MODEL_NAMES,SampledPositionInterface
from trajectory import check_trajectory
from sim2real import Case,contact_xml
from walk_x import build_tables,FLIP


@dataclass(frozen=True)
class Task:
    name:str='reverse'
    kind:str='walk'
    planner:str='abd'
    terrain:str='flat'
    direction:str='backward'
    seconds:float=12.
    torque_nm:float=.3
    period_s:float=4.
    stride_m:float=.02
    clearance_m:float=.008
    slope_deg:float=0.
    step_height_m:float=.005
    step_start_x_m:float=.12
    crouch_m:float=.008
    extension_m:float=.008
    push_duration_s:float=.25
    friction:float=.8

    def validate(self):
        if self.kind not in ('walk','jump') or self.planner not in ('abd','cartesian','terrain_abd'):
            raise ValueError('unsupported task/planner')
        if self.terrain not in ('flat','slope','step') or self.direction not in DIRECTIONS:
            raise ValueError('unsupported terrain/direction')
        for k,v in asdict(self).items():
            if isinstance(v,(int,float)) and not np.isfinite(v):raise ValueError(k+' must be finite')
        if not (5<=self.seconds<=30 and 0<self.torque_nm<=.95 and self.period_s>=2):
            raise ValueError('unsupported duration/torque/period')
        if not (0<self.stride_m<=.06 and 0<self.clearance_m<=.04 and abs(self.slope_deg)<=15):
            raise ValueError('unsupported stride/clearance/slope')
        if not (0<self.step_height_m<=.03 and .105<=self.step_start_x_m<=.3):
            raise ValueError('unsupported step geometry')
        if not (0<self.crouch_m<=.03 and 0<self.extension_m<=.025 and .1<=self.push_duration_s<=1 and 0<self.friction<=2):
            raise ValueError('unsupported jump/friction hypothesis')
        if self.terrain!='flat' and (self.kind!='walk' or self.direction!='forward'):
            raise ValueError('terrain tests currently use forward walking only')
        if self.terrain=='step' and self.planner not in ('cartesian','terrain_abd'):raise ValueError('step needs terrain-aware planner')
        if self.planner=='terrain_abd' and self.terrain!='step':raise ValueError('terrain_abd requires a step')
        if self.planner=='abd' and self.direction in ('left','right') and self.kind=='walk':
            raise ValueError('sidestep requires Cartesian IK')
        if Path(self.name).name!=self.name or self.name in ('.','..'):raise ValueError('invalid name')


DIRECTIONS={'forward':np.array([1.,0.,0.]),'backward':np.array([-1.,0.,0.]),
            'left':np.array([0.,1.,0.]),'right':np.array([0.,-1.,0.])}
PHASE={'A':0.,'D':0.,'B':.5,'C':.5}


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def height(x,task):
    if task.terrain=='step':return task.step_height_m if x>=task.step_start_x_m else 0.
    if task.terrain=='slope':return np.tan(np.deg2rad(task.slope_deg))*x
    return 0.


def scene_xml(xml,task):
    root=ET.fromstring(contact_xml(xml,replace(Case(),friction=task.friction)))
    floor=root.find(".//geom[@name='floor']")
    if task.terrain=='slope':
        a=np.deg2rad(task.slope_deg)/2
        floor.set('quat',f'{np.cos(a)} 0 {-np.sin(a)} 0')
    if task.terrain=='step':
        ET.SubElement(root.find('worldbody'),'geom',{
            'name':'terrain_step','type':'box','size':f'0.5 0.6 {task.step_height_m/2}',
            'pos':f'{task.step_start_x_m+.5} 0 {task.step_height_m/2}',
            'contype':'2','conaffinity':'1','condim':'3','friction':f'{task.friction} 0.005 0.0001',
            'solref':'0.004 1','solimp':'0.95 0.99 0.001 0.5 2','rgba':'0.45 0.6 0.7 1'})
    return ET.tostring(root,encoding='unicode')


class FeetIK:
    """Four bounded local 3D Jacobian IK chains; no changes to the dynamic data."""
    def __init__(self,m,base):
        self.m=m;self.base=base.copy();self.d=mujoco.MjData(m);self.d.qpos[:]=base
        mujoco.mj_forward(m,self.d)
        self.qadr={n:int(m.jnt_qposadr[m.joint(n).id]) for n in MODEL_NAMES}
        self.idx=np.array(list(self.qadr.values()))
        self.didx=np.array([m.jnt_dofadr[m.joint(n).id] for n in MODEL_NAMES])
        self.bounds=np.array([m.jnt_range[m.joint(n).id] for n in MODEL_NAMES])
        self.points={};self.local={};self.ids={};self.max_residual=0.;self.solves=0
        for leg in B.LEGS:
            body=m.body(B.TIP_BODY[leg]).id;self.ids[leg]=body
            points=B.tip_points(m,self.d,leg);p=points[np.argmin(points[:,2])].copy()
            self.points[leg]=p
            self.local[leg]=self.d.xmat[body].reshape(3,3).T@(p-self.d.xpos[body])
        self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv))

    def point(self,leg):
        bid=self.ids[leg]
        return self.d.xpos[bid]+self.d.xmat[bid].reshape(3,3)@self.local[leg]

    def solve(self,leg,target,seed,root_pose=None,iterations=35):
        li=B.LEGS.index(leg);qa=self.idx[4*li:4*li+4];da=self.didx[4*li:4*li+4]
        limits=self.bounds[4*li:4*li+4];reference=self.base[qa]
        q=np.clip(np.asarray(seed),limits[:,0]+1e-5,limits[:,1]-1e-5)
        self.d.qpos[:]=self.base
        if root_pose is not None:self.d.qpos[:7]=root_pose
        error=np.inf
        for _ in range(iterations):
            self.d.qpos[qa]=q;mujoco.mj_forward(self.m,self.d)
            point=self.point(leg);residual=np.asarray(target)-point
            error=float(np.linalg.norm(residual))
            if error<.0001:break
            mujoco.mj_jac(self.m,self.d,self.jp,self.jr,point,self.ids[leg])
            J=self.jp[:,da]
            pinv=J.T@np.linalg.solve(J@J.T+np.eye(3)*1e-6,np.eye(3))
            dq=pinv@residual+.02*(np.eye(4)-pinv@J)@(reference-q)
            q=np.clip(q+np.clip(dq,-.08,.08),limits[:,0]+1e-5,limits[:,1]-1e-5)
        self.d.qpos[qa]=q;mujoco.mj_forward(self.m,self.d)
        error=float(np.linalg.norm(np.asarray(target)-self.point(leg)))
        self.max_residual=max(self.max_residual,error);self.solves+=1
        return q,error


def cartesian_offset(phase,task):
    if phase<.5:
        u=phase/.5;distance=task.stride_m*(.5-u);lift=0.
    else:
        u=(phase-.5)/.5;distance=task.stride_m*(-.5+smooth(u))
        lift=task.clearance_m*np.sin(np.pi*u)
    return DIRECTIONS[task.direction]*distance+np.array([0,0,lift])


class Planner:
    def __init__(self,m,base,task):
        self.task=task;self.ik=FeetIK(m,base);self.base=base.copy()
        self.q=base[self.ik.idx].copy();self.max_residual=0.;self.tables={}
        self.abd_tables=None
        if task.kind=='walk' and task.planner in ('abd','terrain_abd'):
            self.abd_tables=build_tables(m,mujoco.MjData(m),base,self.ik.qadr,
                                        base[self.ik.qadr['unitA-MCP-FLX']],-.6,.6)
        elif task.kind=='walk' and task.terrain=='flat':
            self.phases=np.linspace(0,1,129)
            for leg in B.LEGS:
                li=B.LEGS.index(leg);seed=base[self.ik.idx[4*li:4*li+4]];rows=[]
                for p in self.phases:
                    target=self.ik.points[leg]+cartesian_offset(p%1,task)
                    seed,error=self.ik.solve(leg,target,seed)
                    rows.append(seed.copy())
                self.tables[leg]=np.array(rows)
        self.last_target=self.q.copy()

    def target(self,t,actual):
        task=self.task
        if t<1:return self.q.copy()
        tau=t-1;blend=smooth(tau/2)
        result=self.q.copy()
        for li,leg in enumerate(B.LEGS):
            sl=slice(4*li,4*li+4);seed=self.last_target[sl]
            if task.kind=='jump':
                if t<2:dz=task.crouch_m*smooth(t-1)
                elif t<2+task.push_duration_s:
                    dz=task.crouch_m-(task.crouch_m+task.extension_m)*smooth((t-2)/task.push_duration_s)
                elif t<3.5:dz=-task.extension_m
                elif t<4.5:dz=-task.extension_m*(1-smooth(t-3.5))
                else:dz=0.
                target=self.ik.points[leg]+[0,0,dz]
                result[sl],_=self.ik.solve(leg,target,seed)
                continue
            phase=(tau/task.period_s+PHASE[leg])%1
            if task.planner in ('abd','terrain_abd'):
                swing=phase>=.5;u=phase/.5 if not swing else 1-smooth((phase-.5)/.5)
                abd=(-.6+1.2*u)*(-1 if FLIP[leg] else 1)*DIRECTIONS[task.direction][0]
                table=self.abd_tables[leg]
                lift=.08*np.sin(np.pi*(phase-.5)/.5) if swing else 0.
                raw=np.array([self.q[4*li]+lift,abd,
                              np.interp(abd,table[:,0],table[:,1]),np.interp(abd,table[:,0],table[:,2])])
                result[sl]=self.q[sl]+blend*(raw-self.q[sl])
                if task.planner=='terrain_abd':
                    # Preserve the established steering gait. Only solve a height
                    # correction in a nominal body frame, instead of chasing the
                    # sagging actual root with ground-anchored position targets.
                    self.ik.d.qpos[:]=self.base
                    self.ik.d.qpos[self.ik.idx[sl]]=result[sl]
                    mujoco.mj_forward(self.ik.m,self.ik.d)
                    target=self.ik.point(leg).copy()
                    shift=actual.qpos[0]-self.base[0]
                    ground=height(target[0]+shift,task)
                    if swing:
                        ground=max(ground,height(self.ik.points[leg][0]+shift+.05,task))
                    target[2]+=ground+(task.clearance_m*np.sin(np.pi*(phase-.5)/.5) if swing else 0.)
                    root=self.base[:7].copy()
                    root[2]+=np.mean([height(p[0]+shift,task) for p in self.ik.points.values()])
                    result[sl],_=self.ik.solve(leg,target,seed,root)
            elif task.terrain=='flat':
                raw=np.array([np.interp(phase,self.phases,self.tables[leg][:,j]) for j in range(4)])
                result[sl]=self.q[sl]+blend*(raw-self.q[sl])
            else:
                # Perfect simulated base pose and known height map are explicit oracle inputs.
                offset=cartesian_offset(phase,task)*blend
                neutral=self.ik.points[leg]
                target=neutral+offset
                target[:2]+=actual.qpos[:2]-self.base[:2]
                ground=height(target[0],task)
                if phase>=.5:
                    landing_x=neutral[0]+actual.qpos[0]-self.base[0]+task.stride_m*.5
                    ground=max(ground,height(landing_x,task))
                target[2]=ground+neutral[2]+offset[2]
                result[sl],_=self.ik.solve(leg,target,seed,actual.qpos[:7])
        self.last_target=result.copy();return result


def jump_event_pass(event):
    return bool(event.get('landed') and event['takeoff_com_velocity_z_m_s']>.02 and
                event['duration_s']>=.04 and event['com_rise_m']>=.003 and
                event['max_min_tip_clearance_m']>=.002)


def run(task,output):
    task.validate();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources=['tools/locomotion_tasks.py','tools/sim2real.py','tools/build_model.py','tools/walk_x.py',
             'tools/trajectory.py','tools/hand_interface.py','configs/xc330_hand.json']
    hashes={p:hashlib.sha256((Path(B.ROOT)/p).read_bytes()).hexdigest() for p in sources}
    m,keys=B.build(write=False,xml_transform=lambda xml:scene_xml(xml,task))
    base=keys['stance1'].copy();planner=Planner(m,base,task)
    d=mujoco.MjData(m);d.qpos[:]=base
    normal=np.array([0.,0.,1.]);tangent=DIRECTIONS[task.direction].copy()
    if task.terrain=='slope':
        a=np.deg2rad(task.slope_deg);c,s=np.cos(a),np.sin(a)
        R=np.array([[c,0,-s],[0,1,0],[s,0,c]])
        d.qpos[:3]=R@base[:3];d.qpos[3:7]=[0,np.cos(a/2),0,np.sin(a/2)]
        normal=R@normal;tangent=R@tangent
    mujoco.mj_forward(m,d)
    terrain_ids={m.geom('floor').id}
    if task.terrain=='step':terrain_ids.add(m.geom('terrain_step').id)
    tip_ids=set(planner.ik.ids.values());idx=planner.ik.idx;didx=planner.ik.didx
    servo=SampledPositionInterface(d.qpos[idx],m.opt.timestep,task.torque_nm,60,3,
                                  rate_limits=(2,8),joint_limits=planner.ik.bounds)
    initial=d.qpos[:3].copy();dt=float(m.opt.timestep);previous_com=d.subtree_com[0].copy()
    trace=[];states=[];max_tilt=0.;max_world_tilt=0.;non_tip=False;stop='duration_completed'
    peak_force=0.;max_pen=0.;events=[];air=None;previous_contact=False;step_clear_time=0.;step_clear_max=0.
    requested=d.qpos[idx].copy()
    for k in range(round(task.seconds/dt)):
        if k%10==0:requested=planner.target(k*dt,d)
        try:d.ctrl[:]=servo.step(requested,d.qpos[idx],d.qvel[didx])
        except ValueError as exc:stop='command_rejected: '+str(exc);break
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            stop='nonfinite_state';break
        com=d.subtree_com[0].copy();com_v=(com-previous_com)/dt;previous_com=com
        up=-d.body('platform_mount').xmat.reshape(3,3)[:,2]
        tilt=float(np.degrees(np.arccos(np.clip(up@normal,-1,1))))
        world_tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))))
        max_tilt=max(max_tilt,tilt);max_world_tilt=max(max_world_tilt,world_tilt)
        contacts=0;support=set();bad=0;force=0.;step_support=set()
        for con in d.contact:
            if con.geom1 not in terrain_ids and con.geom2 not in terrain_ids:continue
            terrain=con.geom1 if con.geom1 in terrain_ids else con.geom2
            other=con.geom2 if terrain==con.geom1 else con.geom1
            if other in terrain_ids:continue
            body=int(m.geom_bodyid[other]);contacts+=1
            if body in tip_ids:support.add(body)
            else:bad+=1
            if task.terrain=='step' and terrain==m.geom('terrain_step').id and body in tip_ids:
                step_support.add(body)
            max_pen=max(max_pen,-float(con.dist))
        if task.kind=='jump':
            for i,con in enumerate(d.contact):
                if con.geom1 in terrain_ids or con.geom2 in terrain_ids:
                    f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
                    force+=max(0.,float(f[0]))*abs(float(con.frame[2]))
            peak_force=max(peak_force,force)
            if d.time>=2 and not contacts and previous_contact:
                air={'start_s':float(d.time),'start_com_z_m':float(com[2]),
                     'takeoff_com_velocity_z_m_s':float(com_v[2]),'duration_s':0.,'com_rise_m':0.,
                     'max_min_tip_clearance_m':0.,'landed':False}
            if air is not None:
                air['duration_s']=float(d.time)-air['start_s']
                air['com_rise_m']=max(air['com_rise_m'],float(com[2])-air['start_com_z_m'])
                clearance=min(float(B.tip_points(m,d,leg)[:,2].min()) for leg in B.LEGS)
                air['max_min_tip_clearance_m']=max(air['max_min_tip_clearance_m'],clearance)
                if contacts:
                    air['landed']=True;air['landing_s']=float(d.time);events.append(air);air=None
        previous_contact=contacts>0
        if task.terrain=='step':
            points=[B.tip_points(m,d,leg) for leg in B.LEGS]
            all_across=all(float(p[:,0].min())>task.step_start_x_m+.002 for p in points)
            above=all(float(p[:,2].min())>task.step_height_m-.002 for p in points)
            clear=all_across and above and len(step_support)>0 and tilt<20 and not bad
            step_clear_time=step_clear_time+dt if clear else 0.
            step_clear_max=max(step_clear_max,step_clear_time)
        if k%10==0:
            trace.append({'time_s':float(d.time),'root_xyz_m':d.qpos[:3].tolist(),'com_xyz_m':com.tolist(),
                          'com_velocity_z_m_s':float(com_v[2]),'tilt_relative_terrain_deg':tilt,
                          'supporting_tips':len(support),'non_tip_contacts':bad,'step_supporting_tips':len(step_support),
                          'ground_normal_vertical_force_n':force if task.kind=='jump' else None})
        if k%40==0:states.append({'time_s':float(d.time),'qpos':d.qpos.tolist()})
        if bad:non_tip=True;stop='non_tip_terrain_contact';break
        if tilt>45:stop='tilt_exceeded_45_deg';break
    if air is not None:events.append(air)
    states.append({'time_s':float(d.time),'qpos':d.qpos.tolist()})
    displacement=d.qpos[:3]-initial;progress=float(displacement@tangent)
    cross=float(np.dot(displacement[:2],[-tangent[1],tangent[0]]))
    rate_ok=True
    try:
        rates=check_trajectory([r['time_s'] for r in servo.command_samples],
                              [r['command_native_rad_by_motor_id'] for r in servo.command_samples],2,8)
    except ValueError as exc:rates={'error':str(exc)};rate_ok=False
    completed=stop=='duration_completed';tracking=float(np.degrees(servo.max_tracking_error_rad))
    airborne=any(jump_event_pass(e) for e in events)
    recovered=(completed and max([r['tilt_relative_terrain_deg'] for r in trace if r['time_s']>task.seconds-1] or [180])<20)
    directional=(completed and rate_ok and progress>.02 and abs(cross)<max(.02,.5*progress) and max_tilt<20 and not non_tip)
    if task.kind=='jump':passed=airborne and recovered and rate_ok and not non_tip
    elif task.terrain=='step':passed=directional and step_clear_max>=.2
    else:passed=directional
    if planner.ik.max_residual>.002:passed=False
    R=d.body('platform_mount').xmat.reshape(3,3)
    metrics={'task':asdict(task),'source_sha256':hashes,'simulation_seconds':float(d.time),'stop_reason':stop,
             'task_screen_pass':bool(passed),'progress_mm':progress*1000,'cross_drift_mm':cross*1000,
             'displacement_xyz_mm':(displacement*1000).tolist(),'final_yaw_deg':float(np.degrees(np.arctan2(R[1,0],R[0,0]))),
             'max_tilt_relative_terrain_deg':max_tilt,'max_world_tilt_deg':max_world_tilt,
             'max_tracking_error_deg':tracking,'torque_saturation_fraction':servo.report()['torque_saturation_fraction'],
             'max_ik_residual_mm':planner.ik.max_residual*1000,'max_contact_penetration_mm':max_pen*1000,
             'step_all_feet_clear_duration_s':step_clear_max,'airborne_events':events,
             'jump_airborne_screen_pass':airborne,'peak_ground_vertical_normal_force_n':peak_force,
             'robot_weight_n':float(m.body_subtreemass[0]*np.linalg.norm(m.opt.gravity)),
             'command_bounds':rates,'starts_on_slope':task.terrain=='slope',
             'root_motion':'free dynamics; root initialized once, never prescribed during rollout',
             'state_information':'perfect simulated root pose and known terrain map for step IK; otherwise timed joint commands',
             'qualification':'simulation_exploration_only_not_hardware_validated',
             'limitations':['unmeasured torque/current relationship and servo response','unmeasured mechanical ROM and encoders',
                            'robot self-collision disabled; terrain contacts use CAD convex hulls','single deterministic exploratory trial']}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (output/'interface.json').write_text(json.dumps(servo.report(),indent=2)+'\n')
    (output/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    (output/'states.json').write_text(json.dumps(states)+'\n')
    print(json.dumps({k:metrics[k] for k in ('task_screen_pass','stop_reason','progress_mm','cross_drift_mm','max_tracking_error_deg','max_ik_residual_mm','step_all_feet_clear_duration_s','jump_airborne_screen_pass')}|{'name':task.name}),flush=True)
    return metrics


def tasks():
    return [Task(name='01_reverse'),
            Task(name='02_left',direction='left',planner='cartesian',stride_m=.016),
            Task(name='03_right',direction='right',planner='cartesian',stride_m=.016),
            *[Task(name=f'04_slope_{slope:g}deg',direction='forward',terrain='slope',slope_deg=slope) for slope in (3.,5.,10.)],
            *[Task(name=f'05_step_{h*1000:g}mm',direction='forward',terrain='step',planner='cartesian',seconds=20,
                   step_height_m=h,stride_m=.024,clearance_m=.012) for h in (.005,.01,.02)],
            Task(name='06_jump_030',kind='jump',seconds=6,torque_nm=.3),
            Task(name='07_jump_050',kind='jump',seconds=6,torque_nm=.5,crouch_m=.015,extension_m=.01),
            Task(name='08_jump_095',kind='jump',seconds=6,torque_nm=.95,crouch_m=.015,extension_m=.015)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',type=Path);p.add_argument('--suite',action='store_true');args=p.parse_args()
    if args.suite:
        if args.case:p.error('choose either suite or case')
        args.output.mkdir(parents=True,exist_ok=False)
        results=[run(task,args.output/task.name) for task in tasks()]
        (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    else:
        data=json.loads(args.case.read_text()) if args.case else {}
        run(Task(**data.get('task',data)),args.output)


if __name__=='__main__':main()
