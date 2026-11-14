"""Sequential CPU speed comparison, retaining all successes and failures."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from locomotion_tasks import Task,run,DIRECTIONS


def cases():
    result=[]
    for direction in ('forward','backward','left','right'):
        lateral=direction in ('left','right')
        for period in (4.,3.,2.):
            result.append(Task(name=f'{direction}_period_{period:g}',direction=direction,
                planner='cartesian' if lateral else 'abd',period_s=period,stride_m=.016 if lateral else .02))
        if lateral:
            result.append(replace(result[-1],name=direction+'_period_2_stride_24mm',stride_m=.024))
    return result


def speed_metrics(metrics,trace):
    # Exclude the first 3 seconds of settling/ramp and use actual elapsed time.
    samples=[r for r in trace if r['time_s']>=3.]
    if len(samples)<2:raise ValueError('insufficient post-ramp samples')
    elapsed=samples[-1]['time_s']-samples[0]['time_s']
    delta=np.array(samples[-1]['root_xyz_m'])-np.array(samples[0]['root_xyz_m'])
    direction=DIRECTIONS[metrics['task']['direction']]
    speed=float(delta@direction/elapsed*1000)
    cross=abs(float(delta[:2]@np.array([-direction[1],direction[0]])/elapsed*1000))
    eligible=bool(metrics['task_screen_pass'] and metrics['max_tracking_error_deg']<8 and
                  metrics['max_tilt_relative_terrain_deg']<10 and speed>0 and cross<.35*speed)
    return {'post_ramp_speed_mm_s':speed,'post_ramp_cross_speed_mm_s':cross,
            'measurement_start_s':samples[0]['time_s'],'measurement_end_s':samples[-1]['time_s'],
            'eligible_for_profile':eligible}


def select_profile(results):
    selected={}
    for direction in ('forward','backward','left','right'):
        candidates=[r for r in results if r['task']['direction']==direction and r['speed']['eligible_for_profile']]
        if not candidates:raise ValueError('no eligible speed candidate: '+direction)
        best=max(candidates,key=lambda r:r['speed']['post_ramp_speed_mm_s'])
        baseline=next(r for r in results if r['task']['direction']==direction and r['task']['period_s']==4)
        selected[direction]={'period_s':best['task']['period_s'],'stride_m':best['task']['stride_m'],
            'planner':best['task']['planner'],'source_case':best['task']['name'],
            'measured_speed_mm_s':best['speed']['post_ramp_speed_mm_s'],
            'baseline_speed_mm_s':baseline['speed']['post_ramp_speed_mm_s'],
            'speedup':best['speed']['post_ramp_speed_mm_s']/baseline['speed']['post_ramp_speed_mm_s']}
    return {'qualification':'simulation_only','assumed_torque_nm':.3,'directions':selected,
            'selection':'fastest eligible measured post-ramp speed; no increase of torque or joint rate limits'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);results=[]
    for task in cases():
        destination=args.output/task.name;metrics=run(task,destination)
        metrics['speed']=speed_metrics(metrics,json.loads((destination/'trace.json').read_text()))
        results.append(metrics)
        print(json.dumps({'case':task.name,**metrics['speed']}),flush=True)
    profile=select_profile(results)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'selected_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps(profile),flush=True)


if __name__=='__main__':main()
