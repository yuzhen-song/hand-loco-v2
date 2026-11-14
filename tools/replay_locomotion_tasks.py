"""CPU-only video of recorded dynamics. No controller or physics is rerun."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MUJOCO_GL']='osmesa'
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
os.environ['GALLIUM_DRIVER']='llvmpipe'
os.environ['LP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_model as B
from locomotion_tasks import Task,scene_xml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    records=[];frames=0;fps=12.5
    font=ImageFont.load_default(size=12)
    with imageio.get_writer(str(args.output),fps=fps,quality=7,macro_block_size=1,
                            ffmpeg_params=['-threads','1']) as writer:
        for directory in args.runs:
            metrics=json.loads((directory/'metrics.json').read_text())
            states=json.loads((directory/'states.json').read_text())
            task=Task(**metrics['task']);task.validate()
            for source in ('tools/build_model.py','tools/sim2real.py'):
                actual=hashlib.sha256((Path(B.ROOT)/source).read_bytes()).hexdigest()
                if actual!=metrics['source_sha256'][source]:raise ValueError('model builder changed: '+source)
            model,_=B.build(write=False,xml_transform=lambda xml:scene_xml(xml,task))
            data=mujoco.MjData(model)
            model.vis.quality.shadowsize=256;model.vis.quality.offsamples=0
            camera=mujoco.MjvCamera();camera.distance=.82;camera.azimuth=125;camera.elevation=-25
            camera.lookat[:]=[.10,-.04,.09]
            first_frame=frames
            # Uniform 0.16 s sampling at 12.5 fps gives 2x replay speed.
            times=np.array([s['time_s'] for s in states])
            with mujoco.Renderer(model,height=256,width=480) as renderer:
                from OpenGL import GL
                renderer_name=GL.glGetString(GL.GL_RENDERER).decode()
                if 'llvmpipe' not in renderer_name.lower():raise RuntimeError('expected CPU llvmpipe renderer')
                for stamp in np.arange(times[0],times[-1]+1e-8,.16):
                    row=states[int(np.argmin(abs(times-stamp)))];data.qpos[:]=row['qpos']
                    mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
                    picture=Image.new('RGB',(480,320),'#14212d');picture.paste(Image.fromarray(renderer.render()),(0,32))
                    draw=ImageDraw.Draw(picture)
                    draw.text((6,2),task.name+' | recorded simulation | 2x',font=font,fill='white')
                    draw.text((6,17),f"t={row['time_s']:.2f}s | CPU | assumed torque {task.torque_nm:g} Nm",font=font,fill='#b9cede')
                    label='PASS (narrow simulation screen)' if metrics['task_screen_pass'] else 'NOT PASSED'
                    draw.text((6,290),label+f" | progress {metrics['progress_mm']:.1f} mm",font=font,fill='#ffd285')
                    draw.text((6,305),'Uncalibrated actuators. No real-robot validation.',font=font,fill='white')
                    writer.append_data(np.asarray(picture));frames+=1
            records.append({'run':str(directory),'frames':frames-first_frame,'metrics_sha256':hashlib.sha256((directory/'metrics.json').read_bytes()).hexdigest(),
                            'states_sha256':hashlib.sha256((directory/'states.json').read_bytes()).hexdigest()})
            print(task.name,frames-first_frame,'frames',flush=True)
    decoded=sum(1 for _ in imageio.get_reader(str(args.output)))
    if decoded!=frames:raise RuntimeError('video frame count mismatch')
    report={'frames':frames,'decoded_frames':decoded,'fps':fps,'duration_s':frames/fps,
            'renderer':renderer_name,'runs':records,'mode':'recorded free-dynamics replay, approximately 2x speed'}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
