"""CPU render of one continuous recorded dance at real-time playback speed."""
import argparse,hashlib,json,os
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
from dance import dance_scene,segment_at,ROUTINE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    directory=args.run;output=directory/'dance.mp4'
    if output.exists():raise FileExistsError(output)
    metrics=json.loads((directory/'metrics.json').read_text());states=json.loads((directory/'states.json').read_text())
    for path,expected in metrics['source_sha256'].items():
        if hashlib.sha256((Path(B.ROOT)/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('source changed since dynamics recording: '+path)
    model,_=B.build(write=False,xml_transform=dance_scene);data=mujoco.MjData(model)
    model.vis.quality.shadowsize=256;model.vis.quality.offsamples=0
    camera=mujoco.MjvCamera();camera.distance=.64;camera.azimuth=125;camera.elevation=-28
    camera.lookat[:]=[.025,-.04,.095]
    font=ImageFont.load_default(size=13);small=ImageFont.load_default(size=11)
    heading=ImageFont.load_default(size=19);fps=10;frames=0
    xy=np.array([row['qpos'][:2] for row in states]);xy-=xy[0]
    center=np.array([513,294]);scale=min(360,35/max(.001,float(np.max(abs(xy)))))
    with imageio.get_writer(str(output),fps=fps,quality=8,macro_block_size=1,
                            ffmpeg_params=['-threads','1']) as writer:
        with mujoco.Renderer(model,width=576,height=288) as renderer:
            from OpenGL import GL
            name=GL.glGetString(GL.GL_RENDERER).decode()
            if 'llvmpipe' not in name.lower():raise RuntimeError('CPU renderer required')
            # States are captured at exactly 10 Hz. Include the initial pose;
            # the final frame is held for one frame interval (0.1 s).
            for n,row in enumerate(states):
                data.qpos[:]=row['qpos'];mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                picture=Image.new('RGB',(576,384),'#101c29');picture.paste(Image.fromarray(renderer.render()),(0,48))
                draw=ImageDraw.Draw(picture);i,label,_,local,duration=segment_at(row['time_s'])
                draw.text((12,5),'FOUR FINGERS / ONE DANCE',font=heading,fill='#e4f5ff')
                draw.text((12,29),f'{i+1:02d}  {label.upper()}   |   {row["time_s"]:05.1f} s',font=font,fill='#75e2d1')
                # Path display is measured free-root translation, not a commanded path.
                draw.rounded_rectangle((456,239,570,333),radius=8,fill='#152535',outline='#355368')
                draw.text((463,242),'Actual XY path',font=small,fill='#cadce8')
                trail=[tuple(center+[p[0]*scale,-p[1]*scale]) for p in xy[:n+1]]
                if len(trail)>1:draw.line(trail,fill='#69dbe1',width=2)
                x,y=trail[-1];draw.ellipse((x-3,y-3,x+3,y+3),fill='#ffc56c')
                draw.text((12,340),'Continuous physics | CPU | 1x playback | 0.30 Nm assumed',font=font,fill='#e4f5ff')
                draw.text((12,359),'Simulation only. Actuators uncalibrated. No real-robot test.',font=small,fill='#a6bacb')
                draw.rectangle((0,380,int(576*min(1,row['time_s']/sum(r[2] for r in ROUTINE))),383),fill='#75e2d1')
                writer.append_data(np.asarray(picture));frames+=1
                if n in (0,70,170,300,len(states)-1):picture.save(directory/f'frame_{n:04d}.png')
                if n%100==0:print('rendered',n,'/',len(states),flush=True)
    reader=imageio.get_reader(str(output));decoded=sum(1 for _ in reader);reader.close()
    if decoded!=frames:raise RuntimeError('incomplete video')
    report={'frames':frames,'decoded_frames':decoded,'fps':fps,'duration_s':frames/fps,
            'renderer':name,'playback_speed':1,'continuous_physics':True,
            'states_sha256':hashlib.sha256((directory/'states.json').read_bytes()).hexdigest(),
            'metrics_sha256':hashlib.sha256((directory/'metrics.json').read_bytes()).hexdigest()}
    (directory/'video_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
