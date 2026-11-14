import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from dance import ROUTINE,segment_at,Choreography,dance_scene
import build_model as B
import mujoco


class DanceTests(unittest.TestCase):
    def test_every_boundary_selects_next_segment_and_finishes_in_hold(self):
        t=0.
        for i,(_,_,duration) in enumerate(ROUTINE):
            self.assertEqual(segment_at(t)[0],i)
            self.assertEqual(segment_at(t+duration-1e-7)[0],i)
            t+=duration
        self.assertEqual(segment_at(t)[2],'hold')
        self.assertEqual(t,43.)

    def test_commands_are_continuous_at_all_transitions_without_state_reset(self):
        model,keys=B.build(write=False,xml_transform=dance_scene)
        data=mujoco.MjData(model);data.qpos[:]=keys['stance1']
        before=data.qpos.copy();planner=Choreography(model,keys['stance1'])
        t=0.
        for _,_,duration in ROUTINE[:-1]:
            t+=duration
            left=planner.target(t-1e-7,data);right=planner.target(t+1e-7,data)
            np.testing.assert_allclose(left,right,atol=1e-6)
        np.testing.assert_array_equal(data.qpos,before)


if __name__=='__main__':unittest.main()
