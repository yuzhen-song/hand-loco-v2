import sys,unittest,json
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
        before=data.qpos.copy()
        profile=json.loads((Path(B.ROOT)/'configs/directional_speed.json').read_text())
        for selected in (None,profile):
            planner=Choreography(model,keys['stance1'],selected)
            t=0.
            for _,_,duration in ROUTINE[:-1]:
                t+=duration
                left=planner.target(t-1e-7,data);right=planner.target(t+1e-7,data)
                np.testing.assert_allclose(left,right,atol=1e-6)
        np.testing.assert_array_equal(data.qpos,before)

    def test_selected_profile_applies_to_every_direction(self):
        model,keys=B.build(write=False)
        profile=json.loads((Path(B.ROOT)/'configs/directional_speed.json').read_text())
        planner=Choreography(model,keys['stance1'],profile)
        for direction,walker in planner.walkers.items():
            self.assertEqual(walker.task.period_s,2.)
            if direction in ('left','right'):self.assertEqual(walker.task.stride_m,.024)

    def test_profile_cannot_silently_increase_torque(self):
        model,keys=B.build(write=False)
        profile=json.loads((Path(B.ROOT)/'configs/directional_speed.json').read_text())
        profile['assumed_torque_nm']=.95
        with self.assertRaises(ValueError):Choreography(model,keys['stance1'],profile)


if __name__=='__main__':unittest.main()
