import sys,unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from locomotion_tasks import Task,scene_xml,height,cartesian_offset,jump_event_pass


class LocomotionTaskTests(unittest.TestCase):
    def test_step_height_and_geometry(self):
        task=Task(direction='forward',terrain='step',planner='cartesian',step_height_m=.01)
        task.validate()
        self.assertEqual(height(.119,task),0.)
        self.assertEqual(height(.12,task),.01)
        xml='<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body><geom type="sphere" size=".01" group="0"/><geom type="sphere" size=".01" group="1"/></body></worldbody></mujoco>'
        tree=ET.fromstring(scene_xml(xml,task));step=tree.find(".//geom[@name='terrain_step']")
        self.assertEqual(step.get('contype'),'2');self.assertEqual(step.get('conaffinity'),'1')
        position=list(map(float,step.get('pos').split()));size=list(map(float,step.get('size').split()))
        self.assertAlmostEqual(position[0]-size[0],task.step_start_x_m)
        self.assertAlmostEqual(position[2]+size[2],task.step_height_m)
    def test_slope_geometry_normal(self):
        task=Task(terrain='slope',direction='forward',slope_deg=5)
        xml='<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/></worldbody></mujoco>'
        tree=ET.fromstring(scene_xml(xml,task));quat=np.array(list(map(float,tree.find('.//geom').get('quat').split())))
        import mujoco
        normal=np.zeros(3);mujoco.mju_rotVecQuat(normal,np.array([0.,0.,1.]),quat)
        self.assertAlmostEqual(normal[0],-np.sin(np.deg2rad(5)))
        self.assertAlmostEqual(normal@np.array([.2,0,height(.2,task)]),0.)
    def test_lateral_target_changes_y_not_x(self):
        for direction,sign in [('left',1),('right',-1)]:
            task=Task(direction=direction,planner='cartesian')
            p=cartesian_offset(0,task);self.assertEqual(p[0],0)
            self.assertGreater(p[1]*sign,0)
            np.testing.assert_allclose(cartesian_offset(0,task),cartesian_offset(1,task),atol=1e-12)
            self.assertGreater(cartesian_offset(.75,task)[2],0)
    def test_jump_rejects_initial_fall_and_contact_flicker(self):
        event={'landed':True,'takeoff_com_velocity_z_m_s':.1,'duration_s':.1,'com_rise_m':.01,'max_min_tip_clearance_m':.01}
        self.assertTrue(jump_event_pass(event))
        for field,value in [('takeoff_com_velocity_z_m_s',-.1),('duration_s',.002),('com_rise_m',0.),('max_min_tip_clearance_m',0.),('landed',False)]:
            e=event.copy();e[field]=value;self.assertFalse(jump_event_pass(e))
    def test_ik_reaches_small_height_change_without_mutating_rollout(self):
        import mujoco
        import build_model as B
        from locomotion_tasks import FeetIK
        model,keys=B.build(write=False)
        base=keys['stance1'].copy();data=mujoco.MjData(model);data.qpos[:]=base
        before=data.qpos.copy();ik=FeetIK(model,base)
        for i,leg in enumerate(B.LEGS):
            target=ik.points[leg]+np.array([0.,0.,.003])
            q,error=ik.solve(leg,target,base[ik.idx[4*i:4*i+4]])
            self.assertLess(error,.0002)
            bounds=ik.bounds[4*i:4*i+4]
            self.assertTrue(np.all(q>=bounds[:,0]) and np.all(q<=bounds[:,1]))
        np.testing.assert_array_equal(data.qpos,before)
        np.testing.assert_array_equal(base,keys['stance1'])

    def test_invalid_task_rejected(self):
        for task in [Task(direction='left'),Task(torque_nm=1.2),Task(slope_deg=float('nan')),Task(name='../outside')]:
            with self.assertRaises(ValueError):task.validate()


if __name__=='__main__':unittest.main()
