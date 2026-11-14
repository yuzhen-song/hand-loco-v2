import hashlib
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mujoco
import numpy as np
import build_model as B
from sim2real import Case,contact_xml


class ModelContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path(B.OUT_XML);cls.before=hashlib.sha256(p.read_bytes()).hexdigest()
        cls.model,cls.keys=B.build(write=False,xml_transform=lambda xml:contact_xml(xml,Case()))
        cls.after=hashlib.sha256(p.read_bytes()).hexdigest()
    def test_memory_build_preserves_baseline_xml(self):
        self.assertEqual(self.before,self.after)
    def test_all_collision_parts_contact_floor_without_duplicate_visuals(self):
        m=self.model;floor=m.geom('floor').id
        ids=np.flatnonzero((m.geom_group==0)&(m.geom_bodyid!=0))
        self.assertGreater(len(ids),100)
        self.assertTrue(np.all(m.geom_contype[m.geom_group==1]==0))
        self.assertTrue(np.all(m.geom_contype[ids]==1))
        self.assertTrue(np.all(m.geom_conaffinity[ids]==2))
        self.assertEqual(int(m.geom_contype[floor]),2)
        self.assertEqual(int(m.geom_conaffinity[floor]),1)
        d=mujoco.MjData(m);d.qpos[:]=self.keys['stance1'];d.qpos[2]=.04
        mujoco.mj_forward(m,d)
        tips={m.body(v).id for v in B.TIP_BODY.values()}
        detected=False
        for c in d.contact:
            if floor in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==floor else c.geom1
                detected |= int(m.geom_bodyid[other]) not in tips
        self.assertTrue(detected,'fallen assembly must collide with floor beyond fingertips')


if __name__=='__main__':unittest.main()
