import importlib.util
import unittest
import numpy as np

class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('capture_contracts'))
        import capture_contracts
        self.m=capture_contracts
    def test_feature_alignment(self):
        self.m.validate_features(np.array([[-1,0,1]]),np.ones((2,512)))
        with self.assertRaises(ValueError): self.m.validate_features(np.array([[2]]),np.ones((2,512)))
        with self.assertRaises(ValueError): self.m.validate_features(np.array([[0]]),np.full((1,512),np.nan))
    def test_label_range(self):
        self.m.validate_labels(np.array([0,1,0]),2,3)
        with self.assertRaises(ValueError): self.m.validate_labels(np.array([0,2,0]),2,3)
    def test_budget(self):
        self.m.check_budget(16*2**30,1*2**30)
        with self.assertRaises(ValueError): self.m.check_budget(10*2**30,1)
        with self.assertRaises(ValueError): self.m.check_budget(20*2**30,4*2**30)
    def test_group_repeat_and_separation(self):
        cams=[dict(img_name=str(i),position=p,rotation=np.eye(3).tolist()) for i,p in enumerate([[0,0,0],[.001,0,0],[1,0,0]])]
        self.assertEqual(self.m.geometry_groups(cams)['ids'],[0,0,1])
    def test_forbidden_calls(self):
        self.m.audit_calls('x=1')
        with self.assertRaises(ValueError): self.m.audit_calls('agent.assign_label()')
        with self.assertRaises(ValueError): self.m.audit_calls('agent.merge_small_segs_torch()')

if __name__=='__main__': unittest.main(verbosity=2)
