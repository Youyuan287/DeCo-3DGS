import unittest
import numpy as np
import torch
from evidence_readout_v3 import relative_correction

class RelativeTests(unittest.TestCase):
    def run_rule(self, ids):
        bank=dict(priors=np.ones((3,1),np.float32),valid=np.ones((3,1),bool),region_ids=np.arange(3))
        return relative_correction(torch.zeros(3),torch.tensor([[.4],[.2],[.1]]),
             bank,np.ones((3,1),bool),np.full((3,1),.8,np.float32),ids)
    def test_defined_competition_and_outside_fallback(self):
        x,d=self.run_rule([0,1])
        torch.testing.assert_close(x,torch.tensor([.02,0,0]))
        self.assertNotEqual(float(x[0]-x[1]),0)
    def test_single_or_empty_reverts(self):
        for ids in [[],[0]]:
            x,_=self.run_rule(ids)
            torch.testing.assert_close(x,torch.zeros(3))
    def test_order_and_duplicates_do_not_matter(self):
        a,_=self.run_rule([0,1]);b,_=self.run_rule([1,0,0])
        torch.testing.assert_close(a,b)

if __name__=='__main__':unittest.main()
