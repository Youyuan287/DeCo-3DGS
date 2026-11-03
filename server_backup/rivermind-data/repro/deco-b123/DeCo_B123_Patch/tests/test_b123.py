import sys,unittest,tempfile,json
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidence_modes import BuildConfig,build_bank,rofa_keep
from evidence_readout import combine,replace_root,load_bank,sha256
class Tests(unittest.TestCase):
 def test_rofa_single_and_majority(self):
  self.assertTrue(rofa_keep(np.array([[1.,0.]],np.float32)).all())
  self.assertEqual(len(rofa_keep(np.eye(3,dtype=np.float32))),3)
 def test_b1_b2_construction_and_provenance(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);nag=t/'nag.pt';nag.write_bytes(b'frozen-NAG-test')
   features=np.array([[1,0,0],[.96,.12,0],[0,1,0],[0,.95,.13]],np.float32)
   # separate two views, two spatial support clusters
   ids=np.array([0,0,0,0]); views=np.array([0,1,2,3]); weights=np.ones(4,np.float32)
   ptr=np.array([0,2,4,6,8]); visible=np.array([1,2,1,3,100,101,100,102])
   obs=t/'obs.npz';np.savez(obs,features=features,weights=weights,region_ids=ids,view_ids=views,visible_gaussian_offsets=ptr,visible_gaussian_ids=visible)
   sha=sha256(nag)
   for use_b2 in (False,True):
    out=t/f'b2_{use_b2}.npz';m=build_bank(obs,out,sha,BuildConfig(max_modes=2,b1=True,b2=use_b2,geo_weight=.25))
    b,meta=load_bank(out,nag,1,3)
    self.assertTrue(meta['effective_geometry']);self.assertEqual(bool(meta['b2_enabled']),use_b2)
    self.assertEqual(b['features'].shape,(1,2,3));self.assertTrue(np.allclose(b['priors'].sum(1),1))
    self.assertTrue(out.with_suffix('.members.json').exists())
   nag.write_bytes(b'changed')
   with self.assertRaises(ValueError):load_bank(t/'b2_False.npz',nag,1,3)
 def test_b3_switch_only_scoring(self):
  scores=torch.tensor([[.2,.8],[.8,.2]],dtype=torch.float32)
  p=torch.tensor([[.9,.1],[.9,.1]],dtype=torch.float32)
  valid=torch.ones_like(p,dtype=torch.bool)
  fixed=combine(scores,p,valid,'fixed');adaptive=combine(scores,p,valid,'adaptive',.08)
  self.assertGreater(float(adaptive[0]),float(fixed[0]))
  self.assertGreater(float(adaptive[1]),float(fixed[1]))
  one=combine(torch.tensor([[.5]]),torch.tensor([[1.]]),torch.tensor([[True]]),'adaptive')
  self.assertAlmostEqual(float(one),.5,places=6)
 def test_root_only(self):
  bank={'region_ids':np.array([1]),'priors':np.array([[1.,0.]],np.float32),'valid':np.array([[1,0]],bool)}
  sims=[torch.tensor([.1,.2]),torch.tensor([.6,.7,.8])]
  out=replace_root(sims,bank,torch.tensor([[.9,0.]]),b3=False)
  self.assertTrue(torch.equal(out[0],sims[0]));self.assertEqual(float(out[-1][0]),float(sims[-1][0]));self.assertAlmostEqual(float(out[-1][1]),.9,places=6)
if __name__=='__main__':unittest.main()
