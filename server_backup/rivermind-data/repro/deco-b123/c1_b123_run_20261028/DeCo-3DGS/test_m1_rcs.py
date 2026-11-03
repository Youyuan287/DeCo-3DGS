import tempfile
import unittest
from pathlib import Path
import numpy as np

from evidence_modes_rcs import (BuildConfig, build_bank, compute_cross_view_support,
                                select_rcs_seeds)


class TestM1RCS(unittest.TestCase):
    def test_t1_original_regression(self):
        f=np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0]],np.float32)
        w=np.ones(4,np.float32); r=np.zeros(4,np.int64); v=np.arange(4,dtype=np.int64)
        with tempfile.TemporaryDirectory() as d:
            obs=Path(d)/'o.npz'; np.savez(obs,features=f,weights=w,region_ids=r,view_ids=v)
            a=Path(d)/'a.npz'; b=Path(d)/'b.npz'
            build_bank(obs,a,'nag',BuildConfig(b1=True,geo_weight=0,mode_selection='original'))
            build_bank(obs,b,'nag',BuildConfig(b1=True,geo_weight=0,mode_selection='original'))
            with np.load(a) as x, np.load(b) as y:
                for k in ('features','residuals','priors','valid','base_features','group_count','region_ids'):
                    np.testing.assert_array_equal(x[k],y[k])

    def test_t2_isolated_anomaly_is_not_supported_seed(self):
        x=np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1]],np.float32)
        s,c=compute_cross_view_support(x,np.ones(5,np.float32),np.arange(5))
        self.assertEqual(c[-1],0); self.assertGreaterEqual(c[2],1)
        seeds=select_rcs_seeds(x@x.T,s,c,3)
        self.assertNotIn(4,seeds); self.assertIn(2,seeds)

    def test_t3_minority_two_view_mode_kept(self):
        x=np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[1,0,0]],np.float32)
        s,c=compute_cross_view_support(x,np.ones(5,np.float32),np.array([0,1,2,3,4]))
        seeds=select_rcs_seeds(x@x.T,s,c,3)
        self.assertTrue(any(i in seeds for i in (2,3)))

    def test_t4_duplicate_view_deduplicated(self):
        x=np.array([[1,0],[1,0],[1,0]],np.float32)
        _,c=compute_cross_view_support(x,np.ones(3,np.float32),np.array([0,1,1]))
        self.assertEqual(c[0],1); self.assertEqual(c[1],1)

    def test_t5_single_view_has_no_support(self):
        s,c=compute_cross_view_support(np.array([[1,0],[1,0]],np.float32),np.ones(2),np.array([4,4]))
        self.assertTrue(np.all(c==0)); self.assertEqual(select_rcs_seeds(np.eye(2),s,c,3),[])

    def test_t6_similar_modes_not_repeated(self):
        x=np.tile(np.array([[1.,0.]],np.float32),(5,1)); s,c=compute_cross_view_support(x,np.ones(5),np.arange(5))
        self.assertEqual(len(select_rcs_seeds(x@x.T,s,c,3)),1)

    def test_t7_numeric_stability_and_permutation(self):
        x=np.array([[0,0],[np.nan,1],[1,0]],np.float32); w=np.array([0,1,1],np.float32); v=np.array([0,1,2])
        s,c=compute_cross_view_support(x,w,v)
        self.assertTrue(np.isfinite(s).all()); self.assertTrue(np.isfinite(c).all())
        x2=x[[2,0,1]]; s2,c2=compute_cross_view_support(x2,w[[2,0,1]],v[[2,0,1]])
        self.assertEqual(sorted(s.tolist()),sorted(s2.tolist())); self.assertEqual(sorted(c.tolist()),sorted(c2.tolist()))

    def test_t8_rcs_metadata_does_not_change_readout_inputs(self):
        f=np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0]],np.float32); w=np.ones(4); r=np.zeros(4,dtype=np.int64); v=np.arange(4)
        with tempfile.TemporaryDirectory() as d:
            obs=Path(d)/'o.npz'; np.savez(obs,features=f,weights=w,region_ids=r,view_ids=v)
            a=Path(d)/'a.npz'; b=Path(d)/'b.npz'
            build_bank(obs,a,'nag',BuildConfig(b1=True,geo_weight=0,mode_selection='original'))
            build_bank(obs,b,'nag',BuildConfig(b1=True,geo_weight=0,mode_selection='rcs'))
            with np.load(a) as x, np.load(b) as y:
                self.assertEqual(x['features'].shape,y['features'].shape)
                self.assertEqual(x['region_ids'].tolist(),y['region_ids'].tolist())


if __name__=='__main__': unittest.main()
