"""Synthetic contract tests: catches wrong std, threshold, dedup and edge branches."""
import importlib.util
import unittest
try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, 'Torch runtime unavailable; exact source replay pending')
class ROFATests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('rofa_cpu'), 'CPU implementation missing')
        import rofa_cpu
        self.m = rofa_cpu

    def test_empty_single_zero(self):
        for x, expected in [(torch.zeros(0,2), [0.,0.]),
                            (torch.tensor([[3.,4.]]), [.6,.8]),
                            (torch.zeros(1,2), [0.,0.]),
                            (torch.zeros(3,2), [0.,0.])]:
            out, _ = self.m.rofa(x)
            torch.testing.assert_close(out, torch.tensor(expected))

    def test_sample_std_and_strict_fallback(self):
        x = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        _, d = self.m.rofa(x)
        torch.testing.assert_close(d['mean_sim'], torch.tensor([.5,.5,0.]))
        self.assertAlmostEqual(d['sigma'].item(), (1/12)**.5, places=6)
        out, d = self.m.rofa(torch.eye(2))
        self.assertTrue(d['fallback'])
        self.assertEqual(d['keep'].tolist(), [True,False])
        torch.testing.assert_close(out, torch.tensor([1.,0.]))

    def test_heterogeneous_duplicate_changes_original(self):
        x = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        before, _ = self.m.rofa(x)
        after, _ = self.m.rofa(torch.cat([x,x[2:]]))
        self.assertGreater(torch.linalg.norm(before-after).item(), .3)

    def test_identity_dedup_preserves_not_feature_dedup(self):
        x = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        a, _ = self.m.rofa(x)
        b, d = self.m.rofa_identity(torch.cat([x,x[2:]]), ['a','b','c','c'])
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(d['unique_indices'], [0,1,2])
        with self.assertRaises(ValueError):
            self.m.rofa_identity(x, ['a','b','a'])

    def test_exact_original_source_equivalence(self):
        original = self.m.load_original()
        gen = torch.Generator().manual_seed(19)
        samples = [torch.zeros(0,4), torch.zeros(1,4), torch.zeros(3,4), torch.eye(4)]
        samples += [torch.randn(n,4,generator=gen) for n in [1,2,3,15]]
        for x in samples:
            for tau in [0.,3.]:
                actual,_ = self.m.rofa(x,tau)
                expected = original(x,tau)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

if __name__=='__main__': unittest.main(verbosity=2)
