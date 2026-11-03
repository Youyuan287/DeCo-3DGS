import unittest
import torch

from evidence_readout_v2 import aggregate_advantage


class TestDeCoLIAggregation(unittest.TestCase):
    def test_no_eligible_modes_returns_zero(self):
        scores = torch.tensor([[.9, .8]])
        main = torch.tensor([.5])
        priors = torch.tensor([[.5, .5]])
        eligible = torch.tensor([[False, False]])
        for method in ("softmax", "maxsim", "prior_maxsim"):
            out = aggregate_advantage(scores, main, priors, eligible, method=method)
            self.assertTrue(torch.equal(out, torch.zeros(1)))

    def test_single_mode_agrees(self):
        scores = torch.tensor([[.7, .2]])
        main = torch.tensor([.5])
        priors = torch.tensor([[.8, .2]])
        eligible = torch.tensor([[True, False]])
        outs = [aggregate_advantage(scores, main, priors, eligible, method=m)
                for m in ("softmax", "maxsim", "prior_maxsim")]
        for out in outs:
            self.assertTrue(torch.allclose(out, torch.tensor([.2])))

    def test_maxsim_bounds_softmax_and_prior(self):
        scores = torch.tensor([[.9, .8, .6]])
        main = torch.tensor([.5])
        priors = torch.tensor([[.6, .3, .1]])
        eligible = torch.tensor([[True, True, True]])
        soft = aggregate_advantage(scores, main, priors, eligible, method="softmax")
        maxsim = aggregate_advantage(scores, main, priors, eligible, method="maxsim")
        prior = aggregate_advantage(scores, main, priors, eligible, method="prior_maxsim")
        self.assertGreaterEqual(maxsim.item(), soft.item())
        self.assertLessEqual(prior.item(), maxsim.item())

    def test_nonpositive_advantage_is_zero(self):
        x = torch.tensor([[.4, .5]])
        main = torch.tensor([.6])
        p = torch.tensor([[.5, .5]])
        e = torch.tensor([[True, True]])
        for m in ("softmax", "maxsim", "prior_maxsim"):
            self.assertEqual(aggregate_advantage(x, main, p, e, method=m).item(), 0.0)

    def test_prior_maxsim_suppresses_low_prior_high_advantage(self):
        x = torch.tensor([[.95, .70]])
        main = torch.tensor([.5])
        p = torch.tensor([[.01, .99]])
        e = torch.tensor([[True, True]])
        out = aggregate_advantage(x, main, p, e, method="prior_maxsim")
        self.assertAlmostEqual(out.item(), .2, places=6)

    def test_permutation_invariant_and_invalid_ignored(self):
        x = torch.tensor([[.95, .70, .99]])
        main = torch.tensor([.5])
        p = torch.tensor([[.01, .99, .0]])
        e = torch.tensor([[True, True, False]])
        a = aggregate_advantage(x, main, p, e, method="prior_maxsim")
        b = aggregate_advantage(x[:, [1, 0, 2]], main, p[:, [1, 0, 2]], e[:, [1, 0, 2]], method="prior_maxsim")
        self.assertTrue(torch.allclose(a, b))
        self.assertTrue(torch.isfinite(a).all())

    def test_softmax_matches_frozen_formula(self):
        x = torch.tensor([[.7, .8, .6], [.4, .9, .5]])
        main = torch.tensor([.5, .5])
        p = torch.tensor([[.5, .3, .2], [.1, .7, .2]])
        e = torch.tensor([[True, True, False], [True, False, True]])
        got = aggregate_advantage(x, main, p, e, method="softmax", temperature=.08)
        adv = torch.clamp(x-main[:, None], min=0)
        logits = x/.08 + torch.log(p.clamp_min(1e-12))
        logits = logits.masked_fill(~e, -torch.inf)
        expected = (torch.softmax(logits, dim=1) * adv).sum(dim=1)
        self.assertTrue(torch.allclose(got, expected, atol=1e-7, rtol=0))

    def test_prior_maxsim_with_no_positive_prior_returns_zero(self):
        out = aggregate_advantage(torch.tensor([[.9, .8]]), torch.tensor([.5]),
                                   torch.tensor([[0., 0.]]), torch.tensor([[True, True]]),
                                   method="prior_maxsim")
        self.assertEqual(out.item(), 0.0)

    def test_invalid_high_score_cannot_participate(self):
        out = aggregate_advantage(torch.tensor([[.7, 99.]]), torch.tensor([.5]),
                                   torch.tensor([[.5, .5]]), torch.tensor([[True, False]]),
                                   method="maxsim")
        self.assertAlmostEqual(out.item(), .2, places=6)

    def test_nonfinite_inputs_produce_finite_output(self):
        out = aggregate_advantage(torch.tensor([[float('nan'), .6]]), torch.tensor([.5]),
                                   torch.tensor([[.5, float('inf')]]), torch.tensor([[True, True]]),
                                   method="softmax")
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
