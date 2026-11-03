import unittest

import numpy as np
import torch

from evidence_readout_factorial import factorial_correction
from evidence_readout_v2 import auxiliary_correction


class FactorialReadoutTests(unittest.TestCase):
    def setUp(self):
        self.bank = {
            "priors": np.ones((3, 1), np.float32),
            "valid": np.ones((3, 1), bool),
            "region_ids": np.array([0, 1, 2]),
        }
        self.main = torch.zeros(4)
        self.slots = torch.tensor([[.6], [.3], [.1]])
        self.eligible = np.ones((3, 1), bool)
        self.reliability = np.ones((3, 1), np.float32)

    def test_all_absolute_matches_v2(self):
        expected, _ = auxiliary_correction(
            self.main, self.slots, self.bank, self.eligible, self.reliability
        )
        actual, _ = factorial_correction(
            self.main, self.slots, self.bank, self.eligible, self.reliability,
            "all", "absolute", []
        )
        torch.testing.assert_close(actual, expected)

    def test_selected_absolute_has_frozen_scope(self):
        actual, _ = factorial_correction(
            self.main, self.slots, self.bank, self.eligible, self.reliability,
            "selected", "absolute", [1]
        )
        self.assertEqual(float(actual[0]), 0.0)
        self.assertGreater(float(actual[1]), 0.0)
        self.assertEqual(float(actual[2]), 0.0)

    def test_relative_changes_pairwise_scores(self):
        actual, _ = factorial_correction(
            self.main, self.slots, self.bank, self.eligible, self.reliability,
            "all", "relative", []
        )
        self.assertGreater(float(actual[0] - actual[1]), 0.0)
        self.assertEqual(float(actual[3]), 0.0)


if __name__ == "__main__":
    unittest.main()
