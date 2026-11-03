import unittest
import numpy as np

from diagnostic import (
    project_centers,
    vote_distribution,
    normalized_entropy,
    js_divergence,
    split_half_tv,
    ridge_delta_r2,
    _parent_map,
)


class DiagnosticMathTests(unittest.TestCase):
    def test_projection_uses_camera_and_marks_out_of_frame(self):
        points = np.array([[0.0, 0.0, 2.0], [6.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
        # identity intrinsics, 4x4 image: u=1.5*x/z+1.5, v=1.5*y/z+1.5
        K = np.array([[1.5, 0.0, 1.5], [0.0, 1.5, 1.5], [0.0, 0.0, 1.0]])
        uv, valid, depth = project_centers(points, np.eye(4), K, 4, 4)
        self.assertEqual(valid.tolist(), [True, False, False])
        np.testing.assert_allclose(uv[0], [1.5, 1.5])
        self.assertEqual(depth[2], -1.0)

    def test_distribution_entropy_and_js_are_bounded(self):
        q = vote_distribution(np.array([1, 1, 2, -1]), classes=[1, 2])
        np.testing.assert_allclose(q, [2 / 3, 1 / 3])
        self.assertGreater(normalized_entropy(q), 0.0)
        self.assertAlmostEqual(js_divergence(q, q), 0.0)
        self.assertGreater(js_divergence(q, [1.0, 0.0]), 0.0)

    def test_split_half_tv_is_zero_for_identical_votes(self):
        self.assertAlmostEqual(split_half_tv(np.array([1, 2]), np.array([1, 2]), [1, 2]), 0.0)
        self.assertAlmostEqual(split_half_tv(np.array([1, 1]), np.array([2, 2]), [1, 2]), 1.0)

    def test_ridge_delta_r2_is_positive_for_incremental_signal(self):
        rng = np.random.default_rng(7)
        x0 = rng.normal(size=(30, 2))
        extra = rng.normal(size=(30, 1))
        y = 2.0 * extra[:, 0] + 0.05 * rng.normal(size=30)
        delta = ridge_delta_r2(x0, np.c_[x0, extra], y, folds=5, alpha=1.0)
        self.assertGreater(delta, 0.5)

    def test_parent_map_uses_adjacent_parent_level(self):
        child_points = np.array([0, 0, 1, 1, 1])
        parent_points = np.array([0, 0, 1, 1, 1])
        np.testing.assert_array_equal(_parent_map(child_points, parent_points, 2, 2), [0, 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
