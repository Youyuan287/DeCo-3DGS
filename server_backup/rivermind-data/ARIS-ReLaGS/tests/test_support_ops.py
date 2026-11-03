"""Hand-checked fixtures for cache-only spatial scoring (no model/GPU)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class SupportTests(unittest.TestCase):
    def test_iou_penalizes_extra_extent_not_just_intersection(self):
        from support_ops import mask_iou
        self.assertAlmostEqual(mask_iou([1, 1, 0, 0], [0, 1, 1, 0]), 1/3)
        self.assertEqual(mask_iou([0, 0], [0, 0]), 0.0)
        self.assertEqual(mask_iou([1, 0], [1, 0]), 1.0)

    def test_evidence_invisible_views_zero_and_visibility_control(self):
        from support_ops import evidence_scores
        visibility, spatial = evidence_scores([0.5, 1.0, 1.0, 0.0], [True, True, False, False])
        self.assertEqual(visibility, 0.5)
        self.assertEqual(spatial, 0.375)
        self.assertEqual(evidence_scores([], []), (0.0, 0.0))

    def test_gap_preserves_original_single_candidate_empty_behavior(self):
        from support_ops import gap_select
        self.assertEqual(gap_select(['a'], [0.7]), [])
        self.assertEqual(gap_select(['a', 'b', 'c'], [0.9, 0.8, 0.2]), ['a', 'b'])
        self.assertEqual(gap_select(['a', 'b'], [0.5, 0.5]), [])
        self.assertEqual(gap_select([], []), [])

    def test_rescore_keeps_pool_and_falls_back_only_when_whole_pool_unobserved(self):
        from support_ops import rescore
        base = [0.9, 0.8]
        self.assertEqual(rescore(base, [0.0, 0.0]), [0.0, 0.0])
        self.assertEqual(rescore(base, [0.0, 0.0], all_unobserved=True), base)
        self.assertEqual(rescore(base, [0.0, 0.5]), [0.0, 0.4])
        self.assertEqual(base, [0.9, 0.8])

    def test_permutation_moves_within_level_and_area_and_has_no_fixed_pair_in_nonsingleton(self):
        from support_ops import proposal_permutation
        seg = np.array([[[0, 0, 1, 1, 2, -1]], [[3, 3, 4, 4, -1, -1]]])
        mapping, stats = proposal_permutation(seg, n_features=5, seed=0)
        self.assertEqual(mapping.tolist(), [1, 0, 2, 4, 3])
        self.assertEqual(stats['eligible_features'], 5)
        self.assertEqual(stats['moved_features'], 4)
        self.assertEqual(stats['singleton_features'], 1)
        self.assertEqual(stats['total_mask_area_preserved'], True)
        self.assertTrue(np.array_equal(mapping, proposal_permutation(seg, 5, 0)[0]))

    def test_invalid_feature_ids_rejected_not_silently_wrapped(self):
        from support_ops import proposal_permutation
        with self.assertRaises(ValueError):
            proposal_permutation(np.array([[[0, 7]]]), 2, 0)

    def test_top_proposal_excludes_feature_rows_without_spatial_mask(self):
        from support_ops import select_feature
        self.assertEqual(select_feature([0.9, 0.4, 0.6], np.array([[[-1, 1, 2]]])), 2)
        with self.assertRaises(ValueError):
            select_feature([0.9], np.array([[[-1]]]))

    def test_preflight_rejects_unmatched_eval_names(self):
        from support_ops import validate_run_inputs
        rows=[dict(scene='figurines',image='eval',prompt='apple',iou='0.5',pred_px='2')]
        with self.assertRaises(ValueError):
            validate_run_inputs(['train'],{'eval':['apple']},rows)

    def test_preflight_rejects_missing_or_duplicate_baseline_keys(self):
        from support_ops import validate_run_inputs
        row=dict(scene='figurines',image='eval',prompt='apple',iou='0.5',pred_px='2')
        with self.assertRaises(ValueError):
            validate_run_inputs(['eval','train'],{'eval':['pear']},[row])
        with self.assertRaises(ValueError):
            validate_run_inputs(['eval','train'],{'eval':['apple']},[row,row])
        self.assertEqual(validate_run_inputs(['eval','train'],{'eval':['apple']},[row]),['train'])


if __name__ == '__main__':
    unittest.main()
