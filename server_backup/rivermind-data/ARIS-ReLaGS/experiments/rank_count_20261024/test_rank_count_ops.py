"""Detect swapped axes, silent reordering, and corrupt frozen-pool replays."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import rank_count_ops as ops


class RankCountTests(unittest.TestCase):
    def fixture(self):
        return {'pool': [[2,0,10,.9,3],[2,0,20,.8,4],[3,1,30,.1,5]],
                'modes': {'baseline': {'scores':[.9,.8,.1], 'selected':[[2,0,10,.9,3],[2,0,20,.8,4]]},
                          'spatial': {'scores':[.1,.7,.2], 'selected':[[2,0,20,.8,4]]}}}

    def test_axes_and_original_replay_are_not_swapped(self):
        r=self.fixture(); before=copy.deepcopy(r)
        cells=ops.build_cells(r)
        self.assertEqual([n[2] for n in cells['rank1_count0']], [20,30])
        self.assertEqual([n[2] for n in cells['rank0_count1']], [10])
        self.assertEqual([n[2] for n in cells['baseline']], [10,20])
        self.assertEqual([n[2] for n in cells['spatial']], [20])
        self.assertEqual(r,before)

    def test_tie_preserves_pool_order_and_zero_gap_preserves_empty(self):
        r=self.fixture()
        r['modes']['spatial']={'scores':[.2,.2,.2], 'selected':[]}
        cells=ops.build_cells(r)
        self.assertEqual([n[2] for n in cells['rank1_count0']], [10,20])
        self.assertEqual(cells['rank0_count1'], [])

    def test_corrupt_original_selected_nodes_abort(self):
        r=self.fixture();r['modes']['baseline']['selected']=[]
        with self.assertRaises(ValueError): ops.build_cells(r)

    def test_bad_score_count_or_nan_abort(self):
        for scores in ([.1,.2],[float('nan'),.2,.3]):
            r=self.fixture();r['modes']['spatial']['scores']=scores
            with self.assertRaises(ValueError):ops.build_cells(r)


if __name__=='__main__':unittest.main()
