import unittest
import numpy as np
import pandas as pd
import torch
from vclad.detector import ReconstructionGRU
from vclad.ranking import variable_scores,rank_event
from vclad.envelope import _smooth_fault_envelope
from vclad.quality_metrics import QualityMetrics

class CoreTests(unittest.TestCase):
    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in ReconstructionGRU().parameters()),17481)
    def test_ranking_uses_last_time_step(self):
        x=torch.zeros(1,3,2); y=torch.tensor([[[100.,0.],[0.,0.],[1.,2.]]])
        score=variable_scores(x,y)
        torch.testing.assert_close(score,torch.tensor([[.25,1.]]))
        self.assertEqual(float(variable_scores(x,x).sum()),0.)
    def test_event_overlap_and_earliest_tie(self):
        peak,order=rank_event([1.,1.,20.],torch.tensor([[1.,2.],[2.,1.],[1.,2.]]),[0,1,90],10,20)
        self.assertEqual(peak,0);self.assertEqual(order.tolist(),[1,0])
    def test_envelope(self):
        e=_smooth_fault_envelope(101)
        self.assertAlmostEqual(e[0],0.);self.assertAlmostEqual(e[-1],0.)
        np.testing.assert_allclose(e[30:81],1.)
        self.assertAlmostEqual(e[15],.5);self.assertAlmostEqual(e[90],.5)
    def test_bad_timestamps_are_not_dropped(self):
        f=pd.DataFrame({'数据时间':['2000-01-01 00:00:00','bad','2000-01-01 00:00:20']})
        self.assertAlmostEqual(QualityMetrics().compute_timeliness(f),1/3)

class AveragePrecisionTests(unittest.TestCase):
    def test_tied_scores_are_one_threshold(self):
        from vclad.detector import average_precision_score
        for labels in ([1,0],[0,1]):
            self.assertAlmostEqual(average_precision_score(np.array(labels),np.array([.5,.5])),.5)

    def test_mixed_ties_and_row_permutations(self):
        from itertools import permutations
        from vclad.detector import average_precision_score
        labels=np.array([1,0,1,0]);scores=np.array([.9,.9,.4,.1])
        expected=(.5*.5)+(.5*2/3)
        for order in permutations(range(4)):
            order=list(order)
            self.assertAlmostEqual(average_precision_score(labels[order],scores[order]),expected)

    def test_degenerate_and_invalid_input(self):
        from vclad.detector import average_precision_score
        self.assertEqual(average_precision_score(np.zeros(2),np.ones(2)),0.)
        self.assertEqual(average_precision_score(np.ones(2),np.ones(2)),1.)
        with self.assertRaises(ValueError):
            average_precision_score(np.array([0,2]),np.array([.1,.2]))

class SharedModelTests(unittest.TestCase):
    def test_all_imports_use_shared_model(self):
        from vclad.model import VCLAD,HydraADLiteV2
        from vclad.detector import ReconstructionGRU,VCLAD as TrainingModel
        self.assertIs(VCLAD,HydraADLiteV2)
        self.assertIs(VCLAD,ReconstructionGRU)
        self.assertIs(VCLAD,TrainingModel)

    def test_scoring_entry_points_agree(self):
        from vclad.model import VCLAD,anomaly_scores
        from vclad.detector import _score_batch
        torch.manual_seed(42)
        model=VCLAD().eval();x=torch.randn(2,30,9)
        stats={k:{'median':.2,'iqr':.4} for k in ('p','s')}
        total,p,s=_score_batch(model,x,stats,20.,'last',.8,.2)
        result=anomaly_scores(model,x,score_stats=stats)
        torch.testing.assert_close(total,result['score'])
        torch.testing.assert_close(p,result['p_score'])
        torch.testing.assert_close(s,result['s_score'])

if __name__=='__main__':unittest.main()
