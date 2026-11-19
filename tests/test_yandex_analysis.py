"""Offline tests for scoring boundaries and preregistered comparisons."""
import sys
from pathlib import Path
import unittest
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_yandex import auto_correct, classification_metrics, accuracy_ci, h12_comparison


class AnalysisTests(unittest.TestCase):
    def test_numeric_tolerance_and_wrong_types(self):
        task={'type':'A','answer_kind':'number','reference':{'answer':.5},'tolerance':{'absolute':1e-9,'relative':.01}}
        self.assertTrue(auto_correct(task,{'answer':.504}))
        self.assertFalse(auto_correct(task,{'answer':.506}))
        self.assertFalse(auto_correct(task,{'answer':True}))
        self.assertFalse(auto_correct(task,None))

    def test_false_and_correct_abstention(self):
        task={'type':'B','answer_kind':'boolean','reference':{'answer':False}}
        self.assertTrue(auto_correct(task,{'answer':False}))
        task={'type':'F','answer_kind':'insufficient_data','reference':{'answer':'insufficient_data'}}
        self.assertTrue(auto_correct(task,{'answer':'insufficient_data'}))

    def test_all_abstentions_are_not_majority_predictions(self):
        m=classification_metrics([0,0,1],[-1,-1,-1])
        self.assertEqual(m['coverage'],0)
        self.assertEqual(m['accuracy'],0)
        self.assertEqual(m['balanced_accuracy'],0)
        self.assertEqual(m['f1_minority'],0)

    def test_valid_binary_metrics_match_sklearn(self):
        y=np.array([0,0,0,1,1,1]); p=np.array([0,1,0,1,0,1])
        m=classification_metrics(y,p)
        self.assertAlmostEqual(m['f1_minority'],f1_score(y,p))
        self.assertAlmostEqual(m['balanced_accuracy'],balanced_accuracy_score(y,p))

    def test_bootstrap_extremes(self):
        self.assertEqual(accuracy_ci([0]*8),{'accuracy':0,'ci_low':0,'ci_high':0})
        self.assertEqual(accuracy_ci([1]*8),{'accuracy':1,'ci_low':1,'ci_high':1})

    def test_equal_predictions_support_noninferiority_not_loss(self):
        y=np.array([0,0,1,1]*5)
        self.assertEqual(h12_comparison(y,y,y,5)['decision'],'confirm')
        self.assertEqual(h12_comparison(y,y,y,20)['decision'],'reject')


if __name__=='__main__': unittest.main()
