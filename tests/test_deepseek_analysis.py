"""Offline tests for snapshot boundaries and classification failures."""
from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts')]
import deepseek_runner as r
from analyze_deepseek import select,STAGES
from analysis_common import classification_metrics,raw_text


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'event':'response','stage':stage,'request_id':f'{stage}-{i}','raw_response':{'output':[]}} for stage,n in STAGES.items() for i in range(n)]
        self.snapshot={'request_order':[e['request_id'] for e in self.rows],
                       'response_sha256':{e['request_id']:r.digest(e) for e in self.rows}}

    def test_later_twenty_shot_and_inflight_events_do_not_change_snapshot(self):
        extra=[{'event':'response','stage':'twenty','request_id':'later','raw_response':{}},
               {'event':'started','request_id':'inflight'}]
        self.assertEqual(select(self.rows+extra,self.snapshot),self.rows)

    def test_changed_or_duplicate_response_is_rejected(self):
        with self.assertRaises(ValueError): select(self.rows+[self.rows[0]],self.snapshot)
        modified=[dict(e) for e in self.rows];modified[0]['raw_response']={'changed':True}
        with self.assertRaises(ValueError): select(modified,self.snapshot)

    def test_missing_response_is_not_silently_counted(self):
        with self.assertRaises(ValueError): select(self.rows[:-1],self.snapshot)

    def test_unfinished_predictions_are_not_majority_class(self):
        metrics=classification_metrics([0,0,1,1],[-1,-1,-1,-1])
        self.assertEqual(metrics['balanced_accuracy'],0)
        self.assertEqual(metrics['coverage'],0)
        self.assertEqual(metrics['accuracy'],0)

    def test_reasoning_null_is_not_answer_text(self):
        self.assertEqual(raw_text({'raw_response':{'output':[{'content':None},{'content':[{'type':'output_text','text':'answer'}]}]}}),'answer')


if __name__=='__main__': unittest.main()
