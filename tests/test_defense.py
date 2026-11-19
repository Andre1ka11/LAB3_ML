"""Defense audit must be offline and reproduce the original experiment."""
from pathlib import Path
import sys
import unittest
import numpy as np
import ast
import nbformat
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from defense_analysis import audit,offline,baselines,confusion,explicit_refusal,holm_comparisons,MODELS


class DefenseTests(unittest.TestCase):
    def test_frozen_sources_and_all_1376_requests(self):
        with offline():result=audit()
        self.assertTrue(result['all_prompt_texts_identical'])
        self.assertEqual(sum(r['responses'] for r in result['models']),1376)
        self.assertTrue(all(r['registration_before_main'] for r in result['models']))

    def test_baselines_retrain_offline(self):
        with offline():ids,y,_,rows=baselines()
        self.assertEqual(len(ids),100);self.assertEqual(int(y.sum()),37)
        self.assertTrue(all(r['predictions_reproduced'] for r in rows))

    def test_abstentions_keep_true_class(self):
        self.assertEqual(confusion(np.array([0,0,1,1]),np.array([0,-1,1,-1])),[[1,0,1],[0,1,1]])

    def test_fenced_refusal_stays_a_refusal(self):
        record={'raw_response':{'output':[{'content':[{'type':'output_text','text':'```json\n{"answer":"insufficient_data"}\n```'}]}]}}
        self.assertTrue(explicit_refusal(record))

    def test_offline_guard_restores_cloud_transports(self):
        import yandex_runner
        original=yandex_runner.urlopen
        with offline():
            with self.assertRaisesRegex(RuntimeError,'Network blocked'):
                yandex_runner.urlopen('https://example.invalid')
        self.assertIs(yandex_runner.urlopen,original)

    def test_six_comparisons_and_monotone_holm(self):
        scores=[{'model':m,'task_id':str(i),'correct':m=='DeepSeek'} for m in MODELS for i in range(24)]
        rows=holm_comparisons(scores);self.assertEqual(len(rows),6)
        for r in rows:self.assertGreaterEqual(r['p_holm'],r['p_exact'])

    def test_notebook_cells_compile_and_are_executed(self):
        notebook=nbformat.read(ROOT/'notebooks/defense.ipynb',as_version=4)
        nbformat.validate(notebook)
        cells=[c for c in notebook.cells if c.cell_type=='code']
        self.assertEqual(len(cells),33)
        for cell in cells:
            ast.parse(cell.source);self.assertIsNotNone(cell.execution_count)
            self.assertFalse(any(o.output_type=='error' for o in cell.outputs))
        self.assertIn('LIVE=False',cells[-1].source)

    def test_every_block_and_figure_has_teaching_explanations(self):
        from defense_explanations import BLOCKS,FIGURES
        notebook=nbformat.read(ROOT/'notebooks/defense.ipynb',as_version=4)
        notes=[c.metadata.get('teaching',{}) for c in notebook.cells if c.cell_type=='markdown']
        for role in ['before_block','after_block']:
            self.assertEqual({r['block'] for r in notes if r.get('role')==role},set(BLOCKS))
        for role in ['before_figure','after_figure']:
            self.assertEqual({r['figure'] for r in notes if r.get('role')==role},set(FIGURES))
        for i,cell in enumerate(notebook.cells):
            if cell.cell_type=='code':self.assertEqual(notebook.cells[i-1].cell_type,'markdown')

    def test_expert_review_is_source_bound_and_not_a_pair_of_team_ratings(self):
        from defense_expert_review import review
        from unittest.mock import patch
        import defense_expert_review
        records={m:load[3] for m in MODELS for load in [__import__('defense_analysis').load_model(m)]}
        data=json.loads((ROOT/'data/reviews/defense_expert_review.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['ratings']),92)
        self.assertEqual(data['assessors'],1);self.assertIsNone(data['cohen_kappa'])
        self.assertFalse(data['independent_blind_team_review_completed'])
        data['ratings'][0]['response_sha256']='0'*64
        original=defense_expert_review.read_json
        with patch.object(defense_expert_review,'read_json',side_effect=lambda p:data if p.name=='defense_expert_review.json' else original(p)):
            with self.assertRaises(AssertionError):review(ROOT,records)

    def test_expert_summary_keeps_methodology_separate_from_contract(self):
        data=json.loads((ROOT/'analysis/defense/expert_review.json').read_text(encoding='utf-8'))
        by_model={r['model']:r for r in data['summary']}
        self.assertEqual(by_model['GigaChat-2']['D_wrong_metric'],1)
        self.assertEqual(by_model['GigaChat-2']['D_strict_correct'],0)
        self.assertEqual(by_model['Qwen3-4B']['D_not_assessable'],2)
        self.assertEqual(by_model['DeepSeek']['D_strict_correct'],8)


if __name__=='__main__':unittest.main()
