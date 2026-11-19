"""Defense audit must be offline and reproduce the original experiment."""
from pathlib import Path
import sys
import unittest
import numpy as np
import ast
import nbformat
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


if __name__=='__main__':unittest.main()
