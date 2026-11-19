"""Offline API adapter checks. No real requests or credentials needed."""
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import gigachat_runner as g
import yandex_runner as common


class GigaTests(unittest.TestCase):
    def test_shared_prompts_and_parameters(self):
        config,rs=g.prepare();_,cloud=common.prepare(ROOT,'yandex')
        self.assertEqual(len(rs),344);self.assertEqual(len({r['request_id'] for r in rs}),344)
        for a,b in zip(rs,cloud):
            self.assertEqual(a['payload']['messages'],[{'role':'system','content':b['payload']['instructions']},{'role':'user','content':b['payload']['input']}])
            self.assertEqual(a['payload']['temperature'],0);self.assertEqual(a['payload']['max_tokens'],4096)
            self.assertEqual(set(a['payload']),{'model','messages','temperature','max_tokens','stream'})

    def test_exact_provider_response_preserved(self):
        raw={'choices':[{'message':{'content':'{"answer":false,"explanation":"x","confidence":80}'},'finish_reason':'stop'}],
             'usage':{'prompt_tokens':20,'completion_tokens':8,'total_tokens':28},'model':'GigaChat-2:version'}
        wrapped=g.normalize(raw)
        self.assertIs(wrapped['provider_response'],raw)
        self.assertIs(common.parse_response(wrapped,'boolean')[0]['answer'],False)
        self.assertEqual(wrapped['usage']['input_tokens'],20)

    def test_truncation_and_null_content(self):
        wrapped=g.normalize({'choices':[{'message':{'content':None},'finish_reason':'length'}]})
        self.assertEqual(common.parse_response(wrapped,'text'),(None,'response_not_completed'))

    def test_extra_code_does_not_repair_primary_json(self):
        raw={'choices':[{'message':{'content':'{"answer":{},"explanation":"","confidence":90}\n```python\nprint(1)\n```'},'finish_reason':'stop'}]}
        self.assertEqual(common.parse_response(g.normalize(raw),'python_code'),(None,'invalid_json'))

    def test_mcnemar_family_holm(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import analyze_gigachat as analysis
        def rows(correct):return [{'task_id':f'A{i:02}','correct':correct} for i in range(24)]
        with tempfile.TemporaryDirectory() as temp,patch.object(analysis,'OUT',Path(temp)):
            analysis.paired_family(rows(True),rows(True),rows(False))
            result=json.loads((Path(temp)/'paired_comparisons.json').read_text())['results']
        self.assertEqual(len(result),3)
        for r in result:self.assertGreaterEqual(r['p_holm'],r['p_exact'])
        self.assertFalse(result[0]['significant_05'])
        self.assertTrue(result[1]['significant_05'])
        self.assertTrue(result[2]['significant_05'])


if __name__=='__main__':unittest.main()
