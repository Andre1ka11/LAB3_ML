"""Offline checks of cross-model comparisons and prompt integrity."""
from pathlib import Path
import sys
import unittest
import json
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import local_runner
import yandex_runner
from analyze_local import analysis_package_versions, h5_comparison


class LocalTests(unittest.TestCase):
    def test_identical_shared_requests_except_model(self):
        config,local=local_runner.prepare()
        _,cloud=yandex_runner.prepare(ROOT,'cloud-model')
        self.assertEqual(len(local),344)
        for a,b in zip(local,cloud):
            self.assertEqual({k:v for k,v in a['payload'].items() if k!='model'},
                             {k:v for k,v in b['payload'].items() if k!='model'})

    def test_template_preserves_user_and_system(self):
        prompt=local_runner.chat_prompt('unchanged-system','unchanged-user')
        self.assertIn('<|im_start|system\nunchanged-system<|im_end|>',prompt)
        self.assertIn('<|im_start|user\nunchanged-user<|im_end|>',prompt)
        self.assertTrue(prompt.endswith('<think>\n\n</think>\n\n'))

    def test_h5_direction_and_equal_pairing(self):
        def rows(value):return [{'correct':value} for _ in range(8)]
        large={'A':rows(True),'B':rows(True)}
        small={'A':rows(False),'B':rows(True)}
        result=h5_comparison(large,small)
        self.assertEqual(result['statistic_gap_A_minus_gap_B'],1)
        self.assertEqual(result['decision'],'confirm')
        self.assertEqual(h5_comparison(large,large)['decision'],'reject')

    def test_truncated_answer_does_not_become_success(self):
        raw={'status':'incomplete','output':[{'content':[{'type':'output_text','text':json.dumps({'answer':.2,'explanation':'','confidence':99})}]}]}
        self.assertEqual(yandex_runner.parse_response(raw,'number'),(None,'response_not_completed'))


    def test_offline_analysis_does_not_require_inference_backend(self):
        def available(name):
            if name == 'llama-cpp-python':
                raise PackageNotFoundError(name)
            return 'test-version'
        with patch('analyze_local.version', side_effect=available):
            packages = analysis_package_versions()
        self.assertIsNone(packages['llama-cpp-python'])
        self.assertEqual(packages['numpy'], 'test-version')

    def test_missing_analysis_dependency_is_not_hidden(self):
        with patch('analyze_local.version', side_effect=PackageNotFoundError('numpy')):
            with self.assertRaises(PackageNotFoundError):
                analysis_package_versions()


if __name__=='__main__':unittest.main()
