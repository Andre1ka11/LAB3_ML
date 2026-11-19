"""Execute only manually inspected, hash-pinned model answers; never repair code."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from analyze_yandex import read_lines, read_json, write_json, normalize_traceback_paths

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/local'


def execute():
    rows=[r for r in read_lines(ROOT/'results/models/local/qwen3-4b-q4-v1/responses.jsonl') if r['stage']=='main' and r['task_id'].startswith('D')]
    reviewed=read_json(ROOT/'docs/local_code_review.json')
    tasks={r['id']:r for r in read_lines(ROOT/'data/tasks.jsonl')}
    scratch=OUT/'code_execution';scratch.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k.upper() in {'SYSTEMROOT','WINDIR','PATH','COMSPEC'}}
    env.update(TEMP=str(scratch.resolve()),TMP=str(scratch.resolve()),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    results=[]
    for r in rows:
        task=r['task_id'];parsed=r['parsed'];code=parsed['answer'] if parsed else None
        sha=hashlib.sha256(code.encode()).hexdigest() if code is not None else None
        if sha != reviewed[task]['source_sha256']:raise ValueError('Answer differs from inspected source')
        if code is None:
            results.append({'task_id':task,'source_sha256':None,'exit_code':None,'pass_at_1':False,
                'output_is_json':False,'metrics_match_reference':False,'metrics':None,
                'status':'not_executable_response_format','format_error':r['error'],
                'stderr':'','methodology_review':'pending_two_independent_reviewers'})
            continue
        path=scratch/(task+'.py');path.write_text(code,encoding='utf-8')
        try:
            proc=subprocess.run([sys.executable,'-I','-X','utf8',str(path.resolve())],cwd=scratch,
                env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
            stdout,stderr,exit_code=proc.stdout,proc.stderr,proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout,stderr,exit_code='', 'Timeout after 120 seconds',-1
        original_hash=hashlib.sha256(stderr.encode()).hexdigest();stderr=normalize_traceback_paths(stderr)
        path.with_suffix('.stdout.txt').write_text(stdout,encoding='utf-8')
        path.with_suffix('.stderr.txt').write_text(stderr,encoding='utf-8')
        metrics=None;is_json=False
        try:
            metrics=json.loads(stdout);is_json=True
        except ValueError:pass
        normalized={('accuracy' if k=='test_accuracy' else 'silhouette' if k=='silhouette_score' else k):v for k,v in metrics.items()} if isinstance(metrics,dict) else {}
        ref=tasks[task]['reference']['answer']['metrics']
        matched=all(k in normalized and isinstance(normalized[k],(int,float)) and abs(normalized[k]-v)<=(0 if k in ['best_C','best_k'] else .01) for k,v in ref.items())
        results.append({'task_id':task,'source_sha256':sha,'exit_code':exit_code,'pass_at_1':exit_code==0,
            'output_is_json':is_json,'metrics_match_reference':matched,'metrics':metrics,
            'status':'executed_first_answer','format_error':r['error'],'stderr':stderr,
            'stderr_original_sha256':original_hash,'methodology_review':'pending_two_independent_reviewers'})
    write_json(OUT/'code_execution.json',results)
    return results


if __name__=='__main__':print(json.dumps(execute(),ensure_ascii=False,indent=2))
