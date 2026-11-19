"""Inspect/extract unchanged code regions and execute them only as a secondary diagnostic."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from analyze_yandex import read_lines,read_json,write_json,normalize_traceback_paths,raw_text

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/gigachat'
NOTES={
 'D01':'Inspected numpy/sklearn only. Uses estimator.score (accuracy), labels it balanced_accuracy; stdout is Python dict.',
 'D02':'Inspected sklearn/numpy only; Pipeline used without import; preserve NameError.',
 'D03':'Inspected sklearn only; train-only CV and test score; stdout Python dict.',
 'D04':'Inspected sklearn only; output key balance_accuracy differs from required balanced_accuracy.',
 'D05':'Inspected sklearn/json references only; Pipeline and json imports missing; preserve NameError.',
 'D06':'Inspected sklearn/json only; body of invalid triple-quoted JSON answer, no code edits.',
 'D07':'Inspected sklearn only; descriptive clustering; stdout Python dict.',
 'D08':'Inspected sklearn only; correct original class mapping, JSON stdout rounded to three decimals.'}


def extract():
    rows=[r for r in read_lines(ROOT/'results/models/gigachat/gigachat-2-v1/responses.jsonl') if r['stage']=='main' and r['task_id'].startswith('D')]
    review={};primary=[]
    for row in rows:
        text=raw_text(row);match=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
        method='first fenced region, exact substring'
        if match is None:
            match=re.search('"""(.*?)"""',text,re.S);method='first triple-quoted region, exact substring'
        if match is None:raise ValueError('No diagnostic code region '+row['task_id'])
        code=match.group(1);tree=ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
                if any(m.split('.')[0] not in {'numpy','sklearn','json'} for m in modules):raise ValueError('Unreviewed import')
            if isinstance(node,ast.Attribute) and (node.attr.startswith('__') or node.attr.startswith('fetch_')):raise ValueError('Unreviewed attribute')
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'open','eval','exec','compile','input','getattr','globals','locals','__import__'}:raise ValueError('Unreviewed call')
        review[row['task_id']]={'source_sha256':hashlib.sha256(code.encode()).hexdigest(),
            'raw_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'start':match.start(1),'end':match.end(1),
            'extraction':method,'technical_inspection':NOTES[row['task_id']],
            'reviewer':'Codex technical review; not two independent human ratings'}
        if row['parsed'] is not None:raise ValueError('This diagnostic setup expects primary invalid-format D responses')
        primary.append({'task_id':row['task_id'],'source_sha256':None,'exit_code':None,'pass_at_1':False,
            'output_is_json':False,'metrics_match_reference':False,'metrics':None,'stderr':'',
            'status':'invalid_primary_response_format','format_error':row['error'],
            'methodology_review':'pending_two_independent_reviewers'})
    write_json(ROOT/'docs/gigachat_code_review.json',review)
    write_json(OUT/'code_execution.json',primary)
    return review,rows


def execute():
    review=read_json(ROOT/'docs/gigachat_code_review.json')
    rows=[r for r in read_lines(ROOT/'results/models/gigachat/gigachat-2-v1/responses.jsonl') if r['stage']=='main' and r['task_id'].startswith('D')]
    tasks={r['id']:r for r in read_lines(ROOT/'data/tasks.jsonl')}
    directory=OUT/'diagnostic_code_execution';directory.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k.upper() in {'SYSTEMROOT','WINDIR','PATH','COMSPEC'}}
    env.update(TEMP=str(directory.resolve()),TMP=str(directory.resolve()),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    results=[]
    for row in rows:
        task=row['task_id'];info=review[task];text=raw_text(row);code=text[info['start']:info['end']]
        assert hashlib.sha256(text.encode()).hexdigest()==info['raw_text_sha256']
        assert hashlib.sha256(code.encode()).hexdigest()==info['source_sha256']
        path=directory/(task+'.py');path.write_text(code,encoding='utf-8')
        try:
            proc=subprocess.run([sys.executable,'-I','-X','utf8',str(path.resolve())],cwd=directory,env=env,
                capture_output=True,text=True,encoding='utf-8',timeout=120)
            stdout,stderr,status=proc.stdout,proc.stderr,proc.returncode
        except subprocess.TimeoutExpired:stdout,stderr,status='','Timeout after 120 seconds',-1
        stderr=normalize_traceback_paths(stderr)
        path.with_suffix('.stdout.txt').write_text(stdout,encoding='utf-8');path.with_suffix('.stderr.txt').write_text(stderr,encoding='utf-8')
        metrics=None;is_json=False
        try:metrics=json.loads(stdout);is_json=True
        except ValueError:
            try:metrics=ast.literal_eval(re.sub(r'np\.float64\(([-+0-9.eE]+)\)',r'\1',stdout))
            except (ValueError,SyntaxError):pass
        values={('accuracy' if k=='test_accuracy' else k):v for k,v in metrics.items()} if isinstance(metrics,dict) else {}
        ref=tasks[task]['reference']['answer']['metrics']
        match=all(k in values and isinstance(values[k],(int,float)) and abs(values[k]-v)<=(0 if k in ['best_C','best_k'] else .01) for k,v in ref.items())
        results.append({'task_id':task,'source_sha256':info['source_sha256'],'exit_code':status,'runs_without_error':status==0,
            'output_is_json':is_json,'metrics':metrics,'metrics_match_reference':match,
            'primary_score_changed':False,'assessment':'secondary post-hoc extraction; primary invalid JSON remains failure',
            'inspection_note':info['technical_inspection'],'stderr':stderr})
    write_json(OUT/'diagnostic_code_execution.json',results)
    return results


if __name__=='__main__':
    if '--prepare' in sys.argv:extract();print('Inspected/extracted eight unchanged code regions; no execution')
    else:print(json.dumps(execute(),ensure_ascii=False,indent=2))
