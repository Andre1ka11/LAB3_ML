"""Offline GigaChat-2 analysis; no API calls."""
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import gigachat_runner
import yandex_runner as common
from analyze_yandex import (read_json, read_lines, write_json, write_csv, auto_correct,
    accuracy_ci, classification_metrics, classification_ci, calibration, h12_comparison,
    make_review_pack, review_status, execute_code, raw_text)

OUT = ROOT/'analysis/gigachat'


def h5_comparison(large, small):
    rng=np.random.default_rng(42)
    diffs={t:np.array([int(a['correct'])-int(b['correct']) for a,b in zip(large[t],small[t])]) for t in 'AB'}
    stat=float(diffs['A'].mean()-diffs['B'].mean())
    draws=[]
    for _ in range(1000):
        draws.append(float(np.mean(rng.choice(diffs['A'],len(diffs['A']),replace=True))-
                           np.mean(rng.choice(diffs['B'],len(diffs['B']),replace=True))))
    lo,hi=map(float,np.quantile(draws,[.025,.975]))
    return {'statistic_gap_A_minus_gap_B':stat,'ci_low':lo,'ci_high':hi,
            'decision':'confirm' if lo>0 else 'reject' if hi<=0 else 'insufficient_data',
            'n_A':len(diffs['A']),'n_B':len(diffs['B']),'seed':42,'bootstrap_iterations':1000}


def explicitly_refused(record):
    try:
        return json.loads(raw_text(record)).get('answer')=='insufficient_data'
    except (ValueError,AttributeError):
        return False


def paired_family(yandex,qwen,giga):
    import itertools
    models={'YandexGPT':yandex,'Qwen3-4B':qwen,'GigaChat-2':giga}
    results=[]
    for first,second in itertools.combinations(models,2):
        a={r['task_id']:r['correct'] for r in models[first]};b={r['task_id']:r['correct'] for r in models[second]}
        assert set(a)==set(b)
        wins=sum(a[k] and not b[k] for k in a);losses=sum(b[k] and not a[k] for k in a)
        p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.0
        results.append({'first':first,'second':second,'n':len(a),'first_only_correct':wins,'second_only_correct':losses,'p_exact':p})
    adjusted=0
    for rank,r in enumerate(sorted(results,key=lambda r:r['p_exact'])):
        adjusted=max(adjusted,min(1,r['p_exact']*(len(results)-rank)))
        r['p_holm']=adjusted;r['significant_05']=adjusted<.05
    write_json(OUT/'paired_comparisons.json',{'family':'Three available models, 24 A/B/F tasks; exploratory comparisons','results':results})


def analyze():
    OUT.mkdir(parents=True,exist_ok=True)
    config,requests=gigachat_runner.prepare()
    qrows=read_lines(ROOT/'results/models/local/qwen3-4b-q4-v1/responses.jsonl')
    qt={r['id']:r for r in read_lines(ROOT/'data/tasks.jsonl')}
    q=[{'task_id':r['task_id'],'type':r['task_id'][0],'correct':auto_correct(qt[r['task_id']],common.parse_response(r['raw_response'],qt[r['task_id']]['answer_kind'])[0])} for r in qrows if r['stage']=='main' and r['task_id'][0] in 'ABF']
    write_json(OUT/'comparison_scores_qwen.json',q)
    events=read_lines(gigachat_runner.RUN/'events.jsonl')
    done,_,_=common.replay(events)
    if set(done)!={r['request_id'] for r in requests}:
        raise ValueError('Expected exactly 344 distinct completed requests')
    records=[common.decode_event(done[r['request_id']],r['answer_kind']) for r in requests]
    tasks={r['id']:r for r in read_lines(ROOT/'data/tasks.jsonl')}
    scores=[]
    for r in records:
        if r['stage']=='main' and r['task_id'][0] in 'ABF':
            p=r['parsed']
            scores.append({'task_id':r['task_id'],'type':r['task_id'][0], 'parsed':p is not None,
                'answer':p['answer'] if p else None,'correct':auto_correct(tasks[r['task_id']],p),
                'confidence':p['confidence'] if p else None,'format_error':r['error'],'cost_rub':r['cost_rub']})
    write_csv(OUT/'automatic_scores.csv',scores)
    types=[{'type':t,'n':8,**accuracy_ci([r['correct'] for r in scores if r['type']==t])} for t in 'ABF']
    write_csv(OUT/'type_metrics.csv',types)
    yandex=read_lines(ROOT/'results/models/yandex/yandex-v1/responses.jsonl')
    large=[{'task_id':r['task_id'],'type':r['task_id'][0],
            'correct':auto_correct(tasks[r['task_id']],common.parse_response(r['raw_response'],tasks[r['task_id']]['answer_kind'])[0])}
           for r in yandex if r['stage']=='main' and r['task_id'][0] in 'ABF']
    grouped=lambda rows:{t:sorted([r for r in rows if r['type']==t],key=lambda r:r['task_id']) for t in 'ABF'}
    h5=read_json(ROOT/'analysis/local/hypotheses.json')['H5']
    h5={**h5,'comparison':'Primary: YandexGPT vs Qwen3-4B, unchanged from local experiment'}
    qwen_scores=read_json(ROOT/'analysis/gigachat/comparison_scores_qwen.json')
    h5['secondary_gigachat_vs_qwen']=h5_comparison(grouped(scores),grouped(qwen_scores))
    a={r['task_id']:r['correct'] for r in large}; b={r['task_id']:r['correct'] for r in scores}
    wins=sum(a[k] and not b[k] for k in a); losses=sum(b[k] and not a[k] for k in a)
    mcnemar={'n':24,'yandex_only_correct':wins,'gigachat_only_correct':losses,
             'exact_two_sided_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.0,
             'scope':'Yandex-GigaChat descriptive pair; full available three-model family in paired_comparisons.json.'}
    write_json(OUT/'paired_comparison.json',mcnemar)
    paired_family(large,qwen_scores,scores)
    with (ROOT/'data/classification/test_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        labels={r['row_id']:int(r['label']) for r in csv.DictReader(f)}
    ids=sorted(labels); y=np.array([labels[k] for k in ids])
    predictions={}; classification=[]; export=[]
    for stage,shots in [('zero',0),('five',5),('twenty',20)]:
        rs={r['row_id']:r for r in records if r['stage']==stage}
        if set(rs)!=set(labels): raise ValueError('Missing classification rows')
        p=np.array([rs[k]['parsed']['answer'] if rs[k]['parsed'] else -1 for k in ids])
        predictions[shots]=p
        classification.append({'model':'GigaChat-2','shots':shots,**classification_metrics(y,p),'ci':classification_ci(y,p),
            'explicit_refusals':sum(explicitly_refused(r) for r in rs.values()),
            'other_unscored':sum(r['parsed'] is None and not explicitly_refused(r) for r in rs.values())})
        export += [{'row_id':k,'shots':shots,'true_label':labels[k],'prediction':int(v),'unscored':v==-1,
                    'explicit_refusal':explicitly_refused(rs[k]),'format_error':rs[k]['error']} for k,v in zip(ids,p)]
    write_csv(OUT/'classification_predictions.csv',export)
    with (ROOT/'results/reference/classification_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        baselines=list(csv.DictReader(f))
    comparisons=[]
    for model in ['LogisticRegression','GaussianNB']:
        for shots in [5,20]:
            rs={r['row_id']:int(r['prediction']) for r in baselines if r['model']==model and int(r['n_train'])==shots}
            if set(rs)!=set(labels): raise ValueError('Baseline rows differ')
            p=np.array([rs[k] for k in ids])
            classification.append({'model':model,'shots':shots,**classification_metrics(y,p),'ci':classification_ci(y,p),
                                   'explicit_refusals':0,'other_unscored':0})
            if model=='LogisticRegression': comparisons.append(h12_comparison(y,predictions[shots],p,shots))
    classification.append({'model':'constant_majority','shots':0,**classification_metrics(y,np.zeros(100,dtype=int)),
                           'ci':classification_ci(y,np.zeros(100,dtype=int)),'explicit_refusals':0,'other_unscored':0})
    write_json(OUT/'classification_metrics.json',classification)
    write_csv(OUT/'classification_metrics.csv',[{k:v for k,v in r.items() if k!='ci'} for r in classification])
    h10=calibration(scores,tasks)
    h12={'decision':'reject' if any(r['decision']=='reject' for r in comparisons) else 'confirm' if all(r['decision']=='confirm' for r in comparisons) else 'insufficient_data','comparisons':comparisons}
    hypotheses={'H5':h5,'H10':h10,'H12':h12}
    write_json(OUT/'hypotheses.json',hypotheses)
    code_results=read_json(OUT/'code_execution.json')
    for r in code_results:
        response=next(x for x in records if x['stage']=='main' and x['task_id']==r['task_id'])
        if response['parsed']:
            assert hashlib.sha256(response['parsed']['answer'].encode()).hexdigest()==r['source_sha256']
    write_csv(OUT/'code_metrics.csv',[{k:v for k,v in r.items() if k not in ['metrics','stderr']} for r in code_results])
    make_review_pack(ROOT,records,tasks,OUT)
    reviews=review_status(OUT)
    write_json(OUT/'error_examples.json',[{'task_id':r['task_id'],'prompt':tasks[r['task_id']]['text_ru'],
        'raw_response':raw_text(next(x for x in records if x['stage']=='main' and x['task_id']==r['task_id'])),
        'reference':tasks[r['task_id']]['reference'],'student_analysis':''} for r in scores if not r['correct']][:5])
    costs=[{'stage':s,'n':sum(r['stage']==s for r in records),'api_cost_estimate_rub':sum(r['cost_rub'] for r in records if r['stage']==s),
            'latency_s':sum(r['latency_s'] for r in records if r['stage']==s),
            'billable_input_tokens':sum(r['input_tokens'] for r in records if r['stage']==s),
            'cached_input_tokens':sum(r['raw_response']['provider_response']['usage'].get('precached_prompt_tokens',0) for r in records if r['stage']==s),
            'output_tokens':sum(r['output_tokens'] for r in records if r['stage']==s)} for s in common.STAGES]
    for row in costs:
        row['mean_latency_s']=row['latency_s']/row['n']
        row['mean_billable_tokens']=(row['billable_input_tokens']+row['output_tokens'])/row['n']
        row['mean_full_tokens']=(row['billable_input_tokens']+row['cached_input_tokens']+row['output_tokens'])/row['n']
    write_csv(OUT/'costs.csv',costs)
    summary={'responses':len(records),'pilot':5,'main':39,'classification':300,'model_versions_returned':sorted({r['model_version_returned'] for r in records}),
             'auto_ABF':accuracy_ci([r['correct'] for r in scores]),'auto_ABF_n':24,
             'code_pass_at_1':sum(r['pass_at_1'] for r in code_results)/8,
             'code_metrics_match':sum(r['metrics_match_reference'] for r in code_results),
             'strict_format_errors':sum(not r['strict_format_ok'] for r in records if r['stage']!='pilot'),
             'hypotheses_status':{k:v['decision'] for k,v in hypotheses.items()},
             'overall_39_accuracy':None,'manual_review':reviews,'analysis_model_calls':0,
             'api_cost_estimate_rub':sum(r['cost_rub'] for r in records),'actual_cash_payment_rub':0,'cost_note':'Freemium; published package-equivalent estimate, not actual charge or purchased package.',
             'generation_seconds':sum(r['latency_s'] for r in records),
             'billable_tokens':sum(r['input_tokens']+r['output_tokens'] for r in records),
             'cached_input_tokens':sum(r['raw_response']['provider_response']['usage'].get('precached_prompt_tokens',0) for r in records),
             'token_usage_note':'prompt_tokens excludes precached_prompt_tokens; total_tokens is billable, cached tokens are free',
             'diagnostic_code_runs':sum(r['runs_without_error'] for r in read_json(OUT/'diagnostic_code_execution.json'))}
    write_json(OUT/'summary.json',summary)
    inputs=['data/tasks.jsonl','data/bank_manifest.json','prompts/system_ru.txt','docs/gigachat_registration.json',
            'results/models/gigachat/gigachat-2-v1/events.jsonl','results/models/yandex/yandex-v1/events.jsonl',
            'results/models/local/qwen3-4b-q4-v1/events.jsonl','analysis/local/hypotheses.json',
            'analysis/gigachat/code_execution.json','analysis/gigachat/diagnostic_code_execution.json',
            'docs/gigachat_code_review.json','results/reference/classification_predictions.csv']
    write_json(OUT/'provenance.json',{'hash_policy':'UTF-8 text bytes, CRLF normalized to LF for Git checkout portability',
        'sha256':{p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in inputs},
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'registration_commit':(gigachat_runner.RUN/'registration_commit.txt').read_text().strip(),
        'seed':42,'bootstrap_iterations':1000,'model_calls':0})
    write_json(OUT/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
        'packages':{p:version(p) for p in ['numpy','scipy','scikit-learn','matplotlib','requests','nbformat','nbclient']}})
    plot(types,large,classification,hypotheses,costs)
    return summary



def plot(*args):
    from gigachat_figures import plot as draw
    draw(ROOT,OUT,*args)


if __name__=='__main__':
    print(json.dumps(analyze(),ensure_ascii=False,indent=2))
