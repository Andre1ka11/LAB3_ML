"""Offline local-model analysis using the same scoring functions as Yandex."""
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import local_runner
import yandex_runner as common
from analyze_yandex import (read_json, read_lines, write_json, write_csv, auto_correct,
    accuracy_ci, classification_metrics, classification_ci, calibration, h12_comparison,
    make_review_pack, review_status, execute_code, raw_text)

OUT = ROOT/'analysis/local'


def analysis_package_versions():
    packages = {name: version(name) for name in
                ['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'nbformat', 'nbclient']}
    # The inference backend is optional when recalculating saved responses.
    try:
        packages['llama-cpp-python'] = version('llama-cpp-python')
    except PackageNotFoundError:
        packages['llama-cpp-python'] = None
    return packages


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


def analyze():
    OUT.mkdir(parents=True,exist_ok=True)
    config,requests=local_runner.prepare()
    events=read_lines(local_runner.RUN/'events.jsonl')
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
                'confidence':p['confidence'] if p else None,'format_error':r['error'],'cost_rub':0})
    write_csv(OUT/'automatic_scores.csv',scores)
    types=[{'type':t,'n':8,**accuracy_ci([r['correct'] for r in scores if r['type']==t])} for t in 'ABF']
    write_csv(OUT/'type_metrics.csv',types)
    yandex=read_lines(ROOT/'results/models/yandex/yandex-v1/responses.jsonl')
    large=[{'task_id':r['task_id'],'type':r['task_id'][0],
            'correct':auto_correct(tasks[r['task_id']],common.parse_response(r['raw_response'],tasks[r['task_id']]['answer_kind'])[0])}
           for r in yandex if r['stage']=='main' and r['task_id'][0] in 'ABF']
    grouped=lambda rows:{t:sorted([r for r in rows if r['type']==t],key=lambda r:r['task_id']) for t in 'ABF'}
    h5=h5_comparison(grouped(large),grouped(scores))
    a={r['task_id']:r['correct'] for r in large}; b={r['task_id']:r['correct'] for r in scores}
    wins=sum(a[k] and not b[k] for k in a); losses=sum(b[k] and not a[k] for k in a)
    mcnemar={'n':24,'yandex_only_correct':wins,'local_only_correct':losses,
             'exact_two_sided_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.0,
             'scope':'One available LLM pair on A+B+F; no four-model family yet; exploratory.'}
    write_json(OUT/'paired_comparison.json',mcnemar)
    with (ROOT/'data/classification/test_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        labels={r['row_id']:int(r['label']) for r in csv.DictReader(f)}
    ids=sorted(labels); y=np.array([labels[k] for k in ids])
    predictions={}; classification=[]; export=[]
    for stage,shots in [('zero',0),('five',5),('twenty',20)]:
        rs={r['row_id']:r for r in records if r['stage']==stage}
        if set(rs)!=set(labels): raise ValueError('Missing classification rows')
        p=np.array([rs[k]['parsed']['answer'] if rs[k]['parsed'] else -1 for k in ids])
        predictions[shots]=p
        classification.append({'model':'Qwen3-4B','shots':shots,**classification_metrics(y,p),'ci':classification_ci(y,p),
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
    costs=[{'stage':s,'n':sum(r['stage']==s for r in records),'api_cost_rub':0,
            'latency_s':sum(r['latency_s'] for r in records if r['stage']==s),
            'input_tokens':sum(r['input_tokens'] for r in records if r['stage']==s),
            'output_tokens':sum(r['output_tokens'] for r in records if r['stage']==s)} for s in common.STAGES]
    for row in costs:
        row['mean_latency_s']=row['latency_s']/row['n']
        row['mean_total_tokens']=(row['input_tokens']+row['output_tokens'])/row['n']
    write_csv(OUT/'costs.csv',costs)
    summary={'responses':len(records),'pilot':5,'main':39,'classification':300,
             'auto_ABF':accuracy_ci([r['correct'] for r in scores]),'auto_ABF_n':24,
             'code_pass_at_1':sum(r['pass_at_1'] for r in code_results)/8,
             'code_metrics_match':sum(r['metrics_match_reference'] for r in code_results),
             'strict_format_errors':sum(not r['strict_format_ok'] for r in records if r['stage']!='pilot'),
             'hypotheses_status':{k:v['decision'] for k,v in hypotheses.items()},
             'overall_39_accuracy':None,'manual_review':reviews,'analysis_model_calls':0,
             'api_cost_rub':0,'total_cost_rub':None,'cost_note':'Energy and hardware costs not measured.',
             'generation_seconds':sum(r['latency_s'] for r in records),
             'tokens':sum(r['input_tokens']+r['output_tokens'] for r in records)}
    write_json(OUT/'summary.json',summary)
    inputs=['data/tasks.jsonl','data/bank_manifest.json','prompts/system_ru.txt','docs/local_registration.json',
            'results/models/local/qwen3-4b-q4-v1/events.jsonl','results/models/yandex/yandex-v1/events.jsonl',
            'results/reference/classification_predictions.csv']
    write_json(OUT/'provenance.json',{'hash_policy':'UTF-8 text bytes, CRLF normalized to LF for Git checkout portability',
        'sha256':{p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in inputs},
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'registration_commit':(local_runner.RUN/'registration_commit.txt').read_text().strip(),
        'seed':42,'bootstrap_iterations':1000,'model_calls':0})
    write_json(OUT/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
        'packages':analysis_package_versions(),
        'purpose':'Environment for offline analysis; original generation packages are recorded in package-versions.txt.'})
    plot(types,large,classification,hypotheses,costs)
    return summary


def plot(types,large,classification,hypotheses,costs):
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(directory/(name+'.png'),dpi=150);plt.close()
    fig,ax=plt.subplots(figsize=(7,4))
    for offset,model,rows in [(-.07,'YandexGPT',[{'type':t,**accuracy_ci([r['correct'] for r in large if r['type']==t])} for t in 'ABF']),(.07,'Qwen3-4B',types)]:
        ax.errorbar(np.arange(3)+offset,[r['accuracy'] for r in rows],yerr=[[r['accuracy']-r['ci_low'] for r in rows],[r['ci_high']-r['accuracy'] for r in rows]],fmt='o',capsize=4,label=model)
    ax.set(xticks=range(3),xticklabels=list('ABF'),ylim=(-.05,1.05),ylabel='Accuracy',title='95% bootstrap CI; n=8/type');ax.legend();save('accuracy_ci')
    fig,ax=plt.subplots(figsize=(7,3))
    heat=np.array([[np.mean([r['correct'] for r in large if r['type']==t]) if t in 'ABF' else np.nan for t in 'ABCDF'],[next((r['accuracy'] for r in types if r['type']==t),np.nan) for t in 'ABCDF']])
    cmap=plt.get_cmap('YlGn').copy();cmap.set_bad('lightgray')
    im=ax.imshow(heat,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set(xticks=range(5),xticklabels=list('ABCDF'),yticks=[0,1],yticklabels=['YandexGPT','Qwen3-4B'])
    for (i,j),v in np.ndenumerate(heat):ax.text(j,i,'pending' if np.isnan(v) else f'{v:.1%}',ha='center',va='center')
    fig.colorbar(im,ax=ax);save('heatmap')
    fig,ax=plt.subplots(figsize=(7,4))
    for model in ['Qwen3-4B','LogisticRegression','GaussianNB']:
        rows=[r for r in classification if r['model']==model]
        ax.plot([r['shots'] for r in rows],[r['f1_minority'] for r in rows],'o-',label=model)
    ax.set(xlabel='Shots',ylabel='F1 class 1',ylim=(-.05,1.05));ax.legend();save('classification_f1')
    fig,ax=plt.subplots(figsize=(5,4));h=hypotheses['H10'];bins=[r for r in h['bins'] if r['n']]
    ax.plot([0,1],[0,1],'--',color='gray');ax.scatter([r['mean_confidence'] for r in bins],[r['accuracy'] for r in bins],s=[r['n']*15 for r in bins])
    ax.set(xlim=(0,1.03),ylim=(0,1.03),xlabel='Confidence',ylabel='Accuracy',title=f'H10: {h["n_valid"]}/{h["n_total"]} valid');save('reliability')
    fig,ax=plt.subplots(figsize=(6,4))
    for r in hypotheses['H12']['comparisons']:
        d=r['delta_f1_llm_minus_lr'];ax.errorbar([r['shots']],[d],yerr=[[d-r['one_sided_95_lower']],[r['one_sided_95_upper']-d]],fmt='o',capsize=4)
    ax.axhline(0,color='gray');ax.axhline(-.05,color='red',ls='--');ax.set(xlabel='Shots',ylabel='F1 Qwen - F1 LR',xticks=[5,20],title='H12: one-sided 95% bounds');save('h12_delta')
    fig,ax=plt.subplots(figsize=(6,3));h=hypotheses['H5'];d=h['statistic_gap_A_minus_gap_B']
    ax.errorbar([0],[d],yerr=[[d-h['ci_low']],[h['ci_high']-d]],fmt='o',capsize=5);ax.axhline(0,color='gray');ax.set(xticks=[0],xticklabels=['(Yandex-Qwen) A minus B'],ylabel='Accuracy difference',title='H5: paired bootstrap');save('h5_gap')
    fig,ax=plt.subplots(figsize=(7,3));ax.bar([r['stage'] for r in costs],[r['latency_s'] for r in costs]);ax.set(ylabel='Generation seconds',title='Local inference time (load excluded)');save('runtime')
    fig,ax=plt.subplots(figsize=(6,4))
    ycost=sum(r['cost_rub'] for r in read_lines(ROOT/'results/models/yandex/yandex-v1/responses.jsonl') if r['stage']=='main' and r['task_id'][0] in 'ABF')
    for x,y,label in [(0,np.mean([r['accuracy'] for r in types]),'Qwen: API only'),(ycost,np.mean([r['correct'] for r in large]),'Yandex: API tariff')]:
        ax.scatter([x],[y]);ax.annotate(label,(x,y),xytext=(0,8),textcoords='offset points',ha='center')
    ax.margins(x=.3);ax.set(ylim=(0,1.1),xlabel='API cost A+B+F (RUB); energy/hardware excluded',ylabel='Accuracy',title='API cost vs accuracy; not total cost');save('cost_accuracy')


if __name__=='__main__':
    print(json.dumps(analyze(),ensure_ascii=False,indent=2))
