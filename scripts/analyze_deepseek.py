"""Offline analysis of the complete main-bank and zero/five/twenty-shot snapshot."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import deepseek_runner as runner
from analysis_common import (read_json,write_json,write_csv,auto_correct,accuracy_ci,
    classification_metrics,classification_ci,calibration,h12_comparison,execute_code,
    make_review_pack,review_status,raw_text)

OUT=ROOT/'analysis/deepseek'
RUN=ROOT/'results/models/deepseek/deepseek-v1'
STAGES={'pilot':5,'main':39,'zero':100,'five':100,'twenty':100}


def journal(path):
    data=path.read_bytes()
    # A running writer may have a partially written last line. Only that line is ignored.
    complete=data if data.endswith(b'\n') else data.rsplit(b'\n',1)[0]+b'\n'
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def select(events,snapshot):
    wanted=snapshot['response_sha256']
    selected={}
    for event in events:
        rid=event.get('request_id')
        if event['event']=='response' and rid in wanted:
            if rid in selected: raise ValueError('Duplicate response in snapshot')
            if runner.digest(event)!=wanted[rid]: raise ValueError('Saved response changed')
            selected[rid]=event
    if set(selected)!=set(wanted): raise ValueError('Missing snapshot response')
    rows=[selected[rid] for rid in snapshot['request_order']]
    if dict(Counter(r['stage'] for r in rows))!=STAGES:
        raise ValueError('Snapshot must contain exactly pilot5/main39/zero100/five100/twenty100')
    return rows


def prepare_snapshot():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'snapshot.json'
    if path.exists(): raise ValueError('Snapshot already exists; never silently replace it')
    events=journal(RUN/'events.jsonl')
    responses=[e for e in events if e['event']=='response']
    rows=[e for e in responses if e['stage'] in STAGES]
    if dict(Counter(e['stage'] for e in rows))!=STAGES: raise ValueError('Requested stages not complete')
    config=read_json(RUN/'config.json')
    _,requests=runner.prepare(ROOT,config['model'])
    expected={r['request_id']:r for r in requests if r['stage'] in STAGES}
    if {e['request_id'] for e in rows}!=set(expected): raise ValueError('Unexpected request set')
    starts={e['request_id']:e for e in events if e['event']=='started'}
    for row in rows:
        if starts[row['request_id']]['request']!=expected[row['request_id']]['payload']:
            raise ValueError('Payload differs from the common bank')
    snapshot={'captured_utc':runner.now(),'model':config['model'],'scope':STAGES,
        'evaluated_requests':339,'pilot_excluded':True,'request_order':[e['request_id'] for e in rows],
        'response_sha256':{e['request_id']:runner.digest(e) for e in rows},
        'inventory_at_capture':dict(Counter(e['stage'] for e in responses)),
        'twenty_shot_policy':'Complete paired 100-row test included, using the same fixed rows and demonstrations.',
        'forecasts_commit':(RUN/'registration_commit.txt').read_text().strip()}
    write_json(path,snapshot)
    select(events,snapshot)
    print('Frozen complete snapshot: 344 saved responses, 339 evaluated; pilot excluded.')


def analyze(execute=False):
    snapshot=read_json(OUT/'snapshot.json')
    events=journal(RUN/'events.jsonl')
    original=select(events,snapshot)
    config,requests=runner.prepare(ROOT,snapshot['model'])
    by_id={r['request_id']:r for r in requests}
    records=[runner.decode_event(e,by_id[e['request_id']]['answer_kind']) for e in original]
    tasks={r['id']:r for r in runner.load_lines(ROOT/'data/tasks.jsonl')}
    scores=[]
    for record in records:
        if record['stage']=='main' and record['task_id'][0] in 'ABF':
            parsed=record['parsed']
            scores.append({'task_id':record['task_id'],'type':record['task_id'][0],
                'parsed':parsed is not None,'answer':parsed['answer'] if parsed else None,
                'correct':auto_correct(tasks[record['task_id']],parsed),
                'confidence':parsed['confidence'] if parsed else None,
                'format_error':record['error'],'cost_rub':record['cost_rub']})
    write_csv(OUT/'automatic_scores.csv',scores)
    types=[{'type':t,'n':8,**accuracy_ci([r['correct'] for r in scores if r['type']==t])} for t in 'ABF']
    write_csv(OUT/'type_metrics.csv',types)
    with (ROOT/'data/classification/test_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        labels={r['row_id']:int(r['label']) for r in csv.DictReader(f)}
    ids=sorted(labels);y=np.array([labels[k] for k in ids]);metrics=[];exports=[];predictions={}
    for stage,shots in [('zero',0),('five',5),('twenty',20)]:
        rs={r['row_id']:r for r in records if r['stage']==stage}
        if set(rs)!=set(labels): raise ValueError('Classification rows mismatch')
        p=np.array([rs[k]['parsed']['answer'] if rs[k]['parsed'] else -1 for k in ids])
        predictions[shots]=p
        incomplete=sum(r['raw_response'].get('status')!='completed' for r in rs.values())
        metrics.append({'model':'DeepSeek','shots':shots,**classification_metrics(y,p),
                        'incomplete_responses':incomplete,'ci':classification_ci(y,p)})
        exports.extend({'row_id':k,'shots':shots,'true_label':labels[k],'prediction':int(v),
                        'error':rs[k]['error']} for k,v in zip(ids,p))
    with (ROOT/'results/reference/classification_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        baseline=list(csv.DictReader(f))
    comparisons=[]
    for name in ['LogisticRegression','GaussianNB']:
        for shots in [5,20]:
            rs={r['row_id']:int(r['prediction']) for r in baseline if r['model']==name and int(r['n_train'])==shots}
            if set(rs)!=set(labels): raise ValueError('Baseline split differs')
            p=np.array([rs[k] for k in ids])
            metrics.append({'model':name,'shots':shots,**classification_metrics(y,p),'incomplete_responses':0,'ci':classification_ci(y,p)})
            if name=='LogisticRegression': comparisons.append(h12_comparison(y,predictions[shots],p,shots))
    p=np.zeros(len(ids),dtype=int)
    metrics.append({'model':'constant_majority','shots':0,**classification_metrics(y,p),'incomplete_responses':0,'ci':classification_ci(y,p)})
    write_json(OUT/'classification_metrics.json',metrics)
    write_csv(OUT/'classification_metrics.csv',[{k:v for k,v in r.items() if k!='ci'} for r in metrics])
    write_csv(OUT/'classification_predictions.csv',exports)
    h10=calibration(scores,tasks)
    hypotheses={'H5':{'decision':'not_evaluated_here','reason':'Primary comparison remains YandexGPT vs Qwen; see team analysis.'},
        'H10':h10,'H12':{'decision':'reject' if any(r['decision']=='reject' for r in comparisons) else 'confirm' if all(r['decision']=='confirm' for r in comparisons) else 'insufficient_data',
        'comparisons':comparisons,'primary_metric':'F1 minority class 1; paired bootstrap, seed 42'}}
    write_json(OUT/'hypotheses.json',hypotheses)
    if execute:
        execution=execute_code(ROOT,records,tasks,OUT)
    else:
        execution=read_json(OUT/'code_execution.json')
    for row in execution:
        record=next(r for r in records if r['stage']=='main' and r['task_id']==row['task_id'])
        if row['source_sha256']!=hashlib.sha256(record['parsed']['answer'].encode()).hexdigest():
            raise ValueError('Executed code differs from saved model answer')
    write_csv(OUT/'code_metrics.csv',[{k:v for k,v in r.items() if k not in ['metrics','stderr']} for r in execution])
    make_review_pack(ROOT,records,tasks,OUT)
    reviews=review_status(OUT)
    costs=[{'stage':s,'n':sum(r['stage']==s for r in records),
        'estimated_cost_rub':sum(r['cost_rub'] for r in records if r['stage']==s),
        'input_tokens':sum(r['input_tokens'] for r in records if r['stage']==s),
        'output_tokens':sum(r['output_tokens'] for r in records if r['stage']==s),
        'mean_latency_s':float(np.mean([r['latency_s'] for r in records if r['stage']==s and r['latency_s'] is not None]))} for s in STAGES]
    write_csv(OUT/'costs.csv',costs)
    summary={'model':snapshot['model'],'scope':'39 main + 100 zero + 100 five + 100 twenty; pilot excluded',
        'captured_utc':snapshot['captured_utc'],'snapshot_responses':344,'evaluated_requests':339,
        'inventory_at_capture':snapshot['inventory_at_capture'],'auto_ABF_n':24,
        'auto_ABF':accuracy_ci([r['correct'] for r in scores]),'overall_39_accuracy':None,
        'code_pass_at_1':sum(r['pass_at_1'] for r in execution)/8,
        'code_metrics_matched':sum(r['metrics_match_reference'] for r in execution),
        'format_errors':dict(Counter(r['error'] or 'none' for r in records if r['stage']!='pilot')),
        'scope_estimated_cost_rub':sum(r['cost_rub'] for r in records if r['stage']!='pilot'),
        'known_response_cost_rub':sum(r['cost_rub'] for r in records),
        'budget_ledger_rub':runner.replay(events)[1],
        'uncertain_attempt_reserve_rub':runner.replay(events)[1]-sum(r['cost_rub'] for r in records),
        'cost_note':'Published tariff applied to returned usage; not Billing. Ledger additionally retains conservative reserves for network attempts without responses.',
        'manual_review':reviews,'analysis_api_calls':0,'hypotheses_status':{k:v['decision'] for k,v in hypotheses.items()}}
    write_json(OUT/'summary.json',summary)
    examples=[{'task_id':r['task_id'],'prompt':tasks[r['task_id']]['text_ru'],
        'raw_response':raw_text(next(x for x in records if x['task_id']==r['task_id'] and x['stage']=='main')),
        'reference':tasks[r['task_id']]['reference'],'student_analysis':''} for r in scores if not r['correct']][:5]
    for record in records:
        if len(examples)>=5: break
        if record['stage'] in ('zero','five','twenty') and record['raw_response'].get('status')!='completed':
            raw=record['raw_response']
            examples.append({'task_id':record['task_id'],'row_id':record['row_id'],'shots':record['shots'],
                'prompt':by_id[record['request_id']]['payload']['input'],'raw_response':raw_text(record),
                'true_label':labels[record['row_id']],'provider_status':raw.get('status'),
                'incomplete_details':raw.get('incomplete_details'),'usage':raw.get('usage'),
                'selection':'First saved incomplete classification responses after the automatic A/B/F errors; not an invented reasoning trace.',
                'student_analysis':''})
    write_json(OUT/'error_examples.json',examples)
    write_json(OUT/'environment.json',{'python':platform.python_version(),
        'packages':{p:version(p) for p in ['numpy','scipy','scikit-learn','matplotlib','nbformat','nbclient']}})
    write_json(OUT/'provenance.json',{'snapshot_sha256':runner.digest(read_json(OUT/'snapshot.json')),
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'parser_version':runner.PARSER_VERSION,'registration_commit':snapshot['forecasts_commit'],
        'input_hashes':{p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in ['data/tasks.jsonl','prompts/system_ru.txt','data/classification/test_labels.csv','results/reference/classification_predictions.csv']},
        'model_calls':0,'seed':42,'bootstrap_iterations':1000})
    figures(types,metrics,h10,scores,comparisons)
    return summary


def figures(types,metrics,h10,scores,comparisons):
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(directory/(name+'.png'),dpi=150);plt.close()
    fig,ax=plt.subplots(figsize=(6,4))
    ax.errorbar(range(3),[r['accuracy'] for r in types],yerr=[[r['accuracy']-r['ci_low'] for r in types],[r['ci_high']-r['accuracy'] for r in types]],fmt='o',capsize=4)
    ax.set(xticks=range(3),xticklabels=list('ABF'),ylim=(-.05,1.05),ylabel='Accuracy',title='DeepSeek: 95% bootstrap CI, n=8/type');save('accuracy_ci')
    fig,ax=plt.subplots(figsize=(7,2.5))
    values=np.array([[next((r['accuracy'] for r in types if r['type']==t),np.nan) for t in 'ABCDF']])
    cmap=plt.get_cmap('YlGn').copy();cmap.set_bad('lightgray')
    im=ax.imshow(values,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set(xticks=range(5),xticklabels=list('ABCDF'),yticks=[0],yticklabels=['DeepSeek'])
    for j,value in enumerate(values[0]):ax.text(j,0,'pending\nreview' if np.isnan(value) else f'{value:.1%}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax);save('heatmap')
    fig,ax=plt.subplots(figsize=(11,4.5))
    labels=[r['model']+' '+str(r['shots'])+'-shot' for r in metrics]
    x=np.arange(len(metrics));ax.bar(x-.16,[r['f1_minority'] for r in metrics],.32,label='F1 class 1');ax.bar(x+.16,[r['balanced_accuracy'] for r in metrics],.32,label='Balanced accuracy')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.05));ax.tick_params(axis='x',labelrotation=25,labelsize=8);ax.legend();save('classification')
    fig,ax=plt.subplots(figsize=(5,4));bins=[r for r in h10['bins'] if r['n']]
    ax.plot([0,1],[0,1],'--',color='gray');ax.scatter([r['mean_confidence'] for r in bins],[r['accuracy'] for r in bins],s=[r['n']*15 for r in bins])
    ax.set(xlim=(0,1.03),ylim=(0,1.03),xlabel='Confidence',ylabel='Accuracy',title='H10: A+B+F');save('reliability')
    fig,ax=plt.subplots(figsize=(6,4))
    for row in comparisons:
        delta=row['delta_f1_llm_minus_lr']
        ax.errorbar([row['shots']],[delta],yerr=[[delta-row['one_sided_95_lower']],[row['one_sided_95_upper']-delta]],fmt='o',capsize=4)
    ax.axhline(0,color='gray');ax.axhline(-.05,color='red',ls='--')
    ax.set(xticks=[5,20],xlabel='Shots',ylabel='F1 DeepSeek minus F1 LR',title='H12: paired bootstrap one-sided 95% bounds');save('h12_delta')
    fig,ax=plt.subplots(figsize=(5,4));price=sum(r['cost_rub'] for r in scores);accuracy=np.mean([r['correct'] for r in scores])
    ax.scatter([price],[accuracy]);ax.annotate('DeepSeek A+B+F',(price,accuracy),xytext=(0,10),textcoords='offset points',ha='center')
    ax.set(ylim=(0,1.05),xlabel='Estimated API cost of A+B+F, RUB',ylabel='Accuracy',title='One-model snapshot; comparison added after team merge');save('cost_accuracy')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',action='store_true')
    parser.add_argument('--execute-reviewed-code',action='store_true')
    args=parser.parse_args()
    if args.snapshot: prepare_snapshot()
    else: print(json.dumps(analyze(args.execute_reviewed_code),ensure_ascii=False,indent=2))
