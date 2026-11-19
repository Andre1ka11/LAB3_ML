"""Unified offline audit and analysis of the four first-response experiments."""
import contextlib
import csv
from datetime import datetime,timezone
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from scipy.stats import binomtest
from analysis_common import (read_json,read_lines,write_json,write_csv,raw_text,auto_correct,
    accuracy_ci,classification_metrics,classification_ci,calibration,h12_comparison)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MODELS={
 'YandexGPT':('yandex/yandex-v1','yandex','yandex_runner','yandex_registration.json'),
 'Qwen3-4B':('local/qwen3-4b-q4-v1','local','local_runner','local_registration.json'),
 'GigaChat-2':('gigachat/gigachat-2-v1','gigachat','gigachat_runner','gigachat_registration.json'),
 'DeepSeek':('deepseek/deepseek-v1','deepseek','deepseek_runner','deepseek_registration.json')}
COUNTS={'pilot':5,'main':39,'zero':100,'five':100,'twenty':100}
OUT=ROOT/'analysis/defense'


def sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


@contextlib.contextmanager
def offline():
    """Fail immediately if offline analysis accidentally calls an HTTP transport."""
    # Import before patching: otherwise `from urllib.request import urlopen` would
    # retain a mock after leaving the context and break the optional live probe.
    runners=[importlib.import_module(name) for name in ['yandex_runner','deepseek_runner']]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch('requests.sessions.Session.request',side_effect=RuntimeError('Network blocked in defense analysis')))
        stack.enter_context(patch('urllib.request.urlopen',side_effect=RuntimeError('Network blocked in defense analysis')))
        stack.enter_context(patch('http.client.HTTPConnection.connect',side_effect=RuntimeError('Network blocked in defense analysis')))
        stack.enter_context(patch('http.client.HTTPSConnection.connect',side_effect=RuntimeError('Network blocked in defense analysis')))
        for runner in runners:
            if hasattr(runner,'urlopen'):stack.enter_context(patch.object(runner,'urlopen',side_effect=RuntimeError('Network blocked in defense analysis')))
        yield


def paths(root=ROOT):
    names=['data/bank_manifest.json','data/tasks.jsonl','data/tasks_public.jsonl','data/pilot_tasks.jsonl',
           'prompts/system_ru.txt','docs/protocol.md','docs/hypotheses_template.md',
           'results/reference/classification_baselines.json','results/reference/classification_predictions.csv','data/metadata_cleanup.json']
    names += [str(p.relative_to(root)).replace('\\','/') for d in ['data/requests','data/classification'] for p in (root/d).glob('*') if p.is_file()]
    for folder,analysis,runner,reg in MODELS.values():
        names += [runner+'.py','docs/'+reg,'analysis/'+analysis+'/code_execution.json']
        names += ['analysis/'+analysis+'/review_status.json']
        names += [str(p.relative_to(root)).replace('\\','/') for p in (root/'analysis'/analysis/'code_execution').glob('*') if p.is_file()]
        names += ['results/models/'+folder+'/'+file for file in ['config.json','events.jsonl','responses.jsonl','registration_commit.txt']]
    names += ['analysis/gigachat/diagnostic_code_execution.json','docs/gigachat_code_review.json','docs/local_code_review.json',
              'analysis/deepseek/snapshot.json','analysis/local/hardware.json']
    names += [str(p.relative_to(root)).replace('\\','/') for p in (root/'analysis/gigachat/diagnostic_code_execution').glob('*') if p.is_file()]
    return sorted(set(names))


def freeze_sources(root=ROOT):
    out=root/'analysis/defense';out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    write_json(out/'source_manifest.json',{'source_commit':commit,'hash_policy':'CRLF -> LF, UTF-8 bytes otherwise unchanged',
        'sha256':{name:sha(root/name) for name in paths(root)},
        'meaning':'Frozen local integrity evidence. Not provider signatures and not absolute proof against fabricated logs.'})


def load_model(name,root=ROOT):
    folder,_,module,_=MODELS[name];run=root/'results/models'/folder
    config=read_json(run/'config.json');runner=importlib.import_module(module)
    _,requests=runner.prepare(root,config['model']) if name in ['YandexGPT','DeepSeek'] else runner.prepare(root)
    expected={r['request_id']:r for r in requests}
    events=read_lines(run/'events.jsonl')
    responses=[r for r in events if r['event']=='response']
    assert len(responses)==344 and len({r['request_id'] for r in responses})==344
    assert {r['request_id'] for r in responses}==set(expected)
    starts={}
    for event in events:
        rid=event.get('request_id')
        if event['event']=='started':
            assert rid in expected and event['request']==expected[rid]['payload'],(name,'payload mismatch')
            starts[rid]=event
        elif event['event']=='response':
            assert rid in starts
            assert datetime.fromisoformat(starts[rid]['timestamp_utc'])<=datetime.fromisoformat(event['timestamp_utc'])
    byid={r['request_id']:r for r in responses}
    records=[]
    for request in requests:
        row=byid[request['request_id']]
        # One identical parser; null reasoning content is handled without changing the answer.
        parser=importlib.import_module('deepseek_runner')
        parsed,error=parser.parse_response(row['raw_response'],request['answer_kind'])
        records.append({**row,'parsed':parsed,'error':error,'strict_format_ok':parsed is not None and error is None})
    from collections import Counter
    assert dict(Counter(r['stage'] for r in records))==COUNTS
    return config,requests,events,records


def audit(root=ROOT):
    manifest=read_json(root/'analysis/defense/source_manifest.json')
    for name,expected in manifest['sha256'].items():assert sha(root/name)==expected,('Source changed',name)
    rows=[];payloads=[]
    for name,(folder,analysis,module,regfile) in MODELS.items():
        config,requests,events,records=load_model(name,root)
        for row in records:
            raw=row['raw_response'];usage=raw.get('usage') or {}
            assert row['input_tokens']==usage['input_tokens'] and row['output_tokens']==usage['output_tokens']
            if name=='Qwen3-4B':calculated_cost=0
            elif name=='GigaChat-2':
                provider_usage=raw['provider_response']['usage']
                assert provider_usage['total_tokens']==row['input_tokens']+row['output_tokens']
                calculated_cost=provider_usage['total_tokens']*config['rate_rub_per_token']
            elif name=='YandexGPT':
                calculated_cost=(usage['input_tokens']+usage['output_tokens'])*config['rate_rub_per_token']
            else:
                rates=config['rates_rub_per_token'];cached=(usage.get('input_tokens_details') or {}).get('cached_tokens',0)
                assert 0<=cached<=usage['input_tokens']
                calculated_cost=(usage['input_tokens']-cached)*rates['input']+cached*rates['cached_input']+usage['output_tokens']*rates['output']
            assert np.isclose(row['cost_rub'],calculated_cost,rtol=0,atol=1e-10),(name,'cost/usage mismatch')
        texts=[]
        for r in requests:
            p=r['payload']
            if 'messages' in p:
                assert [m['role'] for m in p['messages']]==['system','user']
                system,user=[m['content'] for m in p['messages']]
                limit=p['max_tokens']
            else:system,user,limit=p['instructions'],p['input'],p['max_output_tokens']
            assert p['temperature']==0 and limit==4096
            assert user==r['user_prompt']
            texts.append((r['stage'],r['task_id'],r.get('row_id'),r.get('shots'),system,user))
        payloads.append(texts)
        regcommit=(root/'results/models'/folder/'registration_commit.txt').read_text().strip()
        git_ok=False;before_main=None;commit_utc=None
        try:
            original=subprocess.check_output(['git','show',regcommit+':docs/'+regfile],cwd=root,stderr=subprocess.DEVNULL)
            assert json.loads(original)==read_json(root/'docs'/regfile)
            timestamp=int(subprocess.check_output(['git','show','-s','--format=%ct',regcommit],cwd=root,text=True))
            commit_utc=datetime.fromtimestamp(timestamp,timezone.utc).isoformat()
            before_main=timestamp<=min(datetime.fromisoformat(r['timestamp_utc']).timestamp() for r in events if r['event']=='started' and r['stage']=='main')
            git_ok=True
        except subprocess.CalledProcessError:pass
        if git_ok:assert before_main,(name,'registration after main')
        uncertainty=sum(e['event']=='uncertain' for e in events)
        authorizations=sum(e['event']=='retry_authorized' for e in events)
        assert uncertainty==authorizations
        if name=='DeepSeek':
            snapshot=read_json(root/'analysis/deepseek/snapshot.json')
            digest=importlib.import_module('deepseek_runner').digest
            assert snapshot['response_sha256']=={r['request_id']:digest(r) for r in events if r['event']=='response'}
        rows.append({'model':name,'responses':len(records),'starts':sum(e['event']=='started' for e in events),
            'uncertain':uncertainty,'authorized_network_retries':authorizations,'registration_commit':regcommit,
            'registration_git_verified':git_ok,'registration_before_main':before_main,'commit_utc':commit_utc,
            'cost_usage_verified':True,
            'first_response_utc':min(r['timestamp_utc'] for r in records),'last_response_utc':max(r['timestamp_utc'] for r in records),
            'first_main_started_utc':min(r['timestamp_utc'] for r in events if r['event']=='started' and r['stage']=='main'),
            'model_id':config['model'],'versions_returned':sorted({r.get('model_version_returned') or r['raw_response'].get('model') or r['model_id'] for r in records}),
            'journal_sha256':sha(root/'results/models'/folder/'events.jsonl')})
    assert all(texts==payloads[0] for texts in payloads[1:]),'Model prompt texts differ'
    labels=read_json(root/'data/classification/split_indices.json')
    assert set(labels['train5'])<=set(labels['train20']) and set(labels['train20']).isdisjoint(labels['test'])
    return {'models':rows,'frozen_files_verified':len(manifest['sha256']),'all_prompt_texts_identical':True,
            'splits_disjoint':True,'source_commit':manifest['source_commit'],
            'limits':'Logs/hashes are internally consistent evidence, not signed provider attestations. Compare Billing externally if needed.'}


def baselines(root=ROOT):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    features=read_json(root/'data/classification/protocol.json')['feature_names']
    test=sorted(read_lines(root/'data/classification/test_features.jsonl'),key=lambda r:r['row_id'])
    with (root/'data/classification/test_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        labels={r['row_id']:int(r['label']) for r in csv.DictReader(f)}
    ids=[r['row_id'] for r in test];y=np.array([labels[k] for k in ids]);X=np.array([[r['features'][k] for k in features] for r in test])
    with (root/'results/reference/classification_predictions.csv').open(encoding='utf-8-sig',newline='') as f:stored=list(csv.DictReader(f))
    rows=[];predictions={}
    for shots in [5,20]:
        train=read_lines(root/f'data/classification/train{shots}.jsonl')
        xt=np.array([[r['features'][k] for k in features] for r in train]);yt=np.array([r['label'] for r in train])
        for name,model in [('LogisticRegression',make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))),('GaussianNB',GaussianNB())]:
            model.fit(xt,yt);p=model.predict(X)
            saved={r['row_id']:int(r['prediction']) for r in stored if r['model']==name and int(r['n_train'])==shots}
            assert np.array_equal(p,[saved[k] for k in ids]),('Baseline mismatch',name,shots)
            predictions[name,shots]=p;rows.append({'model':name,'shots':shots,**classification_metrics(y,p),'predictions_reproduced':True})
    return ids,y,predictions,rows


def confusion(y,p):
    return [[int(np.sum((y==truth)&(p==prediction))) for prediction in [0,1,-1]] for truth in [0,1]]


def explicit_refusal(record):
    text=raw_text(record).strip();lines=text.splitlines();fence='`'*3
    if len(lines)>=3 and lines[0].strip() in [fence,fence+'json'] and lines[-1].strip()==fence:
        text='\n'.join(lines[1:-1])
    try:return json.loads(text).get('answer')=='insufficient_data'
    except (ValueError,AttributeError):return False


def holm_comparisons(scores):
    result=[]
    for a,b in itertools.combinations(MODELS,2):
        aa={r['task_id']:r['correct'] for r in scores if r['model']==a};bb={r['task_id']:r['correct'] for r in scores if r['model']==b}
        assert set(aa)==set(bb) and len(aa)==24
        wins=sum(aa[k] and not bb[k] for k in aa);losses=sum(bb[k] and not aa[k] for k in aa)
        p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.
        result.append({'first':a,'second':b,'first_only_correct':wins,'second_only_correct':losses,'p_exact':p})
    previous=0.
    for rank,row in enumerate(sorted(result,key=lambda r:r['p_exact'])):
        previous=max(previous,min(1.,row['p_exact']*(len(result)-rank)));row['p_holm']=previous;row['significant_05']=previous<.05
    return result


def analyze(root=ROOT):
    root=Path(root);out=root/'analysis/defense';out.mkdir(parents=True,exist_ok=True)
    verification=audit(root);write_json(out/'audit.json',verification)
    ids,y,bp,baseline_rows=baselines(root)
    tasks={r['id']:r for r in read_lines(root/'data/tasks.jsonl')}
    scores=[];types=[];classification=[];pred_export=[];h10={};h12={};costs=[];code_rows=[];statuses=[];manual=[];records_by_model={}
    for name,(folder,analysis,_,_) in MODELS.items():
        cfg,requests,events,records=load_model(name,root);records_by_model[name]=records
        for r in records:
            if r['stage']=='main' and r['task_id'][0] in 'ABF':
                p=r['parsed'];scores.append({'model':name,'task_id':r['task_id'],'type':r['task_id'][0],
                    'correct':auto_correct(tasks[r['task_id']],p),'parsed':p is not None,
                    'answer':p['answer'] if p else None,'confidence':p['confidence'] if p else None,'format_error':r['error']})
        ms=[r for r in scores if r['model']==name]
        for t in ['A','B','F','ABF']:
            selected=[r['correct'] for r in ms if r['type'] in t];types.append({'model':name,'type':t,'n':len(selected),'correct':sum(selected),**accuracy_ci(selected)})
        h10[name]=calibration(ms,tasks);comparisons=[]
        for stage,shots in [('zero',0),('five',5),('twenty',20)]:
            rs={r['row_id']:r for r in records if r['stage']==stage};assert set(rs)==set(ids)
            p=np.array([rs[k]['parsed']['answer'] if rs[k]['parsed'] else -1 for k in ids]);valid=p!=-1
            explicit=sum(explicit_refusal(r) for r in rs.values())
            selective=classification_metrics(y[valid],p[valid]) if valid.any() else None
            classification.append({'model':name,'shots':shots,**classification_metrics(y,p),
                'ci':classification_ci(y,p),'confusion_0_1_abstain':confusion(y,p),'explicit_refusals':int(explicit),
                'incomplete':sum(r['raw_response'].get('status')!='completed' for r in rs.values()),
                'selective_accuracy':selective['accuracy'] if selective else None,
                'selective_f1':selective['f1_minority'] if selective else None,
                'correct_positive':int(np.sum((y==1)&(p==1))),'correct_negative':int(np.sum((y==0)&(p==0)))})
            pred_export += [{'model':name,'shots':shots,'row_id':k,'truth':int(truth),'prediction':int(pred)} for k,truth,pred in zip(ids,y,p)]
            if shots in [5,20]:comparisons.append(h12_comparison(y,p,bp['LogisticRegression',shots],shots))
        h12[name]={'decision':'reject' if any(r['decision']=='reject' for r in comparisons) else 'confirm' if all(r['decision']=='confirm' for r in comparisons) else 'insufficient_data','comparisons':comparisons}
        evaluated=[r for r in records if r['stage']!='pilot'];abf=[r for r in evaluated if r['stage']=='main' and r['task_id'][0] in 'ABF']
        cost=sum(r['cost_rub'] for r in abf);correct=sum(r['correct'] for r in ms)
        module=importlib.import_module('deepseek_runner' if name=='DeepSeek' else 'yandex_runner')
        _,ledger,_=module.replay(events)
        cached=sum(r['raw_response'].get('provider_response',{}).get('usage',{}).get('precached_prompt_tokens',0) for r in records) if name=='GigaChat-2' else 0
        costs.append({'model':name,'ABF_cost_rub':cost,'ABF_correct':correct,'rub_per_correct_ABF':cost/correct if correct else None,
            'estimated_all_344_rub':sum(r['cost_rub'] for r in records),'pilot_cost_rub':sum(r['cost_rub'] for r in records if r['stage']=='pilot'),
            'ledger_all_rub':ledger,'uncertain_reserve_rub':ledger-sum(r['cost_rub'] for r in records),
            'mean_latency_s':float(np.mean([r['latency_s'] for r in evaluated])),
            'reported_input_output_tokens':sum(r['input_tokens']+r['output_tokens'] for r in records),'additional_cached_input_tokens':cached,
            'local_full_cost_known':None if name=='Qwen3-4B' else 'API tariff-equivalent only'})
        from collections import Counter
        statuses += [{'model':name,'status':k or 'none','n':v} for k,v in Counter(r['error'] for r in evaluated).items()]
        executions=read_json(root/'analysis'/analysis/'code_execution.json')
        for e in executions:code_rows.append({'model':name,'task_id':e['task_id'],'pass_at_1':e['pass_at_1'],
            'output_is_json':e['output_is_json'],'metrics_match_reference':e['metrics_match_reference'],'exit_code':e['exit_code']})
        for t,row in read_json(root/'analysis'/analysis/'review_status.json').items():
            manual.append({'model':name,'type':t,'required':row['n_required'],'paired_reviews':row['n_paired_scores'],'status':row['status'],'kappa':row.get('kappa')})
    from analyze_local import h5_comparison
    from defense_expert_review import review
    expert_types=review(root,records_by_model)
    groups=lambda model:{t:[r for r in scores if r['model']==model and r['type']==t] for t in 'AB'}
    h5=h5_comparison(groups('YandexGPT'),groups('Qwen3-4B'))
    hypotheses={'H5_primary_Yandex_vs_Qwen':h5,'H10':h10,'H12':h12}
    paired=holm_comparisons(scores)
    summary={'models':4,'responses':1376,'pilot':20,'evaluated_requests':1356,'ABF_tasks_per_model':24,
             'classification_rows_per_shots':100,'manual_reviews_pending':any(r['paired_reviews']<r['required'] for r in manual),
             'overall_39_accuracy':None,'analysis_api_calls':0,'generated_code_executions':0,'seed':42,'bootstrap_draws':1000}
    summary.update(expert_review_completed=True,expert_reviewed_answers=92,expert_review_assessors=1,
                   independent_blind_team_review_completed=False)
    for filename,data in [('summary.json',summary),('hypotheses.json',hypotheses),('classification.json',classification),('paired_comparisons.json',paired),('baselines.json',baseline_rows),('manual_status.json',manual)]:write_json(out/filename,data)
    for filename,data in [('accuracy.csv',types),('automatic_scores.csv',scores),('predictions.csv',pred_export),('costs.csv',costs),('code_metrics.csv',code_rows),('format_status.csv',statuses),('audit.csv',verification['models'])]:write_csv(out/filename,data)
    write_csv(out/'classification.csv',[{k:v for k,v in r.items() if k not in ['ci','confusion_0_1_abstain']} for r in classification])
    examples=[]
    for model,task in [('YandexGPT','A03'),('Qwen3-4B','A06'),('Qwen3-4B','B03'),('GigaChat-2','B05'),('YandexGPT','F05')]:
        r=next(r for r in records_by_model[model] if r['stage']=='main' and r['task_id']==task)
        examples.append({'model':model,'task_id':task,'prompt':tasks[task]['text_ru'],'raw_text':raw_text(r),'parsed':r['parsed'],'error':r['error'],'reference':tasks[task]['reference']['answer']})
    r=next(r for r in records_by_model['DeepSeek'] if r['stage']=='twenty' and r['raw_response'].get('status')=='incomplete')
    examples.append({'model':'DeepSeek','task_id':'E01','row_id':r['row_id'],'shots':20,'status':r['raw_response']['status'],
        'incomplete_details':r['raw_response'].get('incomplete_details'),'usage':r['raw_response']['usage'],'final_text':raw_text(r)})
    write_json(out/'case_studies.json',examples)
    from defense_figures import figures
    figures(root,out,types+expert_types,classification,hypotheses,paired,costs,code_rows,statuses,baseline_rows)
    return summary


if __name__=='__main__':
    with offline():print(json.dumps(analyze(),ensure_ascii=False,indent=2))
