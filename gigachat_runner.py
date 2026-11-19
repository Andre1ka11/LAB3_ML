"""Resumable GigaChat-2 experiment; credentials never enter the response journal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import requests
import yandex_runner as common

ROOT=Path(__file__).resolve().parent
MODEL='GigaChat-2'
RUN=ROOT/'results/models/gigachat/gigachat-2-v1'
BASE='https://gigachat.devices.sberbank.ru/api/v1'
ENDPOINT=BASE+'/chat/completions'
RATE=1300/20_000_000


def prepare(root=ROOT,model=MODEL):
    config,rows=common.prepare(root,model)
    for row in rows:
        old=row['payload']
        row['payload']={'model':model,'messages':[{'role':'system','content':old['instructions']},
            {'role':'user','content':old['input']}],'temperature':0,'max_tokens':4096,'stream':False}
        row['request_id']=common.digest(row['payload'])
    config.update(endpoint=ENDPOINT,rate_rub_per_token=RATE,tariff_checked='2026-11-19',
        tariff_source='https://developers.sber.ru/docs/ru/gigachat/tariffs/individual-tariffs',
        cost_basis='Published package-equivalent RUB/token including VAT; not actual freemium cash spending',
        thinking_mode='provider default; no tools, search, or explicit reasoning parameter',
        requests_sha256=common.digest([r['request_id'] for r in rows]))
    return config,rows


def normalize(raw):
    choice=(raw.get('choices') or [{}])[0]
    text=(choice.get('message') or {}).get('content') or ''
    usage=raw.get('usage') or {}
    return {'status':'completed' if choice.get('finish_reason')=='stop' else 'incomplete',
        'output':[{'content':[{'type':'output_text','text':text}]}],
        'usage':{'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),
                 'total_tokens':usage.get('total_tokens')},'provider_response':raw}


class Client:
    def __init__(self,root=ROOT):
        cfg={}
        path=root/'.env.gigachat'
        if path.exists():
            cfg=dict(line.split('=',1) for line in path.read_text(encoding='utf-8-sig').splitlines() if '=' in line and not line.startswith('#'))
        self.key=os.environ.get('GIGACHAT_CREDENTIALS') or cfg.get('GIGACHAT_CREDENTIALS')
        if not self.key:raise ValueError('Set GIGACHAT_CREDENTIALS locally')
        self.scope=os.environ.get('GIGACHAT_SCOPE') or cfg.get('GIGACHAT_SCOPE','GIGACHAT_API_PERS')
        ca=os.environ.get('GIGACHAT_CA_BUNDLE_FILE') or str(root/'.cache/gigachat-ca/bundle.pem')
        self.session=requests.Session();self.session.verify=ca if Path(ca).exists() else True
        self.token=None;self.expires=0

    def headers(self):
        if self.token is None or time.time()>self.expires-60:
            response=self.session.post('https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
                headers={'Authorization':'Basic '+self.key,'RqUID':str(uuid.uuid4())},
                data={'scope':self.scope},timeout=60)
            if response.status_code!=200:raise RuntimeError('OAuth HTTP '+str(response.status_code))
            data=response.json();self.token=data['access_token']
            expiry=data.get('expires_at',0);self.expires=expiry/1000 if expiry>1e11 else expiry
        return {'Authorization':'Bearer '+self.token,'Accept':'application/json','RqUID':str(uuid.uuid4())}

    def get(self,name):
        response=self.session.get(BASE+'/'+name,headers=self.headers(),timeout=60)
        if not response.ok:raise RuntimeError('Metadata HTTP '+str(response.status_code))
        return response.json()

    def send(self,payload):
        return self.session.post(ENDPOINT,headers=self.headers(),json=payload,timeout=180)


def run(stage='pilot',budget=100):
    config,rows=prepare();RUN.mkdir(parents=True,exist_ok=True)
    regpath='docs/gigachat_registration.json'
    registration=json.loads((ROOT/regpath).read_text(encoding='utf-8'))
    if registration['config']!=config:raise ValueError('Registration configuration changed')
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',regpath],cwd=ROOT,text=True).strip()
    if json.loads(subprocess.check_output(['git','show',commit+':'+regpath],cwd=ROOT))!=registration:
        raise ValueError('Commit registration before generation')
    client=Client()
    with common.exclusive(RUN/'.lock'):
        path=RUN/'config.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Changed run configuration')
        path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
        journal=RUN/'events.jsonl';done,spent,_=common.replay(common.load_lines(journal))
        if stage!='pilot':
            if not all(r['request_id'] in done for r in rows if r['stage']=='pilot'):
                raise ValueError('Finish all five pilot calls first')
            anchor=RUN/'registration_commit.txt'
            if anchor.exists() and anchor.read_text().strip()!=commit:raise ValueError('Registration changed after main')
            if not anchor.exists():anchor.write_text(commit+'\n',encoding='utf-8')
        selected=[r for r in rows if stage=='all' or r['stage']==stage]
        for i,row in enumerate(selected,1):
            rid=row['request_id']
            if rid in done:continue
            upper=len(json.dumps(row['payload'],ensure_ascii=False).encode())+2048+4096
            held=upper*RATE
            if spent+held>budget:raise ValueError('Cost estimate budget limit reached')
            base={k:row[k] for k in ['request_id','task_id','stage','experiment','language']}
            base.update({k:row[k] for k in ['row_id','shots'] if k in row})
            base.update(run_id=RUN.name,model_id=MODEL,provider='Sber GigaChat API',
                bank_version=config['bank_version'],bank_sha256=config['bank_sha256'],prompt_sha256=config['prompt_sha256'],
                temperature_requested=0,temperature_effective=None,thinking_mode=config['thinking_mode'],protocol_commit=commit)
            for attempt in range(1,4):
                # Refresh OAuth before marking a generation attempt; no tokens are logged.
                client.headers()
                common.append(journal,{**base,'event':'started','timestamp_utc':common.now(),
                    'attempt':attempt,'reserved_rub':held,'request':row['payload']})
                start=time.perf_counter()
                try:response=client.send(row['payload'])
                except requests.RequestException as e:
                    common.append(journal,{**base,'event':'uncertain','timestamp_utc':common.now(),'cost_rub':None,'error':type(e).__name__})
                    raise RuntimeError('Generation outcome unknown; no automatic retry') from None
                if response.status_code!=200:
                    rejected=400<=response.status_code<500
                    common.append(journal,{**base,'event':'rejected' if rejected else 'uncertain',
                        'timestamp_utc':common.now(),'http_status':response.status_code,
                        'error':'HTTP '+str(response.status_code),'cost_rub':0 if rejected else None,
                        'provider_error_body':response.text[:2000]})
                    if response.status_code==429 and attempt<3:time.sleep(min(30,10*attempt));continue
                    raise RuntimeError('Generation HTTP '+str(response.status_code))
                backup=ROOT/'.cache/gigachat_raw';backup.mkdir(parents=True,exist_ok=True)
                (backup/(rid+'.json')).write_bytes(response.content)
                raw=response.json();wrapped=normalize(raw)
                parsed,error=common.parse_response(wrapped,row['answer_kind'])
                usage=raw.get('usage') or {};total=usage.get('total_tokens')
                charge=total*RATE if isinstance(total,int) else None
                record={**base,'event':'response','timestamp_utc':common.now(),'raw_response':wrapped,
                    'parsed':parsed,'error':error,'cost_rub':charge,'input_tokens':usage.get('prompt_tokens'),
                    'output_tokens':usage.get('completion_tokens'),'latency_s':time.perf_counter()-start,
                    'parser_version':common.PARSER_VERSION,'strict_format_ok':parsed is not None and error is None,
                    'model_version_returned':raw.get('model')}
                common.append(journal,record);done[rid]=record
                common.export(RUN,common.load_lines(journal),rows)
                if charge is None:raise ValueError('Missing token usage; response preserved')
                spent+=charge
                print(f'{i}/{len(selected)} {row["stage"]} {row["task_id"]} {row.get("row_id", "")} {error or "OK"} {spent:.4f} RUB',flush=True)
                break


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=[*common.STAGES,'all'],default='pilot')
    parser.add_argument('--budget-rub',type=float,default=100)
    args=parser.parse_args();run(args.stage,args.budget_rub)
