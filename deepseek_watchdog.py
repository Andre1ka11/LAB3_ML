"""Bounded unattended recovery of network failures in the existing run."""
import json
from pathlib import Path
import subprocess
import sys
import time
import deepseek_runner as r

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'results/models/deepseek/deepseek-v1'
MARKER = 'User-approved bounded watchdog'
NETWORK_ERRORS = {'TimeoutError','RemoteDisconnected','URLError','ConnectionResetError','ConnectionAbortedError','SSLError'}


def retry_candidate(events):
    pending, terminals = {}, {}
    for e in events:
        rid=e.get('request_id')
        if e['event']=='started': pending[rid]=e
        elif e['event'] in ('response','uncertain','rejected'):
            pending.pop(rid,None)
            if e['event']!='rejected': terminals[rid]=e
        elif e['event']=='retry_authorized': terminals.pop(rid,None)
    if pending: raise r.StopRun('Unfinished journal entry: manual recovery required; no automatic resend.')
    bad=[e for e in terminals.values() if e['event']=='uncertain' or e.get('cost_rub') is None]
    if not bad: return None
    if len(bad)!=1: raise r.StopRun('Multiple uncertain attempts require manual review.')
    e=bad[0]
    if e['event']!='uncertain' or not (e.get('error') in NETWORK_ERRORS or e.get('http_status') in (500,502,503,504)):
        raise r.StopRun('Failure is not an eligible network interruption.')
    rid=e['request_id']
    retries=[x for x in events if x['event']=='retry_authorized']
    if sum(x.get('request_id')==rid for x in retries)>=2:
        raise r.StopRun('Two retry authorizations already used for this request.')
    if sum(MARKER in x.get('note','') for x in retries)>=10:
        raise r.StopRun('Ten automatic retry authorizations already used.')
    return rid


def log(message):
    with (RUN/'watchdog.log').open('a',encoding='utf-8') as stream:
        stream.write(r.now()+' '+message+'\n')


def main():
    child=None
    before_launch=None
    with r.exclusive(RUN/'.watchdog.lock'):
        log('START: max 2 retries/request, 10 automatic retries total, budget 600 RUB, poll 10 seconds.')
        while True:
            try:
                with r.exclusive(RUN/'.lock'):
                    events=r.load_lines(RUN/'events.jsonl')
            except r.StopRun:
                time.sleep(10)
                continue
            try:
                if child is not None:
                    if child.poll() is None:
                        time.sleep(10)
                        continue
                    if r.digest(events)==before_launch:
                        raise r.StopRun('Runner exited without recording progress. Manual check required.')
                    child=None
                rid=retry_candidate(events)
                if rid:
                    log('NETWORK RETRY authorized: '+rid)
                    r.authorize_retry(RUN,rid,MARKER+': retry one missing network response; keep previous reserved cost; never retry received answers.')
                    events=r.load_lines(RUN/'events.jsonl')
                done,spent,_=r.replay(events)
                values=r.settings(ROOT)
                config,rows=r.prepare(ROOT,f"gpt://{values['YANDEX_FOLDER_ID']}/{r.MODEL_NAME}")
                stored=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
                if stored!={**config,'budget_rub':600,'pilot_budget_rub':10}:
                    raise r.StopRun('Configuration differs from the approved 600 RUB run.')
                r.registration(ROOT,config)
                missing=[row for row in rows if row['request_id'] not in done]
                if not missing:
                    log('COMPLETE: 344 responses saved. Watchdog exits.')
                    return
                if spent+r.reserve(missing[0]['payload'])>600:
                    raise r.StopRun('Budget limit reached. Watchdog will not increase it.')
                if events and events[-1]['event']=='rejected':
                    raise r.StopRun('HTTP rejection requires manual review.')
                r.append(RUN/'events.jsonl',{'event':'watchdog_launch','timestamp_utc':r.now(),'saved_responses':len(done),'budget_rub':600,'policy':MARKER})
                before_launch=r.digest(r.load_lines(RUN/'events.jsonl'))
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'deepseek_runner.py'),'run','--stage','all','--budget-rub','600','--append-logs'],cwd=ROOT)
                (RUN/'.process.json').write_text(json.dumps({'pid':child.pid,'started_utc':r.now(),'run_id':'deepseek-v1','budget_rub':600,'managed_by':'deepseek_watchdog.py'}),encoding='utf-8')
                log(f'RUNNER START pid={child.pid}; saved={len(done)}; ledger={spent:.4f} RUB.')
                time.sleep(10)
            except (r.StopRun,ValueError,KeyError,OSError) as error:
                log('STOP: '+str(error))
                return


if __name__=='__main__':
    main()
