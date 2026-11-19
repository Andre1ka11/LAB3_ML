"""Manually resume the existing DeepSeek run; never resend received responses."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import deepseek_runner as r


class Tee:
    def __init__(self,console,file): self.console,self.file=console,file
    def write(self,text):
        self.console.write(text);self.file.write(text)
        self.flush()
        return len(text)
    def flush(self): self.console.flush();self.file.flush()


def main():
    directory=ROOT/'results/models/deepseek/deepseek-v1'
    try:
        values=r.settings(ROOT)
        if not values.get('YANDEX_API_KEY') or not values.get('YANDEX_FOLDER_ID'):
            raise r.StopRun('Fill YANDEX_API_KEY and YANDEX_FOLDER_ID in the local .env.')
        config,rows=r.prepare(ROOT,f"gpt://{values['YANDEX_FOLDER_ID']}/{r.MODEL_NAME}")
        stored=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        if stored!={**config,'budget_rub':600,'pilot_budget_rub':10}:
            raise r.StopRun('Configuration differs from the approved 600 RUB run.')
        with r.exclusive(directory/'.watchdog.lock'):
            with r.exclusive(directory/'.lock'):
                events=r.load_lines(directory/'events.jsonl')
                pending,terminal={},{}
                for e in events:
                    rid=e.get('request_id')
                    if e['event']=='started': pending[rid]=e
                    elif e['event'] in ('response','uncertain','rejected'):
                        pending.pop(rid,None)
                        if e['event']!='rejected': terminal[rid]=e
                    elif e['event']=='retry_authorized': terminal.pop(rid,None)
                if pending: raise r.StopRun('A request has no terminal journal entry. Inspect manually; no automatic resend.')
                uncertain=[e for e in terminal.values() if e['event']=='uncertain' or e.get('cost_rub') is None]
                if len(uncertain)>1: raise r.StopRun('Multiple uncertain requests require manual review.')
                if uncertain:
                    e=uncertain[0]
                    network=e.get('error') in {'TimeoutError','URLError','RemoteDisconnected','ConnectionResetError','ConnectionAbortedError','SSLError'} or e.get('http_status') in (500,502,503,504)
                    if e['event']!='uncertain' or not network:
                        raise r.StopRun('Failure requires manual review; this command retries network interruptions only.')
            if uncertain:
                r.authorize_retry(directory,e['request_id'],'User explicitly invoked the manual resume command. Authorize one additional retry of the unresolved network attempt only; keep all previous reserves, received responses and the 600 RUB total budget. No unattended retry loop.')
            done,spent,_=r.replay(r.load_lines(directory/'events.jsonl'))
            print(f'Saved: {len(done)}/344; remaining: {344-len(done)}; ledger: {spent:.4f} RUB; budget: 600 RUB.',flush=True)
            (directory/'.process.json').write_text(json.dumps({'pid':os.getpid(),'started_utc':r.now(),'run_id':'deepseek-v1','budget_rub':600,'manual_resume':True}),encoding='utf-8')
            with (directory/'process.stdout.log').open('a',encoding='utf-8',buffering=1) as stdout_log, (directory/'process.stderr.log').open('a',encoding='utf-8',buffering=1) as stderr_log:
                original_out,original_err=sys.stdout,sys.stderr
                try:
                    sys.stdout,sys.stderr=Tee(original_out,stdout_log),Tee(original_err,stderr_log)
                    print('=== MANUAL RESUME '+r.now()+' ===',flush=True)
                    r.run(ROOT,directory,config,rows,'all',values['YANDEX_API_KEY'],600,10)
                finally:
                    sys.stdout,sys.stderr=original_out,original_err
    except (r.StopRun,OSError,ValueError) as error:
        print('STOP:',str(error),file=sys.stderr,flush=True)
        return 2
    return 0


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
