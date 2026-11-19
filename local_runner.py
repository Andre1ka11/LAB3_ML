"""Qwen3 GGUF experiment on CUDA. Saves every first answer; no answer retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yandex_runner as common

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'Qwen/Qwen3-4B-GGUF:Q4_K_M'
REVISION = 'bc640142c66e1fdd12af0bd68f40445458f3869b'
RUN = ROOT/'results/models/local/qwen3-4b-q4-v1'


def prepare(root=ROOT, model=MODEL_ID):
    config, rows = common.prepare(root, model)
    config.update(endpoint='in-process llama-cpp-python CUDA', rate_rub_per_token=0,
                  thinking_mode='disabled by empty think block in Qwen3 chat template',
                  seed=42, n_ctx=8192, n_gpu_layers=-1, quantization='Q4_K_M',
                  revision=REVISION, repeat_penalty=1.0, top_p=1.0,
                  tariff_checked=None)
    return config, rows


def chat_prompt(system, user):
    # Qwen3 non-thinking template: original system/user messages stay identical.
    return ('<|im_start|system\n'+system+'<|im_end|>\n'
            '<|im_start|user\n'+user+'<|im_end|>\n'
            '<|im_start|assistant\n<think>\n\n</think>\n\n')


def run(model_path, stage):
    from llama_cpp import Llama
    config, rows = prepare()
    registration = json.loads((ROOT/'docs/local_registration.json').read_text(encoding='utf-8'))
    if config != registration['config']:
        raise ValueError('Configuration differs from preregistration')
    commit = subprocess.check_output(['git','log','-1','--format=%H','--','docs/local_registration.json'],cwd=ROOT,text=True).strip()
    committed = json.loads(subprocess.check_output(['git','show',commit+':docs/local_registration.json'],cwd=ROOT))
    if registration != committed:
        raise ValueError('Commit registration before any generation')
    sha = hashlib.file_digest(model_path.open('rb'),'sha256').hexdigest()
    if sha != registration['weights_sha256']:
        raise ValueError('Unexpected model weights')
    RUN.mkdir(parents=True,exist_ok=True)
    with common.exclusive(RUN/'.lock'):
        cfg = RUN/'config.json'
        if cfg.exists() and json.loads(cfg.read_text()) != config:
            raise ValueError('Changed configuration')
        cfg.write_text(json.dumps(config,indent=2),encoding='utf-8')
        (RUN/'registration_commit.txt').write_text(commit+'\n',encoding='utf-8')
        events = RUN/'events.jsonl'
        done,_,_ = common.replay(common.load_lines(events))
        selected = [r for r in rows if stage=='all' or r['stage']==stage]
        if stage != 'pilot' and not all(r['request_id'] in done and done[r['request_id']].get('parsed') for r in rows if r['stage']=='pilot'):
            raise ValueError('Complete pilot first')
        llm = Llama(model_path=str(model_path), n_ctx=8192, n_gpu_layers=-1,
                    seed=42, n_batch=512, verbose=False)
        for i,row in enumerate(selected,1):
            if row['request_id'] in done:
                continue
            base = {k:row[k] for k in ['request_id','task_id','stage','experiment','language']}
            base.update({k:row[k] for k in ['row_id','shots'] if k in row})
            base.update(run_id=RUN.name,model_id=MODEL_ID,provider='local CUDA',
                        bank_version=config['bank_version'],bank_sha256=config['bank_sha256'],
                        prompt_sha256=config['prompt_sha256'],temperature_requested=0,
                        temperature_effective=0,thinking_mode=config['thinking_mode'],protocol_commit=commit)
            common.append(events,{**base,'event':'started','timestamp_utc':common.now(),
                                  'attempt':1,'reserved_rub':0,'request':row['payload']})
            start=time.perf_counter()
            result=llm(chat_prompt(row['payload']['instructions'],row['payload']['input']),
                       max_tokens=4096,temperature=0,top_p=1,repeat_penalty=1,
                       stop=['<|im_end|>','<|endoftext|>'],echo=False)
            latency=time.perf_counter()-start
            choice=result['choices'][0]
            raw={'status':'completed' if choice['finish_reason']=='stop' else 'incomplete',
                 'output':[{'content':[{'type':'output_text','text':choice['text']}]}],
                 'usage':{'input_tokens':result['usage']['prompt_tokens'],
                          'output_tokens':result['usage']['completion_tokens']},
                 'local_completion':result}
            parsed,error=common.parse_response(raw,row['answer_kind'])
            record={**base,'event':'response','timestamp_utc':common.now(),'raw_response':raw,
                    'parsed':parsed,'error':error,'parser_version':common.PARSER_VERSION,
                    'strict_format_ok':parsed is not None and error is None,'latency_s':latency,
                    'cost_rub':0,'input_tokens':raw['usage']['input_tokens'],
                    'output_tokens':raw['usage']['output_tokens']}
            common.append(events,record)
            done[row['request_id']]=record
            common.export(RUN,common.load_lines(events),rows)
            print(f'{i}/{len(selected)} {row["stage"]} {row["task_id"]} {row.get("row_id", "")} {latency:.1f}s {error or "OK"}',flush=True)
        llm.close()


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--stage',choices=[*common.STAGES,'all'],default='pilot')
    a=p.parse_args()
    run(a.model_path,a.stage)
