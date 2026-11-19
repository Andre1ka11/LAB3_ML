"""Run/resume inference with process RSS and whole-device VRAM sampling."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import local_runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    args=parser.parse_args()
    stop=threading.Event();samples=[]
    def sample():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                name,total,used,util=raw.strip().split(', ')
                samples.append({'timestamp_utc':local_runner.common.now(),'rss_bytes':psutil.Process().memory_info().rss,
                    'gpu':name,'vram_total_mib':int(total),'device_used_mib':int(used),'device_utilization_pct':int(util)})
            except Exception as error:
                samples.append({'monitor_error':str(error)})
            stop.wait(1)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:local_runner.run(args.model_path,'all')
    finally:
        stop.set();thread.join(timeout=5)
        out=ROOT/'analysis/local';out.mkdir(parents=True,exist_ok=True)
        valid=[r for r in samples if 'rss_bytes' in r]
        report={'cpu':__import__('platform').processor(),'logical_cpus':psutil.cpu_count(),
                'ram_total_bytes':psutil.virtual_memory().total,'samples':samples,
                'sampling_scope':'All/resume invocation; pilot and main before monitoring excluded. VRAM is whole device including desktop, not process allocation.',
                'peak_rss_bytes':max((r['rss_bytes'] for r in valid),default=None),
                'peak_device_used_mib':max((r['device_used_mib'] for r in valid),default=None)}
        (out/'hardware.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
