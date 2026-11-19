"""Optional explicit live check; separate from the frozen experiment and its scores."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def probe(model,consent=False):
    if consent is not True:raise ValueError('Explicit consent=True required; this may use tokens')
    prompt='Чему равна sigmoid(0)?'
    if model=='GigaChat-2':
        import gigachat_runner as g
        client=g.Client(ROOT);_,rs=g.prepare(ROOT);payload=next(r['payload'] for r in rs if r['task_id']=='P01')
        response=client.send(payload);response.raise_for_status();raw=response.json()
    elif model in ['YandexGPT','DeepSeek']:
        import yandex_runner as y
        import deepseek_runner as d
        settings=y.settings(ROOT)
        if not settings.get('YANDEX_API_KEY') or not settings.get('YANDEX_FOLDER_ID'):raise ValueError('Set local YANDEX_API_KEY and YANDEX_FOLDER_ID; do not put them in notebook')
        runner=y if model=='YandexGPT' else d
        model_id=f"gpt://{settings['YANDEX_FOLDER_ID']}/"+('yandexgpt-5.1' if model=='YandexGPT' else 'deepseek-v4.1-flash')
        _,rs=runner.prepare(ROOT,model_id);raw=json.loads(runner.send(next(r['payload'] for r in rs if r['task_id']=='P01'),settings['YANDEX_API_KEY']))
    else:raise ValueError('For Qwen use the original local_runner and documented CUDA environment; no cloud substitution')
    now=datetime.now(timezone.utc);dest=ROOT/'analysis/defense/live_probes';dest.mkdir(parents=True,exist_ok=True)
    record={'timestamp_utc':now.isoformat(),'model':model,'prompt':prompt,'raw_response':raw,'included_in_original_scores':False}
    (dest/(now.strftime('%Y%m%dT%H%M%S%f')+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record
