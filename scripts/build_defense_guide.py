"""Render the personal study guide with tables from the unified analysis."""
import base64
import csv
import html
import json
from pathlib import Path
import re
import markdown
from markdown.extensions.toc import slugify_unicode

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/defense'


def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def csvrows(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(headers,rows):
    clean=lambda v:str(v).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(clean(v) for v in row)+' |' for row in rows]])
def pct(x):return f'{float(x):.1%}'


def build():
    models=['YandexGPT','Qwen3-4B','GigaChat-2','DeepSeek'];acc=csvrows('accuracy.csv');cl=load('classification.json');hyp=load('hypotheses.json');cost=csvrows('costs.csv');code=csvrows('code_metrics.csv')
    values={}
    values['expert_review_table']=table(['Модель','Проверено','B среднее / 2','C среднее / 2','C полностью / 7','D строгий / 8','Дополнительно / 39'],
        [[r['model'],r['reviewed'],f"{r['B_mean_0_2']:.3f}",f"{r['C_mean_0_2']:.3f}",r['C_full_2'],r['D_strict_correct'],r['expert_strict_39_correct']] for r in load('expert_review.json')['summary']])
    values['audit_table']=table(['Модель','Ответов','Started','Неопределённых попыток','Разрешённых повторов','Commit прогнозов'],
        [[r['model'],r['responses'],r['starts'],r['uncertain'],r['authorized_network_retries'],r['registration_commit'][:8]] for r in load('audit.json')['models']])
    rows=[]
    for m in models:
        rs={r['type']:r for r in acc if r['model']==m};a=rs['ABF']
        rows.append([m,*[rs[t]['correct']+'/8' for t in 'ABF'],a['correct']+'/24',pct(a['accuracy']),f"[{pct(a['ci_low'])}; {pct(a['ci_high'])}]"])
    values['accuracy_table']=table(['Модель','A','B вердикт','F','A+B+F','Accuracy','95% bootstrap CI'],rows)
    values['baseline_table']=table(['Baseline','Train','F1 класса 1','BA','Accuracy'],
        [[r['model'],r['shots'],f"{r['f1_minority']:.4f}",f"{r['balanced_accuracy']:.4f}",f"{r['accuracy']:.4f}"] for r in load('baselines.json')])
    values['classification_table']=table(['Модель','Shots','F1 класса 1','BA','Accuracy всех','Coverage','Accuracy отвеченных'],
        [[r['model'],r['shots'],f"{r['f1_minority']:.4f}",f"{r['balanced_accuracy']:.4f}",pct(r['accuracy']),pct(r['coverage']),pct(r['selective_accuracy']) if r['selective_accuracy'] is not None else 'не определена'] for r in cl])
    values['paired_table']=table(['Первая','Вторая','Только первая верна','Только вторая верна','Exact p','Holm p','p Holm < 0.05'],
        [[r['first'],r['second'],r['first_only_correct'],r['second_only_correct'],f"{r['p_exact']:.6g}",f"{r['p_holm']:.6g}",'да' if r['significant_05'] else 'нет'] for r in load('paired_comparisons.json')])
    values['h10_table']=table(['Модель','n valid / 24','Mean confidence','Accuracy valid','Gap','95% CI gap','ECE','Статус'],
        [[m,str(h['n_valid'])+'/24',pct(h['mean_confidence']),pct(h['accuracy_on_valid']),f"{h['gap']:.4f}",f"[{h['ci_low']:.4f}; {h['ci_high']:.4f}]",f"{h['ece']:.4f}",h['decision']] for m,h in hyp['H10'].items()])
    h12=[]
    for m,h in hyp['H12'].items():
        for r in h['comparisons']:
            h12.append([m,r['shots'],f"{r['delta_f1_llm_minus_lr']:.4f}",f"[{r['one_sided_95_lower']:.4f}; {r['one_sided_95_upper']:.4f}]",r['decision'],h['decision']])
    values['h12_table']=table(['Модель','Shots','F1 LLM − F1 LR','Односторонние 95% границы','Часть','Вся H12'],h12)
    rows=[]
    for m in models:
        rs=[r for r in code if r['model']==m]
        rows.append([m,*[str(sum(r[k]=='True' for r in rs))+'/8' for k in ['pass_at_1','output_is_json','metrics_match_reference']], '6/8 извлечённых фрагментов' if m=='GigaChat-2' else 'основной сохранённый код'])
    values['code_table']=table(['Модель','Основной pass','stdout JSON','Числа совпали','Примечание'],rows)
    values['cost_table']=table(['Модель','Цена A/B/F руб','Верных A/B/F','Руб за верный A/B/F','Цена всех 344 руб','Резерв сети руб','Mean latency с'],
        [[r['model'],f"{float(r['ABF_cost_rub']):.4f}",r['ABF_correct'],f"{float(r['rub_per_correct_ABF']):.4f}",f"{float(r['estimated_all_344_rub']):.4f}",f"{float(r['uncertain_reserve_rub']):.4f}",f"{float(r['mean_latency_s']):.2f}"] for r in cost])
    text=(ROOT/'docs/defense_guide_template.md').read_text(encoding='utf-8')
    for key,value in values.items():text=text.replace('{{'+key+'}}',value)
    assert not re.search(r'\{\{[a-z_]+\}\}',text),'Unexpanded table'
    (ROOT/'docs/defense_guide.md').write_text(text,encoding='utf-8')
    rendered=markdown.markdown(text,extensions=['tables','fenced_code','toc'],extension_configs={'toc':{'slugify':slugify_unicode,'title':'Содержание'}})
    def embed(match):
        src=match.group(1);path=(ROOT/'docs'/src).resolve()
        data=base64.b64encode(path.read_bytes()).decode();return 'src="data:image/png;base64,'+data+'"'
    rendered=re.sub(r'src="(\.\./analysis/defense/figures/[^"\n]+\.png)"',embed,rendered)
    style='''html{scroll-behavior:smooth}body{margin:0;color:#18212b;background:#f6f7fa;font:17px/1.68 "Segoe UI",Arial,sans-serif}main{max-width:1040px;margin:0 auto;padding:48px 44px 100px;background:white}h1{font-size:34px;line-height:1.2;color:#111;margin:0 0 28px}h2{font-size:26px;line-height:1.3;margin:52px 0 18px;color:#152b48;scroll-margin-top:20px}h3{font-size:21px;line-height:1.4;margin:30px 0 14px}p{margin:0 0 17px}a{color:#175fac}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.45;margin:24px 0;display:block;overflow-x:auto}th,td{padding:10px 12px;border:1px solid #dbe1e9;text-align:left;vertical-align:top}th{background:#edf2f8}tr:nth-child(even){background:#fafbfd}img{max-width:100%;height:auto;display:block;margin:24px auto}code{font:14px Consolas,monospace;background:#eef2f6;padding:2px 4px;overflow-wrap:anywhere}pre{padding:16px;background:#eef2f6;overflow:auto}pre code{padding:0}strong{color:#122a44}.toc{background:#f1f5fa;padding:20px 26px;border-radius:8px;font-size:15px;margin:32px 0}.toc ul{padding-left:22px}.toctitle{font-size:20px;font-weight:600}@media(max-width:700px){main{padding:25px 18px}body{font-size:16px}h1{font-size:28px}h2{font-size:23px}}@media print{@page{size:A4;margin:18mm}body{font-size:10.5pt;background:white}main{max-width:none;margin:0;padding:0}h1{font-size:24pt}h2{font-size:18pt;break-after:avoid}h3{font-size:13pt;break-after:avoid}p{orphans:3;widows:3}table{font-size:8pt;display:table;overflow:visible;break-inside:avoid}thead{display:table-header-group}tr,img{break-inside:avoid}.toc{font-size:9pt}a{color:inherit;text-decoration:none}}'''
    document='<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Подготовка к защите ЛР3</title><style>'+style+'</style></head><body><main>'+rendered+'</main></body></html>'
    (ROOT/'docs/defense_guide.html').write_text(document,encoding='utf-8')
    print('Guide words:',len(text.split()),'characters:',len(text),'embedded figures:',document.count('data:image/png'))
    return ROOT/'docs/defense_guide.html'


if __name__=='__main__':print(build())
