"""Build the unified defense notebook; frozen inputs are created only explicitly."""
import argparse
from pathlib import Path
import nbformat as nb
from defense_analysis import freeze_sources

ROOT=Path(__file__).resolve().parents[1]


def build(freeze=False):
    if freeze:freeze_sources()
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md('''# ЛР3 Можно ли доверять LLM в классическом машинном обучении

**Единый блокнот для защиты.** Четыре модели, одни условия, сохранённые первые ответы, пересчёт результатов без API. Дополнительный учебный разбор: `docs/defense_guide.html` или `docs/defense_guide.md`.

Маршрут на 5 минут: **1 → 2 → 5 → 7 → 8 → 10 → 11 → 13**. Разделы с подробными доказательствами, формулами и примерами можно раскрывать по вопросам преподавателя. Run All не генерирует ответы и не запускает программы моделей. Обучаются только доверенные классические baseline.

Ключевой результат: DeepSeek решил 24/24 A/B/F, но 138/300 запросов классификации оборвались по лимиту. Qwen при 5/20 примерах имеет F1 0.851/0.904; LR — 0.804/0.912. Это разные виды надёжности, поэтому один общий рейтинг по 1356 запросам не строится. C/D и обоснования B ещё требуют двух независимых оценок; итоговая accuracy по 39 задачам не выдумывается.'''),
    md('''## 1 Подключение и пересчёт

Окружение: `pip install -r requirements-defense.txt`. Для Run All не нужны CUDA, веса или API-ключи. HTTP-транспорты во время анализа блокируются программно. Основные вычисления находятся в `scripts/defense_analysis.py`; ниже также вынесены показательные формулы и статистические расчёты.'''),
    code('''from pathlib import Path
import sys,json,csv,html,inspect,hashlib
import numpy as np
import pandas as pd
from IPython.display import display,HTML,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts/defense_analysis.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from defense_analysis import analyze,audit,offline,baselines,load_model,MODELS,confusion
from analysis_common import read_json,read_lines,raw_text,auto_correct,accuracy_ci,h12_comparison
OUT=ROOT/'analysis/defense'
with offline():summary=analyze(ROOT)
def table(name):display(pd.read_csv(OUT/name))
def show(name):print(json.dumps(read_json(OUT/name),ensure_ascii=False,indent=2))
def figure(name):display(Image(filename=str(OUT/'figures'/name)))
show('summary.json')'''),
    md('''## 2 Откуда взялись ответы и что действительно проверено

Проверяются 1376 уникальных первых ответов, 5+39+100+100+100 на модель, соответствие каждого started готовому запросу, одинаковые тексты и лимиты, SHA256 замороженных входных файлов, коммиты регистрации и их время до main, отсутствие пересечения train/test, snapshot DeepSeek, учёт повторов. Полный список hashes: `analysis/defense/source_manifest.json`; он **не пересоздаётся** при Run All.

Эти проверки выявляют случайное изменение, подмену отдельных файлов и несогласованность истории. Они не являются подписью API-провайдера: человек, контролирующий все файлы и Git, теоретически может сфабриковать согласованную историю. Для внешнего подтверждения нужны Billing/история использования кабинета; здесь проверяется сохранённая экспериментальная цепочка. Новый live-запрос в конце подтверждает доступ сейчас, а не исторический запуск сам по себе.'''),
    code("with offline():verification=audit(ROOT)\nprint('Frozen files checked:',verification['frozen_files_verified'])\nprint('Same system/user texts:',verification['all_prompt_texts_identical'])\nprint('Costs independently recomputed from usage:',all(r['cost_usage_verified'] for r in verification['models']))\ndisplay(pd.DataFrame(verification['models'])[['model','responses','starts','uncertain','authorized_network_retries','registration_commit','registration_before_main','commit_utc','first_main_started_utc','last_response_utc']])"),
    md('''### Показать один исходный запрос и ответ

Поменяйте MODEL и TASK. Для E дополнительно задайте SHOTS и ROW_ID. Печатается источник из events.jsonl, а не готовая таблица правильности. Секреты авторизации в журналах не хранятся.'''),
    code('''MODEL='DeepSeek'; TASK='A03'; SHOTS=20; ROW_ID='bc_007'
with offline():config,requests,events,records=load_model(MODEL,ROOT)
selected=next(r for r in records if r['task_id']==TASK and (r['stage']=='main' if TASK!='E01' else r.get('shots')==SHOTS and r.get('row_id')==ROW_ID))
started=next(e for e in events if e['event']=='started' and e['request_id']==selected['request_id'])
print('request_id:',selected['request_id'],'timestamp:',selected['timestamp_utc'])
print('Request:');print(json.dumps(started['request'],ensure_ascii=False,indent=2))
print('Parsed:',selected['parsed'],'format:',selected['error'])
print('Final text:');print(raw_text(selected))
print('Usage:',selected['raw_response'].get('usage'))
print('Provider ID:',selected['raw_response'].get('id') or selected['raw_response'].get('provider_response',{}).get('id'))'''),
    md('''## 3 Условия эксперимента и правильные знаменатели

40 записей банка = A8+B8+C7+D8+F8 и один протокол E. E разворачивается в 300 вызовов. На модель 344 сохранённых ответа, из них 5 pilot вне оценки. Основных запросов 339, но единицы качества разделены: A/B/F — 24; C/D — ручные критерии; E — 100 строк отдельно для каждого shots. Сотни E не должны подавлять остальные типы.

Температура запрошена 0, лимит 4096. Qwen — 4B, Q4_K_M, CUDA, режим без рассуждений; облачные модели — режим провайдера по умолчанию. У DeepSeek лимит включает reasoning tokens. Равные численные параметры не гарантируют равный вычислительный бюджет и режимы.'''),
    code('''tasks={r['id']:r for r in read_lines(ROOT/'data/tasks.jsonl')}
display(pd.crosstab(pd.Series([r['type'] for r in tasks.values()],name='Type'),pd.Series([r['topic'] for r in tasks.values()],name='Topic')))
config_rows=[]
for model,(folder,_,_,_) in MODELS.items():
    cfg=read_json(ROOT/'results/models'/folder/'config.json')
    config_rows.append({'model':model,'ID':cfg['model'],'temperature':cfg['temperature'],'max_output':cfg['max_output_tokens'],'mode':cfg['thinking_mode']})
display(pd.DataFrame(config_rows))'''),
    md('''## 4 Проверка эталонов и baseline собственным расчётом

Ячейка показывает формулы, а не обращается к другой LLM. A06 — голосование всех трёх соседей; A07 — полная ширина 2/||w||, а не 1/||w||. В E класс 1 означает malignant, тогда как в D08 используется исходная sklearn-кодировка 0=malignant.'''),
    code('''x=np.array([0,1,2,3]);y_line=np.array([1,2,2,5])
gini=lambda counts:1-sum((v/sum(counts))**2 for v in counts)
calculated={
 'A01':(.02*.96)/(.02*.96+.98*.04),
 'A02':gini([36,24])-(36/60)*gini([30,6])-(24/60)*gini([6,18]),
 'A03':2*42/(2*42+18+14),
 'A04':sum((x-x.mean())*(y_line-y_line.mean()))/sum((x-x.mean())**2),
 'A05':1/(1+np.exp(-(-.4+.6*2-.2))),
 'A06':int(sum([0,1,1])>1.5),
 'A07':2/np.linalg.norm([6,8]),
 'A08':(4+3)/(4+3+1)}
checks=[]
for task,value in calculated.items():
    reference=tasks[task]['reference']['answer'];tol=tasks[task]['tolerance']
    ok=abs(value-reference)<=tol['absolute']+tol['relative']*abs(reference)
    checks.append({'task':task,'calculated':value,'reference':reference,'match':ok})
assert all(r['match'] for r in checks)
display(pd.DataFrame(checks))'''),
    code('''with offline():ids,y,baseline_predictions,baseline_rows=baselines(ROOT)
display(pd.DataFrame(baseline_rows))
split=read_json(ROOT/'data/classification/split_indices.json')
assert set(split['train5'])<=set(split['train20'])
assert set(split['train20']).isdisjoint(split['test'])
print('Test class counts:',dict(zip(*np.unique(y,return_counts=True))))
print('Majority constant accuracy:',float(np.mean(y==0)))
print(inspect.getsource(baselines))'''),
    md('''Baseline LR заново обучается как StandardScaler→LogisticRegression на тех же 5/20 строках, scaler видит только train. GaussianNB в исходном baseline обучался **без scaler**; именно этот вариант воспроизводится, с проверкой каждого предсказания по row_id. Масштабирование NB при var_smoothing может изменить результаты, поэтому его нельзя молча добавить и сохранить старые цифры.'''),
    md('''## 5 Основные результаты A B F

Здесь измеряется совпадение итогового answer с эталоном, не убедительность explanation. Для B оценивается вердикт; объяснения отдельно. В разрешённой нормализации снимается только одна внешняя Markdown-оболочка; арифметика JSON, типы и код не исправляются. Непарсируемые ответы входят в знаменатель как неуспех.'''),
    code("table('accuracy.csv')\nfigure('heatmap.png')\nfigure('accuracy_ci.png')\ntable('format_status.csv')"),
    md('''### Пересчитать точность и интервал на выбранном типе

Стандартный percentile bootstrap повторяет задачи с возвращением. При 8/8 все ресэмплы тоже дают 8/8; интервал [1,1] — вырожденность метода на этой малой выборке, а не гарантия отсутствия ошибок в будущем.'''),
    code("scores=pd.read_csv(OUT/'automatic_scores.csv')\nCHOSEN_MODEL='DeepSeek'; CHOSEN_TYPE='A'\nvalues=scores.loc[(scores.model==CHOSEN_MODEL)&(scores.type==CHOSEN_TYPE),'correct'].astype(int).to_numpy()\nprint(accuracy_ci(values))\ndisplay(scores.loc[(scores.model==CHOSEN_MODEL)&(scores.type==CHOSEN_TYPE)])"),
    md('''## 6 Статистическая значимость

На каждой паре моделей важны задачи, где правильность различается. Exact McNemar — двусторонний binomial test discordant ответов с p=0.5. Для четырёх моделей шесть сравнений; Holm контролирует множественность в этом семействе. Он не устраняет зависимость вариантов F или селекцию банка. Результаты относятся к этому набору задач и правилам формата.'''),
    code('''from scipy.stats import binomtest
paired=read_json(OUT/'paired_comparisons.json');display(pd.DataFrame(paired))
row=next(r for r in paired if r['first']=='YandexGPT' and r['second']=='DeepSeek')
b,c=row['first_only_correct'],row['second_only_correct']
print('Discordant:',b,c,'exact p=',binomtest(b,b+c,.5).pvalue)
print('Учебный пример из методички: 3 против 0, p=',binomtest(3,3,.5).pvalue)'''),
    md('''## 7 E как классификатор и роль отсутствующих ответов

Основные показатели F1 класса 1 и balanced accuracy. Дополнительно coverage и точность на ответившем подмножестве. Отсутствующее предсказание = -1, а не класс 0. Для F1 отказ на положительном объекте добавляет FN; отказ на отрицательном не добавляет FP. Поэтому F1 одного класса не симметрично штрафует все отказы; BA, confusion и coverage обязательны.

DeepSeek 20-shot: 35/100 допустимых ответов, все 35 правильны. Следовательно conditional accuracy=100%, но full accuracy=35%. Нельзя выбросить 65 незавершённых ответов и заявить качество 100% на исходном тесте. Giga zero-shot предсказывает класс 1 всем строкам: F1≈0.540, но BA=0.5, то есть разделения классов нет.'''),
    code("table('classification.csv')\nfigure('classification.png')\nfigure('selective_accuracy.png')\nfigure('confusion_20.png')"),
    code('''pred=pd.read_csv(OUT/'predictions.csv')
subset=pred[(pred.model=='DeepSeek')&(pred.shots==20)]
assert len(subset)==100
print('Full accuracy:',np.mean(subset.prediction==subset.truth))
answered=subset[subset.prediction!=-1]
print('Coverage:',len(answered)/100,'selective accuracy:',np.mean(answered.prediction==answered.truth))
print('Confusion rows true 0/1; columns predicted 0/1/no answer:')
print(confusion(subset.truth.to_numpy(),subset.prediction.to_numpy()))'''),
    md('''## 8 H5 H10 H12

Прогноз команды — confirm для каждой гипотезы; правила сохранены до основных запусков каждой модели. Основная H5 заранее выбрана как Yandex vs Qwen, не заменяется более выгодным DeepSeek. H10: signed confidence gap, 95% кластерный bootstrap с сохранением F-пар. H12: при 5 примерах не меньшее качество LR с margin 0.05 F1; при 20 — проигрыш LR. Границы H12 односторонние 95% (перцентили 5/95), это не тот же интервал, что двухсторонний 95% 2.5/97.5.

Гипотеза H12 — сочетание двух утверждений: обе части подтверждены → confirm; хотя бы одна опровергнута → reject; иначе insufficient_data. Отсутствие значимости не доказывает равенство.'''),
    code("hyp=read_json(OUT/'hypotheses.json')\nshow('hypotheses.json')\nfigure('h5_gap.png')\nfigure('h10_gap.png')\nfigure('reliability.png')\nfigure('h12_delta.png')"),
    code('''MODEL='GigaChat-2'; SHOTS=5
subset=pred[(pred.model==MODEL)&(pred.shots==SHOTS)].set_index('row_id').loc[ids]
print(h12_comparison(y,subset.prediction.to_numpy(),baseline_predictions['LogisticRegression',SHOTS],SHOTS))
print(inspect.getsource(h12_comparison))'''),
    md('''## 9 D запуск и методологическая корректность

Читайте три столбца отдельно: завершился ли процесс, напечатан ли JSON, совпали ли численные метрики. Qwen D07 — словарь без print, процесс завершается успешно, задача не выполнена. Giga D01 диагностически печатает accuracy под именем balanced_accuracy; численная близость не исправляет неверную метрику.

В Giga все первичные ответы D имеют недопустимый JSON, поэтому основной end-to-end pass=0. После этого выполнена отдельная диагностика точных фрагментов кода (6/8 запустились), не меняющая первичные оценки. Она явно вторична. DeepSeek 8/8 запусков и метрик не заменяет двух независимых проверок методологии.'''),
    code("table('code_metrics.csv')\nfigure('code_execution.png')\ndisplay(pd.DataFrame(read_json(ROOT/'analysis/gigachat/diagnostic_code_execution.json'))[['task_id','runs_without_error','output_is_json','metrics_match_reference','inspection_note']])"),
    code('''MODEL='DeepSeek'; TASK='D01'
folder,analysis,_,_=MODELS[MODEL]
rs=read_lines(ROOT/'results/models'/folder/'responses.jsonl')
r=next(r for r in rs if r['stage']=='main' and r['task_id']==TASK)
print('Original answer string:');print(r['parsed']['answer'] if r['parsed'] else raw_text(r))
for suffix in ['py','stdout.txt','stderr.txt']:
    p=ROOT/'analysis'/analysis/'code_execution'/(TASK+'.'+suffix)
    if p.exists():print('\\nSAVED '+suffix+'\\n'+p.read_text(encoding='utf-8'))
print('Эта ячейка только читает сохранённый код и вывод, ничего не исполняет.')'''),
    md('''## 10 Разбор конкретных ошибок

Пять содержательных/форматных примеров и один незавершённый запрос DeepSeek. Данные ниже дословно взяты из журналов. Объяснения механизмов и границ интерпретации подробно разобраны в личном отчёте, раздел «Разбор ошибок».

Yandex A03: рассуждение получает 0.7241 и затем ошибочно заменяет результат на 0.6364. Qwen A06: признаёт два голоса за 1, но отвечает 0. Qwen B03: отсутствие меток ошибочно считает отсутствием утечки. Giga B05: путает рост штрафа C с усилением регуляризации. F05: JSON-дробь 2/3 не является числом JSON. DeepSeek E: reasoning_tokens=4096 исчерпывают бюджет до итогового класса.'''),
    code("show('case_studies.json')"),
    md('''## 11 Скорость стоимость память

Рубли за верный ответ рассчитаны **на A/B/F**, с тем же знаменателем, что и accuracy. Цена всех 344 вызовов — другой показатель. Тарифная цена, списание денег и резерв неизвестных сетевых попыток различаются. У Qwen API-цена 0, полная стоимость электричества/оборудования неизвестна.

У Giga prompt_tokens не включает бесплатный кэш: полный объём больше тарифицируемого. У DeepSeek reasoning уже включён в output_tokens; повторно прибавлять его нельзя. Задержка зависит от длины ответов, reasoning, кэша, сервиса и оборудования — это не чистый benchmark скорости архитектуры.'''),
    code("table('costs.csv')\nfigure('cost_accuracy.png')\nhardware=read_json(ROOT/'analysis/local/hardware.json')\nprint(json.dumps({k:v for k,v in hardware.items() if k!='samples'},ensure_ascii=False,indent=2))"),
    md('''## 12 Ручная проверка и ограничения

Ни одна отсутствующая оценка не заменяется нулём. B-вердикт проверен автоматически, обоснование 0–2 — отдельно. C и методология D: два участника независимо, смешанные ответы разных моделей, без ключа соответствия; затем kappa, уточнение рубрики при κ<0.6, разрешение разногласий. Отдельные пакеты по модели сами по себе не обеспечивают слепоту.

Ограничения: восемь задач на тип; F-пары зависимы; публичный breast cancer мог быть в обучении; единственный train5/train20; режимы reasoning и квантование различаются; банк draft и эталоны требуют человеческой проверки; доверительный интервал не учитывает все источники неопределённости; нет полноценных ручных итогов C/D. Поэтому обобщать на все ML-задачи и клиническое применение нельзя.'''),
    code("show('manual_status.json')"),
    md('''## 13 Как сформулировать результат на защите

На нашем фиксированном банке DeepSeek существенно лучше остальных по итоговым A/B/F, но ограничение reasoning-бюджета делает его ненадёжным как классификатор строк в выбранном режиме. Qwen конкурентна few-shot, особенно при 5 примерах; это не подтверждает, что любой маленькой модели достаточно для любого ML. Классическая LR на 20 примерах даёт F1≈0.912 и не имеет проблемы незавершённой генерации. Ни одна confidence не заменяет проверку, особенно когда объяснение противоречит answer.

Практически: расчёты сверять формулой/кодом; советы по методам проверять на утечку и схему оценки; сгенерированный код запускать и отдельно проверять методологию; табличную классификацию сравнивать с простыми baseline. Численно подтверждённые результаты отделять от гипотез о причинах.

Подробный маршрут, объяснение каждого числа, формулы, разбор ошибок, 12 вопросов методички и дополнительные вопросы: `docs/defense_guide.md` / `docs/defense_guide.html`.'''),
    md('''## 14 Необязательная живая проверка подключения

**По умолчанию выключена.** Не входит в Run All и оригинальные оценки. Если преподаватель попросит показать реальный вызов, установите LIVE=True и выберите модель. Может расходовать токены/квоту; ключи читаются только из локального окружения, не вписывайте их в блокнот. Результат сохраняется отдельно. Новый вызов не гарантирует побитовое совпадение исторического ответа. Для Qwen используйте отдельный local_runner с уже скачанными весами и CUDA.'''),
    code('''LIVE=False
LIVE_MODEL='GigaChat-2'
if LIVE:
    from defense_live_probe import probe
    print(json.dumps(probe(LIVE_MODEL,consent=True),ensure_ascii=False,indent=2))
else:
    print('Живой вызов отключён. При Run All токены не расходуются.')''')]
    path=ROOT/'notebooks/defense.ipynb'
    nb.write(nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 LAB3 defense','language':'python'}}),path)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-sources',action='store_true');args=p.parse_args();print(build(args.freeze_sources))
