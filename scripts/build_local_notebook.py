"""Create a notebook with reproducible numeric analysis and original responses."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def build():
    md=nb.v4.new_markdown_cell; code=nb.v4.new_code_cell
    cells=[md('''# ЛР 3: локальная Qwen3-4B, Q4_K_M

Официальные веса Qwen, CUDA, temperature=0, seed=42, лимит 4096 токенов, режим без рассуждений. Пять pilot исключены. Основной эксперимент: 39 задач и 100 строк классификации в каждом режиме 0/5/20-shot. Системная инструкция, условия, train/test совпадают с YandexGPT. Модель не получает эталоны и тестовые метки.

Run All пересчитывает анализ из сохранённых первых ответов. Генерация и исполнение кода модели здесь не выполняются. Прогнозы и параметры зафиксированы до генераций; результаты Yandex и classical baseline уже были известны. Численные статусы гипотез получаются кодом, итоговые выводы и объяснения причин ошибок формулируют участники согласно заданию.'''),
    code('''from pathlib import Path
import sys,json,csv,html
from IPython.display import display,HTML,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'local_runner.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_local import analyze
summary=analyze()
OUT=ROOT/'analysis/local'
def show_json(name): print(json.dumps(json.loads((OUT/name).read_text(encoding='utf-8')),ensure_ascii=False,indent=2))
def show_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    keys=list(rows[0]);display(HTML('<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'))
def figure(name): display(Image(filename=str(OUT/'figures'/name)))
show_json('summary.json')'''),
    md('''## Протокол, окружение и ресурсы

Ревизия и SHA256 весов — в docs/local_registration.json. n_ctx=8192. Пустой think-блок задаёт режим без рассуждений через шаблон Qwen3, не изменяя общий system/user. У Yandex режим провайдера по умолчанию, поэтому режимы не идентичны. temperature=0 принят для общего протокола, хотя авторы Qwen обычно рекомендуют негридовый sampling.

API локальной модели не тарифицируется. Электроэнергия и амортизация не измерены, поэтому полная стоимость и рубли за правильный ответ неизвестны. На графике сопоставляется только цена API. Время — генерация, без загрузки весов.'''),
    code("show_json('environment.json')\nhardware=json.loads((OUT/'hardware.json').read_text(encoding='utf-8'))\nprint(json.dumps({k:v for k,v in hardware.items() if k!='samples'},ensure_ascii=False,indent=2))\nprint('Resource samples:',len(hardware['samples']))\nshow_csv('costs.csv')\nfigure('runtime.png')"),
    md('''## A/B/F: автоматическая оценка

24 задачи; допуск чисел из банка, B — только вердикт. Ошибки формата входят в знаменатель как неуспех. Можно снять лишь внешнее Markdown-ограждение с отдельным флагом. Объяснения B и ответы C/D требуют двух независимых проверяющих. Непараметрический bootstrap (1000, seed=42) на восьми задачах даёт грубые интервалы, на 8/8 — вырожденный интервал, не гарантию качества на новых задачах.'''),
    code("show_csv('type_metrics.csv')\nshow_csv('automatic_scores.csv')\nfigure('accuracy_ci.png')\nfigure('heatmap.png')"),
    md('''## H5: отставание по A относительно B

Сравнение заранее выбрано: YandexGPT Pro 5.1 против Qwen3 4B. Статистика (Yandex_A−Qwen_A)−(Yandex_B−Qwen_B). Парный bootstrap внутри A/B, 95% CI; нижняя граница >0 подтверждает, верхняя ≤0 опровергает, иначе insufficient_data. Размер модели смешан с обучением, квантованием и режимом генерации, поэтому это не причинный эксперимент размера.'''),
    code("hypotheses=json.loads((OUT/'hypotheses.json').read_text(encoding='utf-8'))\nprint(json.dumps(hypotheses['H5'],ensure_ascii=False,indent=2))\nfigure('h5_gap.png')\nshow_json('paired_comparison.json')"),
    md('''## H10: уверенность

Выбраны A+B+F. Средняя confidence и accuracy считаются на разбираемых ответах; coverage и полный знаменатель показаны отдельно. Кластерный bootstrap сохраняет связанные варианты F. Signed gap задаёт направление; ECE с пятью bins — размер расхождения без направления. Три гипотезы исследовательские, интервалы не скорректированы как единое семейство.'''),
    code("print(json.dumps(hypotheses['H10'],ensure_ascii=False,indent=2))\nfigure('reliability.png')"),
    md('''## E и H12: few-shot классификация

100 общих строк, шесть признаков; 1=malignant, 0=benign. Это обратная исходной sklearn кодировка (D08 использует исходную). Отказ/невалидный формат = -1 и ошибка для истинного класса. Train5 вложен в train20. У baseline нет zero-shot режима. Основная метрика F1 класса 1, дополнительная balanced accuracy; интервалы условны на один фиксированный набор демонстраций.

Для H12 delta=F1_Qwen−F1_LR. При 5-shot нижняя односторонняя 95% граница должна быть >−0.05; при 20-shot верхняя <0. Парный bootstrap использует одни и те же тестовые строки. Критерии взяты из регистрации Yandex до ответов Qwen.'''),
    code("show_csv('classification_metrics.csv')\nshow_json('classification_metrics.json')\nfigure('classification_f1.png')\nprint(json.dumps(hypotheses['H12'],ensure_ascii=False,indent=2))\nfigure('h12_delta.png')\nfigure('cost_accuracy.png')"),
    md('''## C: выбор метода — ответы модели

Для каждой задачи показаны условие, исходный ответ, обоснование и уверенность модели. Ответы могут содержать ошибки и отказы: здесь они отображаются без исправлений. Если JSON корректен, но тип answer не соответствует протоколу, ответ всё равно виден с пометкой ошибки формата; его оценка не меняется.'''),
    code('''records=[json.loads(s) for s in (ROOT/'results/models/local/qwen3-4b-q4-v1/responses.jsonl').read_text(encoding='utf-8').splitlines()]
tasks={r['id']:r for r in [json.loads(s) for s in (ROOT/'data/tasks.jsonl').read_text(encoding='utf-8').splitlines()]}
executions={r['task_id']:r for r in json.loads((OUT/'code_execution.json').read_text(encoding='utf-8'))}

def pre(value):
    if not isinstance(value,str):
        value=json.dumps(value,ensure_ascii=False,indent=2)
    return '<pre style="white-space:pre-wrap">'+html.escape(value)+'</pre>'

def show_answers(prefix):
    for r in sorted([r for r in records if r['stage']=='main' and r['task_id'].startswith(prefix)],key=lambda r:r['task_id']):
        text='\\n'.join(part.get('text','') for block in r['raw_response'].get('output',[]) for part in (block.get('content') or []) if part.get('type')=='output_text')
        obj=r.get('parsed')
        if obj is None:
            # Decode valid JSON for display only. Scoring and original records stay unchanged.
            try:
                candidate=json.loads(text)
                obj=candidate if isinstance(candidate,dict) else None
            except ValueError:
                obj=None
        content='<h3>'+r['task_id']+'</h3><b>Условие</b>'+pre(tasks[r['task_id']]['text_ru'])
        if r.get('error'):
            content+='<p><b>Ошибка формата:</b> '+html.escape(r['error'])+'</p>'
        if obj is not None and 'answer' in obj:
            label='Код модели' if prefix=='D' and isinstance(obj['answer'],str) else 'Ответ модели'
            content+='<b>'+label+'</b>'+pre(obj['answer'])
            if prefix=='D' and not isinstance(obj['answer'],str):
                content+='<p>Модель не предоставила код: поле answer имеет другой тип.</p>'
            if 'explanation' in obj:
                content+='<b>Обоснование модели</b>'+pre(obj['explanation'])
            if 'confidence' in obj:
                content+='<p><b>Уверенность модели:</b> '+html.escape(str(obj['confidence']))+'%</p>'
        else:
            content+='<b>Исходный ответ модели (JSON не удалось разобрать)</b>'+pre(text)
        display(HTML(content))
        if prefix=='D':
            result=executions[r['task_id']]
            display(HTML('<b>Сохранённый результат проверки</b>'+pre({k:result[k] for k in ['status','exit_code','pass_at_1','output_is_json','metrics_match_reference']})))
            for suffix,label in [('stdout.txt','Вывод программы (stdout)'),('stderr.txt','Ошибки программы (stderr)')]:
                path=OUT/'code_execution'/(r['task_id']+'.'+suffix)
                if path.exists():
                    value=path.read_text(encoding='utf-8')
                    display(HTML('<b>'+label+'</b>'+(pre(value) if value else '<p>Пустой вывод.</p>')))

show_answers('C')'''),
    md('''## D: код модели и результаты его запуска

Ниже код отображается обычным многострочным текстом. Показаны сохранённые stdout/stderr, факт запуска и совпадение метрик с эталоном. Run All не выполняет код модели повторно. Если модель прислала число вместо кода или некорректный JSON, показан именно этот исходный ответ. Предварительная техническая проверка не заменяет двух независимых оценок методологии.'''),
    code('''show_answers('D')
show_csv('code_metrics.csv')
show_json('code_execution.json')'''),
    md('''## Независимая проверка и разбор ошибок

Пакет analysis/local/review/ содержит рубрики и две пустые формы. Для слепой проверки организатор перемешивает ответы разных моделей. Два участника независимо выставляют оценки B/C и флаги D, затем сравнивают их и считают kappa; пропуски не заменяются нулями. student_analysis в примерах оставлено участникам. Если автоматических ошибок меньше пяти, дополнительные случаи следует выбрать из C/D после ручной проверки, не выдумывать ошибки.

Финальный текст отчёта, практические рекомендации и пятиминутный доклад заполняет команда. Ограничения для обсуждения: восемь задач на тип, публичный датасет, единственный few-shot набор, зависимость F-пар, различие режимов и квантование. Общая accuracy по 39 задачам до ручной проверки не заявляется.'''),
    code("show_json('error_examples.json')\nshow_json('review_status.json')\nshow_json('provenance.json')")]
    target=ROOT/'notebooks/local_analysis.ipynb'
    nb.write(nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (LAB3 local)','language':'python'}}),target)
    return target


if __name__=='__main__': print(build())
