"""Build a single notebook that recalculates saved responses without API calls."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def build():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md('''# ЛР 3: DeepSeek V4.1 Flash — банк задач и классификация 0/5/20-shot

Полный фиксированный снимок: 39 задач A/B/C/D/F и по 100 строк классификации в режимах с 0, 5 и 20 примерами. Пять pilot не входят в оценку. Получены все 344 ответа; в основной анализ входят 339 запросов.

Run All читает сохранённые ответы, пересчитывает таблицы и строит графики. API-ключи, интернет и новые запросы к модели не нужны. Код из ответов D при Run All не выполняется: используются результаты отдельного запуска неизменённых программ. Новые ответы фонового прогона не меняют снимок.

Это завершённый сбор ответов DeepSeek и воспроизводимый численный анализ. Для окончательной сдачи всей лабораторной остаётся независимая проверка и содержательный разбор участников. Оценки C/D, обоснования B, интерпретация ошибок и окончательные рекомендации требуют работы участников.'''),
    code('''from pathlib import Path
import sys,json,csv,html
from IPython.display import display,HTML,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'deepseek_runner.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_deepseek import analyze,select,journal,RUN
from analysis_common import raw_text
summary=analyze()
OUT=ROOT/'analysis/deepseek'
snapshot=json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
records=select(journal(RUN/'events.jsonl'),snapshot)
tasks={r['id']:r for r in [json.loads(s) for s in (ROOT/'data/tasks.jsonl').read_text(encoding='utf-8').splitlines()]}
executions={r['task_id']:r for r in json.loads((OUT/'code_execution.json').read_text(encoding='utf-8'))}

def show_json(name): print(json.dumps(json.loads((OUT/name).read_text(encoding='utf-8')),ensure_ascii=False,indent=2))
def show_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if not rows: return
    keys=list(rows[0]);display(HTML('<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'))
def figure(name): display(Image(filename=str(OUT/'figures'/name)))
def pre(value):
    if not isinstance(value,str): value=json.dumps(value,ensure_ascii=False,indent=2)
    return '<pre style="white-space:pre-wrap">'+html.escape(value)+'</pre>'
def show_answers(prefix,collapsed=False):
    for record in sorted([r for r in records if r['stage']=='main' and r['task_id'].startswith(prefix)],key=lambda r:r['task_id']):
        text=raw_text(record)
        try: obj=json.loads(text)
        except ValueError: obj={}
        content='<h3>'+record['task_id']+'</h3><b>Условие</b>'+pre(tasks[record['task_id']]['text_ru'])
        content+='<b>'+('Код модели' if prefix=='D' else 'Ответ модели')+'</b>'+pre(obj.get('answer',text))
        content+='<b>Обоснование модели</b>'+pre(obj.get('explanation',''))
        content+='<p><b>Уверенность:</b> '+html.escape(str(obj.get('confidence','не извлечена')))+'%; <b>Отметка формата:</b> '+html.escape(str(record.get('error')))+'.</p>'
        if prefix=='D':
            result=executions[record['task_id']]
            content+='<b>Сохранённый результат запуска</b>'+pre(result)
            for suffix in ['stdout','stderr']:
                path=OUT/'code_execution'/(record['task_id']+'.'+suffix+'.txt')
                value=path.read_text(encoding='utf-8')
                content+='<b>'+suffix+'</b>'+(pre(value) if value else '<p>Пустой вывод.</p>')
        content+='<details><summary>Эталон и рубрика</summary>'+pre({'reference':tasks[record['task_id']]['reference'],'rubric':tasks[record['task_id']]['rubric']})+'</details>'
        if collapsed: content='<details><summary>'+record['task_id']+': условие и исходный ответ</summary>'+content+'</details>'
        display(HTML(content))
print('Новых запросов к API:',summary['analysis_api_calls'])'''),
    md('''## Состав снимка и расходы

В анализ входят 339 запросов, но они не объединяются в одну accuracy: 39 разнородных задач и 300 строк классификации оцениваются отдельно. Сводная автоматическая accuracy ниже относится только к 24 задачам A/B/F. Общая оценка 39 задач пока не рассчитана — C/D ещё требуют независимых оценок.

Расходы ниже рассчитаны по тарифу и usage всех полученных ответов. Отдельно показан журнал бюджета с резервами за сетевые попытки без ответа. Резерв — не подтверждённое списание, а оценка по usage — не выписка Billing. Для первого pilot время ответа не восстановлено после локальной ошибки парсинга; оно не подменяется нулём.'''),
    code("show_json('summary.json')\nshow_csv('costs.csv')"),
    md('''## A/B/F: автоматическая оценка и исходные ответы

Числовой допуск берётся из банка; у B проверяется только вердикт. Обоснования оцениваются отдельно. Неверный ответ и нарушение формата не перезапрашиваются. 95% bootstrap-интервалы построены на 1000 выборках, seed=42. При восьми задачах на тип интервалы грубые; вырожденный интервал на 8/8 не гарантирует качество на новых задачах.

Карточки ниже можно раскрыть: внутри сохранённые условие, ответ, объяснение, уверенность и эталон.'''),
    code("show_csv('type_metrics.csv')\nshow_csv('automatic_scores.csv')\nfigure('accuracy_ci.png')\nfigure('heatmap.png')\nfor prefix in 'ABF': show_answers(prefix,collapsed=True)"),
    md('''## E: классификация с 0, 5 и 20 примерами

В каждом из трёх режимов ровно 100 общих тестовых строк. Метка 1 — malignant, 0 — benign. Это обратная кодировка относительно исходного sklearn и D08. Используются те же шесть признаков, train5/train20 и test, что у остальных участников. LogisticRegression и GaussianNB обучены на тех же 5 и 20 примерах; у них нет режима без обучения.

Незавершённый ответ, отказ и невалидный формат обозначены −1 и считаются неуспехом для истинного класса. Они не превращаются в класс 0. Coverage — доля допустимых предсказаний, а не доля полученных HTTP-ответов. Основная метрика — F1 класса 1, дополнительная — balanced accuracy; константа предсказывает частый класс. Интервалы условны на один фиксированный набор демонстраций каждого размера.'''),
    code("show_csv('classification_metrics.csv')\nshow_json('classification_metrics.json')\nfigure('classification.png')"),
    md('''## H10: калибровка уверенности

Используются заранее выбранные 24 задачи A/B/F. Проверяется средняя confidence/100 минус accuracy на валидных ответах; coverage и accuracy с полным знаменателем показаны отдельно. Кластерный bootstrap сохраняет связанные F-пары. ECE оценивает величину расхождения, но сам по себе не доказывает завышение уверенности. Проверка исследовательская; коррекция семейства гипотез не заявляется.'''),
    code("hypotheses=json.loads((OUT/'hypotheses.json').read_text(encoding='utf-8'))\nprint(json.dumps(hypotheses['H10'],ensure_ascii=False,indent=2))\nfigure('reliability.png')"),
    md('''## H12: полное сравнение с 5 и 20 примерами

Показаны delta F1 = F1 DeepSeek − F1 LogisticRegression и односторонние bootstrap-границы 95% для обоих размеров обучения. Допуск не меньшего качества при 5-shot — 0,05; при 20-shot проверяется ожидаемое отставание. H12 подтверждается только при поддержке обеих частей, опровергается при противоречии хотя бы одной, иначе данных недостаточно. Правила зафиксированы до основного прогона. Все 100 ответов каждого режима, включая незавершённые генерации, входят в знаменатель.

Первичная H5 сравнивает YandexGPT и Qwen3 4B и рассчитывается в командном анализе. DeepSeek не подменяет выбранную большую модель.'''),
    code("print(json.dumps(hypotheses['H12'],ensure_ascii=False,indent=2))\nfigure('h12_delta.png')\nfigure('cost_accuracy.png')"),
    md('''## C: выбор метода — все семь исходных ответов

Ответы отображаются без исправлений. Предложения модели могут быть неверными; публикация ответа не означает положительную оценку. Две независимые слепые оценки по рубрике ещё не заполнены.'''),
    code("show_answers('C')"),
    md('''## D: все восемь программ и результаты запуска

Код получен от DeepSeek, просмотрен перед исполнением и запущен без правок отдельно, без API-ключей, в каталоге выполнения, с ограничением времени и числом потоков. Это не полноценная ОС-песочница. При Run All программы повторно не исполняются. Отдельно показаны запускаемость, формат вывода и совпадение метрик с эталоном. Проверка методологии двумя независимыми участниками ещё требуется.'''),
    code("show_csv('code_metrics.csv')\nshow_answers('D')"),
    md('''## Независимая проверка и разбор случаев

Формы находятся в `analysis/deepseek/review/`. Организатор должен перемешать ответы разных моделей; пакет одной модели сам по себе не обеспечивает слепую проверку. Ключ сопоставления проверяющим не передаётся. Оценки и kappa не выдумываются, пропуски остаются пустыми.

Ниже сохранены пять случаев для самостоятельного разбора: сначала автоматически неверные A/B/F, затем первые незавершённые ответы классификации. При отсутствии ошибок A/B/F показаны фактические случаи исчерпания токенов с исходными usage и статусом сервера; внутренние рассуждения модели не выдумываются. Поле student_analysis оставлено для участников. Ограничения: маленький банк, публичный датасет, один few-shot набор, различия режима рассуждений и провайдера, обрывы сети, лимит рассуждений и неполная ручная оценка.'''),
    code("show_json('review_status.json')\nshow_json('error_examples.json')\nshow_json('provenance.json')\nshow_json('environment.json')")]
    path=ROOT/'notebooks/deepseek_analysis.ipynb'
    path.parent.mkdir(exist_ok=True)
    nb.write(nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (LAB3)','language':'python'}}),path)
    return path


if __name__=='__main__': print(build())
