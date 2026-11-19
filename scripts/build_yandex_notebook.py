"""Build the offline Yandex notebook; no cloud requests."""
import argparse
import nbformat as nbf
from pathlib import Path


def build(root):
    root=Path(root)
    md=nbf.v4.new_markdown_cell
    code=nbf.v4.new_code_cell
    cells=[md('''# ЛР 3: анализ ответов YandexGPT Pro 5.1

Ноутбук читает сохранённые ответы; запросов к API и запуска кода из ответов при Run All нет. Все исходные ответы неизменны. Пять отладочных задач исключены из оценки. Восемь программ D ранее выполнены отдельно без исправлений после просмотра исходников; их stdout/stderr и hashes сохранены.

Численные таблицы и статусы гипотез получаются программно. Итоговые выводы отчёта и разбор ошибок команда формулирует самостоятельно. H5 ожидает Qwen; C/D требуют двух независимых проверяющих. Значения отсутствующих оценок не заменяются нулями.'''),
    code('''from pathlib import Path
import sys, json, csv, html
from IPython.display import display, HTML, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'yandex_runner.py').exists() and (p/'results/models/yandex/yandex-v1/events.jsonl').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_yandex import analyze
if not (ROOT/'analysis/yandex/preliminary_review.csv').exists():
    from create_preliminary_review import create
    create(ROOT)
summary=analyze(ROOT)
OUT=ROOT/'analysis/yandex'
def show_json(name):
    print(json.dumps(json.loads((OUT/name).read_text(encoding='utf-8')),ensure_ascii=False,indent=2))
def show_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    keys=list(rows[0])
    content='<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'
    content+=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)
    display(HTML(content+'</table>'))
def figure(name): display(Image(filename=str(OUT/'figures'/name)))
def show_model_answers(prefix):
    tasks={r['id']:r for r in [json.loads(s) for s in (ROOT/'data/tasks.jsonl').read_text(encoding='utf-8').splitlines()]}
    records=[json.loads(s) for s in (ROOT/'results/models/yandex/yandex-v1/responses.jsonl').read_text(encoding='utf-8').splitlines()]
    executions={r['task_id']:r for r in json.loads((OUT/'code_execution.json').read_text(encoding='utf-8'))}
    def pre(value):
        if not isinstance(value,str): value=json.dumps(value,ensure_ascii=False,indent=2)
        return '<pre style="white-space:pre-wrap">'+html.escape(value)+'</pre>'
    for row in sorted([r for r in records if r['stage']=='main' and r['task_id'].startswith(prefix)],key=lambda r:r['task_id']):
        task=tasks[row['task_id']]
        raw='\\n'.join(p.get('text','') for i in row['raw_response'].get('output',[]) for p in i.get('content',[]) if p.get('type')=='output_text')
        obj=row.get('parsed')
        if obj is None:
            # Display valid JSON content even if its answer type failed the scoring schema.
            # This never changes stored results or scores (e.g. C07's list-valued answer).
            lines=raw.strip().splitlines(); fence=chr(96)*3
            body='\\n'.join(lines[1:-1]) if len(lines)>=3 and lines[0].strip() in (fence,fence+'json') and lines[-1].strip()==fence else raw
            try: obj=json.loads(body)
            except ValueError: obj={}
        content='<h3>'+html.escape(row['task_id'])+'</h3><b>Условие</b>'+pre(task['text_ru'])
        content+='<b>Ответ Яндекса (без исправлений)</b>'+pre(obj.get('answer',raw))
        content+='<b>Обоснование модели</b>'+pre(obj.get('explanation',''))
        content+='<p>Уверенность: '+html.escape(str(obj.get('confidence','не извлечена')))+'; отметка формата: '+html.escape(str(row.get('error')))+'.</p>'
        if prefix=='D':
            outcome=executions[row['task_id']]
            content+='<b>Результат запуска</b>'+pre({k:outcome[k] for k in ['exit_code','pass_at_1','output_is_json','metrics_match_reference']})
            for name in ['stdout','stderr']:
                p=OUT/'code_execution'/(row['task_id']+'.'+name+'.txt')
                content+='<b>'+name+'</b>'+pre(p.read_text(encoding='utf-8'))
        content+='<details><summary>Эталон и критерии проверки</summary>'+pre({'reference':task['reference'],'rubric':task['rubric']})+'</details>'
        content+='<p><b>Ручная оценка: ожидаются два независимых проверяющих.</b></p>'
        display(HTML(content))
print('Analysis API calls:',summary['api_calls_during_analysis'])'''),
    md('''## Комплектность и расходы

344 уникальных полученных ответа: pilot=5, main=39, zero/five/twenty по 100. Две оборванные попытки разрешены пользователем к повтору, но не считаются дополнительными ответами. Известная стоимость вычисляется по usage. В бюджетном журнале дополнительно оставлены резервы неизвестных попыток; они не являются подтверждённым списанием. Фактический платёж сверяется с Billing.'''),
    code("show_json('summary.json')\nshow_csv('costs.csv')"),
    md('''## Автоматическая оценка A, B, F

Числа проверяются с допуском банка, B — по вердикту; обоснования B оцениваются отдельно вручную. Неразбираемые ответы считаются неуспехами, а не удаляются из знаменателя. Единственная внешняя Markdown-оболочка снимается по правилу, зарегистрированному до основных запросов; нарушение чистого JSON фиксируется отдельно. Недопустимые JSON-дроби вроде `2/3` не вычисляются и не исправляются.

Сводная оценка здесь относится к 24 задачам A+B+F. Итоговой accuracy по 39 задачам пока нет: C и D ещё не получили две независимые оценки. Для 8/8 верных ответов B обычный непараметрический bootstrap даёт вырожденный интервал [1,1]; это ограничение метода на малой выборке, а не гарантия отсутствия ошибок на новых задачах.'''),
    code("show_csv('type_metrics.csv')\nshow_csv('automatic_scores.csv')\nfigure('accuracy_ci.png')\nfigure('heatmap.png')"),
    md('''## Тип E: классификация

Одни и те же 100 строк и шесть признаков; train5 является подмножеством train20. Положительный редкий класс здесь 1=malignant (в D08 сохранена исходная sklearn-кодировка). У LLM в zero-shot все ответы — `insufficient_data`: это отказы, не предсказания класса 0. При расчёте эффективного качества отказ считается ошибкой для его истинного класса; coverage=0. Метрики на подмножестве ответов в этом режиме не определены, поскольку такого подмножества нет. F1/BA с отказами показаны как эффективные нулевые оценки. Для классических моделей zero-shot не определён.

Интервалы условны на фиксированные train5/train20; они не включают разброс выбора демонстрационных примеров.'''),
    code("show_csv('classification_metrics.csv')\nfigure('classification_f1.png')\nshow_json('classification_metrics.json')"),
    md('''## H10: калибровка уверенности

Предварительно выбраны A+B+F. Уверенность и точность сравниваются только для разбираемых ответов с валидной confidence; coverage и accuracy по полному знаменателю показаны явно. В B измеряется вердикт, а не качество объяснения. ECE имеет пять равных bins; confidence=1 включается в последний. Направление завышения определяется signed gap, а не самой величиной ECE. Bootstrap сохраняет пары F как кластеры.'''),
    code("hypotheses=json.loads((OUT/'hypotheses.json').read_text(encoding='utf-8'))\nprint(json.dumps(hypotheses['H10'],ensure_ascii=False,indent=2))\nfigure('reliability.png')"),
    md('''## H12: LLM и логистическая регрессия

Первичная метрика F1 редкого класса. Заранее выбран допуск не меньшего качества 0.05; delta=F1_LLM−F1_LR. При 5 примерах нижняя односторонняя 95% граница должна превышать −0.05, при 20 верхняя должна быть ниже 0. Парный bootstrap: одни и те же тестовые индексы для моделей, 1000 повторов, seed=42. Эти односторонние границы отличаются от двухсторонних 95% CI для отдельных метрик. Недостаточная доказательность не означает равенство моделей. Результаты классических baseline были известны до регистрации; результаты LLM — нет.'''),
    code("print(json.dumps(hypotheses['H12'],ensure_ascii=False,indent=2))\nfigure('h12_delta.png')"),
    md('''## H5 и сравнение четырёх моделей

Показатели Яндекса на A и B готовы, но проверка H5 требует совпадающих ответов Qwen3 4B. В её отсутствие статус `pending_other_model`. Тест Макнемара между моделями и итоговый сравнительный график не рассчитываются с вымышленными данными. При получении остальных моделей использовать exact McNemar для парной правильности и заранее выбранную поправку Holm для семейства сравнений.'''),
    code("print(json.dumps(hypotheses['H5'],ensure_ascii=False,indent=2))\nfigure('cost_accuracy.png')"),
    md('''## C: все ответы на выбор метода

Ниже приведены все семь исходных ответов C01-C07 с условиями, обоснованиями и уверенностью модели. Ответы получены и сохранены; отсутствуют только два независимых балла по рубрике. C07 показывает исходный список методов и нарушение типа JSON-поля; отображение содержимого не меняет результаты автоматического парсинга.'''),
    code("show_model_answers('C')"),
    md('''## D: запуск исходного кода

Это техническая проверка запуска, не замена двух независимых проверок методологии. В каждом файле сохранена исходная строка answer без исправлений. Проверялись зафиксированные версии sklearn/numpy. Отдельно показаны запуск, формат stdout и совпадение метрик; диагностическое извлечение числа из Python dict не превращает этот вывод в JSON. Алиасы test_accuracy→accuracy и silhouette_score→silhouette применены только при сопоставлении названий метрик.

Повторный Run All этого ноутбука лишь читает результаты уже выполненных программ; новые исполнения неизвестного кода не производятся.'''),
    code("show_csv('code_metrics.csv')\nshow_json('code_execution.json')\nfigure('code_execution.png')"),
    md('''### D01-D08: полный код и фактический результат каждого запуска

Ниже сохранённый код модели, исходный stdout и stderr. Ячейка только показывает ранее полученные результаты и не исполняет этот код заново. Ошибки модели не исправлены.'''),
    code("show_model_answers('D')"),
    md('''## Предварительная проверка B/C/D

Ниже одна предварительная проверка по рубрикам. В B/C выставлены предварительные баллы, в D отдельно отмечены методологические признаки и фактическая запускаемость. Она не является двумя независимыми оценками участников, не заполняет их формы и не используется для расчёта Cohen kappa. Итоговые показатели ручной проверки остаются незаполненными до её фактического проведения.'''),
    code("show_csv('preliminary_review.csv')"),
    md('''## Материалы двум проверяющим

`analysis/yandex/review/REVIEW.md` и `responses.json` содержат 23 ответа B/C/D с анонимными идентификаторами. Каждому проверяющему — отдельный CSV reviewer_1/reviewer_2, независимо. Ключ `review_key.json` не передавать проверяющим. Для настоящей слепой проверки нужно перемешать ответы всех моделей: пакет одной модели сам по себе не скрывает её происхождение.

Рубрики и эталоны сохранены. Пропущенные оценки остаются пропусками. При наличии пар оценок рассчитывается Cohen kappa; при κ<0.6 рубрика уточняется и ответы переоцениваются. Разногласия требуют согласования, а не автоматического усреднения.'''),
    code("show_json('review_status.json')"),
    md('''## Пять кандидатов для разбора ошибок

Выбраны первые пять неверных автоматических ответов в порядке ID, без отбора по удобному выводу. Здесь исходные цитаты и эталон. Поле student_analysis оставлено пустым: разбор причин ошибки и текст отчёта заполняют участники.'''),
    code("show_json('error_examples.json')"),
    md('''## Что осталось для окончательной сдачи

Две независимые проверки B-обоснований/C/D; согласование разногласий и при необходимости переоценка; данные Qwen для H5 и остальных моделей для общего сравнения; финальная сводная оценка; пять разборов ошибок, минимум три ограничения, рекомендации и доклад. Ограничения методики: малый банк, один фиксированный набор few-shot примеров, публичный датасет, парные задачи и пропуски confidence. Это перечень аспектов для обсуждения, а не готовые исследовательские выводы.''')]
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (LAB3)','language':'python','name':'python3'},'language_info':{'name':'python'}})
    target=root/'notebooks/yandex_analysis.ipynb'
    target.parent.mkdir(exist_ok=True)
    nbf.write(notebook,target)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(build(args.root))
