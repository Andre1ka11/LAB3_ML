"""Create an offline GigaChat notebook; never generates or executes model code."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def build():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md('''# ЛР3: GigaChat-2 (Сбер)

Первый ответ на каждый из 344 запросов: 5 pilot (вне оценки), 39 задач банка и 100 тестовых строк в режимах 0/5/20-shot. Модель запрошена как GigaChat-2; точные версии сервера показаны в summary. История, поиск и инструменты не используются. System/user, temperature=0 и верхний лимит 4096 совпадают с общим протоколом. Значение effective temperature сервер не сообщает — оно не выдумывается.

Run All пересчитывает только анализ сохранённых данных. К API не обращается, код модели повторно не запускает. Результаты Qwen/Yandex и classical baseline были известны до регистрации GigaChat. Прогнозы взяты из командной регистрации; это повторение гипотез по тем же правилам. Выводы отчёта и причины ошибок формулируют участники.'''),
    code('''from pathlib import Path
import sys,json,csv,html
from IPython.display import display,HTML,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'gigachat_runner.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_gigachat import analyze
summary=analyze()
OUT=ROOT/'analysis/gigachat'
def show_json(name): print(json.dumps(json.loads((OUT/name).read_text(encoding='utf-8')),ensure_ascii=False,indent=2))
def show_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    keys=list(rows[0]);display(HTML('<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'))
def figure(name):display(Image(filename=str(OUT/'figures'/name)))
show_json('summary.json')'''),
    md('''## Протокол и расходы

Параметры/прогнозы до генераций: docs/gigachat_registration.json, commit 9b810f0. Pilot: 4/5 ответов ожидаемого формата; P04 с JSON и кодом вне JSON сохранён и не повторялся. После пилота параметры и промпты не менялись. В качестве доступа выбран поддерживаемый legacy endpoint: новый api.giga.chat в этом окружении не завершил TLS-подключение, OAuth и прежний адрес работают. CA официальные, TLS-проверка включена; системное хранилище не менялось.

Freemium-квота достаточна, покупок не было. Для сравнения с другими моделями применяется опубликованная эквивалентная цена пакета Lite: 1300 рублей / 20 млн токенов, включая НДС, проверено 19.11.2026. Это условная цена использованных токенов, не фактический платёж и не цена покупки минимального пакета. Usage total_tokens учитывается один раз. Источник: https://developers.sber.ru/docs/ru/gigachat/tariffs/individual-tariffs .

У Сбера prompt_tokens уже исключает precached_prompt_tokens; кэш не тарифицируется. Поэтому расходы считаются по total_tokens, а полный объём входа/выхода и бесплатный кэш показываются отдельно. Кэш префикса не переносит историю решения других задач: в каждый запрос передаются только общий system и один user. Источник: https://developers.sber.ru/docs/ru/gigachat/guides/counting-tokens .'''),
    code("show_json('environment.json')\nshow_csv('costs.csv')\nfigure('runtime.png')"),
    md('''## Автоматическая оценка A/B/F

24 задачи: численный допуск банка; B — только вердикт. Обоснования B, C и методология D требуют двух независимых людей. Неразбираемый ответ считается ошибкой. Снимается только одно внешнее Markdown-ограждение по общему правилу; дополнительный текст, JSON-дроби и неверные типы не исправляются.

95% bootstrap CI: 1000 ресэмплов, seed=42. На восьми задачах интервалы грубые; [1,1] или [0,0] не гарантируют качество на новых задачах. Общая accuracy 39 задач до ручной проверки не заявляется.'''),
    code("show_csv('type_metrics.csv')\nshow_csv('automatic_scores.csv')\nfigure('accuracy_ci.png')\nfigure('heatmap.png')"),
    md('''## Парное сравнение доступных моделей

Точные тесты Макнемара на совпадающих 24 A/B/F задачах. Семейство из трёх доступных пар Yandex/Qwen/GigaChat с поправкой Holm. Это исследовательские сравнения; четвёртая модель DeepSeek ещё не имеет здесь полного прогона. Проверки относятся к первичным ответам и принятым правилам формата.'''),
    code("show_json('paired_comparisons.json')\nfigure('cost_accuracy.png')"),
    md('''## H5: основное сравнение не меняется

В исходной регистрации выбраны YandexGPT vs Qwen3-4B. Это сравнение сохранено. Дополнительно, после просмотра результатов других моделей, рассчитано GigaChat vs Qwen — оно явно вторичное. Статистика gap_A−gap_B, парный bootstrap внутри типов A/B. Модельный размер смешан с обучением, квантованием и режимом; причинный эффект размера не измеряется.'''),
    code("hypotheses=json.loads((OUT/'hypotheses.json').read_text(encoding='utf-8'))\nprint(json.dumps(hypotheses['H5'],ensure_ascii=False,indent=2))\nfigure('h5_gap.png')"),
    md('''## H10: уверенность

A+B+F, только разбираемые ответы с доступной confidence; coverage и полная accuracy показаны отдельно. Направление определяется mean(confidence/100−correct), ECE с 5 bins измеряет абсолютную ошибку калибровки. 1000 кластерных bootstrap с сохранением пар F, seed=42; границы 2.5/97.5%. Статус confirm при нижней границе >0, reject при верхней ≤0, иначе insufficient_data.'''),
    code("print(json.dumps(hypotheses['H10'],ensure_ascii=False,indent=2))\nfigure('reliability.png')"),
    md('''## E и H12: классификация

Общие 100 тестовых строк, 6 признаков; train5 вложен в train20. Метки E: 1=malignant, 0=benign, обратно исходной sklearn-кодировке; D08 сохраняет исходную. Отказы и невалидные ответы получают -1 и считаются ошибкой для истинного класса; явные insufficient_data и остальные ошибки формата считаются раздельно. У классических методов zero-shot нет.

Основная метрика F1 класса 1; дополнительно BA и константа. H12: delta=F1_GigaChat−F1_LR. Для 5-shot нижняя односторонняя 95% граница >−0.05 подтверждает не меньшее качество; для 20-shot верхняя <0 подтверждает проигрыш. Парный bootstrap использует одинаковые тестовые строки. Все интервалы условны на единственные фиксированные train5/train20.'''),
    code("show_csv('classification_metrics.csv')\nshow_json('classification_metrics.json')\nfigure('classification_f1.png')\nprint(json.dumps(hypotheses['H12'],ensure_ascii=False,indent=2))\nfigure('h12_delta.png')"),
    md('''## C/D: исходные ответы

Ниже все ответы без исправлений. В D все восемь первичных ответов нарушили формат JSON; основной end-to-end pass@1 равен 0. Код, присутствующий вне правильного answer, не засчитывается задним числом. Для понимания технического качества выполнена отдельная post-hoc диагностика: точные подстроки fenced/triple-quoted блоков, с offsets и hashes, без правок. Она не изменяет первичные оценки.

Код просмотрен технически и выполнен в отдельных процессах с очищенным окружением. Это не ОС-контейнер и не две независимые человеческие оценки. Диагностические запускаемость, формат stdout, совпадение чисел и методологические замечания разделены; близкое число не доказывает верную метрику (например, D01 использует accuracy под именем balanced_accuracy).'''),
    code('''rows=[json.loads(s) for s in (ROOT/'results/models/gigachat/gigachat-2-v1/responses.jsonl').read_text(encoding='utf-8').splitlines()]
tasks={r['id']:r for r in [json.loads(s) for s in (ROOT/'data/tasks.jsonl').read_text(encoding='utf-8').splitlines()]}
def pre(v):return '<pre style="white-space:pre-wrap">'+html.escape(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2))+'</pre>'
for r in rows:
    if r['stage']=='main' and r['task_id'][0] in 'CD':
        raw=r['raw_response']['provider_response']['choices'][0]['message']['content']
        display(HTML('<h3>'+r['task_id']+'</h3>'+pre(tasks[r['task_id']]['text_ru'])+pre(raw)))
        if r['task_id'].startswith('D'):
            for suffix in ['stdout.txt','stderr.txt']:
                p=OUT/'diagnostic_code_execution'/(r['task_id']+'.'+suffix)
                display(HTML('<b>Вторичная диагностика: '+suffix+'</b>'+pre(p.read_text(encoding='utf-8'))))
show_json('code_execution.json')
show_json('diagnostic_code_execution.json')'''),
    md('''## Материалы для участников

review/ содержит эталоны, рубрики и две пустые независимые формы. Для слепой проверки перемешать ответы разных моделей; ключ соответствия не передавать проверяющим. После заполнения считаются kappa и статусы, пропуски не заменяются нулями. Финальный разбор пяти ошибок и собственные выводы остаются участникам согласно PDF.

Ограничения для обсуждения: малый банк, публичный датасет, фиксированные few-shot примеры, разные режимы провайдеров, влияние форматных ошибок, post-hoc характер диагностики D. Ниже первые пять автоматических ошибок без отбора по удобному выводу; student_analysis оставлено пустым.'''),
    code("show_json('error_examples.json')\nshow_json('review_status.json')\nshow_json('provenance.json')")]
    target=ROOT/'notebooks/gigachat_analysis.ipynb'
    nb.write(nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (LAB3 GigaChat)','language':'python'}}),target)
    return target


if __name__=='__main__':print(build())
