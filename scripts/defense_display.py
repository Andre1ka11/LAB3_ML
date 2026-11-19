"""Readable notebook presentation without changing saved scores or analysis."""
import html
import json
from pathlib import Path
import pandas as pd
from IPython.display import display,HTML

LABELS={
 'model':'Модель','type':'Тип задач','n':'Число случаев','correct':'Верных','accuracy':'Точность на всех случаях',
 'ci_low':'Нижняя граница CI','ci_high':'Верхняя граница CI','shots':'Примеров в запросе',
 'f1_minority':'F1 класса 1','balanced_accuracy':'Balanced accuracy','coverage':'Доля допустимых ответов',
 'abstentions':'Без допустимого класса','status':'Статус ответа','task_id':'Задача',
 'pass_at_1':'Первый запуск / контракт выполнен','output_is_json':'Вывод является JSON',
 'metrics_match_reference':'Числа совпали с эталоном','exit_code':'Код завершения процесса',
 'ABF_cost_rub':'Цена A/B/F, руб.','ABF_correct':'Верных A/B/F','rub_per_correct_ABF':'Руб. за верный A/B/F',
 'estimated_all_344_rub':'Цена всех 344, руб.','uncertain_reserve_rub':'Резерв сетевых попыток, руб.',
 'mean_latency_s':'Среднее время ответа, с'}
VISIBLE={
 'classification.csv':['model','shots','f1_minority','balanced_accuracy','accuracy','coverage','abstentions'],
 'costs.csv':['model','ABF_cost_rub','ABF_correct','rub_per_correct_ABF','estimated_all_344_rub','uncertain_reserve_rub','mean_latency_s']}


def details(value,title='Раскрыть полные технические данные'):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)
    display(HTML('<details><summary>'+html.escape(title)+'</summary><pre style="white-space:pre-wrap">'+html.escape(text)+'</pre></details>'))


def table(out,name):
    frame=pd.read_csv(Path(out)/name);columns=VISIBLE.get(name,list(frame))
    display(frame[columns].rename(columns=LABELS))
    if len(columns)<len(frame):
        display(HTML('<details><summary>Все исходные столбцы таблицы</summary>'+frame.to_html(index=False,escape=True)+'</details>'))


def hypotheses(value):
    rows=[{'Гипотеза':'H5: отставание в расчётах относительно концепций','Модель / пара':'Yandex–Qwen',
           'Результат':value['H5_primary_Yandex_vs_Qwen']['decision']}]
    for name,h in value['H10'].items():rows.append({'Гипотеза':'H10: завышение уверенности','Модель / пара':name,'Результат':h['decision']})
    for name,h in value['H12'].items():rows.append({'Гипотеза':'H12: не хуже LR при 5, хуже при 20','Модель / пара':name,'Результат':h['decision']})
    frame=pd.DataFrame(rows);frame['Результат']=frame['Результат'].map({'confirm':'Критерий подтверждения выполнен','reject':'Гипотеза противоречит данным','insufficient_data':'Данных для определённого вывода недостаточно'})
    display(frame);details(value,'Раскрыть все численные границы и технические статусы гипотез')


def cases(value):
    rows=[]
    for case in value:
        parsed=case.get('parsed')
        issue=('Формат читается, итоговый ответ неверен'+(' (есть внешняя Markdown-оболочка)' if case.get('error')=='markdown_wrapper' else '')) if parsed else (case.get('error') or case.get('status') or 'Нет допустимого ответа')
        rows.append({'Модель':case['model'],'Задача':case['task_id'],'Эталон':case.get('reference','Класс строки из теста'),
                     'Ответ':str(parsed['answer']) if parsed else 'Нет допустимого итогового ответа',
                     'Что зафиксировано':issue})
    display(pd.DataFrame(rows));details(value,'Раскрыть дословные ответы, условия и метаданные ошибок')


def manual(value):
    frame=pd.DataFrame(value).rename(columns={'model':'Модель','type':'Тип','required':'Требуется ответов',
        'paired_reviews':'Ответов с двумя оценками','status':'Статус','kappa':'Коэффициент согласия'})
    display(frame);details(value,'Раскрыть исходный объект состояния ручной проверки')
