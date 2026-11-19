"""One preliminary technical assessment. Does not fill independent reviewer forms."""
import argparse
import csv
import json
from pathlib import Path


RATINGS = [
    ('B01',2,'Вердикт верен: PCA не использует метки и максимизирует дисперсию.'),
    ('B02',2,'Вердикт верен; указана связь некалиброванности с нарушением предположения независимости.'),
    ('B03',1,'Утечка через статистики test объяснена верно. Не указан обязательный способ исправления: fit только на train и внутри CV.'),
    ('B04',2,'Различие ROC-AUC и precision при конкретном пороге объяснено; зависимость от базовой частоты указана.'),
    ('B05',2,'Правильно объяснены штраф за нарушения зазора и более слабая регуляризация при росте C.'),
    ('B06',2,'Правильно связаны масштаб признака, расстояние и возможная смена соседей.'),
    ('B07',1,'Вердикт верен. Фраза об обучении каждого дерева на разных наборах признаков неточна для обычного RandomForest: случайное подмножество выбирается при поиске разбиений в узлах.'),
    ('B08',2,'Верно названы целевая функция, зависимость от инициализации и возможность локального минимума.'),
    ('C01',1,'Логистическая регрессия, масштабирование и риск p>n разумны. Не описаны настройка регуляризации, стратификация и fit preprocessing внутри CV.'),
    ('C02',2,'Предложены KMeans, стандартизация, выбор k методом локтя и инженерная проверка центроидов; это допустимое решение по рубрике.'),
    ('C03',1,'Бустинг и линейный baseline подходят. Порядок preprocessing до разбиения неоднозначен и создаёт риск утечки; не указаны fit только train, CV и доступность признаков в момент прогноза.'),
    ('C04',1,'Модель и отказ от accuracy обоснованы, стоимость ошибок упомянута. Не задан подбор порога на validation/OOF; не описана явная оценка суммарной стоимости.'),
    ('C05',1,'Временное разбиение и прошлые лаги корректны. Не предложен сезонный наивный baseline, не выбрана метрика, не уточнено использование прогноза погоды вместо будущих фактических значений.'),
    ('C06',1,'PCA выбран верно, отсутствие гарантии разделения классов объяснено. Пропущены требуемые preprocessing и оценка explained variance.'),
    ('C07',1,'NB и логистическая регрессия для sparse текста подходят, уплотнение не нужно, macro-метрики упомянуты. Нет схемы разбиения/CV и fit vectorizer на train; список в поле answer нарушает требуемый тип, что фиксируется отдельно.'),
]


def create(root):
    root=Path(root)
    out=root/'analysis/yandex'
    execution={r['task_id']:r for r in json.loads((out/'code_execution.json').read_text(encoding='utf-8'))}
    flags=['leakage','test_tuning','train_evaluation','wrong_metric','missing_stratification','missing_seed']
    rows=[{'task_id':task,'type':task[0],'score_0_2':score,'rationale':reason,
           **{f:'' for f in flags},'pass_at_1':'','assessment_status':'preliminary_single_review'} for task,score,reason in RATINGS]
    notes={
        'D01':'Pipeline и CV построены корректно, C выбран только на train. Запуск успешен, метрики совпали. stdout является Python dict, а не JSON; это нарушение формата, не утечка.',
        'D02':'Imputer/scaler обучаются в Pipeline на train, метрики и seed заданы верно. Программа не запустилась: отсутствует import json. Код не исправлялся.',
        'D03':'Scaler внутри Pipeline/CV, k выбран только на train, stratify и seed заданы. Запуск и метрики корректны.',
        'D04':'Случайный лес обучен на train, test отделён со стратификацией, метрики дисбаланса и seed корректны. Запуск успешен.',
        'D05':'Scaler и PCA обучены внутри Pipeline только на train. Стратификация и seed заданы. Запуск и accuracy корректны.',
        'D06':'Train/test, Pipeline и R2 корректны; стратификация для регрессии неприменима. Запуск прерван: mean_squared_error(...,squared=False) не поддерживается в зафиксированном sklearn 1.7.2. Код не исправлялся.',
        'D07':'Описательная кластеризация: fit на всех 300 объектах допустим по условию; holdout/стратификация неприменимы. Silhouette рассчитан по масштабированным признакам, seed и n_init заданы.',
        'D08':'GaussianNB обучен только на train; исходная кодировка классов и pos_label=0 соблюдены. Стратификация, seed и метрики корректны.'}
    for task in sorted(notes):
        values={f:0 for f in flags}
        if task=='D06': values['missing_stratification']='N/A'
        if task=='D07':
            for f in ['leakage','test_tuning','train_evaluation','missing_stratification']: values[f]='N/A'
        rows.append({'task_id':task,'type':'D','score_0_2':'','rationale':notes[task],**values,
                     'pass_at_1':execution[task]['pass_at_1'],'assessment_status':'preliminary_single_review'})
    with (out/'preliminary_review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report={'assessment_status':'preliminary_single_review','not_counted_as_independent_reviews':True,
            'basis':'Сопоставление исходных ответов с опубликованными рубриками и фактическими результатами запуска кода.',
            'reviewer_forms_modified':False,'cohen_kappa':None,'ratings':rows,
            'note':'Это одна предварительная проверка. Независимость двух проверяющих по ней не подтверждается; общая итоговая оценка до независимых проверок остаётся незаполненной.'}
    (out/'preliminary_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Prepared one preliminary review:',len(rows),'answers. Independent reviewer forms unchanged.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    create(parser.parse_args().root)
