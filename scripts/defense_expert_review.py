"""Validate recorded expert judgments against immutable original responses."""
import hashlib
import json
from pathlib import Path
from analysis_common import read_json, read_lines, raw_text, write_json, write_csv

FLAGS=('leakage','test_tuning','train_evaluation','wrong_metric','missing_stratification','missing_seed')


def review(root,records_by_model):
    root=Path(root);out=root/'analysis/defense'
    data=read_json(root/'data/reviews/defense_expert_review.json')
    assert data['assessors']==1 and data['cohen_kappa'] is None
    assert data['independent_blind_team_review_completed'] is False
    tasks={t['id']:t for t in read_lines(root/'data/tasks.jsonl')}
    expected={(m,t) for m in records_by_model for t in tasks if t[0] in 'BCD'}
    rows=data['ratings'];assert len(rows)==len(expected)==92
    assert {(r['model'],r['task_id']) for r in rows}==expected
    flat=[];summary=[];full_bank=[]
    for model,records in records_by_model.items():
        by_id={r['task_id']:r for r in records if r['stage']=='main'}
        for r in [r for r in rows if r['model']==model]:
            original=by_id[r['task_id']];task=tasks[r['task_id']]
            assert r['request_id']==original['request_id']
            assert r['response_sha256']==hashlib.sha256(raw_text(original).encode()).hexdigest()
            assert r['task_sha256']==hashlib.sha256(json.dumps(task,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            assert r['notes'].strip()
            if r['type'] in 'BC':assert r['score_0_2'] in (0,1,2)
            else:
                assert set(r['flags'])==set(FLAGS)
                assert all(v in (0,1,'N/A') for v in r['flags'].values())
            flat.append({k:v for k,v in r.items() if k!='flags'}|r.get('flags',{}))
        selected=[r for r in rows if r['model']==model]
        folder={'YandexGPT':'yandex','Qwen3-4B':'local','GigaChat-2':'gigachat','DeepSeek':'deepseek'}[model]
        executions={r['task_id']:r for r in read_json(root/f'analysis/{folder}/code_execution.json')}
        d=[r for r in selected if r['type']=='D']
        strict_d=sum(bool(executions[r['task_id']]['pass_at_1'] and executions[r['task_id']]['output_is_json'] and
                          executions[r['task_id']]['metrics_match_reference']) and
                     not any(v==1 for v in r['flags'].values()) and any(v==0 for v in r['flags'].values()) for r in d)
        b=[r['score_0_2'] for r in selected if r['type']=='B'];c=[r['score_0_2'] for r in selected if r['type']=='C']
        from analysis_common import auto_correct
        abf=sum(auto_correct(tasks[t],r['parsed']) for t,r in by_id.items() if t[0] in 'ABF')
        summary.append({'model':model,'reviewed':len(selected),'B_mean_0_2':sum(b)/len(b),'B_full_2':b.count(2),
                        'C_mean_0_2':sum(c)/len(c),'C_full_2':c.count(2),'C_total':len(c),
                        'D_wrong_metric':sum(r['flags']['wrong_metric']==1 for r in d),
                        'D_not_assessable':sum(all(v=='N/A' for v in r['flags'].values()) for r in d),
                        'D_strict_correct':strict_d,'D_total':8,
                        'expert_strict_39_correct':int(abf+c.count(2)+strict_d),
                        'expert_strict_39_accuracy':(abf+c.count(2)+strict_d)/39,'independent_team_review':False})
        # Separate, explicitly post-hoc composite; original registered A/B/F scores are unchanged.
        full_bank.append({'model':model,'type':'C','n':7,'correct':c.count(2),'accuracy':c.count(2)/7,'basis':'single_expert_full_rubric'})
        full_bank.append({'model':model,'type':'D','n':8,'correct':strict_d,'accuracy':strict_d/8,'basis':'primary_contract_execution_metrics_and_expert_methodology'})
    columns=list(dict.fromkeys(k for r in flat for k in r))
    write_csv(out/'expert_ratings.csv',flat,columns);write_csv(out/'expert_review_summary.csv',summary)
    write_json(out/'expert_review.json',{k:v for k,v in data.items() if k!='ratings'}|{'verified_ratings':len(rows),'summary':summary,'review_data_sha256':hashlib.sha256((root/'data/reviews/defense_expert_review.json').read_bytes()).hexdigest()})
    report=['# Экспертная проверка по рубрикам задания','',data['method'],'',data['limitations'],'',data['rubric_policy'],'',
            'Проверено 92 первых ответа: по 8 B, 7 C и 8 D для каждой из четырёх моделей. Хеши ответа, задания и request_id проверяются при каждом пересчёте. Исходные формы двух участников сохранены отдельно и не заполнялись этой проверкой.','',
            '## Сводка','', '| Модель | B среднее / 2 | C среднее / 2 | C полностью / 7 | D строгий / 8 | Дополнительный результат / 39 |',
            '| --- | --- | --- | --- | --- | --- |']
    for r in summary:report.append(f"| {r['model']} | {r['B_mean_0_2']:.3f} | {r['C_mean_0_2']:.3f} | {r['C_full_2']} | {r['D_strict_correct']} | {r['expert_strict_39_correct']} |")
    report+=['','39 = 24 A/B/F с автоматическими вердиктами B + 7 C (только 2 балла считаются полным успехом) + 8 D (первичный контракт, выполнение, численные метрики и методология одновременно). Это дополнительная оценка после просмотра результатов, а не финальная оценка двух слепых участников. Средний балл B отдельно учитывает качество обоснований.','',
             'Успех C строже разумного выбора алгоритма: нужны все обязательные аспекты рубрики. Поэтому подробный ответ также может получить 1 за конкретный пропуск. D=0 по флагу означает отсутствие найденной ошибки в тексте, а не успешный запуск. Для отсутствующего кода стоят N/A.','',
             'Giga D01 показывает, зачем нужен содержательный просмотр: best_estimator_.score выдаёт accuracy, хотя результат подписан balanced_accuracy. У Qwen D04 weighted-F1 заменяет F1 класса 1, D06 корреляция заменяет R². Статический просмотр не исправляет эти программы и не превращает диагностику Giga в primary pass.','',
             '## Построчные решения','']
    for model in records_by_model:
        report+=['### '+model,'']
        for r in [r for r in rows if r['model']==model]:
            report += [f"#### {r['task_id']} — "+(f"{r['score_0_2']}/2" if r['type'] in 'BC' else 'методология кода'),'',r['notes'],'']
            if r['type']=='D':report += ['Флаги: '+', '.join(f'{k}={v}' for k,v in r['flags'].items())+'.','']
            report += [f"Исходный request_id: `{r['request_id']}`; SHA256 ответа: `{r['response_sha256']}`.",'']
    (root/'docs/expert_review.md').write_text('\n'.join(report).rstrip()+'\n',encoding='utf-8')
    return full_bank
