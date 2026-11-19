"""Offline Yandex analysis: reads saved responses, never calls a model."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score


SEED, N_BOOT = 42, 1000


def normalize_traceback_paths(text):
    def replace(match):
        path=match.group(1).replace('\\','/')
        display='<python>/Lib/'+path.split('/Lib/',1)[1] if '/Lib/' in path else '<project>/'+path.rsplit('/',1)[-1]
        return 'File "'+display+'"'
    return re.sub(r'File "([A-Za-z]:[^"\n]+)"',replace,text)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_text(record):
    return "\n".join(p.get("text", "") for i in record["raw_response"].get("output", [])
                     for p in i.get("content", []) if p.get("type") == "output_text")


def auto_correct(task, parsed):
    if parsed is None:
        return False
    answer, reference = parsed["answer"], task["reference"]["answer"]
    if task["type"] == "B":
        return type(answer) is bool and answer == reference
    if task["answer_kind"] == "insufficient_data":
        return answer == "insufficient_data"
    if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer):
        return False
    tolerance = task["tolerance"]
    return abs(answer-reference) <= tolerance["absolute"] + tolerance["relative"]*abs(reference)


def accuracy_ci(correct):
    values = np.asarray(correct, dtype=float)
    indices = np.random.default_rng(SEED).integers(0, len(values), size=(N_BOOT, len(values)))
    lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return {"accuracy": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def classification_metrics(truth, predictions):
    """-1 means abstention, counted as incorrect for its true class."""
    y, p = np.asarray(truth), np.asarray(predictions)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p != 1)))
    denom = 2*tp+fp+fn
    recalls = [float(np.mean(p[y == k] == k)) for k in (0, 1) if np.any(y == k)]
    return {"f1_minority": 2*tp/denom if denom else 0.0,
            "balanced_accuracy": float(np.mean(recalls)), "accuracy": float(np.mean(p == y)),
            "coverage": float(np.mean(p != -1)), "abstentions": int(np.sum(p == -1))}


def classification_ci(y, p):
    indices = np.random.default_rng(SEED).integers(0, len(y), size=(N_BOOT, len(y)))
    draws = [classification_metrics(y[i], p[i]) for i in indices]
    return {key: [float(v) for v in np.quantile([d[key] for d in draws], [.025, .975])]
            for key in ("f1_minority", "balanced_accuracy", "accuracy")}


def h12_comparison(y, p, baseline, shots):
    indices = np.random.default_rng(SEED).integers(0, len(y), size=(N_BOOT, len(y)))
    draws = [classification_metrics(y[i], p[i])["f1_minority"]-classification_metrics(y[i], baseline[i])["f1_minority"] for i in indices]
    low, high = map(float, np.quantile(draws, [.05, .95]))
    delta = classification_metrics(y, p)["f1_minority"]-classification_metrics(y, baseline)["f1_minority"]
    if shots == 5:
        decision = "confirm" if low > -.05 else "reject" if high < -.05 else "insufficient_data"
    else:
        decision = "confirm" if high < 0 else "reject" if low >= 0 else "insufficient_data"
    return {"shots": shots, "delta_f1_llm_minus_lr": delta, "one_sided_95_lower": low,
            "one_sided_95_upper": high, "decision": decision}


def calibration(scores, tasks):
    valid = [r for r in scores if r["parsed"] and r["confidence"] is not None]
    confidence = np.array([r["confidence"]/100 for r in valid])
    correct = np.array([r["correct"] for r in valid], dtype=float)
    bins, ece = [], 0.0
    for k in range(5):
        mask = (confidence >= k/5) & ((confidence < (k+1)/5) if k < 4 else (confidence <= 1))
        count = int(mask.sum())
        c, a = (float(confidence[mask].mean()), float(correct[mask].mean())) if count else (None, None)
        bins.append({"bin_low": k/5, "bin_high": (k+1)/5, "n": count, "mean_confidence": c, "accuracy": a})
        if count:
            ece += count/len(valid)*abs(c-a)
    groups = {}
    for i, row in enumerate(valid):
        groups.setdefault(tasks[row["task_id"]].get("pair_id", row["task_id"]), []).append(i)
    clusters = list(groups.values())
    rng, draws = np.random.default_rng(SEED), []
    for _ in range(N_BOOT):
        sample = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        draws.append(float(np.mean(confidence[sample]-correct[sample])))
    low, high = map(float, np.quantile(draws, [.025, .975]))
    return {"n_total": len(scores), "n_valid": len(valid), "coverage": len(valid)/len(scores),
            "mean_confidence": float(confidence.mean()), "accuracy_on_valid": float(correct.mean()),
            "full_denominator_accuracy": sum(r["correct"] for r in scores)/len(scores),
            "gap": float(np.mean(confidence-correct)), "ci_low": low, "ci_high": high, "ece": ece,
            "decision": "confirm" if low > 0 else "reject" if high <= 0 else "insufficient_data", "bins": bins}


def execute_code(root, records, tasks, out):
    """Only the eight previously inspected, self-contained sklearn snippets."""
    reviewed = {}
    for row in records:
        if row["stage"] == "main" and row["task_id"].startswith("D"):
            code = row["parsed"]["answer"]
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    if any(m.split(".")[0] not in {"sklearn", "numpy", "json"} for m in modules):
                        raise ValueError("Unreviewed import")
                if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr.startswith("fetch_")):
                    raise ValueError("Unreviewed attribute")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile", "open", "input", "getattr", "globals", "locals", "__import__"}:
                    raise ValueError("Unreviewed call")
            reviewed[row["task_id"]] = (code, hashlib.sha256(code.encode()).hexdigest())
    if len(reviewed) != 8:
        raise ValueError("Expected eight inspected code answers")
    env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "COMSPEC"}}
    scratch = out / "code_execution"
    scratch.mkdir(parents=True, exist_ok=True)
    env.update(TEMP=str(scratch.resolve()), TMP=str(scratch.resolve()), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    results = []
    aliases = {"test_accuracy": "accuracy", "silhouette_score": "silhouette"}
    for task_id, (code, sha) in reviewed.items():
        path = scratch / f"{task_id}.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", "-X", "utf8", str(path.resolve())], cwd=scratch,
                                  env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, exit_code = "", "Timeout after 120 seconds", -1
        stderr_original_sha256=hashlib.sha256(stderr.encode()).hexdigest()
        stderr=normalize_traceback_paths(stderr)
        path.with_suffix(".stdout.txt").write_text(stdout, encoding="utf-8")
        path.with_suffix(".stderr.txt").write_text(stderr, encoding="utf-8")
        metrics, json_output = None, False
        if exit_code == 0:
            try:
                metrics = json.loads(stdout)
                json_output = True
            except ValueError:
                # Diagnostic extraction of Python dict output, not an edit to the code.
                try:
                    cleaned = re.sub(r"np\.float64\(([-+0-9.eE]+)\)", r"\1", stdout)
                    metrics = ast.literal_eval(cleaned)
                except (ValueError, SyntaxError):
                    pass
        normalized = {aliases.get(k, k): v for k, v in metrics.items()} if isinstance(metrics, dict) else {}
        reference = tasks[task_id]["reference"]["answer"]["metrics"]
        matched = all(k in normalized and (normalized[k] == value if k in {"best_k", "best_C"} else abs(normalized[k]-value) <= .01) for k, value in reference.items())
        results.append({"task_id": task_id, "source_sha256": sha, "exit_code": exit_code,
                        "pass_at_1": exit_code == 0, "output_is_json": json_output,
                        "metrics": metrics, "metrics_match_reference": matched,
                        "stderr": stderr, "stderr_original_sha256":stderr_original_sha256,
                        "traceback_paths_normalized":True,
                        "methodology_review": "pending_two_independent_reviewers"})
    write_json(out / "code_execution.json", results)
    return results


def make_review_pack(root, records, tasks, out):
    ids = [t for t in tasks if t[0] in "BCD"]
    np.random.default_rng(SEED).shuffle(ids)
    by_id = {r["task_id"]: r for r in records if r["stage"] == "main"}
    pack, mapping, forms = [], [], []
    for i, task_id in enumerate(ids, 1):
        blind_id = f"R{i:03d}"
        row, task = by_id[task_id], tasks[task_id]
        pack.append({"blind_id": blind_id, "type": task["type"], "prompt": task["text_ru"],
                     "raw_answer": raw_text(row), "reference": task["reference"], "rubric": task["rubric"]})
        mapping.append({"blind_id": blind_id, "task_id": task_id, "request_id": row["request_id"]})
        forms.append({"blind_id": blind_id, "score_0_2": "", "leakage": "", "test_tuning": "", "train_evaluation": "", "wrong_metric": "", "missing_stratification": "", "missing_seed": "", "notes": ""})
    review = out / "review"
    review.mkdir(exist_ok=True)
    write_json(out / "review_key.json", mapping)
    write_json(review / "responses.json", pack)
    for name in ("reviewer_1.csv", "reviewer_2.csv"):
        if not (review / name).exists():
            write_csv(review / name, forms)
    md = ["# Материалы независимой проверки", "", "Модель и технические ID скрыты. Для настоящей слепой оценки нужно перемешать этот пакет с ответами остальных моделей; пакет только одного участника сам по себе этого не обеспечивает. Ключ review_key.json проверяющим не передавать.", "", "B и C: score_0_2 по рубрике. D: шесть флагов ошибок 0/1; N/A там, где критерий неприменим. Не исправлять ответ или код. Два проверяющих заполняют отдельные CSV независимо. После сравнения оценок рассчитывается Cohen kappa; при <0.6 уточнить рубрику и переоценить. Не заменять пропуски нулями.", ""]
    for item in pack:
        md += [f"## {item['blind_id']} (тип {item['type']})", "", item["prompt"], "", "Ответ модели:", "", item["raw_answer"], "", "Эталон и рубрика:", "", "```json", json.dumps({"reference": item["reference"], "rubric": item["rubric"]}, ensure_ascii=False, indent=2), "```", ""]
    (review / "REVIEW.md").write_text("\n".join(md), encoding="utf-8")
    return len(forms)


def review_status(out):
    """Read real human ratings only; missing ratings remain missing."""
    key = {r['blind_id']: r['task_id'] for r in read_json(out/'review_key.json')}
    forms=[]
    for number in (1,2):
        with (out/f'review/reviewer_{number}.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        if len(rows)!=len(key) or {r['blind_id'] for r in rows}!=set(key):
            raise ValueError('Review form IDs are missing or duplicated')
        forms.append({r['blind_id']:r for r in rows})
    result={}
    for kind in 'BC':
        ids=[i for i,t in key.items() if t.startswith(kind)]
        pairs=[]
        for i in ids:
            values=[f[i]['score_0_2'].strip() for f in forms]
            if all(values):
                if any(v not in {'0','1','2'} for v in values):
                    raise ValueError('Manual score must be 0, 1 or 2')
                pairs.append(tuple(map(int,values)))
        kappa=None
        if pairs and len(set(v for p in pairs for v in p))>1:
            kappa=float(cohen_kappa_score([p[0] for p in pairs],[p[1] for p in pairs]))
            if not math.isfinite(kappa): kappa=None
        result[kind]={'n_required':len(ids),'n_paired_scores':len(pairs),'kappa':kappa,
                      'status':'pending' if len(pairs)<len(ids) else 'revise_rubric' if kappa is not None and kappa<.6 else 'needs_adjudication' if any(a!=b for a,b in pairs) else 'complete'}
    fields=['leakage','test_tuning','train_evaluation','wrong_metric','missing_stratification','missing_seed']
    ids=[i for i,t in key.items() if t.startswith('D')]
    completed=sum(all(f[i][k].strip() in {'0','1','N/A'} for f in forms for k in fields) for i in ids)
    flag_agreement={}
    for field in fields:
        pairs=[tuple(int(f[i][field].strip()) for f in forms) for i in ids if all(f[i][field].strip() in {'0','1'} for f in forms)]
        kappa=None
        if pairs and len(set(v for p in pairs for v in p))>1:
            kappa=float(cohen_kappa_score([p[0] for p in pairs],[p[1] for p in pairs]))
            if not math.isfinite(kappa): kappa=None
        flag_agreement[field]={'n':len(pairs),'kappa':kappa,'agreement':sum(a==b for a,b in pairs)/len(pairs) if pairs else None}
    result['D']={'n_required':len(ids),'n_paired_scores':completed,'flag_agreement':flag_agreement,
                 'status':'pending' if completed<len(ids) else 'revise_rubric' if any(v['kappa'] is not None and v['kappa']<.6 for v in flag_agreement.values()) else 'needs_human_adjudication'}
    write_json(out/'review_status.json',result)
    return result


def figures(out, type_scores, classification, calibration_result, code_results, main, records, comparisons):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 130})
    paths = out / "figures"
    paths.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [r["type"] for r in type_scores]
    vals = [r["accuracy"] for r in type_scores]
    ax.errorbar(labels, vals, yerr=[[r["accuracy"]-r["ci_low"] for r in type_scores], [r["ci_high"]-r["accuracy"] for r in type_scores]], fmt="o", capsize=5)
    ax.set(ylim=(-.05, 1.05), ylabel="Accuracy", xlabel="Тип задачи", title="YandexGPT: автоматическая оценка, 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(paths / "accuracy_ci.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 2.5))
    heat = np.array([[next((r["accuracy"] for r in type_scores if r["type"] == t), np.nan) for t in "ABCDF"]])
    cmap = plt.get_cmap("YlGn").copy(); cmap.set_bad("lightgray")
    im = ax.imshow(heat, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(5), list("ABCDF")); ax.set_yticks([0], ["YandexGPT"])
    for i, value in enumerate(heat[0]):
        ax.text(i, 0, "ожидает оценки" if np.isnan(value) else f"{value:.1%}", ha="center", va="center", fontsize=9)
    ax.set_title("Модель × тип: C/D ожидают ручной оценки")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(paths / "heatmap.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for model in ("YandexGPT", "LogisticRegression", "GaussianNB"):
        selected = [r for r in classification if r["model"] == model]
        ax.plot([r["shots"] for r in selected], [r["f1_minority"] for r in selected], "o-", label=model)
    ax.set(xlabel="Число обучающих примеров", ylabel="F1 редкого класса", ylim=(-.05, 1.05), title="Тип E: одинаковые train и test; zero-shot LLM — отказы")
    ax.legend(); fig.tight_layout(); fig.savefig(paths / "classification_f1.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    b = [r for r in calibration_result["bins"] if r["n"]]
    ax.plot([0,1], [0,1], "--", color="gray")
    ax.scatter([r["mean_confidence"] for r in b], [r["accuracy"] for r in b], s=[r["n"]*15 for r in b])
    for r in b:
        ax.annotate(f"n={r['n']}", (r["mean_confidence"],r["accuracy"]), xytext=(-35,8), textcoords="offset points")
    ax.set(xlim=(0,1.03), ylim=(0,1.03), xlabel="Средняя уверенность", ylabel="Фактическая точность", title="H10: 22 из 24 ответов с доступной confidence")
    fig.tight_layout(); fig.savefig(paths / "reliability.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for r in comparisons:
        delta, lo, hi = r["delta_f1_llm_minus_lr"], r["one_sided_95_lower"], r["one_sided_95_upper"]
        ax.errorbar([r["shots"]], [delta], yerr=[[delta-lo],[hi-delta]], fmt="o", capsize=5)
    ax.axhline(0, color="gray"); ax.axhline(-.05, color="red", linestyle="--", label="Допуск не хуже: -0.05")
    ax.set(xlabel="Примеры", ylabel="F1 LLM − F1 LR", title="H12: парный bootstrap, односторонние 95% границы")
    ax.set_xticks([5,20])
    ax.legend(); fig.tight_layout(); fig.savefig(paths / "h12_delta.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    auto_cost = sum(r["cost_rub"] for r in records if r["stage"] == "main" and r["task_id"][0] in "ABF")
    auto_accuracy = sum(r["correct"] for r in main)/len(main)
    ax.scatter([auto_cost], [auto_accuracy]); ax.annotate("YandexGPT (A+B+F)", (auto_cost, auto_accuracy), xytext=(6,8), textcoords="offset points")
    ax.set(xlabel="Расчётная стоимость A+B+F, ₽", ylabel="Accuracy A+B+F", ylim=(0,1), title="Стоимость — точность: остальные модели ещё не получены")
    fig.tight_layout(); fig.savefig(paths / "cost_accuracy.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([r["task_id"] for r in code_results], [int(r["pass_at_1"]) for r in code_results])
    ax.set(ylabel="Запуск без ошибок (0/1)", title="D: запуск исходного кода, без исправлений")
    fig.tight_layout(); fig.savefig(paths / "code_execution.png"); plt.close(fig)


def analyze(root, execute_reviewed_code=False):
    root = Path(root).resolve()
    spec = importlib.util.spec_from_file_location("saved_runner", root / "yandex_runner.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    run_dir = root / "results/models/yandex/yandex-v1"
    out = root / "analysis/yandex"
    out.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "config.json")
    _, requests = runner.prepare(root, config["model"])
    events = read_lines(run_dir / "events.jsonl")
    done, accounted, _ = runner.replay(events)
    expected = {r["request_id"]: r for r in requests}
    if set(done) != set(expected):
        raise ValueError("Incomplete or unexpected responses")
    records = [runner.decode_event(done[rid], expected[rid]["answer_kind"]) for rid in expected]
    tasks = {r["id"]: r for r in read_lines(root / "data/tasks.jsonl")}
    scores = []
    for r in records:
        if r["stage"] == "main" and r["task_id"][0] in "ABF":
            parsed = r["parsed"]
            scores.append({"task_id": r["task_id"], "type": r["task_id"][0], "parsed": parsed is not None,
                           "answer": parsed["answer"] if parsed else None, "correct": auto_correct(tasks[r["task_id"]], parsed),
                           "confidence": parsed["confidence"] if parsed else None, "format_error": r["error"], "cost_rub": r["cost_rub"]})
    write_csv(out / "automatic_scores.csv", scores)
    type_scores = [{"type": t, "n": sum(r["type"] == t for r in scores), **accuracy_ci([r["correct"] for r in scores if r["type"] == t])} for t in "ABF"]
    write_csv(out / "type_metrics.csv", type_scores)
    with (root / "data/classification/test_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = {r["row_id"]: int(r["label"]) for r in csv.DictReader(stream)}
    row_ids = sorted(labels)
    y = np.array([labels[i] for i in row_ids])
    classification, predictions, prediction_export = [], {}, []
    for stage, shots in [("zero",0),("five",5),("twenty",20)]:
        rs = {r["row_id"]: r for r in records if r["stage"] == stage}
        assert set(rs) == set(labels)
        p = np.array([rs[i]["parsed"]["answer"] if rs[i]["parsed"] else -1 for i in row_ids])
        predictions[shots] = p
        classification.append({"model": "YandexGPT", "shots": shots, **classification_metrics(y,p), "ci": classification_ci(y,p)})
        prediction_export += [{"row_id": i, "shots": shots, "true_label": int(labels[i]), "prediction": int(pred), "refusal": pred == -1} for i,pred in zip(row_ids,p)]
    write_csv(out / "classification_predictions.csv", prediction_export)
    with (root / "results/reference/classification_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        baseline_rows = list(csv.DictReader(stream))
    comparisons = []
    for model in ("LogisticRegression", "GaussianNB"):
        for shots in (5,20):
            rs = {r["row_id"]: int(r["prediction"]) for r in baseline_rows if r["model"] == model and int(r["n_train"]) == shots}
            assert set(rs) == set(labels)
            p = np.array([rs[i] for i in row_ids])
            classification.append({"model": model,"shots":shots,**classification_metrics(y,p),"ci":classification_ci(y,p)})
            if model == "LogisticRegression":
                comparisons.append(h12_comparison(y,predictions[shots],p,shots))
    constant = read_json(root / "results/reference/classification_baselines.json")[-1]
    classification.append({"model":"constant_majority","shots":0,**classification_metrics(y,np.full(len(y),constant["predicted_class"])),"ci":classification_ci(y,np.full(len(y),constant["predicted_class"]))})
    write_json(out / "classification_metrics.json", classification)
    write_csv(out / "classification_metrics.csv", [{k:v for k,v in r.items() if k!='ci'} for r in classification])
    h10 = calibration(scores,tasks)
    h12_status = "reject" if any(r["decision"]=="reject" for r in comparisons) else "confirm" if all(r["decision"]=="confirm" for r in comparisons) else "insufficient_data"
    hypotheses = {"H5":{"decision":"pending_other_model","needed":"Qwen3 4B answers to A/B","yandex_A":type_scores[0]["accuracy"],"yandex_B":type_scores[1]["accuracy"]}, "H10":h10,"H12":{"decision":h12_status,"comparisons":comparisons}}
    write_json(out / "hypotheses.json",hypotheses)
    if execute_reviewed_code:
        code_results = execute_code(root,records,tasks,out)
    else:
        code_results = read_json(out / "code_execution.json")
        for result in code_results:
            original = next(r["parsed"]["answer"] for r in records if r["stage"]=="main" and r["task_id"]==result["task_id"])
            assert result["source_sha256"]==hashlib.sha256(original.encode()).hexdigest()
    write_csv(out / "code_metrics.csv", [{k:v for k,v in r.items() if k not in {"metrics","stderr"}} for r in code_results])
    review_count = make_review_pack(root,records,tasks,out)
    manual_status = review_status(out)
    figures(out,type_scores,classification,h10,code_results,scores,records,comparisons)
    known_cost = sum(r["cost_rub"] for r in records)
    summary = {"responses":len(records),"unique_requests":len(done),"counts":dict(Counter(r["stage"] for r in records)),
               "known_cost_rub":known_cost,"budget_ledger_rub":accounted,"unknown_attempt_reserve_rub":accounted-known_cost,
               "tokens":sum(r["input_tokens"]+r["output_tokens"] for r in records),
               "main_and_E_cost_rub":sum(r["cost_rub"] for r in records if r["stage"]!='pilot'),
               "auto_ABF":accuracy_ci([r["correct"] for r in scores]),"auto_ABF_n":len(scores),
               "code_pass_at_1":sum(r["pass_at_1"] for r in code_results)/len(code_results),
               "code_json_output_rate":sum(r["output_is_json"] for r in code_results)/len(code_results),
               "format_errors":dict(Counter(r["error"] or "none" for r in records if r["stage"]!='pilot')),
               "refusals_E_zero":classification[0]["abstentions"],"review_items":review_count,
               "overall_39_accuracy":None,"reason":"C and D require two independent reviews; B explanation scores also pending",
               "bootstrap_iterations":N_BOOT,"seed":SEED,"api_calls_during_analysis":0,
               "hypotheses_status":{k:v["decision"] for k,v in hypotheses.items()}}
    summary['manual_review']=manual_status
    write_json(out / "summary.json",summary)
    from importlib.metadata import version
    import platform
    write_json(out/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
        'packages':{name:version(name) for name in ['numpy','scipy','scikit-learn','matplotlib','nbformat','nbclient']}})
    inputs=['results/models/yandex/yandex-v1/events.jsonl','docs/yandex_registration.json','data/tasks.jsonl',
            'data/classification/test_labels.csv','results/reference/classification_predictions.csv']
    write_json(out/'provenance.json',{'sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in inputs},
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parser_version':runner.PARSER_VERSION,'protocol_commit':(run_dir/'registration_commit.txt').read_text().strip(),
        'seed':SEED,'bootstrap_iterations':N_BOOT,'api_calls':0,
        'editorial_metadata_cleanup':read_json(root/'data/metadata_cleanup.json') if (root/'data/metadata_cleanup.json').exists() else None})
    costs = [{"stage":stage,"n":sum(r['stage']==stage for r in records),"cost_rub":sum(r['cost_rub'] for r in records if r['stage']==stage),"mean_latency_s":float(np.mean([r['latency_s'] for r in records if r['stage']==stage]))} for stage in ['pilot','main','zero','five','twenty']]
    write_csv(out / "costs.csv",costs)
    candidates = [r for r in scores if not r["correct"]][:5]
    by_task = {r['task_id']:r for r in records if r['stage']=='main'}
    write_json(out / "error_examples.json",[{"task_id":r['task_id'],"prompt":tasks[r['task_id']]['text_ru'],"raw_response":raw_text(by_task[r['task_id']]),"reference":tasks[r['task_id']]['reference'],"automatic_correct":False,"student_analysis":""} for r in candidates])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--execute-reviewed-code",action="store_true")
    args=parser.parse_args()
    print(json.dumps(analyze(args.root,args.execute_reviewed_code),ensure_ascii=False,indent=2))
