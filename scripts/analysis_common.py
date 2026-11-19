"""Shared offline scoring rules, identical to the Yandex analysis."""
import ast, csv, hashlib, json, math, os, re, subprocess, sys
from pathlib import Path
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
                     for p in (i.get("content") or []) if p.get("type") == "output_text")


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
