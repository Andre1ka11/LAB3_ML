"""Resumable DeepSeek experiment through Yandex AI Studio. Dry-run is the default; stdlib only."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parent
INPUT_RATE = 0.0003
CACHED_INPUT_RATE = 0.000075
OUTPUT_RATE = 0.0005  # RUB/token incl. VAT; checked 2026-11-19.
MODEL_NAME = "deepseek-v4.1-flash"
ENDPOINT = "https://ai.api.cloud.yandex.net/v1/responses"
PARSER_VERSION = "1.1-outer-markdown-only"
STAGES = {"pilot": "pilot_ru", "main": "bank_ru", "zero": "classification_ru",
          "five": "classification_ru", "twenty": "classification_ru"}


class StopRun(Exception):
    pass


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def newline_hashes(data):
    normalized=data.replace(b'\r\n',b'\n')
    return {digest(data),digest(normalized),digest(normalized.replace(b'\n',b'\r\n'))}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_lines(path):
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise StopRun(f"Повреждена строка {i}: {path}. Автоматически файл не изменяется.")
    return rows


def append(path, value):
    data = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def exclusive(path):
    """OS lock is released even if the process crashes; no stale PID lock."""
    stream = path.open("a+b")
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise StopRun("Другой процесс уже выполняет этот прогон.")
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def settings(root):
    values = {}
    env = root / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip("\"'")
    for name in ("YANDEX_API_KEY", "YANDEX_FOLDER_ID", "YANDEX_MODEL"):
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


def metadata_semantic_digest(path):
    """Ignore only the editorial source citation; task content stays protected."""
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            source = json.loads(row["source"])
            source.pop("citation", None)
            row["source"] = source
    else:
        rows = load_lines(path)
        for row in rows:
            row["source"].pop("citation", None)
    return digest(rows)


def verify_manifest_file(root, filename, expected):
    path = root / filename
    variants = newline_hashes(path.read_bytes())
    if expected in variants:
        return
    audit_path = root / "data/metadata_cleanup.json"
    allowed = {"data/tasks.jsonl", "data/tasks.csv", "data/pilot_tasks.jsonl"}
    if filename in allowed and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        item = audit.get("files", {}).get(filename, {})
        if (item.get("before_sha256") == expected and item.get("after_sha256") in variants
                and item.get("semantic_sha256") == metadata_semantic_digest(path)):
            return
    raise StopRun(f"Контрольная сумма не совпала: {filename}")


def prepare(root, model):
    system = (root / "prompts/system_ru.txt").read_text(encoding="utf-8-sig")
    manifest_bytes = (root / "data/bank_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    manifest_hash=digest(manifest_bytes)
    audit_path=root/'data/metadata_cleanup.json'
    if audit_path.exists():
        original_manifest_hash=json.loads(audit_path.read_text(encoding='utf-8')).get('registered_manifest_sha256')
        if original_manifest_hash in newline_hashes(manifest_bytes):
            manifest_hash=original_manifest_hash
    for filename, expected in manifest["sha256"].items():
        verify_manifest_file(root, filename, expected)
    kinds = {t["id"]: t["answer_kind"] for name in ("tasks_public", "pilot_tasks")
             for t in load_lines(root / f"data/{name}.jsonl")}
    rows = []
    for stage, filename in STAGES.items():
        for row in load_lines(root / f"data/requests/{filename}.jsonl"):
            if stage in ("zero", "five", "twenty") and row["shots"] != {"zero": 0, "five": 5, "twenty": 20}[stage]:
                continue
            payload = {"model": model, "instructions": system, "input": row["user_prompt"],
                       "temperature": 0, "max_output_tokens": 4096}
            item = {**row, "stage": stage, "payload": payload,
                    "answer_kind": "class" if stage in ("zero", "five", "twenty") else kinds[row["task_id"]],
                    "request_id": digest(payload)}
            rows.append(item)
    if len(rows) != 344 or len({r["request_id"] for r in rows}) != 344:
        raise StopRun("Ожидались 344 уникальных запроса.")
    config = {"model": model, "endpoint": ENDPOINT, "temperature": 0, "max_output_tokens": 4096,
              "thinking_mode": "default; no explicit thinking parameter requested",
              "bank_version": manifest["bank_version"], "bank_sha256": manifest_hash,
              "prompt_sha256": digest(system.encode("utf-8")), "requests_sha256": digest([r["request_id"] for r in rows]),
              "rates_rub_per_token": {"input": INPUT_RATE, "cached_input": CACHED_INPUT_RATE, "output": OUTPUT_RATE},
              "tariff_checked": "2026-11-19", "tariff_source": "https://aistudio.yandex.ru/ru/docs/ai-studio/pricing",
              "cost_basis": "Published RUB rates and reported usage; not a Billing statement. Missing cache details are charged at the full input rate."}
    return config, rows


def parse_response(raw, kind):
    if raw.get("status") != "completed":
        return None, "response_not_completed"
    text = "\n".join(part.get("text", "") for item in raw.get("output", [])
                     for part in (item.get("content") or []) if part.get("type") == "output_text")
    warning = None
    lines = text.strip().splitlines()
    fence = chr(96) * 3
    if len(lines) >= 3 and lines[0].strip() in (fence, fence + "json") and lines[-1].strip() == fence:
        text = "\n".join(lines[1:-1])
        warning = "markdown_wrapper"
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None, "invalid_json"
    if not isinstance(obj, dict) or set(obj) != {"answer", "explanation", "confidence"}:
        return None, "invalid_fields"
    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 100 or not isinstance(obj["explanation"], str):
        return None, "invalid_confidence_or_explanation"
    answer = obj["answer"]
    valid = {"boolean": lambda: isinstance(answer, bool),
             "number": lambda: not isinstance(answer, bool) and isinstance(answer, (int, float)) and math.isfinite(answer),
             "text": lambda: isinstance(answer, str),
             "python_code": lambda: isinstance(answer, str) and bool(answer.strip()),
             "insufficient_data": lambda: isinstance(answer, (str, int, float)) and not isinstance(answer, bool),
             "class": lambda: type(answer) is int and answer in (0, 1)}[kind]()
    # Incorrect numeric/string answers remain subject-matter errors, never retried.
    return (obj, warning) if valid else (None, "invalid_answer_type")


def decode_event(event, kind):
    """Rebuild a derived view. Never rewrite original events or model content."""
    if event["event"] != "response":
        return dict(event)
    parsed, error = parse_response(event["raw_response"], kind)
    return {**event, "original_error": event.get("original_error", event.get("error")),
            "parsed": parsed, "error": error, "parser_version": PARSER_VERSION,
            "strict_format_ok": error is None and parsed is not None}


def cost(raw):
    usage = raw.get("usage") or {}
    a, b = usage.get("input_tokens"), usage.get("output_tokens")
    if type(a) is not int or type(b) is not int or a < 0 or b < 0:
        return None
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    if type(cached) is not int or not 0 <= cached <= a:
        return None
    # Reasoning tokens are already included in output_tokens; never add them twice.
    return INPUT_RATE * (a - cached) + CACHED_INPUT_RATE * cached + OUTPUT_RATE * b


def reserve(payload):
    # Deliberately conservative estimate, not a provider-enforced billing cap.
    upper_input = len((payload["instructions"] + payload["input"]).encode("utf-8")) + 2048
    return INPUT_RATE * upper_input + OUTPUT_RATE * payload["max_output_tokens"]


def replay(events):
    pending, done = {}, {}
    spent = pilot_spent = 0.0
    for event in events:
        rid = event.get("request_id")
        if event["event"] == "started":
            if rid in pending or rid in done:
                raise StopRun("Журнал содержит повторный запуск завершённого/неопределённого запроса.")
            pending[rid] = event
        elif event["event"] in ("response", "uncertain", "rejected"):
            started = pending.pop(rid, None)
            if started is None:
                raise StopRun("Нарушен порядок событий журнала.")
            if event["event"] == "rejected":
                continue
            charge = event.get("cost_rub")
            if charge is None:
                charge = started["reserved_rub"]
            spent += charge
            if started["stage"] == "pilot":
                pilot_spent += charge
            done[rid] = event
        elif event["event"] == "retry_authorized":
            previous = done.get(rid)
            if previous is None or previous["event"] != "uncertain" or not event.get("note"):
                raise StopRun("Некорректное разрешение повтора неопределённой попытки.")
            # Keep the first attempt's full reserved cost in the budget ledger.
            # Only clear its cache entry, so one explicitly authorized retry can run.
            del done[rid]
    if pending:
        raise StopRun("Обнаружен запрос без сохранённого ответа: сервер мог списать токены. Автоповтор запрещён. Сохраните журнал и проверьте ситуацию вручную.")
    uncertain = [rid for rid,e in done.items() if e["event"] == "uncertain" or e.get("cost_rub") is None]
    if uncertain:
        raise StopRun("Есть неопределённая попытка/ответ без usage. Автопродолжение остановлено; проверьте журнал и Billing, не создавайте новый run_id для обхода проверки.")
    return done, spent, pilot_spent


def authorize_retry(directory, request_id, note):
    if not note or not note.strip():
        raise StopRun("Для ручного повтора требуется причина/разрешение пользователя.")
    with exclusive(directory / ".lock"):
        journal = directory / "events.jsonl"
        events = load_lines(journal)
        matching = [e for e in events if e.get("request_id") == request_id]
        if not matching or matching[-1]["event"] != "uncertain":
            raise StopRun("Разрешить повтор можно только для последней неопределённой попытки; полученные ответы не повторяются.")
        append(journal, {"event": "retry_authorized", "timestamp_utc": now(),
                         "request_id": request_id, "note": note.strip(),
                         "reserved_cost_retained": True})


def send(payload, key):
    request = Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Authorization": f"Api-Key {key}", "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:
        return response.read().decode("utf-8")


def registration(root, config):
    path = "docs/deepseek_registration.json"
    local = root / path
    if not local.exists():
        raise StopRun("Сначала prepare-registration, заполнение прогнозов и коммит docs/deepseek_registration.json до main.")
    data = json.loads(local.read_text(encoding="utf-8-sig"))
    if data.get("config") != config or data.get("criteria_confirmed") is not True:
        raise StopRun("Регистрация не соответствует параметрам либо критерии не согласованы.")
    if set(data.get("forecasts", {})) != {"H5", "H10", "H12"} or not all(v in ("confirm", "reject") for v in data["forecasts"].values()):
        raise StopRun("Запишите прогноз confirm/reject для каждой из H5/H10/H12.")
    proc = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], cwd=root, capture_output=True, text=True)
    commit = proc.stdout.strip()
    if proc.returncode or not commit:
        raise StopRun("Файл прогнозов ещё не закоммичен.")
    committed = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True)
    if committed.returncode or json.loads(committed.stdout.decode("utf-8-sig")) != data:
        raise StopRun("Прогнозы изменены после коммита. До main зафиксируйте их заново.")
    return commit


def run(root, directory, config, rows, stage, key, budget, pilot_budget, transport=send, sleeper=time.sleep):
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive(directory / ".lock"):
        config_path = directory / "config.json"
        stored_config = {**config, "budget_rub": budget, "pilot_budget_rub": pilot_budget}
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != stored_config:
                raise StopRun("Параметры прогона изменились. Нельзя незаметно продолжить старый прогон с новыми параметрами.")
        else:
            config_path.write_text(json.dumps(stored_config, ensure_ascii=False, indent=2), encoding="utf-8")
        journal = directory / "events.jsonl"
        done, spent, pilot_spent = replay(load_lines(journal))
        kinds = {r["request_id"]: r["answer_kind"] for r in rows}
        done = {rid: decode_event(event, kinds[rid]) for rid, event in done.items()}
        selected = [r for r in rows if r["stage"] == stage or stage == "all"]
        commit = None
        if stage != "pilot":
            commit = registration(root, config)
            pilot_ids = [r["request_id"] for r in rows if r["stage"] == "pilot"]
            if not all(rid in done and done[rid].get("parsed") is not None for rid in pilot_ids):
                raise StopRun("Сначала успешно завершите pilot: 5 ответов с корректным форматом. Правильность решения не является условием допуска.")
            anchor = directory / "registration_commit.txt"
            if anchor.exists() and anchor.read_text(encoding="utf-8").strip() != commit:
                raise StopRun("Регистрация изменена после начала основного прогона. Автоматическое продолжение запрещено.")
            if not anchor.exists():
                anchor.write_text(commit + "\n", encoding="utf-8")
        for i, row in enumerate(selected, 1):
            rid = row["request_id"]
            if rid in done:
                continue
            held = reserve(row["payload"])
            if spent + held > budget or (row["stage"] == "pilot" and pilot_spent + held > pilot_budget):
                raise StopRun(f"Стоп по бюджету: учтено {spent:.3f} руб., резерв следующего запроса {held:.3f} руб.")
            base = {"run_id": directory.name, "request_id": rid, "task_id": row["task_id"],
                    "stage": row["stage"], "experiment": row["experiment"], "language": row["language"],
                    "model_id": config["model"],
                    "provider": "Yandex AI Studio", "bank_version": config["bank_version"],
                    "bank_sha256": config["bank_sha256"], "prompt_sha256": config["prompt_sha256"],
                    "temperature_requested": 0, "temperature_effective": None,
                    "thinking_mode": config["thinking_mode"], "protocol_commit": commit}
            if "row_id" in row:
                base.update(row_id=row["row_id"], shots=row["shots"])
            for attempt in range(1, 4):
                append(journal, {**base, "event": "started", "timestamp_utc": now(), "attempt": attempt,
                                 "reserved_rub": held, "request": row["payload"]})
                started = time.perf_counter()
                try:
                    raw_text = transport(row["payload"], key)
                except HTTPError as error:
                    # Keep provider diagnostics, excluding any accidental echo of the key.
                    try:
                        provider_error = error.read(8192).decode("utf-8", errors="replace").replace(key, "<redacted>")
                    except Exception:
                        provider_error = ""
                    known_rejection = 400 <= error.code < 500 and error.code != 408
                    event = "rejected" if known_rejection else "uncertain"
                    append(journal, {**base, "event": event, "timestamp_utc": now(), "http_status": error.code,
                                     "error": f"HTTP {error.code}", "provider_error": provider_error, "cost_rub": 0 if known_rejection else None})
                    if error.code == 429 and attempt < 3:
                        try:
                            delay = min(300, max(2, float(error.headers.get("Retry-After", 10*attempt))))
                        except (ValueError, AttributeError):
                            delay = 10*attempt
                        print(f"429: ожидание {delay} с", flush=True)
                        sleeper(delay)
                        continue
                    raise StopRun(f"HTTP {error.code}: прогон остановлен. Проверьте доступ/лимит; не запускайте бесконечные повторы.")
                except Exception as error:
                    append(journal, {**base, "event": "uncertain", "timestamp_utc": now(),
                                     "error": type(error).__name__, "cost_rub": None})
                    raise StopRun("Ответ не получен надёжно; возможна тарификация. Автоповтора не будет.")
                # Persist response before attempting JSON parsing. If parsing crashes, replay can rebuild it.
                raw_path = directory / f"{rid}.response.txt"
                with raw_path.open("wb") as stream:
                    stream.write(raw_text.encode("utf-8")); stream.flush(); os.fsync(stream.fileno())
                try:
                    raw = json.loads(raw_text)
                    if not isinstance(raw, dict):
                        raise ValueError("not an object")
                except ValueError:
                    append(journal, {**base, "event": "uncertain", "timestamp_utc": now(),
                                     "error": "invalid_provider_json", "cost_rub": None})
                    raise StopRun("Некорректный ответ API сохранён. Нужна ручная проверка.")
                parsed, format_error = parse_response(raw, row["answer_kind"])
                charge = cost(raw)
                usage = raw.get("usage") or {}
                record = {**base, "event": "response", "timestamp_utc": now(), "raw_response": raw,
                          "parsed": parsed, "error": format_error, "cost_rub": charge,
                          "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                          "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                          "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                          "cost_is_billing_statement": False,
                          "latency_s": time.perf_counter()-started}
                record.update(parser_version=PARSER_VERSION, strict_format_ok=format_error is None and parsed is not None)
                if type(raw.get("temperature")) in (int, float):
                    record["temperature_effective"] = raw["temperature"]
                append(journal, record)
                done[rid] = record
                if charge is None:
                    raise StopRun("Ответ сохранён, но usage отсутствует. Остановка для проверки расходов.")
                spent += charge
                if row["stage"] == "pilot":
                    pilot_spent += charge
                print(f"{i}/{len(selected)} {row['stage']} {row['task_id']} {row.get('row_id','')} | {format_error or 'сохранено'} | {spent:.3f} руб.", flush=True)
                break
        export(directory, load_lines(journal), rows)
        print(f"Этап {stage} завершён. Учтено {spent:.3f} руб. Качество решений ещё не оценивалось.")


def export(directory, events, rows):
    """Derived, rebuildable view; events.jsonl remains the source of truth."""
    kinds = {r["request_id"]: r["answer_kind"] for r in rows}
    records = [decode_event(e, kinds[e["request_id"]]) for e in events if e["event"] == "response"]
    temp = directory / "responses.jsonl.tmp"
    temp.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    temp.replace(directory / "responses.jsonl")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="dry-run", choices=["dry-run", "prepare-registration", "run", "status", "authorize-retry"])
    parser.add_argument("--request-id")
    parser.add_argument("--note")
    parser.add_argument("--append-logs", action="store_true", help="Append background stdout/stderr to existing process logs")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="pilot")
    parser.add_argument("--run-id", default="deepseek-v1")
    parser.add_argument("--budget-rub", type=float, default=300)
    parser.add_argument("--pilot-budget-rub", type=float, default=10)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("run-id: только латиница, цифры, дефис и подчёркивание")
    if not (math.isfinite(args.budget_rub) and math.isfinite(args.pilot_budget_rub) and args.budget_rub > 0 and args.pilot_budget_rub > 0):
        parser.error("Бюджет должен быть положительным конечным числом")
    values = settings(ROOT)
    model = f"gpt://{values.get('YANDEX_FOLDER_ID', '<folder_id>')}/{MODEL_NAME}"
    try:
        config, rows = prepare(ROOT, model)
        directory = ROOT / "results/models/deepseek" / args.run_id
        if args.append_logs:
            if args.command != "run":
                raise StopRun("--append-logs применяется только к run")
            directory.mkdir(parents=True, exist_ok=True)
            sys.stdout = (directory / "process.stdout.log").open("a", encoding="utf-8", buffering=1)
            sys.stderr = (directory / "process.stderr.log").open("a", encoding="utf-8", buffering=1)
            print(f"=== RESUME {now()} ===", flush=True)
            print(f"=== RESUME {now()} (previous messages are historical) ===", file=sys.stderr, flush=True)
        if args.command == "dry-run":
            print("API calls: 0. Контрольные суммы проверены.")
            for stage in STAGES:
                print(stage, sum(r["stage"] == stage for r in rows))
            print("Итого 344. Секретный ключ не выводится. run по умолчанию запускает только pilot.")
        elif args.command == "prepare-registration":
            if not values.get("YANDEX_FOLDER_ID"):
                raise StopRun("Сначала заполните YANDEX_FOLDER_ID в локальном .env.")
            path = ROOT / "docs/deepseek_registration.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"config": config, "forecasts": {"H5": None, "H10": None, "H12": None},
                           "criteria_confirmed": False,
                           "criteria": "Заполните согласованные сравнения, метрики, допуски и статистические критерии до основного прогона."}, stream, ensure_ascii=False, indent=2)
            print("Создан шаблон регистрации; прогнозы не заполнены. API calls: 0.")
        elif args.command == "authorize-retry":
            if not args.request_id or not args.note:
                raise StopRun("Укажите --request-id и --note с основанием ручного повтора.")
            authorize_retry(directory, args.request_id, args.note)
            print("Разрешён один ручной повтор. Возможный расход первой попытки остаётся в резерве; API не вызывался.")
        elif args.command == "status":
            events = load_lines(directory / "events.jsonl")
            if directory.exists():
                with exclusive(directory / ".lock"):
                    events = load_lines(directory / "events.jsonl")
                    export(directory, events, rows)
            done, spent, pilot_spent = replay(events)
            print(f"Сохранено {len(done)}, стоимость {spent:.3f} руб., pilot {pilot_spent:.3f} руб.")
            if directory.exists():
                decoded = load_lines(directory / "responses.jsonl")
                print(f"Разобрано: {sum(e['parsed'] is not None for e in decoded)}; строгих нарушений формата: {sum(not e['strict_format_ok'] for e in decoded)}.")
        else:
            if not values.get("YANDEX_FOLDER_ID") or not values.get("YANDEX_API_KEY"):
                raise StopRun("Заполните YANDEX_FOLDER_ID и YANDEX_API_KEY в .env. Не присылайте ключ в чат.")
            run(ROOT, directory, config, rows, args.stage, values["YANDEX_API_KEY"], args.budget_rub, args.pilot_budget_rub)
    except (StopRun, FileExistsError) as error:
        print(f"СТОП: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Остановлено. Ответы сохранены. Перед продолжением проверьте status.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
