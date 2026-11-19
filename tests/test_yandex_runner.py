"""All transports are mocks. These tests never call a cloud API."""
import json
from pathlib import Path
import tempfile
import contextlib
import io
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yandex_runner as r


def response(answer=.5, confidence=90, text=None, usage=True):
    raw = {"status": "completed", "temperature": 0,
           "output": [{"content": [{"type": "output_text", "text": text if text is not None else json.dumps({"answer": answer, "explanation": "test", "confidence": confidence})}]}]}
    if usage:
        raw["usage"] = {"input_tokens": 22, "output_tokens": 3}
    return json.dumps(raw)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config, cls.rows = r.prepare(cls.root, "gpt://test/yandexgpt-5.1")
        cls.pilot = [row for row in cls.rows if row["stage"] == "pilot"]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "run"

    def execute(self, transport, rows=None, **kwargs):
        return r.run(self.root, self.directory, self.config, rows or self.pilot[:1], "pilot", "test-secret", kwargs.get("budget", 1000), 10, transport=transport, sleeper=lambda _: None)

    def test_all_344_unique_and_no_reference_payload(self):
        self.assertEqual(len(self.rows), 344)
        self.assertEqual(len({row["request_id"] for row in self.rows}), 344)
        self.assertTrue(all(set(row["payload"]) == {"model", "instructions", "input", "temperature", "max_output_tokens"} for row in self.rows))

    def test_resume_does_not_repeat_paid_response(self):
        transport = Mock(return_value=response())
        self.execute(transport)
        self.execute(transport)
        self.assertEqual(transport.call_count, 1)
        done, spent, _ = r.replay(r.load_lines(self.directory / "events.jsonl"))
        self.assertEqual(len(done), 1)
        self.assertAlmostEqual(spent, .02)

    def test_invalid_answer_is_saved_not_retried(self):
        transport = Mock(return_value=response(text="this is not JSON"))
        self.execute(transport)
        self.execute(transport)
        self.assertEqual(transport.call_count, 1)
        events = r.load_lines(self.directory / "events.jsonl")
        self.assertEqual(events[-1]["error"], "invalid_json")
        self.assertIn("this is not JSON", (self.directory / "responses.jsonl").read_text())

    def test_budget_stops_before_request(self):
        transport = Mock(return_value=response())
        with self.assertRaisesRegex(r.StopRun, "бюджету"):
            self.execute(transport, budget=.001)
        transport.assert_not_called()

    def test_timeout_blocks_resume(self):
        transport = Mock(side_effect=TimeoutError)
        with self.assertRaises(r.StopRun):
            self.execute(transport)
        with self.assertRaises(r.StopRun):
            self.execute(transport)
        self.assertEqual(transport.call_count, 1)

    def test_crash_after_started_blocks_resume(self):
        self.directory.mkdir()
        r.append(self.directory / "events.jsonl", {"event": "started", "request_id": self.pilot[0]["request_id"], "stage": "pilot", "reserved_rub": 8})
        transport = Mock()
        with self.assertRaisesRegex(r.StopRun, "без сохранённого ответа"):
            self.execute(transport)
        transport.assert_not_called()

    def test_429_retries_only_rejected_call(self):
        transport = Mock(side_effect=[HTTPError("https://test", 429, "limited", {"Retry-After": "0"}, None), response()])
        self.execute(transport)
        self.assertEqual(transport.call_count, 2)
        events = r.load_lines(self.directory / "events.jsonl")
        self.assertEqual([e["event"] for e in events], ["started", "rejected", "started", "response"])
        self.assertAlmostEqual(r.replay(events)[1], .02)

    def test_no_usage_stops_and_preserves_raw(self):
        transport = Mock(return_value=response(usage=False))
        with self.assertRaisesRegex(r.StopRun, "usage"):
            self.execute(transport)
        with self.assertRaises(r.StopRun):
            self.execute(transport)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(len(list(self.directory.glob("*.response.txt"))), 1)

    def test_config_change_stops(self):
        transport = Mock(return_value=response())
        self.execute(transport)
        with self.assertRaisesRegex(r.StopRun, "Параметры"):
            self.execute(transport, budget=999)
        self.assertEqual(transport.call_count, 1)

    def test_main_requires_committed_forecasts(self):
        transport = Mock()
        with self.assertRaises(r.StopRun):
            r.run(self.root, self.directory, self.config, self.rows, "main", "test-secret", 1000, 10, transport=transport)
        transport.assert_not_called()

    def test_secret_is_not_written(self):
        self.execute(Mock(return_value=response()))
        for path in self.directory.iterdir():
            self.assertNotIn(b"test-secret", path.read_bytes())

    def test_nested_run_lock_is_rejected(self):
        self.directory.mkdir()
        with r.exclusive(self.directory / ".lock"):
            with self.assertRaises(r.StopRun):
                with r.exclusive(self.directory / ".lock"):
                    self.fail("lock was not exclusive")

    def test_boolean_false_and_class_zero_are_valid(self):
        self.assertIs(r.parse_response(json.loads(response(False)), "boolean")[0]["answer"], False)
        self.assertEqual(r.parse_response(json.loads(response(0)), "class")[0]["answer"], 0)
        self.assertIsNone(r.parse_response(json.loads(response(True)), "number")[0])
        self.assertIsNone(r.parse_response(json.loads(response(1, confidence=101)), "class")[0])

    def test_full_night_run_and_second_launch_skip_all(self):
        kinds = {row["request_id"]: row["answer_kind"] for row in self.rows}
        def transport(payload, key):
            kind = kinds[r.digest(payload)]
            answer = {"number": .5, "boolean": False, "text": "test", "python_code": "print(4)",
                      "insufficient_data": "insufficient_data", "class": 0}[kind]
            return response(answer)
        sender = Mock(side_effect=transport)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(r, "registration", return_value="c"*40):
            self.execute(sender, rows=self.rows)
            r.run(self.root, self.directory, self.config, self.rows, "all", "test-secret", 1000, 10, transport=sender)
            r.run(self.root, self.directory, self.config, self.rows, "all", "test-secret", 1000, 10, transport=sender)
        self.assertEqual(sender.call_count, 344)
        exported = r.load_lines(self.directory / "responses.jsonl")
        self.assertEqual(len(exported), 344)
        classification = [row for row in exported if row["experiment"] == "classification"]
        self.assertEqual(len(classification), 300)
        self.assertTrue(all("shots" in row and "row_id" in row for row in classification))
        self.assertTrue(all("shots" not in row for row in exported if row["experiment"] != "classification"))

    def test_markdown_wrapper_preserves_answer_and_flags_violation(self):
        code = 'print(f"{{"accuracy": {accuracy}}}")'
        obj = {"answer": code, "explanation": "test", "confidence": 90}
        fence = chr(96)*3
        wrapped = fence + "json\n" + json.dumps(obj) + "\n" + fence
        parsed, error = r.parse_response(json.loads(response(text=wrapped)), "python_code")
        self.assertEqual(parsed, obj)
        self.assertEqual(parsed["answer"], code)
        self.assertEqual(error, "markdown_wrapper")

    def test_no_extraction_from_extra_prose(self):
        fence = chr(96)*3
        wrapped = "Here is the answer:\n"+fence+"json\n"+json.dumps({"answer": .5, "explanation": "test", "confidence": 90})+"\n"+fence
        self.assertIsNone(r.parse_response(json.loads(response(text=wrapped)), "number")[0])

    def test_reparse_preserves_journal_and_does_not_charge_again(self):
        fence = chr(96)*3
        wrapped = fence+"\n"+json.dumps({"answer": .5, "explanation": "test", "confidence": 90})+"\n"+fence
        sender = Mock(return_value=response(text=wrapped))
        self.execute(sender)
        journal = self.directory / "events.jsonl"
        original = journal.read_bytes()
        self.execute(sender)
        self.assertEqual(journal.read_bytes(), original)
        self.assertEqual(sender.call_count, 1)
        result = r.load_lines(self.directory / "responses.jsonl")[0]
        self.assertEqual(result["parsed"]["answer"], .5)
        self.assertFalse(result["strict_format_ok"])
        self.assertEqual(result["error"], "markdown_wrapper")

    def test_authorized_network_retry_keeps_reserve_and_old_answers(self):
        sender = Mock(side_effect=[TimeoutError(), response()])
        def attempt():
            r.run(self.root, self.directory, self.config, self.pilot[:1], "pilot", "test-secret", 1000, 100, transport=sender)
        with self.assertRaises(r.StopRun):
            attempt()
        r.authorize_retry(self.directory, self.pilot[0]["request_id"], "User confirmed network outage and retry")
        attempt()
        attempt()
        self.assertEqual(sender.call_count, 2)
        done, spent, _ = r.replay(r.load_lines(self.directory / "events.jsonl"))
        self.assertEqual(len(done), 1)
        self.assertAlmostEqual(spent, r.reserve(self.pilot[0]["payload"]) + .02)
        with self.assertRaises(r.StopRun):
            r.authorize_retry(self.directory, self.pilot[0]["request_id"], "Do not repeat a received answer")

    def test_retry_requires_note(self):
        self.directory.mkdir()
        with self.assertRaises(r.StopRun):
            r.authorize_retry(self.directory, "unknown", "")

    def test_metadata_cleanup_cannot_change_task_content(self):
        root=Path(self.temp.name)
        (root/'data').mkdir()
        path=root/'data/tasks.jsonl'
        row={'id':'A01','text_ru':'original condition','reference':{'answer':1},'source':{'citation':'old','kind':'custom'}}
        path.write_text(json.dumps(row)+'\n',encoding='utf-8')
        before=r.digest(path.read_bytes())
        semantic=r.metadata_semantic_digest(path)
        row['source']['citation']='neutral citation'
        path.write_text(json.dumps(row)+'\n',encoding='utf-8')
        entry={'before_sha256':before,'after_sha256':r.digest(path.read_bytes()),'semantic_sha256':semantic}
        audit=root/'data/metadata_cleanup.json'
        audit.write_text(json.dumps({'files':{'data/tasks.jsonl':entry}}),encoding='utf-8')
        r.verify_manifest_file(root,'data/tasks.jsonl',before)
        row['reference']['answer']=2
        path.write_text(json.dumps(row)+'\n',encoding='utf-8')
        entry['after_sha256']=r.digest(path.read_bytes())
        audit.write_text(json.dumps({'files':{'data/tasks.jsonl':entry}}),encoding='utf-8')
        with self.assertRaises(r.StopRun):
            r.verify_manifest_file(root,'data/tasks.jsonl',before)

    def test_git_line_endings_do_not_change_registered_content(self):
        root=Path(self.temp.name)
        (root/'data').mkdir()
        path=root/'data/tasks.jsonl'
        lf=b'{"id":"A01"}\n'
        path.write_bytes(lf.replace(b'\n',b'\r\n'))
        r.verify_manifest_file(root,'data/tasks.jsonl',r.digest(lf))
        path.write_bytes(b'{"id":"A02"}\r\n')
        with self.assertRaises(r.StopRun): r.verify_manifest_file(root,'data/tasks.jsonl',r.digest(lf))


if __name__ == "__main__":
    unittest.main()
