"""Round-level User length choices, persistence, and Character isolation; no API calls."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dialogue_generate as dg


class ScriptedClient:
    def __init__(self, fail_character_call=None):
        self.user_calls = 0
        self.character_calls = 0
        self.fail_character_call = fail_character_call
        self.requests = []

    def call(self, role, messages, validator=None):
        self.requests.append((role, copy.deepcopy(messages)))
        if role == "user":
            rows = [("minimal", "嗯"), ("long", "我想补充两点。\n 第一，时间还没定。第二，我希望听你的看法。"),
                    ("short", "好，先到这。")]
            length, content = rows[self.user_calls]
            self.user_calls += 1
            result = {"message": content, "response_length": length, "goal_completed": self.user_calls == 3}
            validator(result)
            return result
        if role != "character":
            raise AssertionError("Only two actors may call models")
        self.character_calls += 1
        if self.character_calls == self.fail_character_call:
            raise RuntimeError("interrupted")
        return "好。\n 再说说。" if self.user_calls < 3 else "再见。"


class DynamicLengthTests(unittest.TestCase):
    def job(self):
        catalog = dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES,
                                  dg.ROOT / "profiles/Character_profile", conversation_mode="open_chat")
        job = dg.make_jobs(catalog, 1, 42, "open_chat")[0]
        dg.assign_user_behaviors([job], dg.load_user_behaviors(dg.DEFAULT_USER_BEHAVIORS, "sharp"), 42)
        return job

    def test_new_contract_requires_valid_selected_label_not_inferred_length(self):
        for label in dg.RESPONSE_LENGTHS:
            dg.validate_user({"message": "嗯", "goal_completed": False, "response_length": label})
        for label in (None, "", "medium", "minimal/short/long", "SHORT", [], {}, 1, True):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "response_length"):
                dg.validate_user({"message": "嗯", "goal_completed": False, "response_length": label})
        with self.assertRaisesRegex(ValueError, "Missing response_length"):
            dg.validate_user({"message": "嗯", "goal_completed": False})
        dg.validate_legacy_user({"message": "嗯", "goal_completed": False})

    def test_interruption_preserves_completed_and_pending_labels_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = self.job()
            client = ScriptedClient(fail_character_call=2)
            save = lambda: dg.save_job(output, job)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                dg.run_job(job, client, save)
            partial = dg.read_json(output / f"{job['id']}.json")
            self.assertEqual(partial["messages"][1], {"role": "user", "response_length": "minimal", "content": "嗯"})
            self.assertEqual(len(partial["messages"]), 3)
            state = dg.read_json(output / f"{job['id']}.progress.log")["state"]
            self.assertEqual(state["pending_message"]["response_length"], "long")
            job = dg.read_job(output, job["id"])
            self.assertEqual([m["response_length"] for m in job["messages"] if m["speaker"] == "user"],
                             ["minimal", "long"])
            dg.run_job(job, client, save)
            self.assertEqual((client.user_calls, client.character_calls), (3, 4))
            restored = dg.read_job(output, job["id"])
            before = (output / f"{job['id']}.json").read_bytes()
            dg.save_job(output, restored)
            self.assertEqual(before, (output / f"{job['id']}.json").read_bytes())
            messages = dg.read_json(output / f"{job['id']}.json")["messages"]
            self.assertEqual([m["response_length"] for m in messages if m["role"] == "user"],
                             ["minimal", "long", "short"])
            self.assertTrue(all("response_length" not in m for m in messages if m["role"] != "user"))
            metrics = [check["length_metrics"] for check in restored["checks"]]
            self.assertEqual(metrics[0], {"user_response_length": "minimal", "previous_character_chars": None,
                                           "user_chars": 1, "character_chars": 6})
            self.assertEqual(metrics[1]["previous_character_chars"], 6)
            self.assertEqual(metrics[1]["user_chars"], sum(not c.isspace() for c in messages[3]["content"]))
            self.assertEqual(metrics[2]["character_chars"], 3)
            self.assertEqual(dg.export_jobs(output), 1)
            training = json.loads((output / "training.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(all(set(m) == {"role", "content", "loss_mask"} for m in training["messages"]))
            self.assertEqual(json.loads(training["messages"][0]["content"][len(dg.CHARACTER_PROMPT) + 1:]),
                             {"profile": restored["character"]["profile"]})
            self.assertEqual(training["messages"][1]["content"], "嗯")

    def test_character_sees_only_text_user_history_keeps_choice(self):
        job = self.job()
        client = ScriptedClient()
        dg.run_job(job, client, lambda: None)
        for role, messages in client.requests:
            self.assertTrue(all(set(m) == {"role", "content"} for m in messages))
            if role == "character":
                context = json.loads(messages[0]["content"][len(dg.CHARACTER_PROMPT) + 1:])
                self.assertEqual(context, {"profile": job["character"]["profile"]})
                for message in messages[1:]:
                    self.assertNotIn("response_length", message["content"])
            elif len(messages) > 2:
                previous = json.loads(messages[1]["content"])
                self.assertEqual(previous, {"message": "嗯", "goal_completed": False, "response_length": "minimal"})
        self.assertNotIn("response_length", job["user_behavior"])

    def test_missing_label_cannot_be_saved_as_a_successful_new_turn(self):
        class MissingLabelClient:
            def call(self, role, messages, validator=None):
                return {"message": "嗯", "goal_completed": False}
        job = self.job()
        with self.assertRaisesRegex(ValueError, "Missing response_length"):
            dg.run_job(job, MissingLabelClient(), lambda: None)
        self.assertEqual(job["messages"], [])
        self.assertEqual(job["phase"], "user")

    def test_legacy_records_keep_missing_labels_and_fixed_behavior_snapshot(self):
        class LegacyClient:
            def call(self, role, messages, validator=None):
                if role == "user":
                    result = {"message": "结束了", "goal_completed": True}
                    validator(result)
                    return result
                return "再见。"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = self.job()
            job.pop("user_response_schema")
            job["user_behavior"] = {"schema_version": "2.0", "id": "sharp_minimal", "tone": "sharp",
                                    "response_length": "minimal"}
            expected = copy.deepcopy(job["user_behavior"])
            dg.save_job(output, job)
            job = dg.read_job(output, job["id"])
            dg.run_job(job, LegacyClient(), lambda: dg.save_job(output, job))
            restored = dg.read_job(output, job["id"])
            self.assertEqual(restored["user_behavior"], expected)
            self.assertNotIn("response_length", restored["messages"][0])
            self.assertIsNone(restored["checks"][0]["length_metrics"]["user_response_length"])
            transcript = dg.read_json(output / f"{job['id']}.json")
            self.assertNotIn("response_length", transcript["messages"][1])

    def test_export_deduplicates_visible_text_despite_different_length_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = self.job()
            dg.run_job(job, ScriptedClient(), lambda: None)
            dg.save_job(output, job)
            job["id"] = "dialogue_00002"
            job["messages"][0]["response_length"] = "long"
            dg.save_job(output, job)
            self.assertEqual(dg.export_jobs(output), 1)

    def test_retired_preset_ids_are_rejected_for_new_plans(self):
        for selection in ("sharp_minimal", "gentle_long", "neutral_short"):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "Unknown user behavior"):
                dg.load_user_behaviors(dg.DEFAULT_USER_BEHAVIORS, selection)

    def test_missing_length_http_response_retries_without_becoming_public(self):
        config = dg.read_json(dg.ROOT / "dialogue_config.json")
        config.update(min_interval_seconds=0, retries=1)
        audit = []
        contents = [{"message": "invalid reply", "goal_completed": False},
                    {"message": "嗯", "goal_completed": False, "response_length": "minimal"}]
        responses = [io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(content)}}], "usage": {"total_tokens": 10}}).encode()) for content in contents]
        with patch.object(dg.ChatClient, "endpoint", return_value=({"json_mode": True}, "https://example.invalid", "mock", "secret")), \
             patch("urllib.request.urlopen", side_effect=responses), patch("time.sleep"):
            result = dg.ChatClient(config, audit.append).call("user", [], dg.validate_user)
        self.assertEqual(result, contents[1])
        self.assertFalse(audit[0]["ok"])
        self.assertIn("Missing response_length", audit[0]["error"])
        self.assertTrue(audit[1]["ok"])
        self.assertEqual(sum(entry["usage"]["total_tokens"] for entry in audit), 20)


if __name__ == "__main__":
    unittest.main()
