"""Offline tests for review selection, isolation, retries and persistence."""

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dialogue_generate as dg
import user_review as ur


def evaluation(grades=None):
    grades = grades or [[3] * 5, [5] * 5, [3] * 5]
    return {"evaluations": [{"candidate_id": f"c{i + 1}", "scores": {
        key: {"score": grade, "reason": "针对当前情境的依据"}
        for key, grade in zip(ur.CRITERIA, values)}} for i, values in enumerate(grades)]}


class Client:
    def __init__(self, failure=None, grades=None):
        self.calls = []
        self.user_count = 0
        self.failure = failure
        self.grades = grades

    def call(self, role, messages, validator=None):
        self.calls.append((role, copy.deepcopy(messages)))
        if self.failure == role:
            self.failure = None
            raise RuntimeError("interrupted")
        if role == "user":
            self.user_count += 1
            number = (self.user_count - 1) % 3 + 1
            result = {"message": f"候选{number}，先聊到这。", "goal_completed": number == 2}
        elif role == "reviewer":
            result = evaluation(self.grades)
        else:
            return "再见。"
        if validator:
            validator(result)
        return result


class ReviewTests(unittest.TestCase):
    def test_complete_output_contract_is_valid_json_for_three_candidates(self):
        contract = ur.output_contract(["c1", "c2", "c3"])
        value = json.loads(contract.split("\n", 1)[1])
        ur.validate_review(value, {"c1", "c2", "c3"})

    def test_json_wrappers_accepted_but_malformed_or_ambiguous_outputs_rejected(self):
        raw = json.dumps(evaluation(), ensure_ascii=False)
        for text in (raw, "```JSON\n" + raw + "\n```", "评分如下：\n```json\n" + raw + "\n```"):
            self.assertEqual(dg.parse_model_json(text), evaluation())
        for text in (raw[:-1], '{"evaluations": ' + raw, raw + raw, "说明\n" + raw + raw,
                     '[{"evaluations": []}] trailing'):
            with self.assertRaises(ValueError):
                dg.parse_model_json(text)

    def test_reviewer_format_retries_do_not_accumulate_bad_answers(self):
        config = dg.read_json(dg.ROOT / "dialogue_config.json")
        config.update(retries=2, min_interval_seconds=0)
        valid = json.dumps(evaluation())
        responses = [io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": content}}]}).encode()) for content in ('{"bad":', '{"bad":', valid)]
        messages = [{"role": "system", "content": "评分规则"}, {"role": "user", "content": "候选"}]
        audit = []
        with patch.object(dg.ChatClient, "endpoint", return_value=({}, "https://example.invalid", "mock", "fake")), \
             patch("urllib.request.urlopen", side_effect=responses) as calls, patch("time.sleep"):
            dg.ChatClient(config, audit.append).call("reviewer", messages,
                lambda x: ur.validate_review(x, {"c1", "c2", "c3"}))
        payloads = [json.loads(call.args[0].data) for call in calls.call_args_list]
        self.assertEqual([len(p["messages"]) for p in payloads], [2, 3, 3])
        self.assertEqual(payloads[2]["messages"][:2], messages)
        self.assertIn("json_diagnostic", audit[0])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        catalog = dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES,
                                  dg.ROOT / "profiles/Character_profile")
        self.job = dg.make_jobs(catalog, 1, 42)[0]
        self.job["user_review_policy"] = ur.make_policy({"enabled": True, "candidate_count": 3, "max_batches": 2})

    def run_job(self, client):
        dg.run_job(self.job, client, lambda: dg.save_job(self.output, self.job))

    def test_selection_history_isolation_examples_and_two_files(self):
        client = Client()
        self.run_job(client)
        self.assertEqual([r for r, _ in client.calls], ["user"] * 3 + ["reviewer", "character"])
        self.assertEqual(client.calls[0][1], client.calls[1][1])
        self.assertEqual(client.calls[1][1], client.calls[2][1])
        data = json.loads(client.calls[3][1][1]["content"])
        self.assertTrue(data["reference_examples"])
        self.assertEqual(data["public_history"], [])
        self.assertNotIn("character", data)
        self.assertNotIn("profile", data["user_context"].get("character", {}))
        character_history = client.calls[4][1][1:]
        self.assertEqual(character_history, [{"role": "user", "content": "候选2，先聊到这。"}])
        self.assertEqual(self.job["stop_reason"], "user_goal_completed")
        transcript = dg.read_json(self.output / f"{self.job['id']}.json")
        self.assertEqual(len(transcript["messages"]), 3)
        self.assertNotIn("user_reviews", transcript)
        self.assertNotIn("user_review_policy", transcript)
        log = dg.read_json(self.output / f"{self.job['id']}.progress.log")
        self.assertNotIn("profile", log["state"]["character"])
        self.assertEqual(log["state"]["user_reviews"][0]["selected_id"], "c2")
        self.assertEqual(len(list(self.output.iterdir())), 2)
        self.assertEqual(dg.export_jobs(self.output), 1)
        exported = (self.output / "training.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("候选1", exported)
        self.assertNotIn("evaluations", exported)

    def test_judge_failure_resume_reuses_candidates_and_policy(self):
        client = Client(failure="reviewer")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_job(client)
        self.assertEqual(self.job["messages"], [])
        self.job = dg.read_job(self.output, self.job["id"])
        with patch.object(ur, "make_policy", side_effect=AssertionError("Do not reload examples")):
            self.run_job(client)
        self.assertEqual(client.user_count, 3)
        self.assertEqual(self.job["rounds"], 1)

    def test_character_failure_does_not_rerun_selection(self):
        client = Client(failure="character")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_job(client)
        self.job = dg.read_job(self.output, self.job["id"])
        self.assertEqual(self.job["messages"][0]["content"], "候选2，先聊到这。")
        self.run_job(client)
        self.assertEqual(client.user_count, 3)
        self.assertEqual(sum(r == "reviewer" for r, _ in client.calls), 1)

    def test_partial_candidate_generation_resumes_at_missing_candidate(self):
        client = Client()
        original = client.call

        def fail_second(role, messages, validator=None):
            if role == "user" and client.user_count == 1:
                raise RuntimeError("interrupted")
            return original(role, messages, validator)

        with patch.object(client, "call", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_job(client)
        self.job = dg.read_job(self.output, self.job["id"])
        self.assertEqual(len(self.job["user_reviews"][0]["batches"][0]["candidates"]), 1)
        self.run_job(client)
        self.assertEqual(client.user_count, 3)

    def test_http_invalid_judge_score_is_retried_before_selection(self):
        config = dg.read_json(dg.ROOT / "dialogue_config.json")
        config.update(retries=1, min_interval_seconds=0)
        invalid = evaluation(); invalid["evaluations"][0]["scores"]["integrity"]["score"] = 4
        responses = [io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(value)}}], "usage": {"total_tokens": 10}}).encode())
                     for value in (invalid, evaluation())]
        audit = []
        with patch.object(dg.ChatClient, "endpoint", return_value=({}, "https://example.invalid", "mock", "fake")), \
             patch("urllib.request.urlopen", side_effect=responses), patch("time.sleep"):
            result = dg.ChatClient(config, audit.append).call("reviewer", [],
                lambda x: ur.validate_review(x, {"c1", "c2", "c3"}))
        self.assertEqual(result, evaluation())
        self.assertEqual([x["ok"] for x in audit], [False, True])
        self.assertEqual(sum(x["usage"]["total_tokens"] for x in audit), 20)

    def test_all_rejected_has_bounded_budget_even_after_resume(self):
        client = Client(grades=[[1, 5, 5, 5, 5]] * 3)
        with self.assertRaisesRegex(RuntimeError, "all candidate batches rejected"):
            self.run_job(client)
        self.assertEqual(client.user_count, 6)
        self.assertEqual(self.job["messages"], [])
        self.job = dg.read_job(self.output, self.job["id"])
        previous = len(client.calls)
        with self.assertRaisesRegex(RuntimeError, "all candidate batches rejected"):
            self.run_job(client)
        self.assertEqual(len(client.calls), previous)

    def test_low_integrity_cannot_win_on_total_and_tie_prefers_no_repetition(self):
        self.job["scene"] = dg.build_scene(self.job)
        grades = [[1, 5, 5, 5, 5], [3, 3, 5, 3, 3], [5, 3, 3, 3, 3]]
        result = ur.select_user(self.job, Client(grades=grades), lambda: None, lambda *a: None,
                                dg.actor_messages(self.job, "user"), {}, dg.validate_user)
        self.assertEqual(result["candidate_id"], "c2")

    def test_validation_rejects_missing_duplicate_invalid_scores_and_rewrites(self):
        ur.validate_review(evaluation(), {"c1", "c2", "c3"})
        for grade in (True, 2, 4, 5.0, "5", None):
            bad = evaluation(); bad["evaluations"][0]["scores"]["integrity"]["score"] = grade
            with self.assertRaises(ValueError):
                ur.validate_review(bad, {"c1", "c2", "c3"})
        for mutate in (lambda x: x["evaluations"].pop(),
                       lambda x: x["evaluations"][0].update(candidate_id="c2"),
                       lambda x: x["evaluations"][0]["scores"].pop("style"),
                       lambda x: x.update(rewritten_message="不要替换回复")):
            bad = evaluation(); mutate(bad)
            with self.assertRaises(ValueError):
                ur.validate_review(bad, {"c1", "c2", "c3"})

    def test_reviewer_endpoint_fallback_and_complete_override(self):
        config = dg.read_json(dg.ROOT / "dialogue_config.json")
        env = {"USER_API_BASE": "https://user.invalid/v1", "USER_MODEL": "user-model", "USER_API_KEY": "fake-u",
               "CHARACTER_API_BASE": "https://char.invalid/v1", "CHARACTER_MODEL": "char-model", "CHARACTER_API_KEY": "fake-c"}
        with patch.dict(os.environ, env, clear=True):
            client = dg.ChatClient(config, lambda x: None)
            self.assertEqual(client.endpoint("reviewer")[1:], client.endpoint("user")[1:])
            self.assertEqual(client.endpoint("reviewer")[0]["temperature"], 0.2)
            with patch.dict(os.environ, {"REVIEWER_MODEL": "partial"}):
                with self.assertRaises(ValueError):
                    client.endpoint("reviewer")
            with patch.dict(os.environ, {"REVIEWER_MODEL": "review", "REVIEWER_API_BASE": "https://review.invalid/v1",
                                        "REVIEWER_API_KEY": "fake-r"}):
                self.assertEqual(client.endpoint("reviewer")[1:], ("https://review.invalid/v1", "review", "fake-r"))

    def test_main_enables_review_and_resume_detects_reviewer_change(self):
        client = Client()
        args = ["dialogue_generate.py", "generate", "--count", "1", "--output", str(self.output)]
        with patch("sys.argv", args), patch("sys.stdout", io.StringIO()), \
             patch.object(dg.ChatClient, "endpoint", return_value=({}, "https://example.invalid", "mock", "fake")), \
             patch.object(dg.ChatClient, "call", side_effect=client.call):
            dg.main()
        restored = dg.read_job(self.output, "dialogue_00001")
        self.assertEqual(restored["user_reviews"][0]["selected_id"], "c2")
        self.assertIn("reviewer", restored["resolved_models"])
        modified = copy.deepcopy(restored["resolved_models"])
        modified["reviewer"]["model"] = "changed"
        with self.assertRaisesRegex(ValueError, "Model/config changed"):
            dg.check_resume_config(restored, restored["generation_config"], modified)


if __name__ == "__main__":
    unittest.main()
