"""Offline evaluation contracts: native format, history boundaries, resume."""

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dialogue_generate as dg
import evaluate_dialogues as ev


def transcript(mode="task"):
    return {
        "messages": [
            {"role": "system", "content": "# 人工评分参考：角色 profile\n\n" + dg.dumps({"name": "角色", "style": "朴素"})},
            {"role": "user", "content": "FIRST_USER"},
            {"role": "assistant", "content": "FIRST_TEACHER"},
            {"role": "user", "content": "FUTURE_USER"},
            {"role": "assistant", "content": "FUTURE_TEACHER"},
        ],
        "scene": {"conversation_mode": mode, "background": None},
        "conversation_start": {"mode": mode, "goal": "PRIVATE_GOAL"},
        "user_goal": "PRIVATE_GOAL", "user_behavior": {"tone": "PRIVATE_TONE"},
    }


class FakeClient:
    calls = []
    fail_once = False

    def __init__(self, config, audit):
        self.audit = audit

    def endpoint(self, role):
        return {}, "https://judge.example/v1", "offline-judge", "secret"

    def call(self, role, messages, validator):
        self.calls.append(messages)
        if self.fail_once:
            type(self).fail_once = False
            raise RuntimeError("simulated failure")
        principle = next(name for d in ev.dimensions() for name in d["principles"]
                         if f"1. {name}：" in messages[1]["content"])
        value = {principle: {"score": 3, "reason": "结合本轮内容的模拟判定"}}
        validator(value)
        self.audit({"ok": True, "usage": {"total_tokens": 10}})
        return value


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.path = self.source / "dialogue_00001.json"
        dg.write_json(self.path, transcript())
        FakeClient.calls = []
        FakeClient.fail_once = False

    def args(self, *extra):
        return ev.parse_args(["--input", str(self.source), "--output", str(self.root / "eval"), *extra])

    def run_eval(self, *extra):
        with patch.object(ev.dg, "ChatClient", FakeClient), patch.object(ev.dg, "load_api_env"), \
                patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            return ev.run(self.args(*extra))

    def test_native_format_modes_history_and_no_private_fields(self):
        for mode in ev.MODE_NAMES:
            dg.write_json(self.path, transcript(mode))
            original = self.path.read_bytes()
            samples, _ = ev.load_samples(self.source)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["conversation_mode"], mode)
            messages = ev.build_messages(samples[0], ev.dimensions()[0], ev.load_prompts())
            prompt = dg.dumps(messages)
            self.assertIn("FIRST_USER", prompt)
            self.assertIn("FIRST_TEACHER", prompt)
            for absent in ("FUTURE_USER", "FUTURE_TEACHER", "PRIVATE_GOAL", "PRIVATE_TONE", "参考回复", "小模型"):
                self.assertNotIn(absent, prompt)
            self.assertEqual(samples[1]["history"][-1]["content"], "FUTURE_USER")
            self.assertEqual(self.path.read_bytes(), original)

    def test_skip_unfinished_and_filter_mode(self):
        dg.write_json(self.path.with_suffix(".progress.log"), {"state": {"status": "pending"}})
        dg.write_json(self.source / "dialogue_00002.json", transcript("open_chat"))
        samples, skipped = ev.load_samples(self.source, ["open_chat"], 1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(samples[0]["dialogue_id"], "dialogue_00002")

    def test_invalid_order_and_partial_turn_rejected(self):
        data = transcript()
        data["messages"].pop()
        dg.write_json(self.path, data)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            ev.load_samples(self.source)
        data = transcript()
        data["messages"][2]["role"] = "user"
        dg.write_json(self.path, data)
        with self.assertRaisesRegex(ValueError, "order"):
            ev.load_samples(self.source)

    def test_strict_scores(self):
        dimension = ev.dimensions()[0]
        name = next(iter(dimension["principles"]))
        for score in (True, 2, 3.0, "5", None):
            with self.assertRaises(ValueError):
                ev.validate_scores({name: {"score": score, "reason": "reason"}}, dimension)
        for reason in (None, 12, "", " "):
            with self.assertRaises(ValueError):
                ev.validate_scores({name: {"score": 5, "reason": reason}}, dimension)
        with self.assertRaises(ValueError):
            ev.validate_scores({name: {"score": 5, "reason": "reason", "extra": 1}}, dimension)
        with self.assertRaises(ValueError):
            ev.validate_scores({"wrong": {"score": 5, "reason": "reason"}}, dimension)

    def test_check_no_network_env_or_writes(self):
        with patch.object(ev.dg, "ChatClient") as client, patch.object(ev.dg, "load_api_env") as env, \
                patch("sys.stdout", io.StringIO()):
            self.assertEqual(ev.run(self.args("--check")), 0)
        client.assert_not_called()
        env.assert_not_called()
        self.assertFalse((self.root / "eval").exists())

    def test_resume_retries_only_failed_tasks_and_preserves_dialogue_fields(self):
        original = dg.read_json(self.path)
        FakeClient.fail_once = True
        self.assertEqual(self.run_eval(), 1)
        summary = dg.read_json(self.root / "eval/summary.json")
        self.assertEqual(summary["completed_tasks"], 9)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["fully_scored_samples"], 1)
        self.assertEqual(self.run_eval("--resume"), 0)
        self.assertEqual(len(FakeClient.calls), 11)
        self.assertEqual(self.run_eval("--resume"), 0)
        self.assertEqual(len(FakeClient.calls), 11)
        summary = dg.read_json(self.root / "eval/summary.json")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["fully_scored_samples"], 2)
        self.assertEqual(summary["by_mode"]["task"]["性格特质"]["mean"], 3)
        data = dg.read_json(self.path)
        self.assertEqual(list(data)[-1], "average_score")
        self.assertEqual(data.pop("average_score"), 3)
        scores = data.pop("scores")
        self.assertEqual(data, original)
        self.assertEqual(len(scores), 2)
        for items in scores.values():
            self.assertEqual(len(items), 5)
            for item in items:
                self.assertEqual(set(item), {"dimension", "score", "reason"})
        self.assertFalse((self.root / "eval/scores.json").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_eval()
        data = transcript()
        data["messages"][2]["content"] = "CHANGED_RESPONSE"
        dg.write_json(self.path, data)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_eval("--resume")

    def test_resume_rejects_changed_rubric_and_judge(self):
        self.run_eval()
        revised = copy.deepcopy(ev.dimensions())
        revised[0]["intro"] = "changed criterion"
        with patch.object(ev, "dimensions", return_value=revised):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.run_eval("--resume")
        with patch.object(FakeClient, "endpoint", return_value=({}, "https://judge.example/v1", "new-model", "secret")):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.run_eval("--resume")

    def test_dialogue_macro_average(self):
        rows = [{"principle": "话题承接", "dialogue_id": "a", "score": 1},
                {"principle": "话题承接", "dialogue_id": "a", "score": 1},
                {"principle": "话题承接", "dialogue_id": "b", "score": 5}]
        result = ev.aggregate(rows)["话题承接"]
        self.assertAlmostEqual(result["mean"], 7 / 3)
        self.assertEqual(result["dialogue_macro_mean"], 3)

    def test_real_http_client_uses_only_judge_and_retries_invalid_score(self):
        requests = []

        def urlopen(request, timeout):
            payload = json.loads(request.data)
            requests.append(payload)
            self.assertEqual(request.full_url, "https://judge.example/v1/chat/completions")
            self.assertEqual(payload["model"], "judge-only")
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            principle = next(name for d in ev.dimensions() for name in d["principles"]
                             if f"1. {name}：" in payload["messages"][1]["content"])
            score = 2 if len(requests) == 1 else 5
            body = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
                {principle: {"score": score, "reason": "有效的模拟评分"}}, ensure_ascii=False)}}],
                "usage": {"total_tokens": 25}}
            return io.BytesIO(json.dumps(body).encode())

        config = dg.read_json(ev.ROOT / "evaluation_config.json")
        config["min_interval_seconds"] = 0
        config_path = self.root / "config.json"
        dg.write_json(config_path, config)
        env = {"EVAL_API_BASE": "https://judge.example/v1", "EVAL_MODEL": "judge-only", "EVAL_API_KEY": "secret"}
        with patch.dict(os.environ, env, clear=True), patch.object(ev.dg, "load_api_env"), \
                patch.object(ev.dg.urllib.request, "urlopen", side_effect=urlopen), \
                patch.object(ev.dg.time, "sleep"), patch("sys.stdout", io.StringIO()):
            self.assertEqual(ev.run(self.args("--limit", "1", "--config", str(config_path))), 0)
        self.assertEqual(len(requests), 6)
        scores = dg.read_json(self.path)["scores"]["round_01"]
        self.assertEqual(len(scores), 5)
        self.assertEqual(scores[0]["score"], 5)
        self.assertEqual(dg.read_json(self.path)["average_score"], 5)
        self.assertEqual(set(scores[0]), {"dimension", "score", "reason"})
        self.assertNotIn("secret", self.path.read_text(encoding="utf-8"))

    def test_inline_scores_do_not_change_prompts_or_samples(self):
        before, _ = ev.load_samples(self.source)
        prompts = [ev.build_messages(s, ev.dimensions()[0], ev.load_prompts()) for s in before]
        self.run_eval()
        after, _ = ev.load_samples(self.source)
        self.assertEqual(before, after)
        self.assertEqual(prompts, [ev.build_messages(s, ev.dimensions()[0], ev.load_prompts()) for s in after])

    def legacy_scores(self):
        self.run_eval()
        samples, _ = ev.load_samples(self.source)
        lookup = {s["sample_id"]: s for s in samples}
        rows = ev.read_saved_scores(samples, ev.dimensions())
        for row in rows:
            row["review_context"] = lookup[row["sample_id"]]
        data = dg.read_json(self.path)
        data.pop("scores")
        data.pop("average_score", None)
        dg.write_json(self.path, data)
        target = self.root / "eval/scores.json"
        dg.write_json(target, rows)
        return target

    def test_offline_legacy_migration_and_idempotence(self):
        target = self.legacy_scores()
        original = target.read_bytes()
        with patch.object(ev.dg, "ChatClient") as client, patch.object(ev.dg, "load_api_env") as env, \
                patch("sys.stdout", io.StringIO()):
            self.assertEqual(ev.run(self.args("--migrate-scores")), 0)
            self.assertEqual(ev.run(self.args("--migrate-scores")), 0)
        client.assert_not_called()
        env.assert_not_called()
        self.assertFalse(target.exists())
        self.assertEqual((self.root / "eval/scores.legacy.json").read_bytes(), original)
        self.assertEqual(len(dg.read_json(self.path)["scores"]["round_01"]), 5)
        self.assertEqual(len(dg.read_json(self.path)["scores"]["round_02"]), 5)
        FakeClient.calls = []
        self.assertEqual(self.run_eval("--resume"), 0)
        self.assertEqual(FakeClient.calls, [])

    def test_migration_validates_all_contexts_before_writing(self):
        target = self.legacy_scores()
        rows = dg.read_json(target)
        rows[-1]["review_context"]["model_response"] = "WRONG_CONTEXT"
        dg.write_json(target, rows)
        original = self.path.read_bytes()
        with patch("sys.stdout", io.StringIO()):
            with self.assertRaisesRegex(ValueError, "context"):
                ev.run(self.args("--migrate-scores"))
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(target.exists())

    def test_save_or_render_retains_scores_only_for_unchanged_context(self):
        job = {"id": "dialogue_00001", "character": {"name": "角色", "profile": {"name": "角色"}},
               "user": {"id": "user"}, "messages": [{"speaker": "user", "content": "你好"},
                                                          {"speaker": "character", "content": "你好呀"}],
               "conversation_start": {"mode": "open_chat", "topic": None, "goal": None},
               "scene": {"public": {"conversation_mode": "open_chat", "background": None}, "user_goal": None},
               "status": "completed"}
        dg.save_job(self.source, job)
        data = dg.read_json(self.path)
        data["scores"] = {"round_01": [{"dimension": "性格特质", "score": 3, "reason": "理由"}]}
        data["average_score"] = 3
        dg.write_json(self.path, data)
        dg.save_job(self.source, job)
        self.assertEqual(dg.read_json(self.path)["scores"], data["scores"])
        self.assertEqual(dg.read_json(self.path)["average_score"], 3)
        job["messages"][1]["content"] = "changed response"
        dg.save_job(self.source, job)
        self.assertNotIn("scores", dg.read_json(self.path))
        self.assertNotIn("average_score", dg.read_json(self.path))

    def test_dialogue_average_weights_each_valid_score_equally(self):
        samples, _ = ev.load_samples(self.source)
        ev.save_score(samples[0], ev.dimensions()[0], {"score": 1, "reason": "低分"})
        ev.save_score(samples[1], ev.dimensions()[0], {"score": 5, "reason": "高分"})
        ev.save_score(samples[1], ev.dimensions()[1], {"score": 5, "reason": "高分"})
        self.assertEqual(dg.read_json(self.path)["average_score"], 3.6667)
        rows = ev.read_saved_scores(samples, ev.dimensions())
        summary = ev.summarize(samples, rows, 10, [], "test")
        result = summary["by_dialogue"]["dialogue_00001"]
        self.assertEqual(result, {"average_score": 3.6667, "completed_tasks": 3,
                                  "expected_tasks": 10, "complete": False})

    def test_offline_average_backfill_includes_unscored_dialogues(self):
        dg.write_json(self.source / "dialogue_00002.json", transcript("open_chat"))
        self.run_eval()
        first = dg.read_json(self.path)
        first.pop("average_score")
        dg.write_json(self.path, first)
        second_path = self.source / "dialogue_00002.json"
        second = dg.read_json(second_path)
        second.pop("scores")
        second.pop("average_score")
        dg.write_json(second_path, second)
        with patch.object(ev.dg, "ChatClient") as client, patch.object(ev.dg, "load_api_env") as env, \
                patch("sys.stdout", io.StringIO()):
            self.assertEqual(ev.run(self.args("--summarize-only")), 0)
            self.assertEqual(ev.run(self.args("--summarize-only")), 0)
        client.assert_not_called()
        env.assert_not_called()
        self.assertEqual(dg.read_json(self.path)["average_score"], 3)
        self.assertEqual(dg.read_json(self.path)["scores"], first["scores"])
        self.assertIsNone(dg.read_json(second_path)["average_score"])
        summary = dg.read_json(self.root / "eval/summary.json")
        self.assertTrue(summary["by_dialogue"]["dialogue_00001"]["complete"])
        self.assertEqual(summary["by_dialogue"]["dialogue_00002"]["completed_tasks"], 0)
        self.assertIsNone(summary["by_dialogue"]["dialogue_00002"]["average_score"])

    def test_output_cannot_overwrite_input(self):
        for output in (self.source, self.root):
            args = self.args("--output", str(output))
            with self.assertRaisesRegex(ValueError, "input directory"):
                ev.run(args)


if __name__ == "__main__":
    unittest.main()
