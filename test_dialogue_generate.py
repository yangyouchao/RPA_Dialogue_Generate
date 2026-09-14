"""Offline contract tests; no paid API calls."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

import dialogue_generate as dg


SCENE = {"compatible": True, "reason": "兼容", "public": {
    "background": "一次虚构的初次聊天", "relationship": "初次交流", "trigger": "讨论休息日", "facts": []},
    "user_goal": "PRIVATE_GOAL", "user_private": "USER_SECRET", "character_private": "CHAR_SECRET"}


def fixture():
    return {"id": "test", "character": {"name": "角色", "source": str(dg.ROOT / "profiles/Character_profile/cai_xukun_profile.json"), "profile": {"name": "角色", "fact": "PROFILE_SECRET"}},
            "user": {"id": "U001", "identity": "用户", "habit": "USER_PROFILE_SECRET"}, "topic": {"name": "休息"},
            "user_source": str(dg.ROOT / "profiles/User_profile/generated/User_profile.json"),
            "seed_situation": "讨论休息", "messages": [], "checks": [], "attempts": [],
            "status": "pending", "phase": "scene"}


class FakeClient:
    def __init__(self, goal_round=None, repetition=False, invalid_flag=False, fail_character=False):
        self.goal_round = goal_round
        self.repetition = repetition
        self.invalid_flag = invalid_flag
        self.fail_character = fail_character
        self.user_calls = 0
        self.character_calls = 0

    def call(self, role, messages, validator=None):
        if role not in ("user", "character"):
            raise AssertionError("Only dialogue actors may call an LLM")
        if validator is dg.validate_user:
            self.user_calls += 1
            done = self.user_calls == self.goal_round
            result = {"message": "疑惑已解决，今天就聊到这里" if done else f"第{self.user_calls}个新问题",
                      "goal_completed": done or self.invalid_flag}
        else:
            if self.fail_character:
                self.fail_character = False
                raise RuntimeError("simulated interruption")
            self.character_calls += 1
            return f"第{self.character_calls}次角色回应"
        validator(result)
        return result


class DialogueTests(unittest.TestCase):
    def test_character_goodbye_and_repeated_text_cannot_end_dialogue(self):
        class RepeatingClient(FakeClient):
            def call(self, role, messages, validator=None):
                if role == "user":
                    return {"message": "再聊聊吧", "goal_completed": False}
                if role == "character":
                    return "今天就到这里，再见。"
                raise AssertionError("Unexpected judge call")
        job = fixture()
        dg.run_job(job, RepeatingClient(), lambda: None)
        self.assertEqual(job["rounds"], 8)
        self.assertEqual(job["stop_reasons"], ["max_turns"])
        self.assertNotIn("quality", job)

    def test_resume_retired_phases_finishes_locally_or_continues(self):
        for phase in ("turn_check", "quality"):
            for end in (True, False):
                job = fixture()
                job.update(phase=phase, scene=copy.deepcopy(SCENE), pending_goal=end,
                           messages=[{"speaker": "user", "content": "公开发言"},
                                     {"speaker": "character", "content": "公开回应"}],
                           checks=[{"round": 1, "user_goal_flag": end, "repetition": True}])
                if phase == "quality":
                    job.update(dialogue_completed_at="old-time", stop_reason="repetition")
                client = FakeClient(goal_round=1)
                dg.run_job(job, client, lambda: None)
                self.assertEqual(job["rounds"], 1 if end else 2)
                self.assertEqual(client.user_calls, 0 if end else 1)
                self.assertEqual(len(job["checks"]), job["rounds"])
                self.assertEqual(job["scene"], SCENE)
                if phase == "quality" and end:
                    self.assertEqual(job["dialogue_completed_at"], "old-time")
                elif not end:
                    self.assertNotEqual(job["dialogue_completed_at"], "old-time")

    def test_resume_ignores_retired_director_model(self):
        job = fixture()
        job.update(phase="user", scene=copy.deepcopy(SCENE),
                   resolved_models={"user": "a", "character": "b", "judge": "b", "director": "old"})
        resolved = {"user": "a", "character": "b"}
        dg.check_resume_config(job, {}, resolved)
        self.assertEqual(job["resolved_models"], resolved)
        with self.assertRaisesRegex(ValueError, "Model/config changed"):
            dg.check_resume_config(job, {}, {**resolved, "character": "changed"})

    def test_old_paths_load_catalog_and_resume_after_directory_rename(self):
        users, topics, characters = dg.load_catalog(
            dg.ROOT / "profile/User_profile/open_source/User_profile.json",
            dg.ROOT / "scheme/open_source", dg.ROOT / "profile/Character_profile")
        self.assertEqual(len(users["profiles"]), 50)
        self.assertEqual(len(topics), 40)
        self.assertTrue(characters)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = fixture()
            job["character"]["source"] = str(dg.ROOT / "profile/Character_profile/cai_xukun_profile.json")
            job["user_source"] = str(dg.ROOT / "profile/User_profile/generated/User_profile.json")
            dg.save_job(output, job)
            restored = dg.read_job(output, "test")
            self.assertEqual(Path(restored["character"]["source"]),
                             dg.ROOT / "profiles/Character_profile/cai_xukun_profile.json")
            self.assertEqual(Path(restored["user_source"]),
                             dg.ROOT / "profiles/User_profile/generated/User_profile.json")
            self.assertEqual(restored["user"]["id"], "U001")
            log_path = output / "test.progress.log"
            log = dg.read_json(log_path)
            log["state"]["character"]["version"] = "different-profile"
            dg.write_json(log_path, log)
            with self.assertRaisesRegex(ValueError, "Character source changed"):
                dg.read_job(output, "test")

    def test_provider_429_distinguishes_balance_and_rate_without_exposing_body(self):
        for code, hint in (("1113", "余额"), (1302, "速率"), ("1305", "访问量")):
            body = json.dumps({"error": {"code": code, "message": "secret-value"}}).encode()
            error = dg.urllib.error.HTTPError("https://example.invalid", 429, "limited", {}, io.BytesIO(body))
            result = dg.http_error_summary(error)
            self.assertIn(f"code={code}", result)
            self.assertIn(hint, result)
            self.assertNotIn("secret-value", result)

    def test_append_generates_only_new_samples_after_highest_id(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", io.StringIO()):
            output = Path(directory)
            with patch("sys.argv", ["dialogue_generate.py", "plan", "--count", "1", "--output", directory]):
                dg.main()
            # Incomplete file pairs also reserve their numeric IDs.
            (output / "dialogue_00009.json").write_text("{}", encoding="utf-8")
            (output / "dialogue_00012.progress.log").write_text("{}", encoding="utf-8")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch("sys.argv", ["dialogue_generate.py", "generate", "--append", "--count", "2",
                                     "--character", "meng_ziyi_profile", "--output", directory]), \
                 patch.object(dg.ChatClient, "endpoint", return_value=("secret", "https://example.invalid", "mock")), \
                 patch.object(dg.ChatClient, "call", side_effect=FakeClient(goal_round=1).call):
                dg.main()
            self.assertEqual(before, {name: (output / name).read_bytes() for name in before})
            for identifier in ("dialogue_00013", "dialogue_00014"):
                job = dg.read_job(output, identifier)
                self.assertEqual(job["phase"], "done")
                self.assertTrue(job["messages"])
                self.assertEqual(job["character"]["id"], "meng_ziyi_profile")
            self.assertEqual(len(list(output.iterdir())), len(before) + 4)

    def test_append_plan_empty_directory_and_legacy_numbering(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", io.StringIO()):
            output = Path(directory)
            argv = ["dialogue_generate.py", "plan", "--append", "--count", "1", "--output", directory]
            with patch("sys.argv", argv):
                dg.main()
                self.assertTrue((output / "dialogue_00001.json").exists())
                dg.write_json(output / ".checkpoints/dialogue_100000.json", {})
                dg.main()
            self.assertTrue((output / "dialogue_100001.json").exists())
            with patch("sys.argv", [a for a in argv if a != "--append"]):
                with self.assertRaisesRegex(ValueError, "Batch exists"):
                    dg.main()

    def test_append_rejects_conflicting_or_unsupported_options(self):
        for options in (["generate", "--append", "--resume"], ["export", "--append"],
                        ["render", "--append"], ["validate", "--append"]):
            with patch("sys.argv", ["dialogue_generate.py", *options]), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    dg.main()
                self.assertEqual(error.exception.code, 2)

    def test_local_scene_uses_original_topic_without_profile_influence(self):
        first = fixture()
        second = copy.deepcopy(first)
        second["character"] = {"name": "另一角色", "profile": {"identity": "音乐人", "specialty": "舞蹈"}}
        second["user"] = {"identity": "音乐爱好者"}
        scene = dg.build_scene(first)
        self.assertEqual(scene, dg.build_scene(second))
        self.assertEqual(scene["public"]["background"], first["topic"]["name"])
        self.assertEqual(scene["public"]["trigger"], first["seed_situation"])
        self.assertNotIn("PROFILE_SECRET", json.dumps(scene))

    def test_character_selection_and_unknown_name(self):
        for name in ("meng_ziyi_profile", "meng_ziyi_profile.json"):
            catalog = dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES,
                                      dg.ROOT / "profiles/Character_profile", name)
            jobs = dg.make_jobs(catalog, 3, 123)
            self.assertEqual({job["character"]["id"] for job in jobs}, {"meng_ziyi_profile"})
        with self.assertRaisesRegex(ValueError, "Available:"):
            dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES,
                            dg.ROOT / "profiles/Character_profile", "missing")

    def test_resume_rejects_character_change_without_modifying_files(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", io.StringIO()):
            with patch("sys.argv", ["dialogue_generate.py", "plan", "--character", "meng_ziyi_profile",
                                     "--count", "1", "--output", directory]):
                dg.main()
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            with patch("sys.argv", ["dialogue_generate.py", "generate", "--resume", "--character",
                                     "cai_xukun_profile", "--output", directory]):
                with self.assertRaisesRegex(ValueError, "differs from the saved batch"):
                    dg.main()
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})

    def test_open_source_catalog_and_topic_provenance(self):
        import extract_open_source as extraction
        users, topics, chars = dg.load_catalog(
            dg.ROOT / "profiles/User_profile/open_source/User_profile.json",
            dg.ROOT / "schema/open_source", dg.ROOT / "profiles/Character_profile")
        self.assertEqual(len(users["profiles"]), 50)
        self.assertEqual([p["source_row_index"] for p in users["profiles"]], list(range(50)))
        self.assertEqual(len(topics), 40)
        services = {s["service_name"]: s for s in dg.read_json(dg.ROOT / "schema.json")}
        for topic, selected in zip(topics, extraction.SELECTION):
            self.assertEqual(topic["service_name"], selected[0])
            intent = next(i for i in services[selected[0]]["intents"] if i["name"] == selected[1])
            self.assertNotIn("original_intent", topic)
            self.assertEqual(topic["intent_name"], intent["name"])
            self.assertEqual(topic["description_en"], intent["description"])
        self.assertEqual(len(dg.make_jobs((users, topics, chars), 2, 123)), 2)
        merged = dg.read_json(dg.ROOT / "schema/open_source/topic_schema_20_zh.json")
        daily = dg.read_json(dg.ROOT / "tmp/dailydialog_20/output/topic_dailydialog_20_zh.json")
        self.assertEqual(merged["subtopics"][20:], daily["subtopics"])
        if "dailydialog_source" in merged:
            self.assertEqual(merged["dailydialog_source"],
                             {key: value for key, value in daily.items() if key != "subtopics"})
        self.assertEqual(len({item["id"] for item in merged["subtopics"]}), 40)
        for item in merged["subtopics"]:
            self.assertEqual(list(item), list(merged["subtopics"][0]))

    def test_dailydialog_matches_service_topic_fields(self):
        import runpy
        builder = runpy.run_path(str(dg.ROOT / "tmp/dailydialog_20/build.py"))
        expected = dg.read_json(dg.ROOT / "schema/open_source/topic_schema_20_zh.json")["subtopics"][0]
        with tempfile.TemporaryDirectory() as directory:
            builder["main"].__globals__["OUT"] = Path(directory)
            builder["main"]()
            document = dg.read_json(Path(directory) / "topic_dailydialog_20_zh.json")
            self.assertEqual(len(document["subtopics"]), 20)
            for item in document["subtopics"]:
                self.assertEqual(list(item), list(expected))
                self.assertEqual({key: type(value) for key, value in item.items()},
                                 {key: type(value) for key, value in expected.items()})
                self.assertEqual(item["service_name"], "")
                self.assertEqual(item["intent_name"], "")
                self.assertNotIn("original_intent", item)
                self.assertIn(item["id"], document["source_records"])
            dg.load_catalog(dg.DEFAULT_USERS, directory, dg.ROOT / "profiles/Character_profile")

    def test_topic_without_retired_tags_loads_but_invalid_seeds_fail(self):
        topic = {"id": "chat", "name": "闲聊", "mode": "兴趣交流",
                 "seeds": ["聊聊兴趣"], "boundary": "不编造共同经历"}
        document = {"schema_version": "1.0", "id": "test", "name": "测试主题",
                    "version": "1", "source": "test fixture", "subtopics": [topic]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topic_test.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            catalog = dg.load_catalog(dg.DEFAULT_USERS, directory, dg.ROOT / "profiles/Character_profile")
            self.assertEqual(dg.make_jobs(catalog, 1, 123)[0]["seed_situation"], "聊聊兴趣")
            for value in (None, "聊聊兴趣", [], [""], [1]):
                with self.subTest(seeds=value):
                    topic["seeds"] = value
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "seeds"):
                        dg.load_catalog(dg.DEFAULT_USERS, directory, dg.ROOT / "profiles/Character_profile")

    def test_legacy_user_path_resolves_after_reorganization(self):
        self.assertEqual(dg.resolve_users_path(dg.ROOT / "User_profile.json"),
                         dg.ROOT / "profiles/User_profile/generated/User_profile.json")

    def test_html_response_explains_wrong_endpoint(self):
        with self.assertRaisesRegex(dg.ResponseFormatError, "HTML.*base path"):
            dg.parse_api_response(b"<!doctype html><html>Gateway</html>")

    def test_unsupported_model_error_does_not_expose_server_body(self):
        error = dg.urllib.error.HTTPError("https://example.invalid", 404, "missing", {},
            io.BytesIO(b'{"error":{"message":"Model is not supported secret-value"}}'))
        result = dg.http_error_summary(error)
        self.assertIn("model is unavailable", result)
        self.assertNotIn("secret-value", result)

    def test_empty_failed_sample_can_change_model_but_started_sample_cannot(self):
        job = fixture()
        job.update(config_digest="old", resolved_models={"user": "old"}, status="error")
        dg.check_resume_config(job, {"new": True}, {"user": "new"})
        self.assertEqual(job["resolved_models"], {"user": "new"})
        self.assertEqual(len(job["configuration_history"]), 1)
        job["phase"] = "scene_review"
        job["scene"] = SCENE
        with self.assertRaises(ValueError):
            dg.check_resume_config(job, {"new": True}, {"user": "another"})

    def test_client_restriction_with_appended_sse_is_reported(self):
        body = (b'{"error":{"message":"This account only allows Codex official clients"}}'
                b'event: error\ndata: {"error":{"message":"Upstream request failed secret-value"}}')
        error = dg.urllib.error.HTTPError("https://example.invalid", 403, "forbidden", {}, io.BytesIO(body))
        result = dg.http_error_summary(error)
        self.assertIn("only allows Codex official clients", result)
        self.assertNotIn("secret-value", result)

    def test_html_forbidden_response_uses_safe_fallback(self):
        error = dg.urllib.error.HTTPError("https://example.invalid", 403, "forbidden", {},
                                         io.BytesIO(b'<html>secret-value</html>'))
        self.assertEqual(dg.http_error_summary(error), "HTTP 403: access denied; check model permissions")

    def test_env_loads_literal_values_without_overriding_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('USER_API_BASE=https://example.invalid/v1\nUSER_MODEL=file-model\n'
                            'USER_API_KEY="key#${LITERAL}"\n', encoding="utf-8")
            with patch.dict(os.environ, {"USER_MODEL": "process-model"}, clear=True):
                dg.load_api_env(path)
                self.assertEqual(os.environ["USER_API_BASE"], "https://example.invalid/v1")
                self.assertEqual(os.environ["USER_MODEL"], "process-model")
                self.assertEqual(os.environ["USER_API_KEY"], "key#${LITERAL}")

    def test_missing_env_keeps_process_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"USER_MODEL": "process-model"}, clear=True):
                dg.load_api_env(Path(directory) / "missing.env")
                self.assertEqual(os.environ["USER_MODEL"], "process-model")

    def test_maximum_eight_complete_rounds(self):
        job, client = fixture(), FakeClient()
        dg.run_job(job, client, lambda: None)
        self.assertEqual(len(job["messages"]), 16)
        self.assertEqual(job["stop_reason"], "max_turns")
        self.assertEqual(client.user_calls, 8)

    def test_goal_completion_has_one_character_farewell(self):
        job, client = fixture(), FakeClient(goal_round=3)
        dg.run_job(job, client, lambda: None)
        self.assertEqual(job["stop_reason"], "user_goal_completed")
        self.assertEqual(client.user_calls, 3)
        self.assertEqual(client.character_calls, 3)

    def test_repetition_does_not_stop_dialogue(self):
        job = fixture()
        dg.run_job(job, FakeClient(repetition=True), lambda: None)
        self.assertEqual(len(job["messages"]), 16)
        self.assertEqual(job["stop_reason"], "max_turns")
        self.assertEqual(job["status"], "completed")

    def test_goal_flag_stops_without_turn_judge_veto(self):
        job = fixture()
        dg.run_job(job, FakeClient(invalid_flag=True), lambda: None)
        self.assertEqual(job["stop_reason"], "user_goal_completed")
        self.assertEqual(len(job["messages"]), 2)
        self.assertEqual(job["checks"][0], {"round": 1, "user_goal_flag": True})

    def test_priority_keeps_all_reasons(self):
        job = fixture()
        job["pending_goal"] = True
        job["messages"] = [{"speaker": "user", "content": "相同"},
                           {"speaker": "character", "content": "回应"}] * 8
        self.assertEqual(dg.stop_reasons(job),
                         ["user_goal_completed", "max_turns"])

    def test_checkpoint_resumes_character_without_duplicate_user(self):
        job, client = fixture(), FakeClient(goal_round=1, fail_character=True)
        saved = []
        with self.assertRaises(RuntimeError):
            dg.run_job(job, client, lambda: saved.append(copy.deepcopy(job)))
        restored = saved[-1]
        self.assertEqual(restored["phase"], "character")
        dg.run_job(restored, client, lambda: None)
        self.assertEqual(client.user_calls, 1)
        self.assertEqual(len(restored["messages"]), 2)

    def test_actor_history_and_information_isolation(self):
        job = fixture()
        job["scene"] = SCENE
        job["messages"] = [{"speaker": "user", "content": "公开提问"},
                           {"speaker": "character", "content": "公开回应"}]
        user = dg.actor_messages(job, "user")
        char = dg.actor_messages(job, "character")
        self.assertEqual([x["role"] for x in user], ["system", "assistant", "user"])
        self.assertEqual([x["role"] for x in char], ["system", "user", "assistant"])
        self.assertEqual(json.loads(user[1]["content"]), {"message": "公开提问", "goal_completed": False})
        self.assertEqual(char[1]["content"], "公开提问")
        for secret in ("CHAR_SECRET", "PROFILE_SECRET"):
            # USER_PROFILE_SECRET is intentionally visible to User.
            if secret == "PROFILE_SECRET":
                self.assertNotIn('"fact"', user[0]["content"])
            else:
                self.assertNotIn(secret, user[0]["content"])
        for secret in ("USER_SECRET", "USER_PROFILE_SECRET", "PRIVATE_GOAL"):
            self.assertNotIn(secret, char[0]["content"])

    def test_catalog_size_sampling_and_no_repeated_combinations(self):
        catalog = dg.load_catalog(dg.ROOT / "profiles/User_profile/generated/User_profile.json",
                                  dg.ROOT / "schema/generated",
                                  dg.ROOT / "profiles/Character_profile")
        self.assertEqual(len(catalog[0]["profiles"]), 50)
        self.assertEqual(len(catalog[1]), 96)
        self.assertEqual(len({x["group_id"] for x in catalog[1]}), 12)
        jobs = dg.make_jobs(catalog, 50, 12)
        self.assertEqual(jobs, dg.make_jobs(catalog, 50, 12))
        self.assertEqual(len({(j["character"]["id"], j["topic"]["id"], j["user"]["id"]) for j in jobs}), 50)

    def test_random_pairing_ignores_profile_matching_tags(self):
        users = {"version": "1", "profiles": [{"id": "u", "tags": ["adult"]}]}
        topic = {"id": "t", "required_user_tags": ["student"], "excluded_character_tags": [], "seeds": ["a"]}
        jobs = dg.make_jobs((users, [topic], [{"id": "c", "tags": []}]), 1, 1)
        self.assertEqual(jobs[0]["user"]["id"], "u")
        self.assertEqual(jobs[0]["topic"]["id"], "t")
        altered = copy.deepcopy(topic)
        altered.update(required_user_tags=[], soft_tags=["adult"])
        self.assertEqual(jobs[0]["seed_situation"],
                         dg.make_jobs((users, [altered], [{"id": "c", "tags": []}]), 1, 1)[0]["seed_situation"])

    def test_export_masks_and_excludes_review_samples_and_duplicate_text(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = fixture()
            dg.run_job(job, FakeClient(goal_round=1), lambda: None)
            dg.write_json(output / "manifest.json", {"jobs": ["a", "b", "c"]})
            dg.write_json(output / "a.json", job)
            dg.write_json(output / "b.json", job)
            dg.write_json(output / "c.json", {**job, "status": "needs_review"})
            self.assertEqual(dg.export_jobs(output), 1)
            record = json.loads((output / "training.jsonl").read_text(encoding="utf-8"))
            self.assertEqual([m["loss_mask"] for m in record["messages"]], [0, 0, 1])
            self.assertNotIn("USER_SECRET", json.dumps(record))
            self.assertNotIn("PRIVATE_GOAL", json.dumps(record))

    def test_invalid_json_response_retries_and_counts_usage(self):
        config = dg.read_json(dg.ROOT / "dialogue_config.json")
        config.update(min_interval_seconds=0, retries=1)
        audit = []
        env = {"USER_API_BASE": "https://example.invalid/v1", "USER_MODEL": "model-a", "USER_API_KEY": "secret",
               "CHARACTER_API_BASE": "https://example.invalid/v1", "CHARACTER_MODEL": "model-b", "CHARACTER_API_KEY": "secret"}
        def response(content):
            return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                         "usage": {"total_tokens": 10}}).encode())
        replies = [response("not json"), response('{"message":"hi","goal_completed":false}')]
        with patch.dict("os.environ", env), patch("urllib.request.urlopen", side_effect=replies) as http, patch("time.sleep"):
            result = dg.ChatClient(config, audit.append).call("user", [], dg.validate_user)
        second_request = json.loads(http.call_args_list[1].args[0].data)
        self.assertEqual(second_request["messages"][-2]["content"], "not json")
        self.assertIn("格式纠正", second_request["messages"][-1]["content"])
        self.assertIn("not valid JSON", audit[0]["error"])
        self.assertEqual(audit[0]["finish_reason"], "stop")
        self.assertEqual(result["message"], "hi")
        self.assertEqual(len(audit), 2)
        self.assertEqual(sum(x["usage"]["total_tokens"] for x in audit), 20)
        self.assertNotIn("secret", json.dumps(audit))

    def test_schema_rejects_string_booleans(self):
        with self.assertRaises(ValueError):
            dg.validate_user({"message": "hello", "goal_completed": "false"})

    def test_cli_plan_generate_resume_export_with_mock_http(self):
        content = [{"message": "疑惑已经解决，今天就聊到这里。", "goal_completed": True},
                   "好，今天就聊到这里。"]
        responses = [io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": item if isinstance(item, str) else json.dumps(item)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode()) for item in content]
        env = {"USER_API_BASE": "https://example.invalid/v1", "USER_MODEL": "model-a", "USER_API_KEY": "secret",
               "CHARACTER_API_BASE": "https://example.invalid/v1", "CHARACTER_MODEL": "model-b", "CHARACTER_API_KEY": "secret"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("sys.argv", ["dialogue_generate.py", "plan", "--count", "1", "--output", directory]), patch("sys.stdout", io.StringIO()):
                dg.main()
            with patch.dict("os.environ", env), patch("urllib.request.urlopen", side_effect=responses) as http, patch("time.sleep"), patch("sys.stdout", io.StringIO()):
                with patch("sys.argv", ["dialogue_generate.py", "generate", "--resume", "--output", directory]):
                    dg.main()
                    dg.main()
                self.assertEqual(http.call_count, 2)
            self.assertEqual(sorted(p.name for p in output.iterdir()),
                             ["dialogue_00001.json", "dialogue_00001.progress.log"])
            self.assertEqual(sum(a["usage"]["total_tokens"] for a in dg.read_job(output, "dialogue_00001")["attempts"]), 30)
            self.assertFalse((output / ".generation.lock").exists())
            self.assertEqual(dg.read_job(output, "dialogue_00001")["rounds"], 1)
            transcript = dg.read_json(output / "dialogue_00001.json")
            self.assertEqual(set(transcript), {"messages", "scene", "user_goal", "completed_at"})
            self.assertEqual([m["role"] for m in transcript["messages"]], ["system", "user", "assistant"])
            self.assertTrue(transcript["completed_at"])

    def test_completed_transcript_and_progress_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = fixture()
            log = dg.ProgressLog(output, job)
            with patch("sys.stdout", io.StringIO()):
                dg.run_job(job, FakeClient(goal_round=1), lambda: dg.save_job(output, job), log.emit)
            transcript = dg.read_json(output / "test.json")
            self.assertEqual(len(transcript["messages"]), 3)
            completed_at = transcript["completed_at"]
            dg.save_job(output, job)
            self.assertEqual(dg.read_json(output / "test.json")["completed_at"], completed_at)
            progress = (output / "test.progress.log").read_text(encoding="utf-8")
            for secret in ("PROFILE_SECRET", "USER_PROFILE_SECRET"):
                self.assertNotIn(secret, progress)
            self.assertIn("PROFILE_SECRET", transcript["messages"][0]["content"])
            self.assertIn("轮次完成", progress)
            self.assertRegex(progress, r"\[\d{4}-\d{2}-\d{2}T")
            self.assertEqual(dg.read_job(output, "test")["phase"], "done")

    def test_partial_round_stays_in_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = fixture()
            job["messages"] = [{"speaker": "user", "content": "未完成轮次"}]
            dg.save_job(output, job)
            self.assertEqual(len(dg.read_json(output / "test.json")["messages"]), 1)
            self.assertEqual(dg.read_json(output / "test.json")["messages"][0]["role"], "system")
            self.assertNotIn("completed_at", dg.read_json(output / "test.json"))
            self.assertEqual(len(dg.read_job(output, "test")["messages"]), 1)

    def test_legacy_record_migration_preserves_full_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = fixture()
            dg.write_json(output / "test.json", job)
            dg.save_job(output, dg.read_job(output, "test"))
            self.assertEqual(dg.read_job(output, "test")["messages"], job["messages"])
            self.assertEqual(dg.read_job(output, "test")["phase"], job["phase"])
            self.assertNotIn("character", dg.read_json(output / "test.json"))

    def test_review_profile_does_not_enter_model_inputs(self):
        job = fixture()
        job["scene"] = copy.deepcopy(SCENE)
        job["messages"] = [{"speaker": "user", "content": "公开问题"},
                           {"speaker": "character", "content": "公开回答"}]
        original = copy.deepcopy(job)
        before_user = dg.actor_messages(job, "user")
        before_character = dg.actor_messages(job, "character")
        transcript = dg.review_transcript(job)
        self.assertEqual(job, original)
        self.assertEqual(dg.actor_messages(job, "user"), before_user)
        self.assertEqual(dg.actor_messages(job, "character"), before_character)
        self.assertNotIn("人工评分参考", json.dumps(before_character, ensure_ascii=False))
        self.assertEqual(transcript["messages"][1:], [
            {"role": "user", "content": "公开问题"},
            {"role": "assistant", "content": "公开回答"}])
        self.assertTrue(all(set(m) == {"role", "content"} for m in transcript["messages"]))
        self.assertTrue(all(isinstance(m["content"], str) for m in transcript["messages"]))

    def test_previously_incompatible_sample_can_generate(self):
        job, client = fixture(), FakeClient(goal_round=1)
        job.update(status="incompatible", phase="done", rejection="时代不兼容",
                   scene={"compatible": False, "reason": "时代不兼容"})
        dg.run_job(job, client, lambda: None)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(client.user_calls, 1)
        self.assertNotIn("compatible", job["scene"])
        self.assertEqual(job["previous_scene_rejection"], "时代不兼容")

    def test_character_topic_exclusion_tags_do_not_block_sampling(self):
        users = {"version": "1", "profiles": [{"id": "u", "tags": []}]}
        topic = {"id": "t", "required_user_tags": [], "excluded_character_tags": ["historic"],
                 "soft_tags": [], "seeds": ["现代日历应用"]}
        jobs = dg.make_jobs((users, [topic], [{"id": "c", "tags": ["historic"]}]), 1, 1)
        self.assertEqual(len(jobs), 1)


if __name__ == "__main__":
    unittest.main()
