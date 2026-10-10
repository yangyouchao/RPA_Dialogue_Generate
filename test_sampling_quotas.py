"""Offline quota allocation and batch persistence tests."""

from collections import Counter
import copy
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import dialogue_generate as dg


SETTINGS = {
    "schema_version": "2.0",
    "conversation_mode_distribution": {"task": 0.4, "topic_chat": 0.3, "open_chat": 0.3},
    "length_condition_distribution": dict.fromkeys(dg.LENGTH_CONDITIONS, 0.25),
}


class QuotaTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "sampling_config.json"
        dg.write_json(self.config, SETTINGS)
        override = patch.object(dg, "SAMPLING_CONFIG", self.config)
        override.start()
        self.addCleanup(override.stop)
        self.presets = [{"id": condition,
                         "length_condition": condition, "content_mode": "natural"}
                        for condition in dg.LENGTH_CONDITIONS]
        behavior_path = self.root / "all_behaviors.json"
        dg.write_json(behavior_path, {"schema_version": "3.0", "presets": self.presets})
        behavior_override = patch.object(dg, "DEFAULT_USER_BEHAVIORS", behavior_path)
        behavior_override.start()
        self.addCleanup(behavior_override.stop)

    def catalog(self, mode="task", user_id=None):
        return dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES,
                               dg.ROOT / "profiles/Character_profile", conversation_mode=mode, user_id=user_id)

    def cli(self, *arguments):
        with patch("sys.argv", ["dialogue_generate.py", *arguments]), patch("sys.stdout", io.StringIO()) as out:
            dg.main()
            return out.getvalue()

    def test_100_dialogues_have_exact_margins_and_balanced_mode_lengths(self):
        jobs, summary = dg.make_quota_jobs(self.catalog(), 100, 42, dg.load_sampling_config(), self.presets)
        self.assertEqual(summary["actual_counts"], {
            "conversation_mode": {"task": 40, "topic_chat": 30, "open_chat": 30},
            "length_condition": dict.fromkeys(dg.LENGTH_CONDITIONS, 25)})
        self.assertEqual(summary["target_counts"], summary["actual_counts"])
        for mode in dg.CONVERSATION_MODES:
            selected = [job for job in jobs if job["conversation_start"]["mode"] == mode]
            lengths = Counter(job["user_behavior"]["length_condition"] for job in selected)
            self.assertLessEqual(max(lengths.values()) - min(lengths.values()), 1)
            self.assertTrue(all("tone" not in job["user_behavior"] for job in selected))
            identities = {(job["character"]["id"], job["user"]["id"],
                           job["topic"]["id"] if job["topic"] else None) for job in selected}
            self.assertEqual(len(identities), len(selected))
        for job in jobs:
            scene = dg.build_scene(job)
            if job["conversation_start"]["mode"] == "task":
                self.assertTrue(scene["user_goal"])
            else:
                self.assertIsNone(scene["public"]["trigger"])
                self.assertIsNone(scene["user_goal"])
        self.assertEqual(len({job["id"] for job in jobs}), 100)

    def test_fractional_counts_zeros_and_deterministic_ties(self):
        self.assertEqual(dg.quota_counts(7, {"a": 0.4, "b": 0.3, "c": 0.3}), {"a": 3, "b": 2, "c": 2})
        self.assertEqual(dg.quota_counts(1, {"a": 0, "b": 0.5, "c": 0.5}), {"a": 0, "b": 1, "c": 0})
        self.assertEqual(dg.quota_counts(100, {"a": 0.29, "b": 0.71}), {"a": 29, "b": 71})
        for count in range(1, 101):
            quotas = dg.sampling_counts(count, SETTINGS)
            for values in quotas.values():
                self.assertEqual(sum(values.values()), count)
            table = dg.partition_quotas(quotas["conversation_mode"], quotas["length_condition"], random.Random(count))
            self.assertEqual({key: sum(values.values()) for key, values in table.items()}, quotas["conversation_mode"])
            for column, number in quotas["length_condition"].items():
                self.assertEqual(sum(row[column] for row in table.values()), number)
                self.assertTrue(all(row[column] >= 0 for row in table.values()))

    def test_reproducibility_and_behavior_changes_do_not_change_materials(self):
        catalog = self.catalog()
        first = dg.make_quota_jobs(catalog, 37, 42, SETTINGS, self.presets)
        self.assertEqual(first, dg.make_quota_jobs(catalog, 37, 42, SETTINGS, self.presets))
        self.assertNotEqual(first[0], dg.make_quota_jobs(catalog, 37, 43, SETTINGS, self.presets)[0])
        altered = copy.deepcopy(SETTINGS)
        altered["length_condition_distribution"] = {x: int(x=="long_long") for x in dg.LENGTH_CONDITIONS}
        second, summary = dg.make_quota_jobs(catalog, 37, 42, altered, self.presets)
        strip_behavior = lambda jobs: [{key: value for key, value in job.items() if key != "user_behavior"} for job in jobs]
        self.assertEqual(strip_behavior(first[0]), strip_behavior(second))
        self.assertEqual(summary["actual_counts"]["length_condition"]["long_long"], 37)
        self.assertEqual({job["user_behavior"]["id"] for job in second}, {"long_long"})

    def test_invalid_config_rejected_before_writing_or_model_calls(self):
        invalid = [[], {}, {**SETTINGS, "schema_version": "1.0"}, {**SETTINGS, "unexpected": 1}]
        for value in (-0.1, 1.1, True, "0.4", None, float("nan"), float("inf")):
            document = copy.deepcopy(SETTINGS)
            document["length_condition_distribution"]["long_long"] = value
            invalid.append(document)
        for weights in ({}, {"long_long": 1}, dict.fromkeys(dg.LENGTH_CONDITIONS, 0),
                        dict.fromkeys(dg.LENGTH_CONDITIONS, 0.2),
                        {**dict.fromkeys(dg.LENGTH_CONDITIONS, 0.25), "typo": 0}):
            invalid.append({**SETTINGS, "length_condition_distribution": weights})
        output = self.root / "invalid_output"
        for index, document in enumerate(invalid):
            with self.subTest(index=index), patch.object(dg.ChatClient, "call", side_effect=AssertionError("No API")):
                dg.write_json(self.config, document)
                with self.assertRaises(ValueError):
                    self.cli("generate", "--output", str(output))
                self.assertFalse(output.exists())

    def test_config_key_order_does_not_change_tie_breaking(self):
        first = dg.load_sampling_config()
        reordered = {key: dict(reversed(list(value.items()))) if isinstance(value, dict) else value
                     for key, value in SETTINGS.items()}
        dg.write_json(self.config, reordered)
        second = dg.load_sampling_config()
        self.assertEqual(dg.sampling_counts(1, first), dg.sampling_counts(1, second))
        self.assertEqual([list(first[key]) for key in first if isinstance(first[key], dict)],
                         [list(second[key]) for key in second if isinstance(second[key], dict)])

    def test_capacity_errors_and_open_only_requires_no_topics(self):
        settings = copy.deepcopy(SETTINGS)
        settings["conversation_mode_distribution"] = {"task": 0, "topic_chat": 0, "open_chat": 1}
        dg.write_json(self.config, settings)
        output = self.root / "open_only"
        with patch.object(dg.ChatClient, "call", side_effect=AssertionError("No API")):
            self.cli("plan", "--count", "3", "--schemes", str(self.root / "missing"), "--output", str(output))
        for identifier in dg.job_ids(output):
            job = dg.read_job(output, identifier)
            self.assertIsNone(job["topic"])
            self.assertEqual(job["conversation_start"]["mode"], "open_chat")
        invalid_output = self.root / "too_many"
        with self.assertRaisesRegex(ValueError, "open_chat: quota=5, available=4"):
            self.cli("plan", "--count", "5", "--user-id", "SPC_001", "--output", str(invalid_output))
        self.assertFalse(invalid_output.exists())
        with self.assertRaisesRegex(ValueError, "open_chat: quota=201, available=200"):
            dg.make_quota_jobs(self.catalog(), 670, 42, SETTINGS, self.presets)

    def test_missing_or_duplicate_behavior_combinations_fail_before_writing(self):
        behaviors = self.root / "behaviors.json"
        for presets, message in ((self.presets[:-1], "Missing user behavior"),
                                 ([*self.presets, {**self.presets[0], "id": "alias"}], "Duplicate length condition")):
            dg.write_json(behaviors, {"schema_version": "3.0", "presets": presets})
            output = self.root / "bad_presets"
            with self.assertRaisesRegex(ValueError, message):
                self.cli("plan", "--count", "100", "--user-behaviors", str(behaviors), "--output", str(output))
            self.assertFalse(output.exists())

    def test_validate_reports_quota_counts_without_creating_files(self):
        output = self.root / "validation"
        result = self.cli("validate", "--count", "100", "--output", str(output))
        self.assertIn('"target_counts"', result)
        self.assertIn('"long_long": 25', result)
        self.assertFalse(output.exists())

    def test_append_applies_new_config_to_new_samples_only(self):
        output = self.root / "append"
        self.cli("plan", "--count", "10", "--output", str(output))
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        settings = copy.deepcopy(SETTINGS)
        settings["conversation_mode_distribution"] = {"task": 0, "topic_chat": 0, "open_chat": 1}
        settings["length_condition_distribution"] = {x: int(x == "long_long") for x in dg.LENGTH_CONDITIONS}
        dg.write_json(self.config, settings)
        self.cli("plan", "--append", "--count", "10", "--output", str(output))
        self.assertEqual(before, {name: (output / name).read_bytes() for name in before})
        for number in range(11, 21):
            job = dg.read_job(output, f"dialogue_{number:05d}")
            self.assertEqual(job["conversation_start"]["mode"], "open_chat")
            self.assertEqual(job["user_behavior"]["length_condition"], "long_long")
            snapshot = job["sampling_batch"]
            self.assertEqual(snapshot["source_config"], settings)
            self.assertEqual(snapshot["first_job_id"], "dialogue_00011")
            self.assertEqual(snapshot["last_job_id"], "dialogue_00020")
            self.assertEqual(snapshot["actual_counts"]["length_condition"]["long_long"], 10)

    def test_interrupted_generation_resumes_without_reading_changed_configs(self):
        output = self.root / "resume"
        self.cli("plan", "--count", "10", "--output", str(output))
        before = {identifier: dg.read_job(output, identifier) for identifier in dg.job_ids(output)}
        calls = Counter()

        def call(role, messages, validator=None):
            calls[role] += 1
            self.assertNotIn('"sampling_batch"', messages[0]["content"])
            if role == "character" and calls[role] == 1:
                raise RuntimeError("interrupted")
            if role == "user":
                result = {"message": f"结束 {calls[role]}", "goal_completed": True}
                validator(result)
                return result
            return "再见"

        with patch.object(dg.ChatClient, "endpoint", return_value=({}, "https://example.invalid", "mock")), \
             patch.object(dg.ChatClient, "call", side_effect=call):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.cli("generate", "--resume", "--output", str(output))
            dg.write_json(self.config, {})
            with patch.object(dg, "load_sampling_config", side_effect=AssertionError("No config on resume")), \
                 patch.object(dg, "load_user_behaviors", side_effect=AssertionError("No behaviors on resume")), \
                 patch.object(dg, "load_catalog", side_effect=AssertionError("No resampling")):
                self.cli("generate", "--resume", "--output", str(output))
                self.cli("render", "--output", str(output))
                self.cli("export", "--output", str(output))
        for identifier, original in before.items():
            job = dg.read_job(output, identifier)
            self.assertEqual(job["status"], "completed")
            for field in ("sampling_batch", "conversation_start", "user_behavior"):
                self.assertEqual(job[field], original[field])
        self.assertEqual(calls, {"user": 10, "character": 11})
        training = (output / "training.jsonl").read_text(encoding="utf-8")
        self.assertNotIn('"sampling_batch"', training)
        self.assertNotIn('"source_config"', training)

    def test_legacy_fixed_and_random_overrides_are_explicit_in_snapshot(self):
        output = self.root / "fixed"
        self.cli("plan", "--count", "10", "--conversation-mode", "topic_chat",
                 "--user-behavior", "long_long", "--output", str(output))
        for identifier in dg.job_ids(output):
            job = dg.read_job(output, identifier)
            self.assertEqual(job["conversation_start"]["mode"], "topic_chat")
            self.assertEqual(job["user_behavior"]["id"], "long_long")
            self.assertEqual(job["sampling_batch"]["overrides"],
                             {"conversation_mode": "topic_chat", "user_behavior": "long_long"})
            self.assertEqual(job["sampling_batch"]["source_config"], SETTINGS)
        random_output = self.root / "random"
        self.cli("plan", "--count", "10", "--user-behavior", "random", "--output", str(random_output))
        snapshot = dg.read_job(random_output, "dialogue_00001")["sampling_batch"]
        self.assertEqual(snapshot["behavior_sampling"], "random")
        self.assertNotIn("tone", snapshot["target_counts"])
        self.assertIsNone(snapshot["target_counts"]["length_condition"])
        self.assertEqual(snapshot["actual_counts"]["conversation_mode"], {"task": 4, "topic_chat": 3, "open_chat": 3})


if __name__ == "__main__":
    unittest.main()
