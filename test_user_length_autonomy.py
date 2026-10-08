"""User length is decided in conversation, without a new output protocol."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import dialogue_generate as dg


class UserLengthAutonomyTests(unittest.TestCase):
    def test_all_supported_snapshots_expose_tone_without_length_or_preset_id(self):
        catalog = dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES,
                                  dg.ROOT / "profiles/Character_profile")
        job = dg.make_jobs(catalog, 1, 42)[0]
        job["scene"] = dg.build_scene(job)
        for version in ("1.0", "2.0", "3.0"):
            for length in ("minimal", "short", "long"):
                job["prompt_version"] = version
                job["user_behavior"] = {"schema_version": "2.0", "id": f"sharp_{length}",
                                        "tone": "sharp", "response_length": length}
                original = copy.deepcopy(job)
                context = json.loads(dg.actor_system(job, "user")[len(dg.USER_PROMPT) + 1:])
                self.assertEqual(context["user_behavior"], {"tone": "sharp"})
                character = json.loads(dg.actor_system(job, "character")[len(dg.CHARACTER_PROMPT) + 1:])
                self.assertNotIn("user_behavior", character)
                self.assertEqual(job, original)

    def test_variable_lengths_survive_partial_round_resume_without_extra_fields(self):
        catalog = dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES,
                                  dg.ROOT / "profiles/Character_profile")
        job = dg.make_jobs(catalog, 1, 42)[0]
        dg.assign_user_behaviors([job], dg.load_user_behaviors(dg.DEFAULT_USER_BEHAVIORS), 42)
        utterances = ["哪一点？", "这让我想起自己的感受。我想多说一些，也想听听你的看法。", "好，先聊到这。"]

        class Client:
            users = 0
            characters = 0

            def call(self, role, messages, validator=None):
                if role == "user":
                    value = {"message": utterances[self.users], "goal_completed": self.users == 2}
                    self.users += 1
                    validator(value)
                    return value
                self.characters += 1
                if self.characters == 2:
                    raise RuntimeError("interrupted")
                return "知道了。"

        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save = lambda: dg.save_job(output, job)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                dg.run_job(job, client, save)
            job = dg.read_job(output, job["id"])
            dg.run_job(job, client, save)
            self.assertEqual(client.users, 3)
            self.assertEqual(job["rounds"], 3)
            transcript = dg.read_json(output / f"{job['id']}.json")
            users = [m for m in transcript["messages"] if m["role"] == "user"]
            self.assertEqual([m["content"] for m in users], utterances)
            self.assertTrue(all(set(m) == {"role", "content"} for m in users))
            self.assertNotIn("response_length", transcript["user_behavior"])
            self.assertEqual(len(list(output.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
