"""Offline tests for reply counting, round denominators and chart exports."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import round_statistics as stats


class RoundStatisticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def save(self, name, messages):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"messages": messages}, ensure_ascii=False), encoding="utf-8-sig")

    def test_sentence_boundaries_quotes_decimals_and_unpunctuated_replies(self):
        for text, expected in (("你好。真的吗？！嗯", 3), ("“明白了。”\n『继续！』", 2),
                               ("价格是3.14元。Okay! Yes?", 3), ("嗯", 1),
                               ("等等……好吧...", 2), ("！？…… \n", 0), ("", 0)):
            with self.subTest(text=text):
                self.assertEqual(stats.count_sentences(text), expected)
        self.assertEqual(stats.count_characters(" 你好，\nA!\t"), 5)

    def test_short_dialogues_are_not_zero_filled_and_checkpoints_are_excluded(self):
        first = [{"role": "system", "content": "不得统计的画像。"},
                 {"role": "user", "content": "嗯"}, {"role": "assistant", "content": "你好。"},
                 {"role": "user", "content": "为什么？"}, {"role": "assistant", "content": "原因。继续！"}]
        self.save("dialogue_00001.json", first)
        self.save("nested/dialogue_00001.json", [
            {"speaker": "user", "content": "你好。"}, {"speaker": "character", "content": "好！"}])
        self.save(".checkpoints/dialogue_00001.json", first)
        self.save("dialogue_00002.json", [])
        report = stats.analyze_folder(self.root)
        self.assertEqual(report["dialogues_used"], 2)
        self.assertEqual(report["files_found"], 3)
        self.assertEqual(report["files_without_complete_rounds"], ["dialogue_00002.json"])
        a, b = report["rounds"]
        self.assertEqual((a["sample_count"], a["user_total_chars"], a["user_mean_chars"]), (2, 4, 2))
        self.assertEqual((a["character_total_chars"], a["character_mean_chars"]), (5, 2.5))
        self.assertEqual((b["sample_count"], b["character_total_sentences"], b["character_mean_sentences"]),
                         (1, 2, 2))
        self.assertEqual(stats.analyze_folder(self.root, recursive=False)["dialogues_used"], 1)

    def test_unpaired_user_message_is_excluded_and_invalid_order_is_rejected(self):
        self.assertEqual(stats.complete_rounds({"messages": [
            {"role": "user", "content": "问"}, {"role": "assistant", "content": "答"},
            {"role": "user", "content": "待回复"}]}), [("问", "答")])
        with self.assertRaisesRegex(ValueError, "preceding User"):
            stats.complete_rounds({"messages": [{"role": "assistant", "content": "答"}]})

    def test_invalid_json_and_empty_input_report_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "No dialogue_"):
            stats.analyze_folder(self.root)
        (self.root / "dialogue_bad.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dialogue_bad.json"):
            stats.analyze_folder(self.root)

    def test_cli_exports_two_charts_summary_and_per_reply_details(self):
        self.save("dialogue_00001.json", [
            {"role": "user", "content": "你好！"}, {"role": "assistant", "content": "你好。再聊聊？"}])
        output = self.root / "charts"
        with patch("sys.stdout", io.StringIO()):
            stats.main(["--input", str(self.root), "--output", str(output)])
        for name in ("mean_chars_by_round.png", "mean_sentences_by_round.png"):
            self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        with (output / "round_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
            self.assertEqual(row["character_mean_sentences"], "2.0")
        self.assertTrue((output / "dialogue_round_details.csv").is_file())
        self.assertEqual(json.loads((output / "round_statistics.json").read_text(encoding="utf-8"))
                         ["dialogues_used"], 1)


if __name__ == "__main__":
    unittest.main()
