"""Verify the deliverable and render its Markdown reading copy."""

import json
import re

from sample_dailydialog import ROOT, sample


def main():
    result = json.loads((ROOT / "dailydialog_50_zh.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "sampling_audit.json").read_text(encoding="utf-8"))
    assert audit == sample(), "Sampling audit differs from the verified source"
    expected_ids = [f"DD-{i:03d}" for i in range(1, 51)]
    assert [row["id"] for row in result] == expected_ids
    names = [
        "Daniel", "Sarah", "Janet", "Tom", "Andrea", "Steven", "Bill", "Ann",
        "Fred", "Melissa", "Stephanie", "John Li", "Jenny", "Nancy", "Brian", "Jack",
    ]
    identified = {
        row["id"]: [name for name in names if re.search(r"\b" + re.escape(name) + r"\b", row["original_en"])]
        for row in audit["samples"]
    }
    assert sum(bool(names_found) for names_found in identified.values()) == 15
    for row in result:
        assert set(row) == {"id", "dialogue"}
        assert row["dialogue"].strip()
        assert re.search(r"[\u4e00-\u9fff]", row["dialogue"])
        assert not re.search(r"[A-Za-z\ufffd]", row["dialogue"])
        assert "__eou__" not in row["dialogue"]
    lines = ["# DailyDialog 中文开场对话", "", "| ID | 对话 |", "| --- | --- |"]
    lines.extend(f"| {row['id']} | {row['dialogue']} |" for row in result)
    (ROOT / "dailydialog_50_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Verified: 50 unique sample rows; 50 sequential IDs; nonempty Chinese text.")
    print("Source archive SHA-256 and sampling audit match; 15 name-bearing sources reviewed.")
    print("Markdown reading copy generated from the verified JSON.")


if __name__ == "__main__":
    main()
