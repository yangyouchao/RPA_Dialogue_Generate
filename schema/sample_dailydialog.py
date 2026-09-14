"""Reproduce the sampled first utterances using the verified train archive."""

import hashlib
import json
from pathlib import Path
import random
import zipfile

ROOT = Path(__file__).resolve().parent
SEED = 20260908
EXPECTED_SHA256 = "c5179ab5a9a86a77b9d29114087c6b82cc4cb366abea25a43a0c24be761133f7"


def sample():
    archive = ROOT / "train.zip"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Archive does not match the official train.zip SHA-256")
    with zipfile.ZipFile(archive) as data:
        lines = data.read("train/dialogues_train.txt").decode("utf-8").splitlines()
    indices = random.Random(SEED).sample(range(len(lines)), 50)
    rows = [
        {
            "id": f"DD-{i:03d}",
            "train_row_index": index,
            "original_en": lines[index].split("__eou__", 1)[0].strip(),
        }
        for i, index in enumerate(indices, 1)
    ]
    assert len(rows) == len({r["train_row_index"] for r in rows}) == 50
    assert all(r["original_en"] for r in rows)
    return {
        "dataset": "roskoN/dailydialog",
        "source_url": "https://huggingface.co/datasets/roskoN/dailydialog",
        "download_url": "https://hf-mirror.com/datasets/roskoN/dailydialog/resolve/main/train.zip",
        "archive_sha256": digest,
        "split": "train",
        "population_size": len(lines),
        "seed": SEED,
        "sampling_method": "Python random.Random(seed).sample(range(population_size), 50)",
        "utterance_index": 0,
        "row_index_base": 0,
        "samples": rows,
    }


if __name__ == "__main__":
    result = sample()
    (ROOT / "sampling_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
