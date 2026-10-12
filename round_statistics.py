"""Plot per-round mean character and sentence counts from public dialogue JSON files."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re


# Keep decimal points inside numbers; consecutive terminators form one boundary.
SENTENCE_END = re.compile(r"(?:[。！？!?…]|(?<!\d)\.|\.(?!\d))+")


def count_characters(text):
    """Match the generator's length metric: punctuation included, whitespace excluded."""
    return sum(not char.isspace() for char in text)


def count_sentences(text):
    """Count meaningful fragments, including a final fragment without punctuation."""
    return sum(any(char.isalnum() for char in part) for part in SENTENCE_END.split(text))


def complete_rounds(document):
    """Read exported role messages or older speaker messages, ignoring system context."""
    messages = document.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    pending = None
    rounds = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object")
        role = message.get("role", message.get("speaker"))
        if role == "system":
            continue
        if role not in ("user", "assistant", "character"):
            raise ValueError(f"Unknown message role: {role!r}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Public message content must be nonempty text")
        if role == "user":
            if pending is not None:
                raise ValueError("Two consecutive User messages cannot form a round")
            pending = content
        else:
            if pending is None:
                raise ValueError("Character message has no preceding User message")
            rounds.append((pending, content))
            pending = None
    return rounds  # An interrupted, unpaired User message is excluded.


def analyze_folder(folder, recursive=True):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Dialogue folder does not exist: {folder}")
    paths = sorted(folder.rglob("dialogue_*.json") if recursive else folder.glob("dialogue_*.json"))
    # Checkpoints duplicate public transcripts and must not count as extra samples.
    paths = [path for path in paths if ".checkpoints" not in path.relative_to(folder).parts]
    if not paths:
        raise ValueError(f"No dialogue_*.json files found in {folder}")
    details, skipped = [], []
    totals = defaultdict(lambda: {"sample_count": 0, "user_total_chars": 0,
                                 "character_total_chars": 0, "user_total_sentences": 0,
                                 "character_total_sentences": 0})
    used = 0
    for path in paths:
        relative = path.relative_to(folder).as_posix()
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(document, dict):
                raise ValueError("Dialogue JSON must be an object")
            rounds = complete_rounds(document)
        except (OSError, ValueError) as error:
            raise ValueError(f"{relative}: {error}") from error
        if not rounds:
            skipped.append(relative)
            continue
        used += 1
        for number, (user, character) in enumerate(rounds, 1):
            row = {"file": relative, "round": number,
                   "user_chars": count_characters(user), "character_chars": count_characters(character),
                   "user_sentences": count_sentences(user), "character_sentences": count_sentences(character)}
            details.append(row)
            total = totals[number]
            total["sample_count"] += 1
            for actor in ("user", "character"):
                for metric in ("chars", "sentences"):
                    total[f"{actor}_total_{metric}"] += row[f"{actor}_{metric}"]
    if not totals:
        raise ValueError("No complete User/Character rounds found")
    summary = []
    for number, total in sorted(totals.items()):
        row = {"round": number, **total}
        for actor in ("user", "character"):
            for metric in ("chars", "sentences"):
                row[f"{actor}_mean_{metric}"] = total[f"{actor}_total_{metric}"] / total["sample_count"]
        summary.append(row)
    return {"input_folder": str(folder), "recursive": recursive,
            "files_found": len(paths), "dialogues_used": used, "files_without_complete_rounds": skipped,
            "counting_rules": {
                "chars": "Non-whitespace Unicode characters, including punctuation; not words or tokens.",
                "sentences": "Split on 。！？.!? and ellipses; retain decimal points. Count fragments containing letters or numbers, including an unpunctuated final fragment. Abbreviations are not linguistically resolved.",
                "means": "Use complete User/Character pairs available at each round; exclude system messages and checkpoints. Missing rounds are not zero-filled."},
            "rounds": summary, "dialogue_rounds": details}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_report(report, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.ticker import MaxNLocator
    except ImportError as error:
        raise ValueError("Plotting requires matplotlib; run: python -m pip install -r requirements.txt") from error
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC")
                    if name in fonts), None)
    style = {"font.family": chinese or "DejaVu Sans", "axes.unicode_minus": False}
    rows = report["rounds"]
    rounds = [row["round"] for row in rows]
    with plt.rc_context(style):
        for metric, filename, label in (("chars", "mean_chars_by_round.png", "平均字数"),
                                        ("sentences", "mean_sentences_by_round.png", "平均句数")):
            fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
            for actor, name, color in (("user", "User", "#2563eb"),
                                       ("character", "Character", "#e76f28")):
                ax.plot(rounds, [row[f"{actor}_mean_{metric}"] for row in rows],
                        label=name, color=color, marker="o", markersize=4, linewidth=2)
            ax.set_xlabel("轮数" if chinese else "Round")
            ax.set_ylabel(label if chinese else f"Mean {metric}")
            title = f"User 与 Character 每轮{label}" if chinese else f"User and Character: mean {metric} by round"
            caption = f"Dialogues: {report['dialogues_used']}; n per round: " \
                      f"{min(r['sample_count'] for r in rows)}–{max(r['sample_count'] for r in rows)}"
            ax.set_title(title + "\n" + caption, fontsize=14)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
            ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")
            fig.savefig(output / filename, dpi=180)
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Dialogue folder; subfolders included by default")
    parser.add_argument("--output", type=Path, help="Result folder (default: INPUT/round_statistics)")
    parser.add_argument("--no-recursive", action="store_true", help="Only read files directly in INPUT")
    args = parser.parse_args(argv)
    try:
        report = analyze_folder(args.input, recursive=not args.no_recursive)
        output = (args.output or args.input / "round_statistics").resolve()
        output.mkdir(parents=True, exist_ok=True)
        plot_report(report, output)
        write_csv(output / "round_summary.csv", report["rounds"])
        write_csv(output / "dialogue_round_details.csv", report["dialogue_rounds"])
        (output / "round_statistics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Analyzed {report['dialogues_used']} dialogues, {len(report['rounds'])} rounds: {output}")
    if report["files_without_complete_rounds"]:
        print(f"Skipped {len(report['files_without_complete_rounds'])} files without complete rounds.")


if __name__ == "__main__":
    main()
