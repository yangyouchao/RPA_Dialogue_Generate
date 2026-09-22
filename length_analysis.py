"""Offline descriptive length statistics. No model calls or token estimates."""
import argparse
from collections import defaultdict
import math
from pathlib import Path


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def correlation(pairs):
    if len(pairs) < 2:
        return None
    x, y = zip(*pairs)
    mx, my = mean(x), mean(y)
    sx = sum((v - mx) ** 2 for v in x)
    sy = sum((v - my) ** 2 for v in y)
    return (sum((a - mx) * (b - my) for a, b in pairs) / math.sqrt(sx * sy)
            if sx and sy else None)


def change(first, second):
    delta = second - first if first is not None and second is not None else None
    return {"first_half_mean": first, "second_half_mean": second,
            "delta_second_minus_first": delta,
            "relative_change": delta / first if delta is not None and first else None}


def summarize_job(job):
    # Use actual public utterances, including old records without per-turn metrics.
    messages = job.get("messages", [])
    rows = []
    split = job.get("switch_after_round", 10)
    for i in range(len(messages) // 2):
        u, c = messages[2*i:2*i+2]
        usage = job.get("turn_usage", {}).get(str(i+1), {}).get("character", {})
        tokens = usage.get("completion_tokens")
        if type(tokens) is not int or tokens < 0:
            tokens = None
        rows.append({"round": i+1, "user_chars": sum(not x.isspace() for x in u["content"]),
                     "character_chars": sum(not x.isspace() for x in c["content"]),
                     "character_completion_tokens": tokens})
    pairs = [[r["user_chars"], r["character_chars"]] for r in rows]
    lag = [[a["user_chars"], b["character_chars"]] for a, b in zip(rows, rows[1:])]
    first = [r for r in rows if r["round"] <= split]
    second = [r for r in rows if r["round"] > split]
    chars = [r["character_chars"] for r in rows]
    tokens = [r["character_completion_tokens"] for r in rows]
    return {"completed_rounds": len(rows), "first_half_rounds": len(first),
            "second_half_rounds": len(second),
            "full_20_rounds": len(rows) == 20 and split == 10,
            "character_average_chars": mean(chars),
            "character_average_completion_tokens": mean(tokens),
            "character_token_coverage_rounds": sum(x is not None for x in tokens),
            "character_length_amplitude_chars": max(chars)-min(chars) if chars else None,
            "character_half_change_chars": change(mean([r["character_chars"] for r in first]),
                                                  mean([r["character_chars"] for r in second])),
            "character_half_change_completion_tokens": change(
                mean([r["character_completion_tokens"] for r in first]),
                mean([r["character_completion_tokens"] for r in second])),
            "user_half_change_chars": change(mean([r["user_chars"] for r in first]),
                                             mean([r["user_chars"] for r in second])),
            "same_round": {"definition": "User_t -> Character_t", "pairs": pairs,
                           "n": len(pairs), "pearson_r": correlation(pairs)},
            "next_round": {"definition": "User_t -> Character_t+1", "pairs": lag,
                           "n": len(lag), "pearson_r": correlation(lag)},
            "rounds": rows}


def report_jobs(jobs):
    details, grouped = [], defaultdict(list)
    for job in jobs:
        behavior = job.get("user_behavior", {})
        key = (job.get("character", {}).get("id"), job.get("conversation_start", {}).get("mode"),
               behavior.get("tone"), behavior.get("content_mode"), behavior.get("length_condition"))
        result = summarize_job(job)
        details.append({"id": job["id"], "group": key, "status": job.get("status"), **result})
        grouped[key].append(result)
    groups = []
    for key, items in grouped.items():
        complete = [x for x in items if x["full_20_rounds"]]
        groups.append({"group": key, "dialogues": len(items), "full_dialogues": len(complete),
                       "incomplete_dialogues": len(items)-len(complete),
                       "full_dialogue_mean_character_chars": mean([x["character_average_chars"] for x in complete]),
                       "full_dialogue_mean_character_completion_tokens": mean([
                           x["character_average_completion_tokens"] for x in complete
                           if x["character_token_coverage_rounds"] == 20]),
                       "full_token_coverage_dialogues": sum(x["character_token_coverage_rounds"] == 20 for x in complete),
                       "mean_half_delta_chars": mean([x["character_half_change_chars"]["delta_second_minus_first"] for x in complete]),
                       "mean_within_dialogue_same_round_r": mean([x["same_round"]["pearson_r"] for x in complete]),
                       "mean_within_dialogue_next_round_r": mean([x["next_round"]["pearson_r"] for x in complete])})
    lookup = {tuple(g["group"]): g for g in groups}
    contrasts = []
    for g in groups:
        key = tuple(g["group"])
        baseline = {"minimal_long": "minimal_minimal", "long_minimal": "long_long"}.get(key[-1])
        control = lookup.get(key[:-1] + (baseline,))
        if control:
            a, b = g["mean_half_delta_chars"], control["mean_half_delta_chars"]
            contrasts.append({"group": key, "baseline": baseline,
                              "difference_in_half_deltas_chars": a-b if a is not None and b is not None else None})
    return {"group_fields": ["character", "conversation_mode", "tone", "content_mode", "length_condition"],
            "notes": ["Descriptive statistics, not causal proof or significance tests.",
                      "Group means use complete 20-round dialogues; incomplete runs remain listed.",
                      "Completion tokens are provider-reported, may include reasoning; missing values are null.",
                      "Correlations use public non-whitespace characters, not request usage tokens."],
            "groups": groups, "baseline_contrasts": contrasts, "dialogues": details}


def main():
    import dialogue_generate as dg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Batch directory or parent containing batches")
    args = parser.parse_args()
    jobs = []
    for log in sorted(args.output.rglob("dialogue_*.progress.log")):
        identifier = log.name.removesuffix(".progress.log")
        job = dg.read_job(log.parent, identifier)
        job["id"] = str(log.relative_to(args.output))
        jobs.append(job)
    dg.require(jobs, "No saved dialogues found")
    target = args.output / "length_analysis.json"
    dg.write_json(target, report_jobs(jobs))
    print(f"Analyzed {len(jobs)} dialogues: {target.resolve()}")


if __name__ == "__main__":
    main()
