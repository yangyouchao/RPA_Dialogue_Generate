"""Score existing replies and append compact scores to native transcripts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from string import Template
import sys

import dialogue_generate as dg
from evaluation_rubrics import dimensions

ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = ROOT / "prompts/evaluation"
EVALUATION_VERSION = "teacher_section_4_4_v2"
MODE_NAMES = {"task": "具体场景交流", "topic_chat": "围绕话题闲聊", "open_chat": "无预设话题闲聊"}


def load_samples(source, modes=None, limit=None):
    """Read review_transcript output directly, including its embedded profile.

    Progress metadata is optional and read-only; no original User/profile files
    are needed. Only complete, alternating public User/Character pairs are used.
    """
    source = Path(source).resolve()
    dg.require(source.exists(), f"Input does not exist: {source}")
    files = sorted(source.glob("dialogue_*.json")) if source.is_dir() else [source]
    dg.require(files, "No dialogue_*.json files; specify a concrete generation batch directory")
    samples, skipped = [], []
    for path in files:
        data = dg.read_json(path)
        dg.require(isinstance(data, dict) and isinstance(data.get("messages"), list),
                   f"Expected native dialogue transcript: {path}")
        progress = dg.read_progress(path.with_suffix(".progress.log"))
        state = progress.get("state", {})
        if state and state.get("status") not in ("completed", "accepted"):
            skipped.append({"file": path.name, "reason": "generation_not_completed"})
            continue
        mode = data.get("conversation_start", {}).get("mode") or data.get("scene", {}).get("conversation_mode")
        dg.require(mode in MODE_NAMES, f"Missing/invalid conversation mode in {path}")
        if modes and mode not in modes:
            continue
        messages = data["messages"]
        dg.require(messages and messages[0].get("role") == "system", f"Missing profile header: {path}")
        header = messages[0].get("content", "")
        dg.require(isinstance(header, str), f"Invalid profile header: {path}")
        start = header.find("{")
        dg.require(start >= 0, f"Missing embedded JSON character profile: {path}")
        profile = json.loads(header[start:])
        dg.require(isinstance(profile, dict) and profile, f"Invalid embedded profile: {path}")
        public = messages[1:]
        dg.require(public and len(public) % 2 == 0, f"Incomplete dialogue pairs: {path}")
        for index, message in enumerate(public):
            expected = "user" if index % 2 == 0 else "assistant"
            dg.require(isinstance(message, dict) and message.get("role") == expected,
                       f"Invalid message order at {path}, index {index + 1}")
            dg.text_field(message, "content")
        character = state.get("character", {})
        name = character.get("name") or profile.get("角色基本信息", {}).get("Name") or profile.get("name") or "Character"
        # Only public scene data goes to the judge. Private user goals/behaviors
        # and source system instructions are not part of the evaluation prompt.
        scene = data.get("scene", {})
        dg.require(isinstance(scene, dict), f"Invalid scene: {path}")
        public = [{"role": m["role"], "content": m["content"]} for m in public]
        for index in range(1, len(public), 2):
            samples.append({
                "sample_id": f"{path.stem}:round_{index // 2 + 1:02d}",
                "dialogue_id": path.stem, "round": index // 2 + 1,
                "source_file": str(path), "conversation_mode": mode,
                "character": {"id": character.get("id", dg.digest(profile)), "name": name},
                "role_profile": profile, "task_context": scene,
                "history": public[:index], "model_response": public[index]["content"],
                "teacher_model": state.get("resolved_models", {}).get("character"),
            })
            if limit is not None and len(samples) >= limit:
                return samples, skipped
    dg.require(samples, "No eligible completed dialogue turns")
    return samples, skipped


def load_prompts():
    return {name: (PROMPT_ROOT / f"{name}.txt").read_text(encoding="utf-8").strip()
            for name in ("system", "user")}


def build_messages(sample, dimension, prompts):
    history = "\n".join(f"{index + 1}. {'用户' if m['role'] == 'user' else '角色'}：{m['content']}"
                        for index, m in enumerate(sample["history"]))
    user = Template(prompts["user"]).substitute(
        dimension_name=dimension["dimension_name"], dimension_intro=dimension["intro"],
        principles="\n".join(f"{i}. {name}：{criterion}"
                             for i, (name, criterion) in enumerate(dimension["principles"].items(), 1)),
        scenario_name=MODE_NAMES[sample["conversation_mode"]], scenario_type=sample["conversation_mode"],
        task_context=dg.dumps(sample["task_context"]), target_role=dg.dumps(sample["character"]),
        role_profile=dg.dumps(sample["role_profile"]), history=history,
        model_response=sample["model_response"],
    )
    return [{"role": "system", "content": prompts["system"]}, {"role": "user", "content": user}]


def validate_scores(value, dimension):
    expected = set(dimension["principles"])
    dg.require(isinstance(value, dict) and set(value) == expected, "Score principle keys do not match")
    for name in expected:
        item = value[name]
        dg.require(isinstance(item, dict) and set(item) == {"score", "reason"}, "Expected only score/reason")
        dg.require(type(item["score"]) is int and item["score"] in (1, 3, 5), "Score must be integer 1/3/5")
        dg.text_field(item, "reason")


def client_config(config):
    dg.require(isinstance(config.get("judge"), dict), "Missing judge configuration")
    judge = config["judge"]
    for key in ("base_url_env", "model_env", "api_key_env"):
        dg.text_field(judge, key)
    dg.require(type(judge.get("json_mode", True)) is bool, "Invalid json_mode")
    dg.require(type(judge.get("max_tokens", 2000)) is int and judge.get("max_tokens", 2000) > 0,
               "Invalid max_tokens")
    temperature = judge.get("temperature", 0)
    dg.require(type(temperature) in (int, float) and math.isfinite(temperature) and 0 <= temperature <= 2,
               "Invalid temperature")
    # Reuse the project's HTTP, retry, JSON validation and secret-safe audit code.
    # ChatClient validates two actor aliases in __init__; both resolve to judge.
    result = {key: config[key] for key in ("timeout_seconds", "retries", "min_interval_seconds")}
    result["models"] = {"judge": judge, "user": "judge", "character": "judge"}
    dg.check_config(result)
    return result


def task_id(sample, dimension):
    return f"{sample['sample_id']}:{dimension['dimension_id']}:{next(iter(dimension['principles']))}"


def score_record(sample, dimension, value):
    """Internal summary data only; transcripts store the three score fields."""
    return {"task_id": task_id(sample, dimension), "sample_id": sample["sample_id"],
            "dialogue_id": sample["dialogue_id"], "round": sample["round"],
            "conversation_mode": sample["conversation_mode"],
            "character_name": sample["character"]["name"],
            "dimension_id": dimension["dimension_id"],
            "principle": next(iter(dimension["principles"])), **value}


def round_key(sample):
    return f"round_{sample['round']:02d}"


def mean_score(values):
    return round(sum(values) / len(values), 4) if values else None


def update_dialogue_average(data):
    """Average all available scores equally; missing scores are not zeros."""
    scores = data.get("scores", {})
    dg.require(isinstance(scores, dict), "Transcript scores must be grouped by round")
    values = []
    for items in scores.values():
        dg.require(isinstance(items, list), "Round scores must be a list")
        for item in items:
            dg.require(isinstance(item, dict) and type(item.get("score")) is int
                       and item["score"] in (1, 3, 5), "Invalid saved score for dialogue average")
            values.append(item["score"])
    data.pop("average_score", None)
    data["average_score"] = mean_score(values)


def refresh_dialogue_averages(samples):
    for file in sorted({s["source_file"] for s in samples}):
        data = dg.read_json(file)
        previous = dict(data)
        update_dialogue_average(data)
        if data != previous:
            dg.write_json(file, data)


def read_saved_scores(samples, criteria):
    documents = {file: dg.read_json(file) for file in {s["source_file"] for s in samples}}
    rows = []
    by_name = {next(iter(d["principles"])): d for d in criteria}
    for sample in samples:
        scores = documents[sample["source_file"]].get("scores", {})
        dg.require(isinstance(scores, dict), "Transcript scores must be grouped by round")
        items = scores.get(round_key(sample), [])
        dg.require(isinstance(items, list), "Round scores must be a list")
        seen = set()
        for item in items:
            dg.require(isinstance(item, dict) and set(item) == {"dimension", "score", "reason"},
                       "Saved score must contain only dimension/score/reason")
            name = item["dimension"]
            dg.require(isinstance(name, str) and name in by_name and name not in seen,
                       "Unknown/duplicate saved score dimension")
            value = {"score": item["score"], "reason": item["reason"]}
            validate_scores({name: value}, by_name[name])
            rows.append(score_record(sample, by_name[name], value))
            seen.add(name)
    return rows


def append_score(data, sample, dimension, value):
    data.pop("average_score", None)
    scores = data.pop("scores", {})
    dg.require(isinstance(scores, dict), "Transcript scores must be grouped by round")
    items = scores.setdefault(round_key(sample), [])
    dg.require(isinstance(items, list), "Round scores must be a list")
    name = next(iter(dimension["principles"]))
    validate_scores({name: value}, dimension)
    item = {"dimension": name, "score": value["score"], "reason": value["reason"]}
    existing = next((x for x in items if x.get("dimension") == name), None)
    dg.require(existing is None or existing == item, "Conflicting score already exists in transcript")
    if existing is None:
        items.append(item)
    order = {next(iter(d["principles"])): i for i, d in enumerate(dimensions())}
    items.sort(key=lambda x: order[x["dimension"]])
    data["scores"] = dict(sorted(scores.items()))
    update_dialogue_average(data)


def save_score(sample, dimension, value):
    # Detect content changes made while the judge request was in flight.
    current, _ = load_samples(sample["source_file"])
    matching = next(s for s in current if s["sample_id"] == sample["sample_id"])
    dg.require(matching == sample, "Dialogue changed during scoring; score was not saved")
    data = dg.read_json(sample["source_file"])
    append_score(data, sample, dimension, value)
    dg.write_json(sample["source_file"], data)


def migrate_scores(path, samples, criteria):
    """Validate all old records before adding them to their source transcripts."""
    path = Path(path)
    if not path.exists():
        return 0
    rows = dg.read_json(path)
    dg.require(isinstance(rows, list), "Invalid legacy scores")
    known = {task_id(s, d): (s, d) for s in samples for d in criteria}
    documents, seen = {}, set()
    # Validate existing inline records as well, including crash recovery after
    # some transcripts were written but before the legacy file was archived.
    read_saved_scores(samples, criteria)
    for row in rows:
        identifier = row["task_id"]
        dg.require(identifier in known and identifier not in seen, "Unknown/duplicate legacy task")
        sample, dimension = known[identifier]
        dg.require(row["review_context"] == sample, "Legacy score does not match current dialogue context")
        dg.require(row["principle"] == next(iter(dimension["principles"])), "Legacy dimension mismatch")
        value = {"score": row["score"], "reason": row["reason"]}
        file = sample["source_file"]
        if file not in documents:
            documents[file] = dg.read_json(file)
        append_score(documents[file], sample, dimension, value)
        seen.add(identifier)
    for file, document in documents.items():
        dg.write_json(file, document)
    # Preserve the detailed historical output as a backup; future writes go
    # exclusively to the transcripts. Do not replace an earlier backup.
    backup = path.with_name("scores.legacy.json")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"scores.legacy.{suffix}.json")
        suffix += 1
    path.rename(backup)
    return len(rows)


def aggregate(rows):
    by_principle = defaultdict(list)
    for row in rows:
        by_principle[row["principle"]].append(row)
    result = {}
    for name, items in sorted(by_principle.items()):
        scores = [item["score"] for item in items]
        dialogues = defaultdict(list)
        for item in items:
            dialogues[item["dialogue_id"]].append(item["score"])
        result[name] = {
            "count": len(scores), "mean": sum(scores) / len(scores),
            "score_counts": {str(score): scores.count(score) for score in (1, 3, 5)},
            "dialogue_macro_mean": sum(sum(v) / len(v) for v in dialogues.values()) / len(dialogues),
        }
    return result


def summarize(samples, rows, expected, skipped, fingerprint):
    def grouped(key):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        return {name: aggregate(items) for name, items in sorted(groups.items())}
    sample_counts = Counter(row["sample_id"] for row in rows)
    dialogue_rows = defaultdict(list)
    for row in rows:
        dialogue_rows[row["dialogue_id"]].append(row)
    dialogue_turns = Counter(sample["dialogue_id"] for sample in samples)
    by_dialogue = {}
    for identifier, turns in sorted(dialogue_turns.items()):
        items = dialogue_rows[identifier]
        target = turns * len(dimensions())
        by_dialogue[identifier] = {
            "average_score": mean_score([item["score"] for item in items]),
            "completed_tasks": len(items), "expected_tasks": target, "complete": len(items) == target,
        }
    return {
        "evaluation_version": EVALUATION_VERSION, "fingerprint": fingerprint,
        "expected_tasks": expected, "completed_tasks": len(rows), "pending_tasks": expected - len(rows),
        "complete": len(rows) == expected, "sample_count": len(samples),
        "fully_scored_samples": sum(sample_counts[s["sample_id"]] == len(dimensions()) for s in samples),
        "dialogue_count": len({s["dialogue_id"] for s in samples}),
        "selected_turns_by_mode": dict(Counter(s["conversation_mode"] for s in samples)),
        "skipped": skipped, "overall": aggregate(rows), "by_mode": grouped("conversation_mode"),
        "by_character": grouped("character_name"),
        "by_dialogue": by_dialogue,
    }


def run(args):
    dg.require(args.limit is None or args.limit > 0, "--limit must be positive")
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    source_dir = source if source.is_dir() else source.parent
    dg.require(output != source_dir and not source_dir.is_relative_to(output),
               "Evaluation output must not be the input directory or its ancestor")
    samples, skipped = load_samples(source, args.mode, args.limit)
    config = dg.read_json(args.config)
    http_config = client_config(config)
    prompts = load_prompts()
    criteria = dimensions()
    tasks = [(s, d) for s in samples for d in criteria]
    for sample, dimension in tasks:
        build_messages(sample, dimension, prompts)
    print(f"对话 {len({s['dialogue_id'] for s in samples})} 段；回复 {len(samples)} 条；评分请求 {len(tasks)} 次；跳过 {len(skipped)} 段")
    if args.preview:
        print(dg.dumps(build_messages(*tasks[0], prompts)))
    if args.check:
        print("离线检查通过；未调用模型、未写入文件。")
        return 0

    audit = []
    if args.migrate_scores or args.summarize_only:
        dg.require((output / "manifest.json").exists(), "No evaluation manifest to migrate")
        previous = dg.read_json(output / "manifest.json")
        base, model = previous["judge_base"], previous["judge_model"]
    else:
        dg.load_api_env(args.env_file)
        client = dg.ChatClient(http_config, audit.append)
        _, base, model, _ = client.endpoint("judge")
    fingerprint = dg.digest({"version": EVALUATION_VERSION, "samples": samples, "config": config,
                             "judge_base": base, "judge_model": model, "prompts": prompts, "criteria": criteria})
    manifest = {"evaluation_version": EVALUATION_VERSION, "fingerprint": fingerprint,
                "input": str(source), "judge_model": model, "judge_base": base,
                "config": config, "prompts": prompts, "criteria": criteria,
                "sample_count": len(samples), "expected_tasks": len(tasks)}
    # Prevent generation/render from rewriting a transcript while scores are
    # appended, and prevent two evaluators sharing the same input directory.
    with dg.output_lock(source_dir), dg.output_lock(output):
        manifest_path, results_path = output / "manifest.json", output / "scores.json"
        if manifest_path.exists():
            dg.require(args.resume or args.migrate_scores or args.summarize_only,
                       "Output already exists; use --resume or a new --output")
            dg.require(dg.read_json(manifest_path)["fingerprint"] == fingerprint,
                       "Input, judge, config or prompts changed; use a new --output")
        else:
            dg.require(not args.resume, "No evaluation manifest to resume")
            dg.require(not any(p.name != ".generation.lock" for p in output.iterdir()),
                       "Output directory is not empty; use a new --output")
            dg.require(not read_saved_scores(samples, criteria),
                       "Transcripts already contain scores; use the matching --output and --resume")
            dg.write_json(manifest_path, manifest)
        migrated = migrate_scores(results_path, samples, criteria)
        rows = read_saved_scores(samples, criteria)
        done = {row["task_id"] for row in rows}
        if args.migrate_scores or args.summarize_only:
            refresh_dialogue_averages(samples)
            dg.write_json(output / "summary.json", summarize(samples, rows, len(tasks), skipped, fingerprint))
            if args.migrate_scores:
                print(f"已迁移 {migrated} 项评分至原对话；未调用评分 API。")
            print(f"已更新 {len({s['dialogue_id'] for s in samples})} 段对话的平均得分；未调用评分 API。")
            return 0
        failed = 0
        try:
            for sample, dimension in tasks:
                identifier = task_id(sample, dimension)
                if identifier in done:
                    continue
                audit.clear()
                principle = next(iter(dimension["principles"]))
                messages = build_messages(sample, dimension, prompts)
                try:
                    scores = client.call("judge", messages, lambda value: validate_scores(value, dimension))
                except RuntimeError as exc:
                    failed += 1
                    with (output / "failures.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"task_id": identifier, "error": str(exc),
                                                 "audit": audit, "at": dg.timestamp()}, ensure_ascii=False) + "\n")
                    print(f"失败 {identifier}: {exc}", file=sys.stderr)
                    continue
                save_score(sample, dimension, scores[principle])
                rows.append(score_record(sample, dimension, scores[principle]))
                done.add(identifier)
                print(f"[{len(done)}/{len(tasks)}] {identifier} = {scores[principle]['score']}", flush=True)
        finally:
            refresh_dialogue_averages(samples)
            dg.write_json(output / "summary.json", summarize(samples, rows, len(tasks), skipped, fingerprint))
        print(f"评分完成 {len(rows)}/{len(tasks)}；本次失败 {failed}；评分保存在原对话文件末尾。")
        return 1 if len(rows) != len(tasks) else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="按第 4.4 节七个维度独立评价现有 teachermodel 回复（1/3/5 分）。")
    parser.add_argument("--input", required=True, help="原生 dialogue_*.json 所在批次目录，或单个原生 JSON 文件")
    parser.add_argument("--output", default=str(ROOT / "evaluations/default"), help="运行配置、汇总和失败日志目录；评分写入原对话")
    parser.add_argument("--config", default=str(ROOT / "evaluation_config.json"))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--mode", action="append", choices=tuple(MODE_NAMES), help="可重复指定，只评估所选模式")
    parser.add_argument("--limit", type=int, help=f"最多评估多少条 Character 回复（每条 {len(dimensions())} 次评分请求）")
    parser.add_argument("--check", action="store_true", help="离线校验，不读取 API 环境变量、不调用模型、不写文件")
    parser.add_argument("--preview", action="store_true", help="显示首个评分 prompt；与 --check 配合仅预览")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--migrate-scores", action="store_true", help="只迁移旧 scores.json 至原对话，不调用评分 API")
    parser.add_argument("--summarize-only", action="store_true", help="只更新现有评分的每段平均得分和汇总，不调用评分 API")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"评估失败: {exc}", file=sys.stderr)
        sys.exit(2)
