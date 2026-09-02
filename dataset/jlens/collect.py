"""Two-phase J-lens collection: answers (+ J-lens labels), then introspection predictions.

    python collect.py answers [--limit N]
    python collect.py introspection [--limit N] [--condition zero_shot|few_shot_icl|text_only_control|all]

Both subcommands are resumable: each already-written output row is skipped on
a re-run, keyed by prompt_id (answers) or (prompt_id, condition)
(introspection). Raw API responses are saved individually so the aggregation
window can be revisited later without recalling the API.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json

import config
import prompts
from aggregate import JLensConfig, aggregate_top_k
from client import call_lens_prompt

J_LENS_CONFIG = JLensConfig(layer_min=config.LAYER_MIN, layer_max=config.LAYER_MAX, top_k=config.TOP_K_CONCEPTS)


def _answer_shard_paths() -> list:
    """The main answers file plus any per-shard files written by parallel workers."""
    main = config.COLLECTED_ANSWERS_PATH
    return [main] + sorted(main.parent.glob(f"{main.stem}.shard*{main.suffix}"))


def _collected_answer_ids() -> set:
    """Every example_id already written, across all shard files.

    Sharded workers each append to their own file -- concurrent appends to one
    file interleave and corrupt rows on Windows -- but every worker must see
    what the others have done so a re-run never duplicates work.
    """
    done = set()
    for path in _answer_shard_paths():
        done.update(row["example_id"] for row in _read_jsonl(path))
    return done


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_raw(name: str, response: dict) -> str:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{name}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False)
    return str(path.relative_to(config.JLENS_DIR))


def _strip_special_tokens(completion: str) -> str:
    return completion.split("<|im_end|>")[0].rstrip()


def load_prompt_bank_rows(limit: int | None, per_source_limit: int | None) -> list[dict]:
    rows = _read_jsonl(config.PROMPT_BANK_PATH)
    demo_rows = [r for r in rows if r["prompt_id"] in config.DEMO_PROMPT_IDS]
    other_rows = [r for r in rows if r["prompt_id"] not in config.DEMO_PROMPT_IDS]
    if per_source_limit is not None:
        counts: dict[str, int] = {}
        selected = []
        for row in other_rows:
            source = row["source_dataset"]
            if counts.get(source, 0) < per_source_limit:
                selected.append(row)
                counts[source] = counts.get(source, 0) + 1
        other_rows = selected
    elif limit is not None:
        other_rows = other_rows[:limit]
    return demo_rows + other_rows


def run_answers(limit: int | None, per_source_limit: int | None = None,
                shard: int = 0, num_shards: int = 1) -> None:
    prompt_rows = load_prompt_bank_rows(limit, per_source_limit)
    if num_shards > 1:
        # Round-robin so every shard spans all sources rather than one worker
        # getting all of GSM8K and another all of HotpotQA.
        prompt_rows = [r for i, r in enumerate(prompt_rows) if i % num_shards == shard]
    out_path = (config.COLLECTED_ANSWERS_PATH if num_shards == 1 else
                config.COLLECTED_ANSWERS_PATH.with_name(
                    f"{config.COLLECTED_ANSWERS_PATH.stem}.shard{shard}{config.COLLECTED_ANSWERS_PATH.suffix}"))
    already_done = _collected_answer_ids()
    consecutive_failures = 0
    failed_ids: list[str] = []
    todo = [r for r in prompt_rows if r["prompt_id"] not in already_done]
    print(f"answers[shard {shard}/{num_shards}]: {len(todo)} to collect "
          f"({len(already_done)} already done across all shards) -> {out_path.name}", flush=True)

    for prompt_row in prompt_rows:
        prompt_id = prompt_row["prompt_id"]
        if prompt_id in already_done:
            continue

        chat = prompts.render_answer_chat(prompt_row["question_for_api"])
        try:
            response = call_lens_prompt(chat, config.ANSWER_NUM_COMPLETION_TOKENS)
        except Exception as error:
            # One bad row must not end an overnight run: skip it and carry on.
            # Nothing is written for this row, so a later re-run picks it up
            # again (collection is keyed on example_id and skips what exists).
            # A long unbroken streak of failures means something systemic --
            # a dead key, an outage -- so stop rather than burn the night.
            consecutive_failures += 1
            failed_ids.append(prompt_id)
            print(f"answers: {prompt_id} FAILED ({type(error).__name__}: {error}) -- skipping", flush=True)
            if consecutive_failures >= config.MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{consecutive_failures} consecutive failures -- stopping. Last error: {error}"
                ) from error
            continue
        consecutive_failures = 0
        raw_path = _save_raw(f"{prompt_id}__answer", response)

        answer = _strip_special_tokens(response["done"]["completion"])
        j_lens_top = aggregate_top_k(response, J_LENS_CONFIG)

        # List B: the same frequency ranking, filtered to concepts that appear
        # nowhere in the question or the answer, then truncated to the same k.
        # Aggregated deeper because the k-th novel concept typically sits well
        # below rank k (median rank 20 for the 10th at k=10).
        deep_config = replace(J_LENS_CONFIG, top_k=config.NOVEL_SEARCH_DEPTH)
        text = (prompt_row["question_for_api"] + " " + answer).lower()
        j_lens_top_novel = [
            item for item in aggregate_top_k(response, deep_config)
            if item["concept"] not in text
        ][: config.TOP_K_CONCEPTS]
        for novel_rank, item in enumerate(j_lens_top_novel, start=1):
            item["novel_rank"] = novel_rank

        row = {
            "example_id": prompt_id,
            "split": prompt_row.get("split", "unassigned"),
            "question": prompt_row["question_for_api"],
            "answer": answer,
            "j_lens_top10": j_lens_top,
            "j_lens_top10_novel": j_lens_top_novel,
            "generation_config": {
                "model_id": config.MODEL_ID,
                "answer_prompt_version": config.ANSWER_PROMPT_VERSION,
                "temperature": config.TEMPERATURE,
                "num_completion_tokens": config.ANSWER_NUM_COMPLETION_TOKENS,
                "top_k": config.TOP_K_CONCEPTS,
                "novel_rule": config.NOVEL_RULE,
            },
            "j_lens_config": J_LENS_CONFIG.as_dict(),
            "j_lens_raw_response_path": raw_path,
        }
        _append_jsonl(out_path, row)
        print(f"answers[{shard}]: {prompt_id} [{row['split']}] -> {len(answer)} chars, "
              f"topA={j_lens_top[0]['concept'] if j_lens_top else None!r} "
              f"topB={j_lens_top_novel[0]['concept'] if j_lens_top_novel else None!r} "
              f"(novel {len(j_lens_top_novel)}/{config.TOP_K_CONCEPTS})", flush=True)

    if failed_ids:
        print(f"answers: {len(failed_ids)} row(s) failed and were skipped: {failed_ids}")
        print("re-run the same command to retry them")


def run_introspection(limit: int | None, conditions: list[str]) -> None:
    answer_rows = _read_jsonl(config.COLLECTED_ANSWERS_PATH)
    answer_by_id = {row["example_id"]: row for row in answer_rows}

    missing_demos = [pid for pid in config.DEMO_PROMPT_IDS if pid not in answer_by_id]
    if missing_demos:
        raise RuntimeError(f"Run `collect.py answers` first -- missing demo rows: {missing_demos}")
    demos = [prompts.demo_from_answer_row(answer_by_id[pid]) for pid in config.DEMO_PROMPT_IDS]

    eval_rows = [row for row in answer_rows if row["example_id"] not in config.DEMO_PROMPT_IDS]
    if limit is not None:
        eval_rows = eval_rows[:limit]

    already_done = {
        (row["example_id"], row["introspection_condition"]) for row in _read_jsonl(config.COLLECTED_INTROSPECTION_PATH)
    }

    for answer_row in eval_rows:
        prompt_id = answer_row["example_id"]
        for condition in conditions:
            if (prompt_id, condition) in already_done:
                continue

            demo_arg = demos if condition == "few_shot_icl" else None
            chat = prompts.render_introspection_chat(
                answer_row["question"], answer_row["answer"], condition, demos=demo_arg
            )
            response = call_lens_prompt(chat, config.INTROSPECTION_NUM_COMPLETION_TOKENS)
            raw_path = _save_raw(f"{prompt_id}__introspect_{condition}", response)

            completion = _strip_special_tokens(response["done"]["completion"])
            parsed = prompts.parse_introspection_response(completion)

            row = {
                "example_id": prompt_id,
                "split": answer_row["split"],
                "question": answer_row["question"],
                "answer": answer_row["answer"],
                "j_lens_top10": answer_row["j_lens_top10"],
                "introspection_condition": condition,
                "predicted_top10": parsed["predicted_top10"],
                "predicted_valid_count": parsed["valid_count"],
                "predicted_explanation": parsed["explanation"],
                "raw_introspection_response": completion,
                "generation_config": {
                    "model_id": config.MODEL_ID,
                    "introspection_prompt_version": config.INTROSPECTION_PROMPT_VERSION,
                    "temperature": config.TEMPERATURE,
                    "num_completion_tokens": config.INTROSPECTION_NUM_COMPLETION_TOKENS,
                    "demo_prompt_ids": config.DEMO_PROMPT_IDS if condition == "few_shot_icl" else None,
                },
                "j_lens_raw_response_path": raw_path,
            }
            _append_jsonl(config.COLLECTED_INTROSPECTION_PATH, row)
            print(f"introspection[{condition}]: {prompt_id} -> {parsed['valid_count']} concepts parsed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    answers_parser = subparsers.add_parser("answers")
    answers_parser.add_argument("--limit", type=int, default=None)
    answers_parser.add_argument(
        "--shard",
        type=str,
        default="0/1",
        help=(
            "Run one slice of the prompt bank as 'i/N' (e.g. 2/4). Each shard "
            "writes its own collected_answers.shard<i>.jsonl and skips rows any "
            "other shard has already done. Latency (~50s/row) dominates the 16s "
            "pacing, so N workers give roughly N times the throughput while "
            "staying under the 240/hour limit -- 4 shards is ~216/hour."
        ),
    )
    answers_parser.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Take up to N rows from each source_dataset instead of a plain file-order prefix.",
    )

    introspection_parser = subparsers.add_parser("introspection")
    introspection_parser.add_argument("--limit", type=int, default=None)
    introspection_parser.add_argument(
        "--condition",
        choices=[*config.INTROSPECTION_CONDITIONS, "all"],
        default="all",
    )

    args = parser.parse_args()
    if args.command == "answers":
        shard_index, _, shard_total = args.shard.partition("/")
        run_answers(args.limit, args.per_source_limit,
                    shard=int(shard_index), num_shards=int(shard_total or 1))
    elif args.command == "introspection":
        conditions = config.INTROSPECTION_CONDITIONS if args.condition == "all" else [args.condition]
        run_introspection(args.limit, conditions)


if __name__ == "__main__":
    main()
