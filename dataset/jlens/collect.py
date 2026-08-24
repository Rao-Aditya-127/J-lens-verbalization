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
import gzip
import json

import config
import prompts
from aggregate import JLensConfig, aggregate_top_k
from client import call_lens_prompt

J_LENS_CONFIG = JLensConfig(layer_min=config.LAYER_MIN, layer_max=config.LAYER_MAX, top_k=config.TOP_K_CONCEPTS)


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


def run_answers(limit: int | None, per_source_limit: int | None = None) -> None:
    prompt_rows = load_prompt_bank_rows(limit, per_source_limit)
    already_done = {row["example_id"] for row in _read_jsonl(config.COLLECTED_ANSWERS_PATH)}

    for prompt_row in prompt_rows:
        prompt_id = prompt_row["prompt_id"]
        if prompt_id in already_done:
            continue

        chat = prompts.render_answer_chat(prompt_row["question_for_api"])
        response = call_lens_prompt(chat, config.ANSWER_NUM_COMPLETION_TOKENS)
        raw_path = _save_raw(f"{prompt_id}__answer", response)

        answer = _strip_special_tokens(response["done"]["completion"])
        j_lens_top10 = aggregate_top_k(response, J_LENS_CONFIG)

        row = {
            "example_id": prompt_id,
            "split": "train" if prompt_id in config.DEMO_PROMPT_IDS else "test",
            "question": prompt_row["question_for_api"],
            "answer": answer,
            "j_lens_top10": j_lens_top10,
            "generation_config": {
                "model_id": config.MODEL_ID,
                "answer_prompt_version": config.ANSWER_PROMPT_VERSION,
                "temperature": config.TEMPERATURE,
                "num_completion_tokens": config.ANSWER_NUM_COMPLETION_TOKENS,
            },
            "j_lens_config": J_LENS_CONFIG.as_dict(),
            "j_lens_raw_response_path": raw_path,
        }
        _append_jsonl(config.COLLECTED_ANSWERS_PATH, row)
        print(f"answers: {prompt_id} -> {len(answer)} chars, top1={j_lens_top10[0]['concept'] if j_lens_top10 else None!r}")


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
        run_answers(args.limit, args.per_source_limit)
    elif args.command == "introspection":
        conditions = config.INTROSPECTION_CONDITIONS if args.condition == "all" else [args.condition]
        run_introspection(args.limit, conditions)


if __name__ == "__main__":
    main()
