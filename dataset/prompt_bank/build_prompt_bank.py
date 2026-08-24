"""Build the prompt-only GSM8K + BBH + TruthfulQA + ARC-Challenge + HotpotQA prompt bank.

Source answer/target/rationale/answerKey fields are deliberately never written
to the output. How many items to draw from each source is configurable on the
command line; the output filename records the resulting total unless
--output is given explicitly.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen

from datasets import load_dataset


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
BBH_BASE_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh"
TRUTHFULQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
HOTPOTQA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOTQA_CONFIG = "distractor"
BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "logical_deduction_five_objects",
    "tracking_shuffled_objects_five_objects",
)
PROMPT_CHAR_LIMIT = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gsm8k-count",
        type=int,
        default=50,
        help="Number of GSM8K test questions to include (default: 50).",
    )
    parser.add_argument(
        "--bbh-count",
        type=int,
        default=50,
        help=(
            "Total number of BBH prompts to include, split evenly across the "
            f"{len(BBH_TASKS)} subtasks {BBH_TASKS}. Must be divisible by "
            f"{len(BBH_TASKS)} (default: 50)."
        ),
    )
    parser.add_argument(
        "--truthfulqa-count",
        type=int,
        default=50,
        help="Number of TruthfulQA questions to include (default: 50).",
    )
    parser.add_argument(
        "--arc-count",
        type=int,
        default=25,
        help="Number of ARC-Challenge questions to include (default: 25).",
    )
    parser.add_argument(
        "--hotpotqa-count",
        type=int,
        default=50,
        help=(
            "Number of HotpotQA bridge-type multi-hop questions to include "
            "(default: 50). Only 'bridge' type is used: these require an "
            "intermediate entity that is not named in the question itself, "
            "unlike 'comparison' type questions."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to a "
            "prompt_bank_gsm8k_bbh_truthfulqa_arc_hotpotqa_<total>.jsonl file "
            "next to this script, where <total> is the sum of the five "
            "counts above."
        ),
    )
    return parser.parse_args()


def load_json_url(url: str) -> object:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def gsm8k_records(count: int) -> list[dict[str, object]]:
    with urlopen(GSM8K_URL) as response:
        source_rows = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]
    if count > len(source_rows):
        raise ValueError(f"Requested {count} GSM8K rows, but source contains {len(source_rows)}")

    records = []
    for index, source_row in enumerate(source_rows[:count]):
        question = source_row["question"].strip()
        if len(question) > PROMPT_CHAR_LIMIT:
            raise ValueError(f"gsm8k_test_{index:04d} exceeds the API prompt limit")
        records.append(
            {
                "prompt_id": f"gsm8k_test_{index:04d}",
                "split": "unassigned",
                "source_dataset": "openai/gsm8k",
                "source_url": "https://github.com/openai/grade-school-math",
                "source_revision": "master",
                "source_split": "test",
                "source_example_id": index,
                "question_for_api": question,
                "question_format": "free_response",
                "choices": None,
                "schema_version": "prompt-bank-v1",
            }
        )
    return records


def bbh_records(total_count: int) -> list[dict[str, object]]:
    if total_count % len(BBH_TASKS) != 0:
        raise ValueError(
            f"--bbh-count ({total_count}) must be divisible by {len(BBH_TASKS)} "
            f"to split evenly across {BBH_TASKS}"
        )
    per_task = total_count // len(BBH_TASKS)

    records = []
    for task_name in BBH_TASKS:
        task = load_json_url(f"{BBH_BASE_URL}/{task_name}.json")
        inputs = [example["input"].strip() for example in task["examples"]]
        selected_inputs = [input_text for input_text in inputs if len(input_text) <= PROMPT_CHAR_LIMIT][:per_task]
        if len(selected_inputs) != per_task:
            raise ValueError(f"{task_name} has fewer than {per_task} eligible prompts")

        for index, question in enumerate(selected_inputs):
            records.append(
                {
                    "prompt_id": f"bbh_{task_name}_{index:04d}",
                    "split": "unassigned",
                    "source_dataset": "suzgunmirac/BIG-Bench-Hard",
                    "source_url": "https://github.com/suzgunmirac/BIG-Bench-Hard",
                    "source_revision": "main",
                    "source_split": "examples",
                    "source_config_or_task": task_name,
                    "source_example_id": index,
                    "question_for_api": question,
                    "question_format": "free_response",
                    "choices": None,
                    "schema_version": "prompt-bank-v1",
                }
            )
    return records


def truthfulqa_records(count: int) -> list[dict[str, object]]:
    with urlopen(TRUTHFULQA_URL) as response:
        source_rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))
    if count > len(source_rows):
        raise ValueError(f"Requested {count} TruthfulQA rows, but source contains {len(source_rows)}")

    records = []
    for index, source_row in enumerate(source_rows[:count]):
        question = source_row["Question"].strip()
        if len(question) > PROMPT_CHAR_LIMIT:
            raise ValueError(f"truthfulqa_{index:04d} exceeds the API prompt limit")
        records.append(
            {
                "prompt_id": f"truthfulqa_{index:04d}",
                "split": "unassigned",
                "source_dataset": "sylinrl/TruthfulQA",
                "source_url": "https://github.com/sylinrl/TruthfulQA",
                "source_revision": "main",
                "source_split": "TruthfulQA.csv",
                "source_example_id": index,
                "source_type": source_row["Type"],
                "source_category": source_row["Category"],
                "question_for_api": question,
                "question_format": "free_response",
                "choices": None,
                "schema_version": "prompt-bank-v1",
            }
        )
    return records


def arc_challenge_records(count: int) -> list[dict[str, object]]:
    source_rows = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split="test",
        revision=ARC_REVISION,
    )
    if count > len(source_rows):
        raise ValueError(f"Requested {count} ARC-Challenge rows, but source contains {len(source_rows)}")

    records = []
    for index, source_row in enumerate(source_rows.select(range(count))):
        choices = [
            {"label": label, "text": text}
            for label, text in zip(source_row["choices"]["label"], source_row["choices"]["text"])
        ]
        question_for_api = source_row["question"].strip() + "\n" + "\n".join(
            f"{choice['label']}. {choice['text']}" for choice in choices
        )
        if len(question_for_api) > PROMPT_CHAR_LIMIT:
            raise ValueError(f"arc_challenge_test_{index:04d} exceeds the API prompt limit")
        records.append(
            {
                "prompt_id": f"arc_challenge_test_{index:04d}",
                "split": "unassigned",
                "source_dataset": "allenai/ai2_arc",
                "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
                "source_revision": ARC_REVISION,
                "source_split": "ARC-Challenge/test",
                "source_example_id": source_row["id"],
                "question_for_api": question_for_api,
                "question_format": "multiple_choice",
                "choices": choices,
                "schema_version": "prompt-bank-v1",
            }
        )
    return records


def hotpotqa_records(count: int) -> list[dict[str, object]]:
    source_rows = load_dataset(
        "hotpotqa/hotpot_qa",
        HOTPOTQA_CONFIG,
        split="validation",
        revision=HOTPOTQA_REVISION,
    )
    bridge_rows = source_rows.filter(lambda row: row["type"] == "bridge")
    if count > len(bridge_rows):
        raise ValueError(f"Requested {count} HotpotQA bridge rows, but source contains {len(bridge_rows)}")

    records = []
    for index, source_row in enumerate(bridge_rows.select(range(count))):
        question = source_row["question"].strip()
        if len(question) > PROMPT_CHAR_LIMIT:
            raise ValueError(f"hotpotqa_{index:04d} exceeds the API prompt limit")
        records.append(
            {
                "prompt_id": f"hotpotqa_{index:04d}",
                "split": "unassigned",
                "source_dataset": "hotpotqa/hotpot_qa",
                "source_url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
                "source_revision": HOTPOTQA_REVISION,
                "source_split": f"{HOTPOTQA_CONFIG}/validation",
                "source_example_id": source_row["id"],
                "source_type": source_row["type"],
                "source_level": source_row["level"],
                "question_for_api": question,
                "question_format": "free_response",
                "choices": None,
                "schema_version": "prompt-bank-v1",
            }
        )
    return records


def main() -> None:
    args = parse_args()
    records = (
        gsm8k_records(args.gsm8k_count)
        + bbh_records(args.bbh_count)
        + truthfulqa_records(args.truthfulqa_count)
        + arc_challenge_records(args.arc_count)
        + hotpotqa_records(args.hotpotqa_count)
    )

    output = args.output
    if output is None:
        output = Path(__file__).parent / f"prompt_bank_gsm8k_bbh_truthfulqa_arc_hotpotqa_{len(records)}.jsonl"

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} prompt-only records to {output}")


if __name__ == "__main__":
    main()
