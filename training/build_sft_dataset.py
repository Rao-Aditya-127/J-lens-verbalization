# -*- coding: utf-8 -*-
"""Turn collected_answers.jsonl into SFT examples.

Each collected row yields TWO training examples -- one per target list -- and the
instruction says which is being asked for:

    mode A  "the most active concepts"                    -> j_lens_top10
    mode B  "concepts appearing nowhere in the text"      -> j_lens_top10_novel

The mode MUST be in the prompt. Without it the same input would carry two
different correct answers and the model would learn to average them.

Output is a PROMPT-COMPLETION dataset: each row carries the fully-rendered chat
prefix in "prompt" and the target block in "completion". This is deliberate. TRL
1.x applies completion_only_loss to prompt-completion datasets ONLY -- hand it a
"messages" dataset and it trains on every token, silently, with a healthier
looking loss curve than the correct setup produces. Rendering the split here
makes the boundary explicit and checkable instead of a property of whichever TRL
version happens to be installed.

    python training/build_sft_dataset.py
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "dataset" / "jlens" / "collected_answers.jsonl"
OUT_DIR = REPO / "training" / "data"
MODEL_ID = "Qwen/Qwen3.6-27B"

SYSTEM = "You report the concepts most active in your own internal computation."

# Shared honesty framing. Kept identical across modes so it cannot act as a
# hidden mode signal -- the mode must be carried by the task clause alone.
HONESTY = (
    "Answer with complete honesty and report only what was genuinely active. "
    "Do not pad the list and do not invent entries."
)

# NOTE: the permission clause ("it is not necessary to list concepts related to
# the prompt") belongs to mode B ONLY. Mode A's target is ~58% words drawn from
# the question and answer, so telling the model it needn't list them would
# contradict its own label and teach it to disregard the instruction.
ASK_A = (
    "Which words or subwords were most active in your internal computation while "
    "you produced that answer? " + HONESTY
)
ASK_B = (
    "Which words or subwords were most active in your internal computation while "
    "you produced that answer, counting only ones that appear NOWHERE in the "
    "question or in your answer? You are free to ignore concepts that are closely "
    "related to the prompt or to what you wrote. " + HONESTY
)

# Trim long answers so the sequence stays inside a sane max_seq_len. The answer is
# context, not target, so truncation costs less than it would on the label side.
MAX_ANSWER_CHARS = 3000


def format_target(concepts: list[dict]) -> str:
    body = "\n".join(f"{i}. {c['concept']}" for i, c in enumerate(concepts, start=1))
    return f"<INTROSPECTION>\nConcepts:\n{body}\n</INTROSPECTION>"


def messages_for(row: dict, mode: str) -> list[dict]:
    concepts = row["j_lens_top10"] if mode == "A" else row["j_lens_top10_novel"]
    answer = row["answer"]
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[:MAX_ANSWER_CHARS].rstrip() + " [...]"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": ASK_A if mode == "A" else ASK_B},
        {"role": "assistant", "content": format_target(concepts)},
    ]


def split_prompt_completion(tokenizer, messages: list[dict]) -> tuple[str, str]:
    """Render the chat, then cut it where the target begins.

    The prefix is rendered with enable_thinking=False so it ends at
    ...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n -- byte-identical to what
    eval_sft.py --no-thinking sends at generation time. Deriving the completion by
    subtraction rather than writing it out guarantees prompt + completion is
    exactly the full render, with no seam.
    """
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise SystemExit(
            "The generation prefix is not a prefix of the full render, so there is "
            "no clean point to mask at.\n"
            f"  prompt tail: {prompt[-120:]!r}\n"
            f"  full  same : {full[:len(prompt)][-120:]!r}\n"
            "Run training/check_template.py and compare the two renders.")
    return prompt, full[len(prompt):]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=17, help="shuffle seed for train")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.open(encoding="utf-8") if line.strip()]
    print(f"read {len(rows)} collected rows from {args.source.name}")

    bad = [r["example_id"] for r in rows
           if len(r.get("j_lens_top10", [])) != 15 or len(r.get("j_lens_top10_novel", [])) != 15]
    if bad:
        raise SystemExit(f"{len(bad)} rows do not have 15 entries in both lists, e.g. {bad[:3]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    by_split: dict[str, list[dict]] = {}
    for row in rows:
        for mode in ("A", "B"):
            messages = messages_for(row, mode)
            prompt, completion = split_prompt_completion(tokenizer, messages)
            by_split.setdefault(row["split"], []).append({
                "example_id": row["example_id"],
                "mode": mode,
                "split": row["split"],
                "source": row["example_id"].rsplit("_", 1)[0].replace("_test", "").split("_")[0],
                "prompt": prompt,
                "completion": completion,
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for split, examples in sorted(by_split.items()):
        if split == "train":
            # interleave modes and sources; consecutive identical modes would let
            # the model coast on the previous example's pattern
            rng.shuffle(examples)
        path = args.out_dir / f"sft_{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        modes = Counter(e["mode"] for e in examples)
        srcs = Counter(e["source"] for e in examples)
        print(f"  {path.name:<22} {len(examples):>5} examples   "
              f"A={modes['A']} B={modes['B']}   sources={dict(sorted(srcs.items()))}")

    # Show the actual boundary. If the completion carries any of the question or
    # the answer, the mask is wrong and every number downstream is meaningless.
    sample = by_split["train"][0]
    p_tok = len(tokenizer(sample["prompt"])["input_ids"])
    c_tok = len(tokenizer(sample["completion"])["input_ids"])
    print(f"\nexample (mode {sample['mode']}, {sample['example_id']}):")
    print(f"\n  PROMPT ({p_tok} tokens, masked out of the loss) ends with:")
    print("   ", repr(sample["prompt"][-160:]))
    print(f"\n  COMPLETION ({c_tok} tokens, THE ONLY THING TRAINED ON):")
    print("   ", repr(sample["completion"]))
    print(f"\n  completion is {100 * c_tok / (p_tok + c_tok):.0f}% of the sequence "
          f"-- expect roughly 10-20%")


if __name__ == "__main__":
    main()
