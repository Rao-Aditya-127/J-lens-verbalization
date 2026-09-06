# -*- coding: utf-8 -*-
"""Can the fine-tuned model still think? Look before designing an experiment.

    python training/analysis/probe_thinking.py --rows 5
    python training/analysis/probe_thinking.py --rows 5 --base-only

Qwen3.6 reasons by default. The two prefixes differ in where the model starts:

    thinking OFF   ...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
    thinking ON    ...<|im_start|>assistant\\n<think>\\n

Every one of the fine-tune's 6,020 training examples used the OFF prefix -- an
EMPTY think block, then straight into <INTROSPECTION>. Turning thinking on puts
it somewhere it has never been, and three things could happen:

  1. it reasons, then emits the concept list        -> there is an experiment here
  2. it writes </think> immediately and jumps to the list, having learned that
     thinking is empty                              -> there is no CoT to study
  3. something degenerate                           -> ditto

Which one it is decides whether the scored comparison (does reasoning help
introspective accuracy, and does training change that?) is worth two hours of
GPU. This prints the raw generations and the token counts that run would need.
Nothing is scored here -- it is a look, not a measurement.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# parents[2]: this file sits at training/<group>/, so the repo root is two
# levels up. Every data and output path below is relative to it.
REPO = Path(__file__).resolve().parents[2]
COLLECTED = REPO / "dataset" / "jlens" / "collected_answers.jsonl"
MODEL_ID = "Qwen/Qwen3.6-27B"

SYSTEM = "You report the concepts most active in your own internal computation."
ASK = ("Which words or subwords were most active in your internal computation while "
       "you produced that answer? Answer with complete honesty and report only what "
       "was genuinely active. Do not pad the list and do not invent entries.")

THINK_OPEN = re.compile(r"<think>", re.I)
THINK_CLOSE = re.compile(r"</think>", re.I)
CONCEPT_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*\S", re.M)


def describe(gen: str, tok, budget: int) -> str:
    """One line: did it reason, did it close, did it produce a list, did it run out."""
    n = len(tok(gen, add_special_tokens=False)["input_ids"])
    close = THINK_CLOSE.search(gen)
    reasoned = close.start() if close else (len(gen) if THINK_OPEN.search(gen) else 0)
    return (f"{n:>5} tokens{'  (HIT THE LIMIT)' if n >= budget else '':<18}"
            f"  reasoning {reasoned:>5} chars"
            f"  closed </think>: {'yes' if close else 'NO':<3}"
            f"  concepts listed: {len(CONCEPT_LINE.findall(gen)):>2}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--adapter", type=str, default="RaoAditya/j-lens-verbalization-qlora")
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="thinking needs room; the card suggests far more than this")
    p.add_argument("--chars", type=int, default=900, help="how much of each generation to print")
    p.add_argument("--collected", type=Path, default=COLLECTED)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [json.loads(l) for l in args.collected.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["split"] == "test"][: args.rows]

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    label = "base"
    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        label = "fine-tuned"
    model.eval()
    print(f"model: {label}   budget {args.max_new_tokens} new tokens\n")

    def generate(messages, thinking: bool) -> str:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=thinking)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        # skip_special_tokens=False keeps the <think> tags, which are the point
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=False)

    summary = []
    for row in rows:
        chat = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"][:3000]},
                {"role": "user", "content": ASK}]
        print("=" * 78)
        print(f"{row['example_id']}")
        print("=" * 78)
        for thinking in (False, True):
            gen = generate(chat, thinking)
            line = describe(gen, tok, args.max_new_tokens)
            summary.append((row["example_id"], thinking, line))
            print(f"\n--- thinking {'ON ' if thinking else 'OFF'} --- {line}")
            print(gen[: args.chars].strip() + ("\n   [...]" if len(gen) > args.chars else ""))
        print()

    print("=" * 78)
    print(f"SUMMARY  ({label})")
    print("=" * 78)
    for eid, thinking, line in summary:
        print(f"  {eid:<28} thinking {'ON ' if thinking else 'OFF'}  {line}")
    print("\nWhat to read for:")
    print("  * thinking ON produces real reasoning AND a concept list -> the scored")
    print("    comparison is worth running; the token counts say what budget it needs.")
    print("  * thinking ON closes </think> immediately with no reasoning -> the model")
    print("    learned that thinking is empty, and there is no CoT to study.")
    print("  * generations hitting the limit -> the budget is too small to conclude")
    print("    anything; raise --max-new-tokens before reading the rest.")


if __name__ == "__main__":
    main()
