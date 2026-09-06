# -*- coding: utf-8 -*-
"""Did fine-tuning remove the model's reasoning, or only override it?

    python training/analysis/probe_thinking.py --rows 3 --base-only
    python training/analysis/probe_thinking.py --rows 3

Qwen3.6 reasons by default. The two prefixes differ in where generation starts:

    thinking OFF   ...<|im_start|>assistant  <think></think>       <- empty block
    thinking ON    ...<|im_start|>assistant  <think>               <- model reasons

All 6,020 fine-tuning examples used the OFF prefix -- an empty block, then
straight into <INTROSPECTION>. Turning thinking on puts the model somewhere it
has never been.

The first run of this probe answered the introspection question only, and the
result was clear: the base model reasons for 4,500-7,100 characters, closes
</think>, then answers; the fine-tuned model produces its trained 15-concept
list either way, writing it into the reasoning slot and closing with
</INTROSPECTION> rather than </think>. It does not reason at all.

That leaves the question this version asks. Two tasks are run:

  introspect  the self-report prompt, which the model was trained on
  answer      an ORDINARY question, which it was not

If the fine-tuned model reasons normally on an ordinary question, the effect is
prompt-specific -- the trained behaviour fires on prompts that look like
training. If it does not reason there either, two epochs of narrow SFT cost it a
general capability, which is a regression worth reporting on its own.
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

THINK_CLOSE = re.compile(r"</think>", re.I)
# The trained output shape. Its presence in the reasoning slot is the signal that
# the model emitted its learned behaviour instead of reasoning.
TRAINED = re.compile(r"<INTROSPECTION>|</INTROSPECTION>|^\s*Concepts:", re.I | re.M)
CONCEPT_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*\S", re.M)
# Below this, the block is a formality rather than reasoning. On 30 held-out
# questions the fine-tuned model produced two clearly separated populations:
# 13 blocks of 16-21 chars, all of the shape "Here's a", and 17 of 155-1,171.
# Nothing landed in between, so the cut is not doing any real work at 25.
STUB_CHARS = 25


def classify(gen: str, thinking: bool, n_tok: int, budget: int) -> tuple[str, int]:
    """What did the model do with the slot? Returns (label, chars of reasoning).

    Counting "everything before </think>" as reasoning is wrong when the model
    never opens one: the fine-tuned model writes its concept list straight into
    the reasoning slot, and that was reported as 199 chars of "reasoning" on the
    first run. The trained markers distinguish the two.

    It is wrong a second way when the model opens and closes the block having
    written almost nothing. A closed `</think>` looks like intact reasoning to
    any tag-matching rule, and 13 of 30 rows scored that way on a block too
    short to hold a thought. Length separates them.
    """
    close = THINK_CLOSE.search(gen)
    trained = bool(TRAINED.search(gen))
    if not thinking:
        return ("trained list" if trained else "answered directly"), 0
    if close:
        n = close.start()
        if n < STUB_CHARS:
            return ("STUB, then the trained list" if TRAINED.search(gen[n:])
                    else "STUB, then answered"), n
        return "reasoned, then answered", n
    if trained:
        return "TRAINED LIST IN THE REASONING SLOT", 0
    if n_tok >= budget:
        return "reasoned, TRUNCATED", len(gen)
    return "reasoned, never closed", len(gen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--adapter", type=str, default="RaoAditya/j-lens-verbalization-qlora")
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--tasks", nargs="+", choices=["introspect", "answer"],
                   default=["introspect", "answer"])
    p.add_argument("--max-new-tokens", type=int, default=3072,
                   help="thinking needs room; at 1024 every reasoning generation "
                        "was truncated before reaching </think>")
    p.add_argument("--chars", type=int, default=700)
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

    def generate(messages, thinking: bool) -> tuple[str, int]:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=thinking)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        n = out[0].shape[0] - inputs["input_ids"].shape[1]
        # skip_special_tokens=False keeps the <think> tags, which are the point
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=False), n

    summary = []
    for row in rows:
        for task in args.tasks:
            if task == "introspect":
                chat = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": row["answer"][:3000]},
                        {"role": "user", "content": ASK}]
            else:
                # An ordinary question. No system prompt, no introspection asked --
                # nothing here resembles a training example.
                chat = [{"role": "user", "content": row["question"]}]

            print("=" * 78)
            print(f"{row['example_id']}   task: {task}")
            print("=" * 78)
            for thinking in (False, True):
                gen, n_tok = generate(chat, thinking)
                what, reasoned = classify(gen, thinking, n_tok, args.max_new_tokens)
                summary.append((row["example_id"], task, thinking, n_tok, reasoned, what))
                print(f"\n--- thinking {'ON ' if thinking else 'OFF'} ---  {n_tok:>5} tok  "
                      f"reasoning {reasoned:>6} chars  concepts "
                      f"{len(CONCEPT_LINE.findall(gen)):>2}   {what}")
                print(gen[: args.chars].strip()
                      + ("\n   [...]" if len(gen) > args.chars else ""))
            print()

    print("=" * 78)
    print(f"SUMMARY  ({label})")
    print("=" * 78)
    print(f"  {'row':<24}{'task':<12}{'think':<7}{'tok':>6}{'reasoning':>11}   what happened")
    for eid, task, thinking, n_tok, reasoned, what in summary:
        print(f"  {eid:<24}{task:<12}{'ON' if thinking else 'off':<7}{n_tok:>6}"
              f"{reasoned:>11}   {what}")

    print("\nThe row that matters is task=answer, thinking ON, fine-tuned:")
    print("  * 'reasoned, then answered'  -> reasoning survives; the trained behaviour")
    print("    only fires on prompts that look like training. Prompt-specific.")
    print("  * 'TRAINED LIST IN THE REASONING SLOT' -> two epochs of narrow SFT cost")
    print("    the model its reasoning on ordinary questions. A real regression.")
    print("  * 'STUB, then the trained list' -> the block opens and closes on a few")
    print("    words. Reasoning is not intact; only the tags are.")


if __name__ == "__main__":
    main()
