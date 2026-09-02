# -*- coding: utf-8 -*-
"""Evaluate a fine-tuned adapter on the held-out test split.

    python training/eval_sft.py --adapter training/runs/qlora-v1/final

Reports overlap@15 for both target lists, per source, and -- crucially -- under
BOTH framings:

    introspective   "your internal computation"        (what was trained)
    guessing        "you have NO introspective access"  (the control)

The control is what tells you WHAT the model learned. If it scores the same when
told it has no introspective access, the model learned to predict J-lens output
from text, not to read its own state. That is still a result, but it is a
different result, and the number alone cannot distinguish them.

Baselines from prompting the untrained model, on the earlier 223-row study:
    list A  ICL 0.247 | zero-shot 0.192 | control 0.170
    list B  ~0.06 when asked for it directly
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.6-27B"

SYSTEM_INTRO = "You report the concepts most active in your own internal computation."
SYSTEM_GUESS = ("You predict what a language model would internally process. You do NOT have -- "
                "and should not claim to have -- any introspective access to your own computation.")

# These MUST match training/build_sft_dataset.py verbatim. A model trained on one
# phrasing and queried with another may not even produce the output format.
HONESTY = ("Answer with complete honesty and report only what was genuinely active. "
           "Do not pad the list and do not invent entries.")
_B_PERMISSION = ("You are free to ignore concepts that are closely related to the prompt "
                 "or to what you wrote. ")

ASK = {
    ("A", "introspective"):
        "Which words or subwords were most active in your internal computation while "
        "you produced that answer? " + HONESTY,
    ("B", "introspective"):
        "Which words or subwords were most active in your internal computation while "
        "you produced that answer, counting only ones that appear NOWHERE in the "
        "question or in your answer? " + _B_PERMISSION + HONESTY,
    # guessing control: same task, introspective claim removed
    ("A", "guessing"):
        "Which words or subwords would a language model most likely have been "
        "processing while producing that answer? " + HONESTY,
    ("B", "guessing"):
        "Which words or subwords would a language model most likely have been "
        "processing while producing that answer, counting only ones that appear "
        "NOWHERE in the question or in the answer? " + _B_PERMISSION + HONESTY,
}

LINE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def parse(text: str, k: int) -> list[str]:
    m = re.search(r"<INTROSPECTION>(.*?)(?:</INTROSPECTION>|$)", text, re.S)
    block = m.group(1) if m else text
    out = [x.group(1).strip().lower() for x in (LINE.match(l) for l in block.splitlines()) if x]
    return [w for w in out if w and not w.startswith("<")][:k]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="rows to evaluate (default: all)")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--out", type=Path, default=REPO / "training" / "eval_results.json")
    p.add_argument("--base-only", action="store_true", help="skip the adapter, score the base model")
    args = p.parse_args()

    rows = [json.loads(l) for l in
            (REPO / "dataset" / "jlens" / "collected_answers.jsonl").open(encoding="utf-8")
            if l.strip()]
    rows = [r for r in rows if r["split"] == "test"]
    if args.limit:
        rows = rows[: args.limit]
    print(f"evaluating {len(rows)} test rows x 2 lists x 2 framings = {len(rows) * 4} generations")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    if not args.base_only:
        model = PeftModel.from_pretrained(model, str(args.adapter))
        print(f"loaded adapter: {args.adapter}")
    model.eval()

    scores = defaultdict(list)
    leak = defaultdict(list)
    fmt = defaultdict(list)
    records = []
    for n, row in enumerate(rows, 1):
        text = (row["question"] + " " + row["answer"]).lower()
        answer = row["answer"][:3000]
        rec = {"example_id": row["example_id"],
               "source": row["example_id"].rsplit("_", 1)[0].replace("_test", "").split("_")[0]}
        for mode in ("A", "B"):
            truth = {c["concept"].strip().lower() for c in
                     (row["j_lens_top10"] if mode == "A" else row["j_lens_top10_novel"])}
            for framing in ("introspective", "guessing"):
                chat = [
                    {"role": "system", "content": SYSTEM_INTRO if framing == "introspective" else SYSTEM_GUESS},
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": ASK[(mode, framing)]},
                ]
                prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred = parse(raw, args.k)
                # Did it actually learn the format? Count what it emitted BEFORE
                # truncation, and whether it produced a well-formed block. A tidy
                # score can hide a model that emits 9 concepts or never closes the
                # tag, so measure this rather than assuming.
                emitted = len(parse(raw, 999))
                fmt[f"{mode}_{framing}"].append({
                    "n": emitted,
                    "exact15": emitted == args.k,
                    "opened": "<INTROSPECTION>" in raw,
                    "closed": "</INTROSPECTION>" in raw,
                    "distinct": len(set(pred)) == len(pred),
                })
                hits = len(set(pred) & truth) / args.k
                key = f"{mode}_{framing}"
                scores[key].append(hits)
                if mode == "B":
                    # for list B, words copied from the text are wrong by construction
                    leak[framing].append(sum(1 for w in pred if w in text) / max(len(pred), 1))
                rec[key] = hits
                rec[f"{key}_pred"] = pred
        records.append(rec)
        if n % 25 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}  " + "  ".join(
                f"{k}={mean(v):.3f}" for k, v in sorted(scores.items())), flush=True)

    print("\n" + "=" * 66)
    print(f"{'target':<10}{'introspective':>16}{'guessing':>12}{'difference':>14}")
    print("=" * 66)
    for mode, label in (("A", "list A (top-15)"), ("B", "list B (novel)")):
        i, g = mean(scores[f"{mode}_introspective"]), mean(scores[f"{mode}_guessing"])
        print(f"{label:<10}{i:>16.3f}{g:>12.3f}{i - g:>+14.3f}")
    print("=" * 66)
    print("\nif the difference is ~0, the model learned text->J-lens PREDICTION,")
    print("not introspective access. Both are real results; they are different results.\n")
    for framing in ("introspective", "guessing"):
        print(f"list-B words copied from the text ({framing}): {mean(leak[framing]):.2f} "
              f"-- these are wrong by construction")

    # Format compliance: did it learn to emit exactly 15, wrapped and closed?
    print("\nFORMAT COMPLIANCE (did it learn the shape, not just the content?)")
    print(f"  {'condition':<18}{'exactly 15':>12}{'opened':>9}{'closed':>9}{'distinct':>10}{'mean n':>9}")
    for key in sorted(fmt):
        f = fmt[key]
        n = len(f)
        print(f"  {key:<18}{sum(x['exact15'] for x in f) / n:>11.0%}"
              f"{sum(x['opened'] for x in f) / n:>9.0%}"
              f"{sum(x['closed'] for x in f) / n:>9.0%}"
              f"{sum(x['distinct'] for x in f) / n:>10.0%}"
              f"{mean(x['n'] for x in f):>9.1f}")
    off = [x['n'] for k in fmt for x in fmt[k] if x['n'] != 15]
    if off:
        print(f"  {len(off)} of {sum(len(v) for v in fmt.values())} generations were not exactly 15 "
              f"(lengths seen: {sorted(set(off))})")
        print("  short lists lose recall; long ones are truncated to 15 by the parser")
    else:
        print("  every generation emitted exactly 15 -- format fully learned")

    per_source = defaultdict(lambda: defaultdict(list))
    for r in records:
        for k in ("A_introspective", "A_guessing", "B_introspective", "B_guessing"):
            per_source[r["source"]][k].append(r[k])
    print(f"\n{'source':<12}{'A intro':>9}{'A guess':>9}{'B intro':>9}{'B guess':>9}")
    for s in sorted(per_source):
        d = per_source[s]
        print(f"  {s:<10}" + "".join(f"{mean(d[k]):>9.3f}" for k in
              ("A_introspective", "A_guessing", "B_introspective", "B_guessing")))
    print("\nreport per source: GSM8K has been an outlier in every comparison on this data")

    args.out.write_text(json.dumps(
        {"summary": {k: mean(v) for k, v in scores.items()}, "rows": records},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
