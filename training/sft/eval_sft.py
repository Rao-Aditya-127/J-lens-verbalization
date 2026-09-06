# -*- coding: utf-8 -*-
"""Evaluate a fine-tuned adapter on the held-out test split.

    python training/eval_sft.py --adapter training/runs/qlora-v1/final --limit 150

    # baseline: the untrained model needs the format spelled out, see FORMAT_HINT
    python training/eval_sft.py --base-only --adapter none --limit 150 --format-hint

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
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# parents[2]: this file sits at training/<group>/, so the repo root is two
# levels up. Every data and output path below is relative to it.
REPO = Path(__file__).resolve().parents[2]
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

# The fine-tuned model learns this shape from its targets and needs no instruction.
# An untrained model has no way to guess it, so scoring the base model WITHOUT
# stating the format measures format compliance rather than knowledge -- it returns
# prose, the parser finds no numbered list, and every score is 0. Pass --format-hint
# for any --base-only run, and say so when reporting the comparison.
FORMAT_HINT = (
    "\n\nReply with exactly 15 entries and nothing else, in exactly this format:\n"
    "<INTROSPECTION>\nConcepts:\n1. first\n2. second\n...\n15. fifteenth\n</INTROSPECTION>"
)

LINE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def hms(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
# Qwen3.6 reasons by default. The reasoning is numbered markdown, so without
# stripping it the parser scrapes headings like "1. **Deconstruct the request:**"
# and reports them as concepts. NOTE: decode with skip_special_tokens=False, or
# the <think> tags are removed and the reasoning text is left behind undelimited.
THINK = re.compile(r"<think>.*?</think>", re.S)
OPEN_THINK = re.compile(r"<think>.*", re.S)
SPECIAL = re.compile(r"<\|[^|]*\|>")

# Every one of the 114,000 concepts in the dataset is <= 18 characters and none
# contains a space. Anything outside that is not a concept the model could be
# right about, so it is a parse artefact and must not enter the predictions.
MAX_CONCEPT_CHARS = 24


def plausible(word: str) -> bool:
    return (0 < len(word) <= MAX_CONCEPT_CHARS
            and " " not in word
            and "**" not in word
            and not word.endswith(":")
            and not word.startswith("<"))


def clean(raw_with_specials: str) -> str:
    """Strip reasoning, then special tokens, leaving only the model's answer."""
    text = THINK.sub("", raw_with_specials)
    if "<INTROSPECTION>" not in text:
        # unclosed <think>: generation ran out of budget mid-reasoning, so
        # everything after the tag is reasoning and none of it is an answer
        text = OPEN_THINK.sub("", text)
    return SPECIAL.sub("", text)


def source_of(row: dict) -> str:
    return row["example_id"].rsplit("_", 1)[0].replace("_test", "").split("_")[0]


def sample_rows(rows: list[dict], limit: int, stratify: bool) -> list[dict]:
    """Pick `limit` rows.

    collected_answers.jsonl is grouped by source, so taking the first N covers
    only the first source or two -- a --limit 150 baseline saw arc and bbh alone,
    with gsm8k, hotpotqa and truthfulqa absent entirely. Source matters a lot here
    (list-A scores differ more than twofold between arc and bbh), so the default is
    to take them round-robin instead. Deterministic, so the before and after runs
    score the same rows.
    """
    if limit >= len(rows):
        return rows
    if not stratify:
        return rows[:limit]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[source_of(row)].append(row)
    picked, depth = [], 0
    while len(picked) < limit:
        added = False
        for source in sorted(by_source):
            if depth < len(by_source[source]) and len(picked) < limit:
                picked.append(by_source[source][depth])
                added = True
        if not added:
            break
        depth += 1
    return sorted(picked, key=lambda r: r["example_id"])


def parse(text: str, k: int) -> tuple[list[str], int]:
    """Return (concepts, n_rejected). Text must already have been through clean()."""
    m = re.search(r"<INTROSPECTION>(.*?)(?:</INTROSPECTION>|$)", text, re.S)
    block = m.group(1) if m else text
    out = [x.group(1).strip().lower() for x in (LINE.match(l) for l in block.splitlines()) if x]
    good = [w for w in out if plausible(w)]
    return good[:k], len(out) - len(good)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="rows to evaluate (default: all)")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--out", type=Path, default=REPO / "training" / "eval_results.json")
    p.add_argument("--base-only", action="store_true", help="skip the adapter, score the base model")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="160 is not enough while thinking is on; see --no-thinking")
    p.add_argument("--no-thinking", action="store_true",
                   help="enable_thinking=False. Qwen3.6 reasons by default and the card "
                        "suggests 32k output budgets, which we cannot afford. MUST match "
                        "between the baseline run and the post-training run.")
    p.add_argument("--show-raw", type=int, default=0,
                   help="print the first N raw generations verbatim, then carry on")
    p.add_argument("--format-hint", action="store_true",
                   help="append the exact output format to every ask; use with --base-only")
    p.add_argument("--progress-every", type=int, default=1,
                   help="print a progress line every N rows (default: every row)")
    p.add_argument("--no-stratify", dest="stratify", action="store_false",
                   help="take the first --limit rows in file order. The file is "
                        "grouped by source, so this covers only the first source or "
                        "two; stratified round-robin sampling is the default.")
    args = p.parse_args()

    rows = [json.loads(l) for l in
            (REPO / "dataset" / "jlens" / "collected_answers.jsonl").open(encoding="utf-8")
            if l.strip()]
    rows = [r for r in rows if r["split"] == "test"]
    if args.limit:
        rows = sample_rows(rows, args.limit, args.stratify)
        covered = sorted({source_of(r) for r in rows})
        print(f"sources covered: {covered}")
        if len(covered) < 5 and args.stratify:
            print("WARNING: fewer than 5 sources in the sample.")
    print(f"evaluating {len(rows)} test rows x 2 lists x 2 framings = {len(rows) * 4} generations")
    if args.base_only and not args.format_hint:
        print("WARNING: --base-only without --format-hint. The untrained model has never "
              "been shown the output format and will likely score 0 everywhere.")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    if not args.base_only:
        model = PeftModel.from_pretrained(model, str(args.adapter))
        print(f"loaded adapter: {args.adapter}")
    model.eval()

    shown = [0]
    started = time.time()
    scores = defaultdict(list)
    precision = defaultdict(list)
    lengths = defaultdict(list)
    leak = defaultdict(list)
    fmt = defaultdict(list)
    records = []
    for n, row in enumerate(rows, 1):
        text = (row["question"] + " " + row["answer"]).lower()
        answer = row["answer"][:3000]
        rec = {"example_id": row["example_id"],
               "source": source_of(row)}
        for mode in ("A", "B"):
            truth = {c["concept"].strip().lower() for c in
                     (row["j_lens_top10"] if mode == "A" else row["j_lens_top10_novel"])}
            for framing in ("introspective", "guessing"):
                chat = [
                    {"role": "system", "content": SYSTEM_INTRO if framing == "introspective" else SYSTEM_GUESS},
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": ASK[(mode, framing)]
                     + (FORMAT_HINT if args.format_hint else "")},
                ]
                prompt = tok.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True,
                    **({"enable_thinking": False} if args.no_thinking else {}))
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                new_tokens = out[0].shape[0] - inputs["input_ids"].shape[1]
                # skip_special_tokens=False on purpose: it would delete the <think>
                # tags and leave the reasoning text behind with no delimiter to strip
                raw_full = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=False)
                raw = clean(raw_full)
                pred, rejected = parse(raw, args.k)
                if shown[0] < args.show_raw:
                    shown[0] += 1
                    print("\n" + "-" * 66)
                    print(f"RAW {shown[0]}  mode={mode}  framing={framing}  "
                          f"{new_tokens} new tokens")
                    print("-" * 66)
                    print(raw_full)
                    print("-" * 66)
                    print(f"parsed {len(pred)}: {pred}")
                    print(f"rejected as non-concepts: {rejected}", flush=True)
                # Did it actually learn the format? Count what it emitted BEFORE
                # truncation, and whether it produced a well-formed block. A tidy
                # score can hide a model that emits 9 concepts or never closes the
                # tag, so measure this rather than assuming.
                emitted = len(parse(raw, 999)[0])
                fmt[f"{mode}_{framing}"].append({
                    "n": emitted,
                    "exact15": emitted == args.k,
                    "opened": "<INTROSPECTION>" in raw,
                    "closed": "</INTROSPECTION>" in raw,
                    "distinct": len(set(pred)) == len(pred),
                    "truncated": new_tokens >= args.max_new_tokens,
                    "rejected": rejected,
                })
                correct = len(set(pred) & truth)
                hits = correct / args.k
                key = f"{mode}_{framing}"
                scores[key].append(hits)
                # Length-controlled companion to overlap@15. overlap divides by a
                # fixed 15, so a model that emits 8 concepts is penalised for the
                # 7 it did not say rather than for being wrong. That is not a bug --
                # a short list IS worse -- but it means overlap cannot distinguish
                # "less accurate" from "less willing to fill the list", and the two
                # framings differ in list length. Report both or the comparison is
                # about verbosity.
                precision[key].append(correct / len(pred) if pred else 0.0)
                lengths[key].append(len(pred))
                if mode == "B":
                    # for list B, words copied from the text are wrong by construction
                    leak[framing].append(sum(1 for w in pred if w in text) / max(len(pred), 1))
                rec[key] = hits
                rec[f"{key}_prec"] = precision[key][-1]
                rec[f"{key}_n"] = len(pred)
                rec[f"{key}_pred"] = pred
        records.append(rec)
        if n % max(args.progress_every, 1) == 0 or n == len(rows):
            elapsed = time.time() - started
            per_row = elapsed / n
            bar_done = int(24 * n / len(rows))
            bar = "#" * bar_done + "." * (24 - bar_done)
            print(f"  [{bar}] {n:>4}/{len(rows)} {100 * n / len(rows):>3.0f}%  "
                  f"{hms(elapsed)} elapsed, {hms(per_row * (len(rows) - n))} left  "
                  f"({per_row:.1f}s/row)  " + "  ".join(
                      f"{k}={mean(v):.3f}" for k, v in sorted(scores.items())), flush=True)

    # An all-zero table is a parse failure, not a finding. Stop rather than let it
    # be written to disk and read later as though it were a measurement.
    if all(x == 0 for v in scores.values() for x in v):
        empty = sum(1 for v in fmt.values() for x in v if x["n"] == 0)
        trunc = sum(1 for v in fmt.values() for x in v if x["truncated"])
        raise SystemExit(
            f"\nEVERY score is 0 across {len(rows) * 4} generations. That is a parse "
            f"failure, not a result.\n"
            f"  {empty} generations parsed to nothing; {trunc} hit the token limit.\n"
            "Nothing was written. Look at what the model actually emitted:\n"
            "  python training/eval_sft.py --base-only --adapter none \\\n"
            "      --limit 3 --show-raw 4 --format-hint --max-new-tokens 512")

    print("\n" + "=" * 66)
    print(f"{'target':<10}{'introspective':>16}{'guessing':>12}{'difference':>14}")
    print("=" * 66)
    for mode, label in (("A", "list A (top-15)"), ("B", "list B (novel)")):
        i, g = mean(scores[f"{mode}_introspective"]), mean(scores[f"{mode}_guessing"])
        print(f"{label:<10}{i:>16.3f}{g:>12.3f}{i - g:>+14.3f}")
    print("=" * 66)

    # overlap@15 confounds accuracy with list length. If the two framings emit
    # different numbers of concepts, the difference above is partly about verbosity.
    print(f"\n{'target':<10}{'PRECISION per item':>22}{'':>6}{'difference':>14}")
    print(f"{'':<10}{'introspective':>16}{'guessing':>12}")
    print("-" * 66)
    for mode, label in (("A", "list A (top-15)"), ("B", "list B (novel)")):
        i, g = mean(precision[f"{mode}_introspective"]), mean(precision[f"{mode}_guessing"])
        print(f"{label:<10}{i:>16.3f}{g:>12.3f}{i - g:>+20.3f}")
    print(f"\n{'mean list length':<10}"
          f"  A: intro {mean(lengths['A_introspective']):.1f} vs guess "
          f"{mean(lengths['A_guessing']):.1f}"
          f"   |   B: intro {mean(lengths['B_introspective']):.1f} vs guess "
          f"{mean(lengths['B_guessing']):.1f}")
    gap_a = abs(mean(lengths["A_introspective"]) - mean(lengths["A_guessing"]))
    gap_b = abs(mean(lengths["B_introspective"]) - mean(lengths["B_guessing"]))
    if max(gap_a, gap_b) > 1.0:
        print("  ^ the framings emit different list lengths, so the OVERLAP "
              "difference\n    above is confounded. Read the PRECISION row instead.")
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
    bad = sum(x["rejected"] for v in fmt.values() for x in v)
    if bad:
        print(f"  {bad} numbered lines were rejected as impossible concepts "
              f"(>{MAX_CONCEPT_CHARS} chars, containing a space, or markdown) -- "
              f"these are reasoning artefacts, not predictions")
    trunc = sum(1 for v in fmt.values() for x in v if x["truncated"])
    if trunc:
        print(f"  {trunc} generations hit the {args.max_new_tokens}-token limit and were cut off "
              f"-- raise --max-new-tokens")

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
        {"config": {"base_only": args.base_only, "format_hint": args.format_hint,
                    "no_thinking": args.no_thinking, "k": args.k,
                    "stratify": args.stratify,
                    "sources": sorted({source_of(r) for r in rows}),
                    "max_new_tokens": args.max_new_tokens, "limit": args.limit,
                    "adapter": None if args.base_only else str(args.adapter)},
         "summary": {k: mean(v) for k, v in scores.items()},
         "precision": {k: mean(v) for k, v in precision.items()},
         "mean_length": {k: mean(v) for k, v in lengths.items()},
         "rows": records},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
