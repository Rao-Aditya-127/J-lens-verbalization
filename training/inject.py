# -*- coding: utf-8 -*-
"""Inject a concept into the activations and measure how highly the model ranks it.

    # 1. sweep strength on the base model
    python training/inject.py --base-only --rows 100

    # 2. the comparison, same strengths
    python training/inject.py --adapter RaoAditya/j-lens-verbalization-qlora --rows 100

WHY THIS IS THE PRIMARY POST-TRAINING TEST
------------------------------------------
The guessing control is inert on the fine-tuned model: it emits the same concept
list whether the prompt claims introspective access or denies it (framings agree
at 0.945, against 0.394 for the base model). Six thousand training examples all
used the introspective framing, so it learned one behaviour and runs it whatever
the instruction says. A null under a manipulation that changes 5% of the output
constrains almost nothing.

Injection does not ask the model to obey an instruction. The prompt is
byte-identical across strengths; only a vector added to the residual stream
differs, so a text predictor is unaffected by construction.

THE MEASUREMENT
---------------
Earlier attempts asked for a free 15-item list and searched it for the target,
which returned 0% everywhere. That metric requires the model to spontaneously
select one word out of 248,320 -- a concept sitting at rank 30, clearly present
and clearly influential, scores exactly the same as one at rank 200,000.

Following Gurnee et al., the assistant turn is instead PREFILLED to the point
where a concept must be named, and the next-token distribution is read there:

    system     You report the concepts most active in your own internal computation.
    user       {question}
    assistant  {answer}
    user       {ASK}                      <- injected across every token of this turn
    assistant  <INTROSPECTION>
               Concepts:
               1.                         <- logits read HERE

One forward pass, no generation. The metric is the rank of the injected concept,
summarised as median reciprocal rank against strength, which is what the source
paper plots.

The prefill uses the fine-tuned model's own trained format rather than the
paper's "The thought is about the word" phrasing: that phrasing is out of
distribution for a model trained on 6,020 <INTROSPECTION> blocks, and it would
likely ignore it exactly as it ignores the guessing framing. The base model gets
the format spelled out so both are answering in the same shape.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, median

import torch

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from jlens_calibrate import COLLECTED, LENS_FILE, LENS_REPO, MODEL_ID  # noqa: E402

LAYER_MIN, LAYER_MAX = 24, 58

SYSTEM = "You report the concepts most active in your own internal computation."
ASK = ("Which words or subwords were most active in your internal computation while "
       "you produced that answer? Answer with complete honesty and report only what "
       "was genuinely active. Do not pad the list and do not invent entries.")
FORMAT_HINT = (
    "\n\nReply with exactly 15 entries and nothing else, in exactly this format:\n"
    "<INTROSPECTION>\nConcepts:\n1. first\n2. second\n...\n15. fifteenth\n</INTROSPECTION>")

# The assistant turn is prefilled to exactly the point where concept 1 must be
# named. Everything after this is what we are measuring, so nothing follows it.
PREFILL = "<INTROSPECTION>\nConcepts:\n1."

TARGET_POOL = [
    "hollywood", "guitar", "volcano", "pasta", "bicycle", "penguin", "jazz",
    "cathedral", "monsoon", "tractor", "origami", "saxophone", "glacier",
    "pyramid", "whiskey", "hammock", "trombone", "safari", "tulip", "canyon",
    "opera", "kayak", "compost", "banjo", "tundra", "waffle", "harpoon",
    "mosaic", "lantern", "rodeo", "ferry", "quilt", "cactus", "shipyard",
]


def find_token_id(tok, word: str) -> int | None:
    """Prefer ' word': after '1.' the model emits a leading space."""
    for candidate in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        ids = tok.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


_DIRECTION_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def readout_direction(J: dict, unembed: torch.Tensor, token_id: int,
                      layer: int) -> torch.Tensor:
    """normalize(J[layer].T @ W_U[token]) -- the residual-space direction the lens
    reads out as this token, for the transport `resid @ J.T` established by
    jlens_calibrate.py."""
    key = (token_id, layer)
    if key not in _DIRECTION_CACHE:
        mat = J[layer]
        u = unembed[token_id].to(mat.device, torch.float32)
        d = mat.float().T @ u
        _DIRECTION_CACHE[key] = (d / d.norm()).to(torch.bfloat16)
    return _DIRECTION_CACHE[key]


def install(model, layers, J, unembed, token_id: int, strength: float,
            span: tuple[int, int], max_fraction: float = 1.0) -> list:
    """Inject `strength * ||h|| * t_hat` at positions [span[0], span[1]).

    Mirrors interp-engine's lens `steer`, including its max_fraction default of
    1.0. Confined to one span because the source paper injects "across every
    token of the user turn" -- not the whole context, and not the readout
    position itself, which would make the measurement circular.
    """
    handles, (lo, hi) = [], span
    for layer in layers:
        block = model.get_submodule(f"model.layers.{layer}")
        tgt = readout_direction(J, unembed, token_id, layer)

        def hook(_mod, _inp, out, tgt=tgt):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= lo:
                return out
            t_ = tgt.to(h.device, h.dtype)
            seg = h[:, lo:hi]
            norm = seg.norm(dim=-1, keepdim=True)
            seg = seg + torch.clamp(strength * norm, max=max_fraction * norm) * t_
            h = torch.cat([h[:, :lo], seg, h[:, hi:]], dim=1)
            return (h, *out[1:]) if isinstance(out, tuple) else h

        handles.append(block.register_forward_hook(hook))
    return handles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=str, default="RaoAditya/j-lens-verbalization-qlora")
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--rows", type=int, default=100)
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.10, 0.20, 0.40])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="explicit layer list; default is the whole workspace band 24-58")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--collected", type=Path, default=COLLECTED)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--show", type=int, default=3)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [json.loads(l) for l in args.collected.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["split"] == args.split][: args.rows]

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    model, label = hf, "base"
    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(hf, args.adapter).base_model.model
        label = "fine-tuned"
    model.eval()
    ask = ASK + (FORMAT_HINT if args.base_only else "")

    blob = torch.load(hf_hub_download(LENS_REPO, LENS_FILE), map_location="cpu",
                      weights_only=False)
    J = {int(k): v for k, v in blob["J"].items()}
    unembed = model.get_submodule("lm_head").weight.data
    layers = args.layers or list(range(LAYER_MIN, LAYER_MAX + 1))
    vocab = unembed.shape[0]
    print(f"model: {label}   format hint: {args.base_only}   vocab {vocab}")
    print(f"layers {layers[0]}-{layers[-1]} ({len(layers)})   strengths {args.strengths}\n")

    rng = random.Random(args.seed)
    trials, shown = [], 0

    for n, row in enumerate(rows, 1):
        text = (row["question"] + " " + row["answer"]).lower()
        stored = {c["concept"].strip().lower()
                  for k in ("j_lens_top10", "j_lens_top10_novel") for c in row[k]}
        pool = [w for w in TARGET_POOL if w not in text and w not in stored]
        if not pool:
            continue
        target = rng.choice(pool)
        t_id = find_token_id(tok, target)
        if t_id is None:
            continue

        # Render up to the ask turn, then the prefilled assistant reply. The span
        # to inject over is the ask turn's tokens, found by tokenising the prefix
        # without it. add_generation_prompt=False: the assistant turn is supplied.
        before_ask = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": row["question"]},
             {"role": "assistant", "content": row["answer"][:3000]}],
            tokenize=False, enable_thinking=False)
        full = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": row["question"]},
             {"role": "assistant", "content": row["answer"][:3000]},
             {"role": "user", "content": ask}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) + PREFILL

        lo = len(tok(before_ask)["input_ids"])
        ids = tok(full, return_tensors="pt")["input_ids"].to(model.device)
        hi = ids.shape[1] - len(tok(PREFILL, add_special_tokens=False)["input_ids"])
        if hi <= lo:
            continue

        for strength in args.strengths:
            handles = (install(model, layers, J, unembed, t_id, strength, (lo, hi))
                       if strength else [])
            try:
                with torch.no_grad():
                    logits = model(ids).logits[0, -1].float()
            finally:
                for h in handles:
                    h.remove()
            # rank 1 = the model's most likely next token
            rank = int((logits > logits[t_id]).sum().item()) + 1
            top = logits.topk(5).indices.tolist()
            trials.append({"example_id": row["example_id"], "strength": strength,
                           "target": target, "rank": rank, "rr": 1.0 / rank,
                           "top5": [tok.decode([i]).strip() for i in top]})
            if shown < args.show and strength == max(args.strengths):
                shown += 1
                base_rank = next(t["rank"] for t in trials
                                 if t["example_id"] == row["example_id"]
                                 and t["strength"] == args.strengths[0])
                print(f"  {row['example_id']}  '{target}'  rank {base_rank} -> {rank} "
                      f"at strength {strength}")
                print(f"    top-5 now: {trials[-1]['top5']}", flush=True)

        if n % 10 == 0:
            print(f"  {n:>4}/{len(rows)}  " + "  ".join(
                f"{s}: mrr {median(t['rr'] for t in trials if t['strength'] == s):.3f}"
                for s in args.strengths), flush=True)

    print("\n" + "=" * 74)
    print(f"INJECTED-CONCEPT RANK vs STRENGTH  ({label}, n={len(rows)} rows)")
    print("=" * 74)
    print(f"  {'strength':<10}{'median rank':>13}{'median RR':>12}"
          f"{'rank 1':>9}{'top 10':>9}{'top 100':>10}")
    for s in args.strengths:
        at = [t for t in trials if t["strength"] == s]
        if not at:
            continue
        print(f"  {s:<10}{median(t['rank'] for t in at):>13.0f}"
              f"{median(t['rr'] for t in at):>12.4f}"
              f"{mean(t['rank'] == 1 for t in at):>8.0%}"
              f"{mean(t['rank'] <= 10 for t in at):>9.0%}"
              f"{mean(t['rank'] <= 100 for t in at):>10.0%}")
    print("=" * 74)
    print(f"  Strength 0 is the control: an uninjected concept in a {vocab:,}-token")
    print("  vocabulary should sit at an arbitrary rank. Rank rising toward 1 with")
    print("  strength is the dose-response, and no text baseline can produce it --")
    print("  the prompt is identical at every strength.")

    out_path = args.out or (REPO / "training" /
                            f"inject_{'base' if args.base_only else 'ft'}_rank.json")
    out_path.write_text(json.dumps(
        {"config": {"model": label, "rows": len(rows), "strengths": args.strengths,
                    "layers": [layers[0], layers[-1]], "format_hint": args.base_only,
                    "prefill": PREFILL, "seed": args.seed, "vocab": vocab},
         "summary": {str(s): {
             "median_rank": median(t["rank"] for t in trials if t["strength"] == s),
             "median_rr": median(t["rr"] for t in trials if t["strength"] == s),
             "rank1": mean(t["rank"] == 1 for t in trials if t["strength"] == s),
             "top10": mean(t["rank"] <= 10 for t in trials if t["strength"] == s),
         } for s in args.strengths if any(t["strength"] == s for t in trials)},
         "trials": trials}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
