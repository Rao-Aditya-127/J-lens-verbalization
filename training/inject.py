# -*- coding: utf-8 -*-
"""Activation injection: change what is INSIDE the model, leave the text identical.

    python training/inject.py --adapter RaoAditya/j-lens-verbalization-qlora --rows 30
    python training/inject.py --base-only --rows 30            # the comparison

Every other experiment in this project can be answered with "the text explains
it". A bag-of-words nearest-neighbour predictor reaches 0.427 on concepts that
appear nowhere in the question or answer, so absence from the text does not mean
unavailability from the text.

This one cannot be answered that way. The prompt is byte-identical in both
conditions; the only difference is a vector added to the residual stream. A text
predictor therefore scores exactly 0 by construction -- nothing in its input
changed. If the model names the injected concept above the un-injected rate, it
read something that was not in the text.

The intervention is the `swap` from interp-engine's lens interventions, which
mirror Neuronpedia's `endpoints/lens/prompt.py` -- the same path behind the
`swapToken` used for the base-model result this compares against:

    h += (h . s_hat)(t_hat - s_hat)

replacing the component along the SOURCE concept's readout direction with the
TARGET's. Readout directions come from the fitted lens: for the transport
`resid @ J.T`, the residual-space direction for token t at a layer is
`J[layer].T @ W_U[t]`.

Applied at prefill only, so the conversation is encoded with the injection
active and the self-report is then generated normally -- matching
`steerGeneratedTokens=False`.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from statistics import mean

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

# Targets must be unrelated to anything the model is being asked about, so that
# naming one cannot be topical inference. Checked per row against the question,
# the answer and both stored readouts before use.
TARGET_POOL = [
    "hollywood", "guitar", "volcano", "pasta", "bicycle", "penguin", "jazz",
    "cathedral", "monsoon", "tractor", "origami", "saxophone", "glacier",
    "pyramid", "whiskey", "hammock", "trombone", "safari", "tulip", "canyon",
]


def find_token_id(tok, word: str) -> int | None:
    """The vocab id whose decoded form normalizes to `word`, preferring ' word'."""
    for candidate in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        ids = tok.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def readout_directions(J: dict, unembed: torch.Tensor, token_id: int,
                       layers: list[int]) -> dict[int, torch.Tensor]:
    """d[layer] = normalize(J[layer].T @ W_U[token]) -- the residual-space direction
    whose presence the lens reads out as this token."""
    u = unembed[token_id].float()
    out = {}
    for layer in layers:
        d = J[layer].float().T @ u
        out[layer] = (d / d.norm()).to(torch.bfloat16)
    return out


def install(model, layers: list[int], d_src: dict, d_tgt: dict,
            mode: str, strength: float, max_fraction: float = 0.5) -> list:
    """Register prefill-only intervention hooks on the given decoder layers.

    swap  -- h += (h . s_hat)(t_hat - s_hat)
             Replaces the source concept's readout with the target's. Faithful to
             Neuronpedia's swapToken, but its magnitude is whatever the source
             projection happens to be, which on some rows is very little.
    steer -- h += clamp(strength * ||h||) * t_hat
             An explicit dose, for when swap turns out too weak to register.
    """
    handles = []
    for layer in layers:
        block = model.get_submodule(f"model.layers.{layer}")
        src, tgt = d_src[layer], d_tgt[layer]

        def hook(_mod, _inp, out, src=src, tgt=tgt):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1:          # decode step: leave generated tokens alone
                return out
            s_, t_ = src.to(h.device, h.dtype), tgt.to(h.device, h.dtype)
            if mode == "swap":
                coef = (h * s_).sum(-1, keepdim=True)
                h = h + coef * (t_ - s_)
            else:
                norm = h.norm(dim=-1, keepdim=True)
                h = h + torch.clamp(strength * norm, max=max_fraction * norm) * t_
            return (h, *out[1:]) if isinstance(out, tuple) else h

        handles.append(block.register_forward_hook(hook))
    return handles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=str, default="RaoAditya/j-lens-verbalization-qlora")
    p.add_argument("--base-only", action="store_true", help="no adapter: the comparison arm")
    p.add_argument("--rows", type=int, default=30)
    p.add_argument("--split", default="test")
    p.add_argument("--doses", type=int, nargs="+", default=[0, 6, 12, 24, 35],
                   help="number of layers intervened on, counting up from layer 24")
    p.add_argument("--mode", choices=["swap", "steer"], default="swap",
                   help="swap mirrors Neuronpedia's swapToken; steer gives an explicit "
                        "dose when swap is too weak to register")
    p.add_argument("--strength", type=float, default=0.4,
                   help="steer only: injected norm as a fraction of the residual norm")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--collected", type=Path, default=COLLECTED)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--show", type=int, default=2, help="print this many raw generations")
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

    model = hf
    label = "base"
    if not args.base_only:
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(hf, args.adapter)
        model = peft_model.base_model.model
        label = "fine-tuned"
    model.eval()
    print(f"model: {label}  ({args.adapter if not args.base_only else MODEL_ID})")

    blob = torch.load(hf_hub_download(LENS_REPO, LENS_FILE), map_location="cpu",
                      weights_only=False)
    J = {int(k): v for k, v in blob["J"].items()}
    unembed = model.get_submodule("lm_head").weight.data
    print(f"lens loaded; unembed {tuple(unembed.shape)}\n")

    rng = random.Random(args.seed)
    trials, shown = [], 0

    for n, row in enumerate(rows, 1):
        text = (row["question"] + " " + row["answer"]).lower()
        stored = {c["concept"].strip().lower()
                  for k in ("j_lens_top10", "j_lens_top10_novel") for c in row[k]}
        # a target the model has no reason to say: not in the text, not in its readout
        pool = [w for w in TARGET_POOL if w not in text and w not in stored]
        source = row["j_lens_top10"][0]["concept"].strip().lower()
        if not pool:
            continue
        target = rng.choice(pool)
        t_id, s_id = find_token_id(tok, target), find_token_id(tok, source)
        if t_id is None or s_id is None:
            continue

        chat = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"][:3000]},
                {"role": "user", "content": ASK}]
        prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        for dose in args.doses:
            layers = list(range(LAYER_MIN, LAYER_MIN + dose))
            layers = [l for l in layers if l <= LAYER_MAX]
            handles = []
            if layers:
                d_src = readout_directions(J, unembed, s_id, layers)
                d_tgt = readout_directions(J, unembed, t_id, layers)
                handles = install(model, layers, d_src, d_tgt, args.mode, args.strength)
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
            finally:
                for h in handles:
                    h.remove()
            gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Detection is substring-in-generation, not "parsed into the list": it has
            # to work for the base model too, which does not emit the trained format.
            detected = bool(re.search(rf"\b{re.escape(target)}", gen, re.I))
            trials.append({"example_id": row["example_id"], "dose": dose,
                           "target": target, "source": source, "detected": detected})
            if shown < args.show and dose == max(args.doses):
                shown += 1
                print("-" * 70)
                print(f"{row['example_id']}  inject '{target}' over {len(layers)} layers "
                      f"(replacing '{source}')  detected={detected}")
                print(gen[:600])
                print("-" * 70, flush=True)

        if n % 5 == 0:
            rates = {d: mean(t["detected"] for t in trials if t["dose"] == d)
                     for d in args.doses}
            print(f"  {n:>3}/{len(rows)}  " +
                  "  ".join(f"{d}L={r:.0%}" for d, r in rates.items()), flush=True)

    print("\n" + "=" * 70)
    print(f"DOSE-RESPONSE  ({label}, n={len(rows)} rows)")
    print("=" * 70)
    print(f"  {'layers swapped':<18}{'detection rate':>16}")
    for dose in args.doses:
        sub = [t["detected"] for t in trials if t["dose"] == dose]
        print(f"  {dose:<18}{mean(sub):>15.0%}   ({sum(sub)}/{len(sub)})")
    print("=" * 70)
    print("\n  0 layers is the control: same prompt, no intervention. A text predictor")
    print("  scores 0 at EVERY dose, because its input never changes. Detection rising")
    print("  with dose is evidence the report tracks the activations.")

    out_path = args.out or (REPO / "training" /
                            f"inject_{'base' if args.base_only else 'ft'}.json")
    out_path.write_text(json.dumps(
        {"config": {"model": label, "rows": len(rows), "doses": args.doses,
                    "mode": args.mode, "strength": args.strength,
                    "layer_min": LAYER_MIN, "seed": args.seed},
         "rates": {str(d): mean([t["detected"] for t in trials if t["dose"] == d])
                   for d in args.doses},
         "trials": trials}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
