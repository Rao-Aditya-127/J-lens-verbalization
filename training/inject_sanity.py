# -*- coding: utf-8 -*-
"""Does the injection do anything at all? Check the lens before trusting the model.

    python training/inject_sanity.py --rows 3

inject.py returned 0% detection at every dose, including 35 layers, and the
generations came back completely normal -- coherent, on-topic, undisturbed. Two
very different things produce that:

  A. the intervention works and the model cannot report it
  B. the intervention is too weak to change anything

Only A is a result. Distinguishing them needs a readout that does not depend on
the model's willingness to say anything, and the lens is exactly that: inject
along the target concept's direction, then read the lens. If the target does not
climb the lens readout, the intervention is not landing and no detection rate
means anything.

Also reports the perturbation size. A unit direction in 5120 dimensions collects
a projection of roughly ||h||/71 from an arbitrary residual, so `swap` -- whose
magnitude is exactly that projection -- may be moving the residual by about 1%.
`steer` sets the size explicitly and is the fallback if so.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from jlens_calibrate import (  # noqa: E402
    COLLECTED, LENS_FILE, LENS_REPO, MODEL_ID, readout,
)
from inject import (  # noqa: E402
    LAYER_MIN, LAYER_MAX, TARGET_POOL, find_token_id, install, readout_directions,
)


def measure_perturbation(model, layers, d_src, d_tgt, ids, mode, strength):
    """Record ||delta|| / ||h|| at one layer, so 'no effect' can be told from 'no report'."""
    stats = {}
    probe = layers[len(layers) // 2]
    block = model.get_submodule(f"model.layers.{probe}")
    src, tgt = d_src[probe], d_tgt[probe]

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1:
            return out
        s_, t_ = src.to(h.device, h.dtype), tgt.to(h.device, h.dtype)
        edit = h[:, 1:]
        if mode == "swap":
            delta = (edit * s_).sum(-1, keepdim=True) * (t_ - s_)
        else:
            n = edit.norm(dim=-1, keepdim=True)
            delta = torch.clamp(strength * n, max=0.5 * n) * t_
        stats["layer"] = probe
        stats["ratio"] = float((delta.norm(dim=-1) / edit.norm(dim=-1)).mean())
        return out

    handle = block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--mode", choices=["swap", "steer"], default="swap")
    p.add_argument("--strengths", type=float, nargs="+", default=[0.2, 0.4, 0.8])
    p.add_argument("--collected", type=Path, default=COLLECTED)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from interp_engine import EagerModel

    rows = [json.loads(l) for l in args.collected.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["split"] == "test"][: args.rows]

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf.eval()
    model = EagerModel(MODEL_ID, hf_model=hf, tokenizer=tok)

    blob = torch.load(hf_hub_download(LENS_REPO, LENS_FILE), map_location="cpu",
                      weights_only=False)
    J = {int(k): v for k, v in blob["J"].items()}
    unembed = hf.get_submodule("lm_head").weight.data
    layers = list(range(LAYER_MIN, LAYER_MAX + 1))
    rng = random.Random(args.seed)

    for row in rows:
        text = (row["question"] + " " + row["answer"]).lower()
        stored = {c["concept"].strip().lower()
                  for k in ("j_lens_top10", "j_lens_top10_novel") for c in row[k]}
        pool = [w for w in TARGET_POOL if w not in text and w not in stored]
        source = row["j_lens_top10"][0]["concept"].strip().lower()
        t_id, s_id = find_token_id(tok, pool[0]), find_token_id(tok, source)
        if t_id is None or s_id is None:
            continue
        target = pool[0]

        prompt = tok.apply_chat_template(
            [{"role": "user", "content": row["question"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = torch.cat([
            tok(prompt, return_tensors="pt")["input_ids"][0],
            tok(row["answer"], add_special_tokens=False, return_tensors="pt")["input_ids"][0],
        ]).unsqueeze(0).to(model.device)

        d_src = readout_directions(J, unembed, s_id, layers)
        d_tgt = readout_directions(J, unembed, t_id, layers)

        print("=" * 74)
        print(f"{row['example_id']}   inject '{target}'  (replacing '{source}')")
        print("=" * 74)
        base = readout(model, tok, J, prompt, row["answer"], True)
        print(f"  no injection      : {', '.join(base[:8])}")
        print(f"  '{target}' in list: {target in base}")

        settings = ([("swap", None)] if args.mode == "swap"
                    else [("steer", s) for s in args.strengths])
        if args.mode == "swap":
            settings += [("steer", s) for s in args.strengths]

        for mode, strength in settings:
            handles = install(model.hf_model if hasattr(model, "hf_model") else hf,
                              layers, d_src, d_tgt, mode, strength or 0.0)
            try:
                stats = measure_perturbation(hf, layers, d_src, d_tgt, ids,
                                             mode, strength or 0.0)
                got = readout(model, tok, J, prompt, row["answer"], True)
            finally:
                for h in handles:
                    h.remove()
            rank = got.index(target) + 1 if target in got else None
            tag = mode if strength is None else f"{mode} {strength}"
            print(f"  {tag:<14} perturbation {stats.get('ratio', float('nan')):.3f} of ||h||"
                  f"   '{target}' rank: {rank if rank else '-- not in top 15'}")
            if rank:
                print(f"                 readout: {', '.join(got[:8])}")
        print()

    print("=" * 74)
    print("If the target never enters the lens readout at any strength, the")
    print("intervention is not landing and no detection rate is interpretable.")
    print("If it does enter but the model still never says it, THAT is the result:")
    print("the concept is demonstrably present and demonstrably not reported.")


if __name__ == "__main__":
    main()
