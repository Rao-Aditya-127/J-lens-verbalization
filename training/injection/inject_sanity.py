# -*- coding: utf-8 -*-
"""Does the injection actually reach the workspace? Ask the lens, not the model.

    python training/injection/inject_sanity.py --rows 3

Run this before trusting any number from `inject.py`. A flat dose-response has
two very different causes:

  A. the intervention works and the model cannot report it
  B. the intervention is too weak to change anything

Only A is a result. The lens settles it without depending on the model saying
anything: inject along the target concept's readout direction, then read the
lens at the answer positions. If the target does not climb the readout, nothing
is landing and no detection rate downstream means anything.

This is how the original 0% was diagnosed. The `swap` intervention perturbs the
residual by only 0.2% of ‖h‖ -- yet that was enough to put an unrelated concept
at **rank 1** of the lens readout while leaving the row's genuine concepts in
place. So the concept was reaching the workspace and the model simply never said
it, which is a finding rather than a bug.

It also exposes a limitation worth carrying into any conclusion: the lens is far
more sensitive to a nudge along a specific direction than the model's own
downstream computation is. Rank in the lens is not the same as functional
importance in the forward pass.
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

# parents[2]: this file sits at training/<group>/, so the repo root is two
# levels up. Every data and output path below is relative to it.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training" / "lens"))
sys.path.insert(0, str(REPO / "training" / "injection"))
from jlens_calibrate import (  # noqa: E402
    COLLECTED, LENS_FILE, LENS_REPO, MODEL_ID, readout,
)
from inject import (  # noqa: E402
    LAYER_MIN, LAYER_MAX, TARGET_POOL, find_token_ids, install, readout_direction,
)

DEPTH = 300          # how far down the readout to look before calling it "absent"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.10, 0.20])
    p.add_argument("--adapter", type=str, default=None,
                   help="omit for the base model")
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
    inner, label = hf, "base"
    if args.adapter:
        from peft import PeftModel
        inner = PeftModel.from_pretrained(hf, args.adapter).base_model.model
        label = "fine-tuned"
    inner.eval()
    model = EagerModel(MODEL_ID, hf_model=inner, tokenizer=tok)

    blob = torch.load(hf_hub_download(LENS_REPO, LENS_FILE), map_location="cpu",
                      weights_only=False)
    J = {int(k): v for k, v in blob["J"].items()}
    unembed = inner.get_submodule("lm_head").weight.data
    layers = list(range(LAYER_MIN, LAYER_MAX + 1))
    print(f"model: {label}   layers {LAYER_MIN}-{LAYER_MAX}\n")

    rng = random.Random(args.seed)
    single = [w for w in TARGET_POOL if find_token_ids(tok, w)]

    for row in rows:
        text = (row["question"] + " " + row["answer"]).lower()
        stored = {c["concept"].strip().lower()
                  for k in ("j_lens_top10", "j_lens_top10_novel") for c in row[k]}
        pool = [w for w in single if w not in text and w not in stored]
        if not pool:
            continue
        target = rng.choice(pool)
        t_id = find_token_ids(tok, target)[0]

        prompt = tok.apply_chat_template(
            [{"role": "user", "content": row["question"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

        print("=" * 74)
        print(f"{row['example_id']}   injecting '{target}'")
        print("=" * 74)
        print(f"  {'strength':<12}{'rank in the lens readout':>26}   top of the readout")

        for strength in args.strengths:
            # Span (1, huge) = every position except 0, whose attention-sink norm
            # would take a disproportionate share of a norm-scaled injection.
            handles = (install(model.hf_model if hasattr(model, "hf_model") else inner,
                               layers, J, unembed, t_id, strength, (1, 10 ** 9))
                       if strength else [])
            try:
                got = readout(model, tok, J, prompt, row["answer"], True, top_k=DEPTH)
            finally:
                for h in handles:
                    h.remove()
            where = (str(got.index(target) + 1) if target in got
                     else f"not in top {DEPTH}")
            print(f"  {strength:<12}{where:>26}   {', '.join(got[:6])}")
        print()

    print("=" * 74)
    print("If the target never enters the readout at any strength, the intervention")
    print("is not landing and no detection rate from inject.py is interpretable.")
    print("If it does enter but the model still never names it, THAT is the result:")
    print("the concept is demonstrably present and demonstrably not reported.")


if __name__ == "__main__":
    main()
