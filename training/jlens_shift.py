# -*- coding: utf-8 -*-
"""Did fine-tuning move the model's J-space?

    python training/jlens_shift.py --adapter RaoAditya/j-lens-verbalization-qlora --rows 50

This is a validity check on the entire evaluation, not a side question.

Every score reported for the fine-tuned model was computed against J-lens
readouts collected from the BASE model, through Neuronpedia, before training
existed. That is only the fine-tuned model's internal state if training left the
J-space alone. If two epochs of LoRA moved it, those labels describe a model that
no longer exists -- a perfectly introspective fine-tuned model would score badly
against them, and the only way to score well would be to predict from the text.
The null would then be an artefact of a stale target rather than a finding.

Method: hold the input text fixed -- same question, same stored answer, so the
same token positions -- and toggle only the adapter. PEFT's disable_adapter()
does that on one loaded model, so base and fine-tuned readouts come from the same
weights, the same code path and the same GPU, differing in nothing else.

Read jlens_calibrate.py first: the local readout reproduces the API at 0.837
overlap@15, with disagreement concentrated in low-count entries (missed concepts
average rank 12.0 of 15). That figure is the noise floor here -- two readouts of
the SAME model through this path would not agree perfectly either, so it is
printed alongside for comparison rather than left implicit.
"""
from __future__ import annotations

import argparse
import json
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

from jlens_calibrate import (  # noqa: E402
    COLLECTED, LENS_FILE, LENS_REPO, MODEL_ID, TOP_K, readout,
)

TRANSPOSE = True      # resid @ J.T -- established by jlens_calibrate.py (0.837 vs 0.000)
THINKING = False      # ditto: 0.837 vs 0.793


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=str, default="RaoAditya/j-lens-verbalization-qlora")
    p.add_argument("--rows", type=int, default=50)
    p.add_argument("--split", default="test", help="which split to draw from")
    p.add_argument("--collected", type=Path, default=COLLECTED)
    p.add_argument("--out", type=Path, default=REPO / "training" / "jlens_shift.json")
    args = p.parse_args()

    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from interp_engine import EagerModel

    rows = [json.loads(l) for l in args.collected.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["split"] == args.split][: args.rows]
    print(f"{len(rows)} rows from split={args.split}")

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    peft_model = PeftModel.from_pretrained(hf, args.adapter)
    peft_model.eval()
    model = EagerModel(MODEL_ID, hf_model=peft_model.base_model.model, tokenizer=tok)
    print(f"adapter {args.adapter} on {model.device}")

    live = [m for n, m in peft_model.base_model.model.named_modules()
            if n.endswith("lora_B.default")]
    if not any(m.weight.abs().sum() > 0 for m in live):
        raise SystemExit("adapter weights are all zero -- both conditions would be the base model")
    print(f"adapter live: {len(live)} lora_B modules\n")

    blob = torch.load(hf_hub_download(LENS_REPO, LENS_FILE), map_location="cpu",
                      weights_only=False)
    J = {int(k): v for k, v in blob["J"].items()}

    records = []
    for n, row in enumerate(rows, 1):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": row["question"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=THINKING)

        ft = readout(model, tok, J, prompt, row["answer"], TRANSPOSE)
        with peft_model.disable_adapter():
            base = readout(model, tok, J, prompt, row["answer"], TRANSPOSE)

        api = [c["concept"].strip().lower() for c in row["j_lens_top10"]]
        rec = {
            "example_id": row["example_id"],
            "ft_vs_base": len(set(ft) & set(base)) / TOP_K,
            "base_vs_api": len(set(base) & set(api)) / TOP_K,
            "ft_vs_api": len(set(ft) & set(api)) / TOP_K,
            "ft_vs_base_top5": len(set(ft[:5]) & set(base[:5])) / 5,
            "ft": ft, "base": base,
        }
        records.append(rec)
        if n % 5 == 0 or n == len(rows):
            print(f"  {n:>3}/{len(rows)}  ft~base={mean(r['ft_vs_base'] for r in records):.3f}"
                  f"  base~api={mean(r['base_vs_api'] for r in records):.3f}"
                  f"  ft~api={mean(r['ft_vs_api'] for r in records):.3f}", flush=True)

    ft_base = mean(r["ft_vs_base"] for r in records)
    base_api = mean(r["base_vs_api"] for r in records)
    ft_api = mean(r["ft_vs_api"] for r in records)

    print("\n" + "=" * 70)
    print("DID FINE-TUNING MOVE THE J-SPACE?")
    print("=" * 70)
    print(f"  fine-tuned readout vs base readout   {ft_base:.3f}   (top-5: "
          f"{mean(r['ft_vs_base_top5'] for r in records):.3f})")
    print(f"  base readout vs Neuronpedia          {base_api:.3f}   <- the noise floor")
    print(f"  fine-tuned readout vs Neuronpedia    {ft_api:.3f}")
    print("=" * 70)

    # The base-vs-API figure is what this path scores when NOTHING has changed but
    # the implementation, so it is the ceiling any two-readout comparison can reach.
    if ft_base >= base_api - 0.03:
        print("\n  The adapter moved the J-space no more than the local readout differs")
        print("  from the API. The collected labels remain a fair description of the")
        print("  fine-tuned model's internal state, and the evaluation stands as run.")
    elif ft_base > base_api - 0.15:
        print("\n  A modest shift, above the noise floor. Worth stating: the labels are")
        print("  a slightly stale description of the fine-tuned model's state, which")
        print("  costs a genuinely introspective model some score.")
    else:
        print("\n  A LARGE shift. The collected labels no longer describe the fine-tuned")
        print("  model's internal state, so introspection could not score well against")
        print("  them even in principle, and text prediction is the only route left.")
        print("  The introspective-vs-guessing null cannot be read as evidence about")
        print("  introspection until this is accounted for.")

    worst = min(records, key=lambda r: r["ft_vs_base"])
    print(f"\nmost-changed row: {worst['example_id']} (overlap {worst['ft_vs_base']:.2f})")
    print("  base:", ", ".join(worst["base"]))
    print("  ft  :", ", ".join(worst["ft"]))

    args.out.write_text(json.dumps(
        {"config": {"adapter": args.adapter, "rows": len(rows), "split": args.split,
                    "transpose": TRANSPOSE, "thinking": THINKING},
         "summary": {"ft_vs_base": ft_base, "base_vs_api": base_api, "ft_vs_api": ft_api},
         "rows": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
