# -*- coding: utf-8 -*-
"""Can interp-engine run the J-lens on OUR fine-tuned model? Answer in ~5 minutes.

    pip install interp-engine
    python training/probe_interp_engine.py --adapter training/runs/qlora-v1/final_fixed

`load_model()` takes a model *id*, and nothing in the package mentions lora or
peft -- so the documented entry point cannot reach an adapter. But
`EagerModel.__init__` accepts `hf_model: nn.Module`, and PEFT injects its LoRA
layers IN PLACE inside the base model. So the inner module carries the adapter
while keeping the original dotted paths that interp-engine resolves hook points
against. Handing over the PeftModel wrapper instead would shift every path by
two levels (`base_model.model.…`) -- the same class of mismatch that silently
zeroed the adapter once already.

This probe answers, in order:
  1. is the adapter live in the module we hand over?
  2. does EagerModel accept it and resolve points?
  3. what shape is the published lens checkpoint, so the readout can be wired up?

It does NOT run a lens readout yet -- that needs the transport wired to the
checkpoint's actual layout, which step 3 reports.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.6-27B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path,
                   default=REPO / "training" / "runs" / "qlora-v1" / "final_fixed")
    p.add_argument("--skip-lens", action="store_true", help="do not download the lens")
    args = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=" * 70)
    print("1. LOAD BASE + ADAPTER")
    print("=" * 70)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    peft_model = PeftModel.from_pretrained(base, str(args.adapter))
    peft_model.eval()

    # PEFT wraps as PeftModel -> base_model (LoraModel) -> model (the real CausalLM).
    # The LoRA layers live inside that innermost tree, so it is both adapted AND
    # named the way interp-engine expects.
    inner = peft_model.base_model.model
    print(f"  PeftModel      : {type(peft_model).__name__}")
    print(f"  inner module   : {type(inner).__name__}")

    live = {n: m for n, m in inner.named_modules() if n.endswith("lora_B.default")}
    nonzero = sum(1 for m in live.values() if m.weight.abs().sum() > 0)
    print(f"  lora_B modules in the inner tree: {len(live)}, non-zero: {nonzero}")
    if not live or nonzero == 0:
        raise SystemExit(
            "The inner module carries no live adapter weights. Handing this to "
            "interp-engine would silently read out the BASE model.")
    print(f"  sample path    : {next(iter(live))}")

    print("\n" + "=" * 70)
    print("2. HAND IT TO interp-engine")
    print("=" * 70)
    from interp_engine import EagerModel

    model = EagerModel(MODEL_ID, hf_model=inner, tokenizer=tok)
    print(f"  n_layers {model.n_layers} | d_model {model.d_model} | "
          f"vocab {model.vocab_size}")
    print(f"  device {model.device} | dtype {model.dtype} | "
          f"quant_method {model.quant_method}")
    print(f"  hooks_available: {model.hooks_available}")
    if str(model.device) == "cpu":
        print("\n  WARNING: the model is on CPU. A lens readout there is far too slow to")
        print("  be useful. Almost always torch lost its CUDA build -- check")
        print("  torch.cuda.is_available() and that torch.version.cuda matches the driver.")

    # The point the lens reads from. If this resolves, the paths line up.
    for layer in (0, 24, 58, model.n_layers - 1):
        try:
            mod, side = model.resolve_point("resid_post", layer)
            mark = "OK  "
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            mod, side, mark = type(exc).__name__, str(exc)[:60], "FAIL"
        print(f"  {mark} resid_post layer {layer:>2}: "
              f"{getattr(mod, '__class__', type(mod)).__name__} / {side}")

    # `points` is a method on some versions and a property on others
    pts = model.points() if callable(model.points) else model.points
    print("\n  points advertised:", [str(s) for s in list(pts)[:8]], "...")

    if args.skip_lens:
        return

    print("\n" + "=" * 70)
    print("3. WHAT IS IN THE PUBLISHED LENS CHECKPOINT")
    print("=" * 70)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LENS_REPO, LENS_FILE)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  file  : {path}")
    print(f"  type  : {type(blob).__name__}")
    if isinstance(blob, dict):
        for k, v in list(blob.items())[:12]:
            desc = (f"Tensor{tuple(v.shape)} {v.dtype}" if torch.is_tensor(v)
                    else f"{type(v).__name__} {str(v)[:70]}")
            print(f"    {k:<32} {desc}")
        if len(blob) > 12:
            print(f"    ... {len(blob) - 12} more keys")
    else:
        print("   ", {a: getattr(blob, a) for a in dir(blob)
                      if not a.startswith('_') and not callable(getattr(blob, a))})

    print("\nIf every point resolved and the adapter is live, the readout can be")
    print("wired to layer_logits(model, tokens, {'jacobian_lens': list(range(24, 59))},")
    print("transport=...) once the checkpoint layout above is known.")


if __name__ == "__main__":
    main()
