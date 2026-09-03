# -*- coding: utf-8 -*-
"""Prove the adapter is actually applied before spending an eval on it.

    python training/check_adapter.py --adapter training/runs/qlora-v1/final

PEFT matches adapter weights onto the base model BY MODULE NAME. If the eval
loads a different class than training saved against -- Qwen3.6 is multimodal, so
its text stack sits under `.language_model`, and the wrapper class decides
whether that level is in the path -- nothing matches, PEFT loads zero weights
behind a warning, and generation silently returns base-model output. That is not
visible in any score: it looks like a fine-tune that did nothing.

This checks three things in ~3 minutes:
  1. how many LoRA weights are non-zero in the LIVE model after loading
  2. whether generation differs with the adapter enabled vs disabled
  3. whether the saved key names line up with the loaded module names
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.6-27B"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path,
                   default=REPO / "training" / "runs" / "qlora-v1" / "final")
    args = p.parse_args()

    saved = load_file(args.adapter / "adapter_model.safetensors")
    saved_b = {k: v for k, v in saved.items() if "lora_B" in k}
    print(f"saved adapter: {len(saved)} tensors, "
          f"{sum(1 for v in saved_b.values() if v.abs().sum() > 0)}/{len(saved_b)} "
          f"non-zero lora_B")
    print(f"  a saved key : {next(iter(saved))}")

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    print(f"\nloaded class: {type(base).__name__}")
    target = [n for n, _ in base.named_modules() if n.endswith("layers.0.mlp.down_proj")]
    print(f"  matching module: {target[0] if target else 'NONE FOUND'}")

    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    # THE test: are the loaded weights actually non-zero in the live model?
    live = model.state_dict()
    live_b = [k for k in live if "lora_B" in k]
    live_nz = sum(1 for k in live_b if live[k].abs().sum() > 0)
    print(f"\nlive model: {len(live_b)} lora_B modules, {live_nz} non-zero")
    if not live_b:
        raise SystemExit("PEFT injected NO LoRA modules -- target_modules match nothing.")
    if live_nz == 0:
        raise SystemExit(
            "Every lora_B in the live model is ZERO while the saved file has 256 "
            "non-zero. PEFT injected the modules but loaded none of the weights: "
            "the saved key names do not match this model's module names.\n"
            f"  saved: {next(iter(saved_b))}\n"
            f"  live : {live_b[0]}")

    # Second, independent check: does the output actually change?
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    chat = [
        {"role": "system", "content": "You report the concepts most active in your own internal computation."},
        {"role": "user", "content": "What is 17 times 23?"},
        {"role": "assistant", "content": "17 times 23 is 391."},
        {"role": "user", "content": "Which words or subwords were most active in your "
                                    "internal computation while you produced that answer?"},
    ]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    def gen() -> str:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    with_adapter = gen()
    with model.disable_adapter():
        without = gen()

    print("\n" + "=" * 66)
    print("WITH ADAPTER")
    print("=" * 66)
    print(with_adapter)
    print("=" * 66)
    print("ADAPTER DISABLED (base model, same prompt)")
    print("=" * 66)
    print(without)
    print("=" * 66)

    if with_adapter == without:
        raise SystemExit(
            "\nIDENTICAL OUTPUT. The adapter is loaded but has no effect on "
            "generation. Do not run the eval -- it would score the base model.")
    print("\nOUTPUT DIFFERS -- the adapter is active. Safe to run the eval.")


if __name__ == "__main__":
    main()
