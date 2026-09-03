# -*- coding: utf-8 -*-
"""Rewrite adapter keys so PEFT can match them onto Qwen3_5ForCausalLM.

    python training/fix_adapter_keys.py --adapter training/runs/qlora-v1/final

Qwen3.6 is multimodal. Loaded through the multimodal wrapper -- which is what TRL
did during training -- the text stack sits at `model.language_model.layers`;
loaded as Qwen3_5ForCausalLM, which is what eval_sft.py does, it sits at
`model.layers`. PEFT matches adapter weights by module name, so the saved keys
match nothing, every LoRA module loads as zero, and generation returns
base-model output behind a warning that no score can reveal.

Both paths address the same 64 text layers, so stripping the `language_model.`
segment is a rename, not a reinterpretation. The original is never modified: the
rewritten adapter is written to a new directory, and check_adapter.py verifies it
actually takes effect before any eval is run.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]
SEGMENT = "language_model."


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path,
                   default=REPO / "training" / "runs" / "qlora-v1" / "final")
    p.add_argument("--out", type=Path, default=None,
                   help="default: <adapter>_fixed, alongside the original")
    args = p.parse_args()
    out = args.out or args.adapter.with_name(args.adapter.name + "_fixed")

    src = args.adapter / "adapter_model.safetensors"
    tensors = load_file(src)
    renamed = {k.replace(SEGMENT, "", 1): v for k, v in tensors.items()}

    changed = sum(1 for k in tensors if SEGMENT in k)
    print(f"{len(tensors)} tensors, {changed} contained '{SEGMENT}'")
    if not changed:
        raise SystemExit(
            f"No key contains '{SEGMENT}', so this is not the mismatch this script "
            "fixes. Run check_adapter.py and compare the saved and live key names.")
    if len(renamed) != len(tensors):
        raise SystemExit(
            f"Renaming collapsed {len(tensors)} keys into {len(renamed)}: two "
            "different modules would map onto the same name. Stopping rather than "
            "silently dropping trained weights.")

    nonzero = sum(1 for k, v in renamed.items() if "lora_B" in k and v.abs().sum() > 0)
    total_b = sum(1 for k in renamed if "lora_B" in k)
    print(f"non-zero lora_B after rename: {nonzero}/{total_b}")
    if nonzero == 0:
        raise SystemExit("Every lora_B is zero -- this adapter carries no training.")

    out.mkdir(parents=True, exist_ok=True)
    # metadata format=pt is what peft looks for when it reads the file back
    save_file(renamed, out / "adapter_model.safetensors", metadata={"format": "pt"})
    for name in ("adapter_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "chat_template.jinja", "vocab.json",
                 "merges.txt", "added_tokens.json", "log_history.json"):
        source = args.adapter / name
        if source.exists():
            shutil.copy(source, out / name)

    print(f"\nbefore: {next(iter(tensors))}")
    print(f"after : {next(iter(renamed))}")
    print(f"\nwrote {out}")
    print("\nVERIFY before evaluating -- a rename that matches nothing looks "
          "exactly like one that works:")
    print(f"  python training/check_adapter.py --adapter {out}")


if __name__ == "__main__":
    main()
