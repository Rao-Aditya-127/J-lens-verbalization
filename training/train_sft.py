# -*- coding: utf-8 -*-
"""QLoRA SFT: teach Qwen3.6-27B to report its own J-lens concepts.

    # smoke test first -- catches template/masking bugs in ~10 minutes
    python training/train_sft.py --smoke

    # full run
    python training/train_sft.py

Loss is computed ONLY on the final assistant turn (the concept list). Everything
before it is context and is masked, otherwise most of the gradient goes into
re-learning the question and answer the model was already given.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "training" / "data"

MODEL_ID = "Qwen/Qwen3.6-27B"
# Qwen3.6 is multimodal (image-text-to-text). Loading the CausalLM class reads only
# the text_config sub-block, so the ~0.5B vision tower is never materialised.
# Small saving (~1 GB bf16) but it also avoids surprises in the forward pass.
TEXT_ONLY_CLASS = "Qwen3_5ForCausalLM"


def load_split(name: str) -> Dataset:
    path = DATA / f"sft_{name}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return Dataset.from_list([{"messages": r["messages"]} for r in rows])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="200 train examples, 1 epoch, tiny eval")
    p.add_argument("--output-dir", type=Path, default=REPO / "training" / "runs" / "qlora-v1")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    # Qwen3.6's vocabulary is 248,320 tokens, so the logits tensor is ~2 GB at
    # batch 4 and doubles in the backward pass. Liger's fused cross-entropy never
    # materialises it. Off by default because it silently requires liger-kernel to
    # support this architecture -- turn it on if the run OOMs.
    p.add_argument("--liger", action="store_true",
                   help="fused cross-entropy via liger-kernel; cuts activation memory")
    p.add_argument("--track", choices=["auto", "wandb", "tensorboard", "none"], default="auto",
                   help="auto = wandb if WANDB_API_KEY is set, else none")
    p.add_argument("--run-name", default=None, help="name shown in the tracker")
    args = p.parse_args()

    # A rented pod is ephemeral: anything written locally dies with it. Cloud
    # tracking survives, which is why wandb is the default when a key is present.
    if args.track == "auto":
        args.track = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if args.track == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "jlens-verbalization")
    run_name = args.run_name or args.output_dir.name
    print(f"tracking: {args.track}" + (f"  (project jlens-verbalization, run {run_name})"
                                       if args.track == "wandb" else ""))

    train = load_split("train")
    val = load_split("validation")
    if args.smoke:
        train, val = train.select(range(200)), val.select(range(32))
        args.epochs, args.output_dir = 1.0, args.output_dir.with_name("smoke")
    print(f"train {len(train)} | validation {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit base weights. nf4 + double quant is the standard QLoRA recipe; compute
    # in bf16 because the adapters and activations stay full-precision.
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        use_liger_kernel=args.liger,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_len,
        # Mask everything but the final assistant turn. Targets are only ~14% of
        # each sequence, so without this most of the gradient would go into
        # re-predicting the question and answer already present in the context.
        completion_only_loss=True,
        packing=False,          # packing would blur the completion mask across examples
        report_to=[] if args.track == "none" else [args.track],
        run_name=run_name,
        # trainer_state.json records every logged metric regardless of tracker,
        # so the curves are recoverable even if tracking was off or the pod died.
        logging_first_step=True,
        seed=17,
    )

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=cfg,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        peft_config=peft_config,
        model_init_kwargs={
            "quantization_config": quant,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "sdpa",
        },
    )

    # Print one fully-rendered example with its label mask before training. This is
    # the single most valuable check: a chat-template or masking bug silently wastes
    # the whole run, and it is invisible in the loss curve.
    batch = next(iter(trainer.get_train_dataloader()))
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    kept = [i for i, l in enumerate(labels.tolist()) if l != -100]
    print("\n" + "=" * 70)
    print("MASKING CHECK -- only this text should contribute to the loss:")
    print("=" * 70)
    print(tokenizer.decode([ids[i] for i in kept]))
    print("=" * 70)
    print(f"{len(kept)} of {len(ids)} tokens are loss-bearing "
          f"({100 * len(kept) / len(ids):.0f}%) -- expect roughly 10-20%")
    if not kept:
        raise SystemExit("no loss-bearing tokens: the completion mask is wrong, stopping")
    print("=" * 70 + "\n")

    trainer.train()
    final = args.output_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    # Keep the metric history beside the adapter. This is the fallback that
    # survives a dead pod even if no tracker was configured -- plot_training.py
    # renders the curves from it.
    history = final / "log_history.json"
    history.write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")

    losses = [h for h in trainer.state.log_history if "loss" in h]
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    print(f"\nadapter saved to  {final}")
    print(f"metric history    {history}  ({len(losses)} train points, {len(evals)} eval points)")
    if losses:
        print(f"train loss  first {losses[0]['loss']:.4f}  ->  last {losses[-1]['loss']:.4f}")
    if evals:
        best = min(evals, key=lambda h: h["eval_loss"])
        print(f"eval  loss  first {evals[0]['eval_loss']:.4f}  ->  last {evals[-1]['eval_loss']:.4f}"
              f"   best {best['eval_loss']:.4f} at step {best.get('step')}")
        if evals[-1]["eval_loss"] > best["eval_loss"] * 1.02:
            print("  NOTE: eval loss ended above its best -- overfitting; a shorter run may score better")
    print("\nDOWNLOAD the adapter and log_history.json before terminating the pod --")
    print("local files do not survive it.")


if __name__ == "__main__":
    main()
