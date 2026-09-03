# -*- coding: utf-8 -*-
"""Show exactly how the chat template renders, before spending GPU hours on it.

    python training/check_template.py

Qwen3.6 reasons by default. That matters twice:

  * at eval, an untrained model spends its whole token budget reasoning and never
    reaches the answer (observed: 512/512 tokens, zero concepts parsed);
  * at training, if the template primes a <think> block before the assistant turn,
    the target the model learns to produce sits in a different position than the
    one eval asks it for.

Training and eval MUST render identically. This prints both so the difference is
visible rather than assumed. CPU only, no model download, runs in seconds.
"""
from __future__ import annotations

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-27B"

TRAIN_LIKE = [
    {"role": "system", "content": "SYSTEM"},
    {"role": "user", "content": "QUESTION"},
    {"role": "assistant", "content": "ANSWER"},
    {"role": "user", "content": "ASK"},
    {"role": "assistant", "content": "<INTROSPECTION>\nConcepts:\n1. county\n</INTROSPECTION>"},
]
EVAL_LIKE = TRAIN_LIKE[:-1]


def show(tok, label: str, messages, **kwargs) -> str:
    text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(repr(text))
    return text


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    train_default = show(tok, "TRAINING (full conversation, thinking default)", TRAIN_LIKE)
    try:
        train_nothink = show(tok, "TRAINING (enable_thinking=False)", TRAIN_LIKE,
                             enable_thinking=False)
    except TypeError as e:
        train_nothink = None
        print(f"\n  enable_thinking not accepted for a full conversation: {e}")

    eval_default = show(tok, "EVAL (generation prompt, thinking default)", EVAL_LIKE,
                        add_generation_prompt=True)
    eval_nothink = show(tok, "EVAL (generation prompt, enable_thinking=False)", EVAL_LIKE,
                        add_generation_prompt=True, enable_thinking=False)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  '<think>' in training render (default)  : {'<think>' in train_default}")
    if train_nothink is not None:
        print(f"  '<think>' in training render (no-think) : {'<think>' in train_nothink}")
        print(f"  enable_thinking changes training render : {train_default != train_nothink}")
    print(f"  '<think>' in eval prompt (default)      : {'<think>' in eval_default}")
    print(f"  '<think>' in eval prompt (no-think)     : {'<think>' in eval_nothink}")
    print(f"  enable_thinking changes eval prompt     : {eval_default != eval_nothink}")

    # The target must sit in the same position in both renders, or the model is
    # trained to emit it somewhere eval never looks.
    tail = train_default.split("ASK", 1)[-1]
    print(f"\n  what training puts between the ask and the target:\n    {tail[:200]!r}")

    try:
        from trl import SFTConfig
        fields = set(SFTConfig.__dataclass_fields__)
        opts = sorted(f for f in fields if "chat" in f or "template" in f)
        print(f"\n  SFTConfig template-related options: {opts or 'none'}")
    except Exception as e:  # pragma: no cover - informational only
        print(f"\n  could not inspect SFTConfig: {e}")


if __name__ == "__main__":
    main()
