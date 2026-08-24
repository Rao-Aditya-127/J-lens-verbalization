"""Locked Phase-0 configuration for the J-lens collection pipeline.

See think/understanding.md ("Phase 0: lock the experimental configuration")
and the approved plan in this session. Every value here is written into each
collected row's generation_config / j_lens_config so the run stays
reproducible if any of these are changed later.
"""

from __future__ import annotations

from pathlib import Path

MODEL_ID = "qwen3.6-27b"
LENS_TYPE = "JACOBIAN_LENS"

# Confirmed live: qwen3.6-27b exposes JACOBIAN_LENS layers 0-63.
LAYER_MIN = 22
LAYER_MAX = 63

TOP_N_API = 8  # API ceiling for `topN` is 8, not 10.
TOP_K_CONCEPTS = 10

TEMPERATURE = 0
FILTER_NON_WORD_TOKENS = True

ANSWER_NUM_COMPLETION_TOKENS = 256
INTROSPECTION_NUM_COMPLETION_TOKENS = 200

ANSWER_PROMPT_VERSION = "answer-v1"
INTROSPECTION_PROMPT_VERSION = "introspection-v2"  # v2: hand-written per-demo grounded explanations, not generic boilerplate

# Fixed few-shot ICL demonstrations. Deliberately not GSM8K: arithmetic
# answers risk teaching the model "just list numbers", the shortcut-agreement
# failure mode understanding.md warns about. Real answer + j_lens_top10 for
# these rows are collected once (like every other row) and then reused
# unchanged as the ICL demonstrations for every few-shot-condition call in
# the run.
#
# The original 2 demos (ARC, BBH date_understanding) were both multiple-choice
# questions whose answers spend most of their text evaluating and ruling out
# wrong options -- "incorrect" was consequently the #1 or #2 concept in BOTH,
# teaching the model to reflexively predict "incorrect" regardless of the
# held-out question, which showed up as an over-represented prediction across
# the collected data. This is a structural property of the evaluate-and-reject
# MCQ answer pattern, not a property of those two specific rows, so the fix
# was to move to genuinely free-response sources rather than pick different
# MCQ rows:
#   - truthfulqa_0010: misconception-correction (popular July 4 belief vs. the
#     actual August 2 signing date) -- directly on-theme with the project's
#     motivation, and its j_lens_top10 includes "september", a plausible date
#     that never appears in the answer text -- a concrete example of a
#     concept being internally active without being stated in the output.
#   - hotpotqa_0013: multi-hop bridge retrieval, with concrete named-entity
#     grounding (the real answer "nixon" plus "eisenhower" as an honest
#     example of a plausible-but-not-correct neighboring concept).
#
# A 3rd demo (bbh_causal_judgement, philosophical/causal reasoning) was tried
# for extra diversity but dropped: adding it pushed some held-out rows over
# the model's total-conversation limit (2048 tokens -- a separate, harder cap
# than the per-message character limits, discovered when this first 400'd).
# The held-out row's own question+answer can't be shortened without
# invalidating the comparison (its j_lens_top10 was measured from the FULL
# untruncated generation), so the only lever was demo overhead, and even after
# trimming it (shorter row, tighter MAX_DEMO_ANSWER_CHARS, no repeated format
# instruction per demo -- prompts.py) it still wasn't reliably safe across all
# 223 eval rows. 2 demos fit comfortably and already solve the "incorrect"
# bias; a 3rd can be revisited later against a model/deployment with more
# context budget.
DEMO_PROMPT_IDS = ["truthfulqa_0010", "hotpotqa_0013"]

INTROSPECTION_CONDITIONS = ["zero_shot", "few_shot_icl", "text_only_control"]

RATE_LIMIT_MIN_INTERVAL_SECONDS = 16.0  # ~225 calls/hour, under the 240/hour cap.
MAX_RETRIES = 5
RETRY_BACKOFF_BASE_SECONDS = 30.0
# urllib.request.urlopen has no timeout by default (blocks forever). A real
# collection run hit this: one ordinary request stalled and the whole process
# sat "running" with zero progress for ~53 minutes until manually killed.
REQUEST_TIMEOUT_SECONDS = 90.0

PROMPT_BANK_PATH = Path(__file__).resolve().parents[1] / "prompt_bank" / (
    "prompt_bank_gsm8k_bbh_truthfulqa_arc_hotpotqa_225.jsonl"
)
JLENS_DIR = Path(__file__).resolve().parent
RAW_DIR = JLENS_DIR / "raw"
COLLECTED_ANSWERS_PATH = JLENS_DIR / "collected_answers.jsonl"
COLLECTED_INTROSPECTION_PATH = JLENS_DIR / "collected_introspection.jsonl"

API_URL = "https://www.neuronpedia.org/api/lens/prompt"
