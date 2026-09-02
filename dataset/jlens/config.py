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
#
# Window locked to the workspace band. Gurnee et al. (2026), "Verbalizable
# Representations Form a Global Workspace in Language Models", identify three
# functional regions on layers reindexed to 0-100: sensory (~0-38),
# workspace (~38-92), and motor (~92-100), where motor-band J-lens readouts
# are simply next-token predictions rather than abstract workspace content.
# Mapped onto 64 layers that is roughly 24-58. The previous 22-63 window
# included five motor layers; measured effect of excluding them: 8% of the
# top-10 changes, and the share of target concepts already present in the
# prompt/answer text drops 63.1% -> 58.2%.
LAYER_MIN = 24
LAYER_MAX = 58

TOP_N_API = 8  # API ceiling for `topN` is 8, not 10.
TOP_K_CONCEPTS = 15  # both lists are top-15: more novel content per row (41% vs 37%)
NOVEL_SEARCH_DEPTH = 120  # how deep to look when collecting 15 novel concepts
NOVEL_RULE = "absent_as_substring_of_lowercased_question_plus_answer"  # both lists are top-15: more novel content per row (41% vs 37%)

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

RATE_LIMIT_MIN_INTERVAL_SECONDS = 16.0  # per WORKER. With 4 shards and ~50s
# API latency this is ~78s/row each = ~184 calls/hour combined, safely under the
# 240/hour cap. At 16s the four workers collectively exceeded the cap, and the
# flat backoff resynchronised them into a wait-retry-fail cycle: throughput
# collapsed to 12 rows/hour. Pacing below the limit beats colliding with it.
MAX_RETRIES = 5
# A 429 needs wall-clock waiting, not fast retries: the limit is a sliding
# 60-minute window, so requests have to age out of it. The shared
# exponential schedule tops out at ~15 min total, which is not enough.
# The window (60 min, 240 req) releases ~4 slots/minute. A 300s sleep means
# workers snooze through their own replenishment: measured throughput fell to
# 57/hour against a 240/hour allowance. Retry sooner, with jitter so workers
# do not resynchronise, and let the pacing keep the average legal.
# Retries are REQUESTS: a rejected call still counts against the 240/hour
# window. Four workers retrying every 45s made ~320 attempts/hour on their own
# and locked us out permanently (2 rows/hour for 9 hours). Budget TOTAL attempts:
# 3 workers x 16s pacing = ~162 successes/hr, plus ~60/hr of retries = ~222 < 240.
# 60s, not 180s. Measured 2026-09-01: with 180s the three workers slept through
# a window that was 95% free (X-Limit-Remaining=229 of 240) and threw away two
# hours at ~18 rows/hour. A 10-retry chain at 180s is 30 min asleep on ONE row.
# 3 workers x 60s = at most 180 retry attempts/hr, which the measured headroom
# absorbs. Read X-Limit-Remaining before changing this again -- if it is high,
# the workers are idling, not throttled.
RATE_LIMIT_BACKOFF_SECONDS = 60.0
RATE_LIMIT_MAX_RETRIES = 10
# Stop an unattended run if this many rows fail back-to-back: that means
# something systemic (dead key, outage), not an unlucky row.
MAX_CONSECUTIVE_FAILURES = 15
RETRY_BACKOFF_BASE_SECONDS = 30.0
# urllib.request.urlopen has no timeout by default (blocks forever). A real
# collection run hit this: one ordinary request stalled and the whole process
# sat "running" with zero progress for ~53 minutes until manually killed.
REQUEST_TIMEOUT_SECONDS = 90.0

PROMPT_BANK_PATH = Path(__file__).resolve().parents[1] / "prompt_bank" / (
    "prompt_bank_gsm8k_bbh_truthfulqa_arc_hotpotqa_3800.jsonl"
)
JLENS_DIR = Path(__file__).resolve().parent
RAW_DIR = JLENS_DIR / "raw"
COLLECTED_ANSWERS_PATH = JLENS_DIR / "collected_answers.jsonl"
COLLECTED_INTROSPECTION_PATH = JLENS_DIR / "collected_introspection.jsonl"

API_URL = "https://www.neuronpedia.org/api/lens/prompt"
