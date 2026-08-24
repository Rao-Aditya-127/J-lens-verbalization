"""Introspection prompt rendering and response parsing.

Each Q/A/introspection unit is split across alternating user/assistant turns
rather than packed into one user message, because the API caps every USER
message at 1024 chars but ASSISTANT messages at 10000 -- and a generated
answer can easily exceed 1024 chars on its own. Assistant turns are meant for
exactly this ("replayed back through this endpoint for re-analysis"), per
route_prompt.ts's own comment.

    user:      <QUESTION>...</QUESTION>
    assistant: <ANSWER>...</ANSWER>                       (replayed)
    user:      <INTROSPECTION_QUESTION>...</INTROSPECTION_QUESTION> + format
    assistant: <INTROSPECTION>...</INTROSPECTION>          (omitted for the
                                                             held-out row)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INTROSPECTIVE_SYSTEM_PROMPT = (
    "You are answering a task and then reporting selected information about your current internal state."
)

TEXT_ONLY_SYSTEM_PROMPT = (
    "You are answering a task. Afterward you will be asked to guess, from the question and answer text "
    "alone, which words or subwords a language model would likely have been internally processing while "
    "producing that answer. You do not have -- and should not claim to have -- any introspective access "
    "to your own internal computation for this task; just make your best guess from the text."
)

INTROSPECTIVE_QUESTION_TEXT = (
    "What are the ten most active words or subwords that you were thinking about while answering the "
    "question?"
)

TEXT_ONLY_QUESTION_TEXT = (
    "Based only on the question and answer text above (not on any introspective access to your own "
    "processing), guess 10 words or subwords a language model would likely have been internally "
    "processing while producing this answer."
)

FORMAT_INSTRUCTION = """Respond using exactly this format:

<INTROSPECTION>
Concepts:
1. <first concept>
2. <second concept>
3. <third concept>
4. <fourth concept>
5. <fifth concept>
6. <sixth concept>
7. <seventh concept>
8. <eighth concept>
9. <ninth concept>
10. <tenth concept>

Explanation:
<brief explanation of what these concepts represent>
</INTROSPECTION>"""

# Hand-written per demo rather than a generic template: a real explanation of
# why those specific concepts appeared teaches the model something about
# grounding; a content-free boilerplate sentence repeated for every demo does
# not. Grounded in the actual answer text and j_lens_top10 counts for these
# rows (see decision_log.md for the full selection rationale). Kept short
# deliberately -- these explanations are fixed overhead on every few_shot_icl
# call and count against the model's 2048-token total-conversation limit.
DEMO_EXPLANATIONS: dict[str, str] = {
    "truthfulqa_0010": (
        "Splits into the popular belief being corrected (july, declaration, congress) and the real "
        "signing timeline (later, officially, signing, date, dates). 'September' was active internally "
        "even though the final answer says August -- a concept can be active without being stated."
    ),
    "hotpotqa_0013": (
        "Splits into the question's framing (president, presidential, secretary, department, health) "
        "and the identified answer (nixon). 'Eisenhower' was active without being the final answer -- "
        "a concept can be active without being the stated answer."
    ),
}


@dataclass(frozen=True)
class Demo:
    question: str
    answer: str
    concepts: list[str]
    explanation: str


# Demo answers are truncated to a fixed cap so the fixed few-shot overhead is
# predictable regardless of which specific rows are chosen as demos -- the
# model's total-conversation budget (2048 tokens, discovered when the 3-demo
# prompt started 400ing) leaves limited room, and a held-out row's own
# question+answer can itself be long. The truncation only shortens what is
# REPLAYED in the ICL prompt; it does not change the j_lens_top10 label, which
# is always computed from the demo's full, untruncated real generation.
MAX_DEMO_ANSWER_CHARS = 300


def demo_from_answer_row(row: dict) -> Demo:
    concepts = [item["concept"] for item in row["j_lens_top10"]]
    explanation = DEMO_EXPLANATIONS.get(row["example_id"])
    if explanation is None:
        raise ValueError(f"No grounded explanation authored for demo row {row['example_id']!r}")
    answer = row["answer"]
    if len(answer) > MAX_DEMO_ANSWER_CHARS:
        answer = answer[:MAX_DEMO_ANSWER_CHARS].rstrip() + " [...]"
    return Demo(question=row["question"], answer=answer, concepts=concepts, explanation=explanation)


def render_answer_chat(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}]


def _qa_introspection_turns(
    question: str,
    answer: str,
    introspection_question_text: str,
    include_format_instruction: bool = True,
) -> list[dict[str, str]]:
    # Demo turns skip the format spec: the demo's own worked <INTROSPECTION>
    # answer immediately after already shows the format, so repeating the
    # ~340-char instruction block in every demo's ask turn is pure redundancy
    # that eats into the 2048-token conversation budget for no benefit. Only
    # the held-out row's ask turn -- the one actually being generated -- needs
    # the explicit spec.
    ask_content = f"<INTROSPECTION_QUESTION>\n{introspection_question_text}\n</INTROSPECTION_QUESTION>"
    if include_format_instruction:
        ask_content += f"\n\n{FORMAT_INSTRUCTION}"
    return [
        {"role": "user", "content": f"<QUESTION>\n{question}\n</QUESTION>"},
        {"role": "assistant", "content": f"<ANSWER>\n{answer}\n</ANSWER>"},
        {"role": "user", "content": ask_content},
    ]


def _demo_introspection_answer(demo: Demo) -> dict[str, str]:
    numbered = "\n".join(f"{i}. {concept}" for i, concept in enumerate(demo.concepts, start=1))
    content = f"<INTROSPECTION>\nConcepts:\n{numbered}\n\nExplanation:\n{demo.explanation}\n</INTROSPECTION>"
    return {"role": "assistant", "content": content}


def render_introspection_chat(
    question: str,
    answer: str,
    condition: str,
    demos: list[Demo] | None = None,
) -> list[dict[str, str]]:
    """Build the chat turns for one introspection call.

    `condition` is one of "zero_shot", "few_shot_icl", "text_only_control".
    `demos` is only used (and required) for "few_shot_icl"; demonstrations
    always use the introspective framing regardless of the held-out row's
    condition, since they demonstrate genuine grounded reporting.
    """
    if condition == "text_only_control":
        system_prompt = TEXT_ONLY_SYSTEM_PROMPT
        question_text = TEXT_ONLY_QUESTION_TEXT
    else:
        system_prompt = INTROSPECTIVE_SYSTEM_PROMPT
        question_text = INTROSPECTIVE_QUESTION_TEXT

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if condition == "few_shot_icl":
        if not demos:
            raise ValueError("few_shot_icl requires at least one demo")
        for demo in demos:
            messages.extend(
                _qa_introspection_turns(
                    demo.question, demo.answer, INTROSPECTIVE_QUESTION_TEXT, include_format_instruction=False
                )
            )
            messages.append(_demo_introspection_answer(demo))

    messages.extend(_qa_introspection_turns(question, answer, question_text, include_format_instruction=True))
    return messages


_INTROSPECTION_BLOCK_RE = re.compile(r"<INTROSPECTION>(.*?)</INTROSPECTION>", re.DOTALL | re.IGNORECASE)
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")
_EXPLANATION_SPLIT_RE = re.compile(r"(?im)^\s*explanation\s*:?\s*$")
_EXPLANATION_CAPTURE_RE = re.compile(r"(?is)explanation\s*:?\s*\n(.*)")


def parse_introspection_response(text: str) -> dict[str, object]:
    """Parse a model's introspection completion into predicted concepts + explanation.

    Tolerant of missing tags/markers (falls back to scanning the whole text)
    since this is real model output, not something we control the shape of.
    Does not assume exactly 10 valid items; understanding.md's validation
    rule is to record however many were actually parseable.
    """
    block_match = _INTROSPECTION_BLOCK_RE.search(text)
    block = block_match.group(1) if block_match else text

    concepts_part = _EXPLANATION_SPLIT_RE.split(block)[0]
    explanation_match = _EXPLANATION_CAPTURE_RE.search(block)
    explanation = explanation_match.group(1).strip() if explanation_match else None

    concepts = []
    for line in concepts_part.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            concepts.append(match.group(1).strip())

    top10 = concepts[:10]
    return {
        "predicted_top10": top10,
        "explanation": explanation,
        "valid_count": len(top10),
        "raw": text,
    }
