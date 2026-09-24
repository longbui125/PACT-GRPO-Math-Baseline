"""Minimal prompt representation for radical-equation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ANSWER_PREFIX = r"\boxed{"
SYSTEM_PROMPT = (
    "Check the supplied candidate roots in the original radical equation. "
    "The reply starts inside a box: write only the valid integer roots separated "
    "by commas, or none, then close the brace. "
    "No text is needed before the answer."
)
MATH_SYSTEM_PROMPT = (
    "Solve the math problem. The reply starts inside a box: write the final "
    "answer and close the brace. You may explain afterward."
)


@dataclass(frozen=True)
class MathExample:
    uid: str
    question: str
    answer: str
    source: str


def _format_chat(question: str, tokenizer: Any, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
    return f"System: {system_prompt}\nUser: {question}\nAssistant: {ANSWER_PREFIX}"


def format_prompt(question: str, tokenizer: Any) -> str:
    return _format_chat(question, tokenizer, SYSTEM_PROMPT)


def format_math_prompt(question: str, tokenizer: Any) -> str:
    return _format_chat(question, tokenizer, MATH_SYSTEM_PROMPT)
