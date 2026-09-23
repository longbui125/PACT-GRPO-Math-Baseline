from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from datasets import load_dataset

SYSTEM_PROMPT = (
    "Solve the mathematics problem carefully. Show concise reasoning and end "
    "with exactly one final answer in the form \\boxed{answer}."
)


@dataclass(frozen=True)
class MathExample:
    uid: str
    question: str
    answer: str
    source: str


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    try:
        number = Decimal(cleaned)
        result = format(number.normalize(), "f")
        return result.rstrip("0").rstrip(".") if "." in result else result
    except InvalidOperation:
        return cleaned.replace("\\left", "").replace("\\right", "").replace(" ", "").lower() or None


def _last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    cursor, depth = start + len(marker), 1
    content_start = cursor
    while cursor < len(text):
        depth += int(text[cursor] == "{")
        depth -= int(text[cursor] == "}")
        if depth == 0:
            return text[content_start:cursor]
        cursor += 1
    return None


def extract_final_answer(text: str) -> str | None:
    boxed = _last_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)
    marked = re.findall(r"####\s*([^\n]+)", text)
    if marked:
        return normalize_answer(marked[-1])
    explicit = re.findall(r"(?:final answer|answer is|answer)\s*[:=]?\s*(-?[\d,]+(?:\.\d+)?)", text, re.I)
    if explicit:
        return normalize_answer(explicit[-1])
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return normalize_answer(numbers[-1]) if numbers else None


def format_prompt(question: str, tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\nUser: {question}\nAssistant:"


def load_gsm8k(split: str, limit: int | None, seed: int) -> list[MathExample]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None and limit < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(limit))
    rows = []
    for index, row in enumerate(dataset):
        answer = extract_final_answer(row["answer"])
        if answer is None:
            raise ValueError(f"Cannot parse GSM8K answer at row {index}")
        rows.append(MathExample(f"gsm8k-{split}-{index}", row["question"], answer, "gsm8k"))
    return rows


def load_math500(limit: int | None, seed: int) -> list[MathExample]:
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit is not None and limit < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(limit))
    rows = []
    for index, row in enumerate(dataset):
        answer = normalize_answer(str(row.get("answer"))) if row.get("answer") is not None else extract_final_answer(str(row.get("solution", "")))
        if answer is None:
            raise ValueError(f"Cannot parse MATH-500 answer at row {index}")
        question = row.get("problem") or row.get("question")
        rows.append(MathExample(f"math500-{index}", str(question), answer, "math500"))
    return rows

