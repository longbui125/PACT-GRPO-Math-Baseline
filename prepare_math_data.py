"""Optional VS Code entry point to regenerate the pinned MATH-Algebra SFT file."""

import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

from datasets import load_dataset
from pact_grpo.config import ExperimentConfig
from pact_grpo.data import format_math_prompt
from pact_grpo.modeling import load_tokenizer

DATASET_ID = "HuggingFaceH4/MATH"
REVISION = "9bbe1fc38f097a38e3bdf5bbec8d6feee21318c9"
MAX_SAMPLES = 600
MAX_TOTAL_TOKENS = 256


def last_boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    opening = start + len(r"\boxed{")
    depth = 1
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening:index].strip()
    return None


def prepare() -> None:
    config = ExperimentConfig.from_json(ROOT / "configs" / "radical.json")
    tokenizer = load_tokenizer(config.model_name)
    source = load_dataset(DATASET_ID, "algebra", split="train", revision=REVISION)
    eligible = []
    for index, item in enumerate(source):
        answer = last_boxed(item["solution"])
        if answer is None or not re.fullmatch(r"-?\d+", answer):
            continue
        prompt = format_math_prompt(item["problem"], tokenizer)
        response = f"{answer}}}\nExplanation: {item['solution']}" + tokenizer.eos_token
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        response_tokens = len(tokenizer(response, add_special_tokens=False).input_ids)
        if prompt_tokens > config.max_prompt_tokens or (
            prompt_tokens + response_tokens > MAX_TOTAL_TOKENS
        ):
            continue
        eligible.append({
            "source_index": index, "problem": item["problem"],
            "answer": answer, "solution": item["solution"],
            "prompt_tokens": prompt_tokens, "response_tokens": response_tokens,
            "radical_related": r"\sqrt" in item["problem"],
        })
    rng = random.Random(config.seed)
    related = [row for row in eligible if row["radical_related"]]
    other = [row for row in eligible if not row["radical_related"]]
    rng.shuffle(related)
    rng.shuffle(other)
    chosen = (related + other)[:MAX_SAMPLES]
    if len(chosen) < MAX_SAMPLES:
        raise RuntimeError(f"Only {len(chosen)} eligible MATH training examples")
    rng.shuffle(chosen)
    destination = ROOT / "data" / "math_algebra_sft.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": DATASET_ID, "subset": "algebra", "split": "train",
        "revision": REVISION, "selected": len(chosen),
        "radical_related": sum(row["radical_related"] for row in chosen),
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "selection": "integer boxed answer and token budget; radical-containing examples prioritized",
        "seed": config.seed,
    }
    (destination.parent / "math_algebra_sft_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(manifest)


if __name__ == "__main__":
    prepare()
