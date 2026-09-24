"""On-policy groups and completion-token log probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import ANSWER_PREFIX, MathExample, format_prompt


@dataclass
class RolloutGroup:
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    texts: list[str]


@torch.no_grad()
def generate_group(model, tokenizer, example: MathExample, config: ExperimentConfig) -> RolloutGroup:
    model.eval()
    prompt = format_prompt(example.question, tokenizer)
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=config.max_prompt_tokens
    ).to(next(model.parameters()).device)
    generated = model.generate(
        **encoded,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        num_return_sequences=config.group_size,
        max_new_tokens=config.max_completion_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    prompt_length = encoded["input_ids"].shape[1]
    completion_ids = generated[:, prompt_length:]
    valid = completion_ids.ne(tokenizer.pad_token_id)
    if tokenizer.eos_token_id == tokenizer.pad_token_id:
        valid = valid.cumprod(dim=1).bool()
    completion_mask = torch.zeros_like(generated[:, 1:], dtype=torch.bool)
    completion_mask[:, prompt_length - 1:] = valid
    prompt_attention = encoded["attention_mask"].expand(config.group_size, -1)
    full_attention = torch.cat(
        [prompt_attention, completion_ids.ne(tokenizer.pad_token_id).long()], dim=1
    )
    texts = [
        ANSWER_PREFIX + suffix
        for suffix in tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    ]
    model.train()
    return RolloutGroup(
        token_ids=generated,
        attention_mask=full_attention,
        completion_mask=completion_mask,
        texts=texts,
    )


def completion_logprobs(model, group: RolloutGroup) -> torch.Tensor:
    """Return log p(next token) aligned with group.completion_mask."""
    logits = model(
        input_ids=group.token_ids,
        attention_mask=group.attention_mask,
        use_cache=False,
    ).logits[:, :-1]
    targets = group.token_ids[:, 1:]
    batch, length = targets.shape
    negative_log_likelihood = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    )
    return -negative_log_likelihood.reshape(batch, length)
