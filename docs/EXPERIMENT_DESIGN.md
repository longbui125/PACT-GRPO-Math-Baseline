# Controlled comparison

All training runs start from the same model and use the same shuffled GSM8K
questions, prompt, decoding, reward, LoRA configuration, optimizer and number
of generated rollouts. Only the update rule changes.

| Method | Update rule |
|---|---|
| GRPO | Mean rollout surrogate gradient |
| Dr. GRPO | GRPO with fixed length denominator and no group-std normalization |
| PCGrad-GRPO | Pairwise gradient surgery on rollout surrogate gradients |
| PACT-GRPO | Closest mean gradient satisfying each rollout ascent half-space |

Evaluation uses greedy generation on held-out GSM8K and MATH-500. A valid
claim of improvement needs higher test accuracy or better stability over
multiple seeds. A lower RSVR alone shows only that the projection changed
the local gradient geometry. Compare training time and update norm ratio
alongside accuracy, because projection may reduce useful movement.

The optimizer receives the projected raw LoRA gradient. AdamW then applies
its own momentum and preconditioner. The code logs the raw-direction RSVR.
It does not claim zero violation in the actual parameter displacement.

First run the pilot to verify model loading, reward diversity, valid answer
parsing and nonzero update norms. Run the main configuration and repeated
seeds for final numbers. Do not compare local scores directly to published
scores obtained with other models or datasets.
