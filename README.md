# PACT-GRPO math baseline

One shared math RL pipeline compares GRPO, Dr. GRPO, PCGrad-GRPO and
PACT-GRPO. It uses Qwen2.5-Math-1.5B, GSM8K training questions, LoRA updates
and the same answer reward for every method.

## Project structure

```text
pact_grpo_math/
├── train.py              Train one selected method
├── evaluate.py           Evaluate base model or one trained method
├── compare.py            Summarize available results
├── visualize_results.ipynb  Open benchmark plots in VS Code
├── run_verified_pilot.py Reproduce or inspect the completed 6 GB GPU pilot
├── configs/              Pilot and main experiment settings
├── docs/                 Design and related work
├── src/pact_grpo/         Data, reward, objectives, gradients and projection
├── tests/                Objective, parser and projection checks
└── outputs/              One directory per scale and method
```

In VS Code, select the existing interpreter
`C:\Users\slywi\anaconda3\envs\tf_gpu\python.exe`. The code never
installs or changes packages, CUDA or TensorFlow.

## What to run

1. Open `evaluate.py`, leave `METHOD = "base"`, then run it.
2. Open `train.py`. Set `METHOD = "grpo"`, run, then change it successively
   to `"dr_grpo"`, `"pcgrad_grpo"` and `"pact_grpo"`. Each run starts
   from the same pretrained model and saves its own LoRA adapter.
3. Open `evaluate.py`. Set `METHOD` to each trained method and run it.
4. Run `compare.py` to create `comparison.csv` and `comparison.md`.

Keep `SCALE = "pilot"` for the first pass. Change `SCALE` in all three
files to `"main"` only after the pilot is checked. For repeated seeds,
change `seed` in the config and preserve each previous output directory
before rerunning; training currently replaces that method's metrics file.

Outputs are in `outputs/<scale>/<method>/`: `adapter/`,
`train_metrics.jsonl`, `summary.json`, and `evaluation/`.

## Completed 6 GB GPU pilot

The controlled comparison is saved under `outputs/verified_pilot/`. Open
`visualize_results.ipynb` with the `tf_gpu` kernel and run its cells to display
and regenerate the figures. `outputs/verified_pilot/RESULTS.md` explains the
numbers and limitations; `comparison.csv` contains the machine-readable table.

`run_verified_pilot.py` defaults to a read-only summary. Change `MODE` in that
file to `"train"` or `"evaluate"` if rerunning the experiment. Its matching
settings are in `configs/verified_pilot.json`. The model and dataset caches
used for this local run are under
`C:\Users\slywi\Documents\Codex\2026-09-11\h-y`; set `DATN_CACHE_ROOT` if
those files have been moved. The prompt used for this run is pinned in
`run_verified_pilot.py`.

## Method boundary

All training methods use the same clipped, token-level GRPO surrogate with
two policy epochs per sampled group. Dr. GRPO removes reward-standard-deviation
normalization and uses a fixed completion-length denominator. PCGrad projects
pairwise conflicting rollout objective gradients. PACT projects their mean
onto the intersection of rollout ascent half-spaces.

PACT's sign constraint applies to the **current rollout surrogate gradients
of the LoRA parameters**. At an unclipped on-policy step, these gradients
equal advantage-signed log-probability gradients up to the shared sequence
normalization. Projection is applied before AdamW; AdamW's final parameter
displacement is not guaranteed to satisfy the same half-spaces.

The local experiment is a compact implementation of these algorithmic
mechanisms, not a reproduction of any paper's published training scale.
MATH-500 is currently scored by strict normalized answer string equality;
symbolically equivalent but differently written answers can be marked wrong.
Report its metric as strict exact match. The primary numerical benchmark is
GSM8K.
