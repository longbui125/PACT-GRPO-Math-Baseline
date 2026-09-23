# Related methods

- DeepSeekMath introduces GRPO for mathematical reasoning.
- Dr. GRPO removes length and group-standard-deviation normalization biases.
- PCGrad projects pairwise-conflicting task gradients.
- GEM projects updates onto constraints protecting prior tasks; its QP is the
  closest mathematical predecessor of PACT's projection.
- CAGrad optimizes a worst local improvement objective near the mean gradient.
- DaGRPO addresses GRPO gradient conflict using sample distinctiveness masks and
  off-policy anchors. PACT instead retains rollouts and constrains the aggregate
  update according to each rollout's advantage sign.

Published scores must not be compared directly with local scores when model,
data and compute differ. The primary table is a controlled local comparison.

