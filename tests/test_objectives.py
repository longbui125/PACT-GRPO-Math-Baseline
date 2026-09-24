import torch

from pact_grpo.objectives import group_advantages, rollout_surrogates


def test_clipping_stops_over_increasing_a_positive_rollout():
    old = torch.zeros((2, 1))
    current = torch.log(torch.tensor([[1.5], [0.5]])).requires_grad_()
    mask = torch.ones_like(old)
    advantage = torch.tensor([1.0, -1.0])
    values = rollout_surrogates(
        current, old, mask, advantage,
        clip_epsilon=0.2, dr_grpo=False, max_completion_tokens=1
    )
    gradients = torch.autograd.grad(values.sum(), current)[0]
    assert torch.allclose(gradients, torch.zeros_like(gradients))


def test_dr_grpo_uses_fixed_length_denominator():
    old = torch.zeros((2, 2))
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    advantage = torch.tensor([1.0, 1.0])
    values = rollout_surrogates(
        old, old, mask, advantage,
        clip_epsilon=0.2, dr_grpo=True, max_completion_tokens=2
    )
    assert torch.allclose(values, torch.tensor([0.5, 1.0]))


def test_group_advantages_zero_for_identical_rewards():
    rewards = torch.ones(4)
    assert torch.equal(group_advantages(rewards, normalize_std=True), torch.zeros(4))


def test_clip_higher_keeps_positive_gradient_between_bounds():
    old = torch.zeros((1, 1))
    current = torch.log(torch.tensor([[1.24]])).requires_grad_()
    mask = torch.ones_like(old)
    values = rollout_surrogates(
        current, old, mask, torch.ones(1), clip_epsilon=0.2,
        clip_epsilon_high=0.28, dr_grpo=False, max_completion_tokens=1,
    )
    gradient = torch.autograd.grad(values.sum(), current)[0]
    assert gradient.item() > 0


def test_dr_grpo_uses_unscaled_group_advantage():
    rewards = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert torch.allclose(
        group_advantages(rewards, normalize_std=False),
        torch.tensor([-0.5, 0.5, 0.5, -0.5]),
    )
