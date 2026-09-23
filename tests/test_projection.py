import torch

from pact_grpo.projections import project_halfspaces


def test_projection_satisfies_constraints():
    direction = torch.tensor([-1.0, -2.0])
    constraints = torch.eye(2)
    result = project_halfspaces(direction, constraints, 1e-8, 500, 1e-8)
    assert torch.all(constraints @ result.direction >= -1e-5)
    assert result.applied


def test_no_projection_when_feasible():
    direction = torch.tensor([1.0, 2.0])
    result = project_halfspaces(direction, torch.eye(2), 1e-8, 100, 1e-8)
    assert not result.applied
    assert torch.equal(result.direction, direction)

