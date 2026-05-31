import pytest
import torch

from qwen_scope_lab.hooks import apply_steering_to_hidden


def test_strength_zero_equivalent_to_unsteered():
    hidden = torch.randn(1, 3, 4)
    vector = torch.randn(4)
    steered, norm = apply_steering_to_hidden(hidden, vector, 0.0)
    assert torch.allclose(steered, hidden)
    assert norm == 0.0


def test_positive_and_negative_strengths_move_opposite_directions():
    hidden = torch.zeros(1, 1, 4)
    vector = torch.tensor([1.0, -2.0, 0.5, 0.0])
    pos, pos_norm = apply_steering_to_hidden(hidden, vector, 3.0)
    neg, neg_norm = apply_steering_to_hidden(hidden, vector, -3.0)
    assert torch.allclose(pos, -neg)
    assert pos_norm == pytest.approx(neg_norm)
    assert pos_norm > 0
