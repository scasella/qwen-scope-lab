import torch

from qwen_scope_lab.sae_math import compute_pre_activations, topk_features


def test_topk_activations_expected_indices_and_values():
    residual = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    w_enc = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 2.4],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    b_enc = torch.tensor([0.0, 0.5, 0.0, 0.0, -10.0])
    pre = compute_pre_activations(residual, w_enc, b_enc)
    assert torch.allclose(pre, torch.tensor([[1.0, 2.5, 1.0, 1.2, -7.5]]))

    vals, idx = topk_features(residual, w_enc, b_enc, top_k=2)
    assert idx.tolist() == [[1, 3]]
    assert torch.allclose(vals, torch.tensor([[2.5, 1.2]]))
