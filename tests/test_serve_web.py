"""serve_web CLI wiring: the --mlx / --mlx-base model resolution (pure, no model load)."""
from __future__ import annotations

import serve_web


def test_resolve_mlx_model_prefers_explicit_then_base_then_none():
    # neither flag → fall through to --dev / --config
    assert serve_web.resolve_mlx_model(None, False) is None
    # --mlx-base → the base model the SAE was trained on
    assert serve_web.resolve_mlx_model(None, True) == serve_web.BASE_MLX_MODEL
    # explicit --mlx REPO is used as-is
    assert serve_web.resolve_mlx_model("org/repo", False) == "org/repo"
    # an explicit --mlx REPO wins over --mlx-base
    assert serve_web.resolve_mlx_model("org/repo", True) == "org/repo"
    # bare --mlx (argparse const) resolves to the instruct default
    assert serve_web.resolve_mlx_model(serve_web.DEFAULT_MLX_MODEL, False) == serve_web.DEFAULT_MLX_MODEL


def test_base_and_default_pairing_is_consistent():
    # The default SAE is the *base-model* SAE, so --mlx-base must point at the base model it was
    # trained on, and the instruct default must be a distinct repo.
    assert serve_web.BASE_MLX_MODEL == "Qwen/Qwen3.5-2B-Base"
    assert "Base" in serve_web.DEFAULT_MLX_SAE
    assert serve_web.BASE_MLX_MODEL != serve_web.DEFAULT_MLX_MODEL
