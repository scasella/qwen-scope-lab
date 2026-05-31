"""Serve the Lab web app over the FastAPI backend.

Dev (CPU, no downloads):   python serve_web.py --dev
Local on Apple Silicon:    python serve_web.py --mlx          (the full lab on-device; no Modal/CUDA)
Real 2B (needs CUDA):      python serve_web.py --config configs/qwen35_2b_dev_l0_100.yaml
Real 27B (needs CUDA):     python serve_web.py --config configs/qwen35_27b_l0_100.yaml
"""
from __future__ import annotations

import argparse

import uvicorn

from qwen_scope_lab.web_api import create_app

# The default on-device pairing for `--mlx` — bf16 (not 4-bit) for activation fidelity. Bare
# `python serve_web.py --mlx` runs the full lab (this model + its Qwen-Scope SAE) with no other args.
DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"
DEFAULT_MLX_SAE = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100"
DEFAULT_MLX_D_SAE = 32768
DEFAULT_MLX_LAYER = 12


def build(config: str | None, dev: bool, recipes: str, mlx: str | None = None,
          mlx_layer: int = DEFAULT_MLX_LAYER, mlx_sae: str | None = DEFAULT_MLX_SAE,
          mlx_d_sae: int = DEFAULT_MLX_D_SAE):
    if mlx:
        # `--mlx-sae none` (or empty) → capture-only mode: probes / jailbreak / steering / manifold
        # run without the SAE-feature path (no SAE download).
        sae = None if (mlx_sae or "").strip().lower() in ("", "none", "mlx://none") else mlx_sae
        from qwen_scope_lab.mlx_backend import build_mlx_service

        service = build_mlx_service(mlx, default_layer=mlx_layer, sae_repo=sae,
                                    d_sae=mlx_d_sae if sae else 0)
    elif dev or not config:
        from qwen_scope_lab.dev_backend import build_dev_service

        service = build_dev_service()
    else:
        from qwen_scope_lab.service import SteeringService

        service = SteeringService.from_config_path(config)
    return create_app(service, recipes_root=recipes, experiments_root="experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to a real SteeringConfig YAML (needs CUDA)")
    parser.add_argument("--dev", action="store_true", help="use the GPU-free dev backend")
    parser.add_argument("--mlx", nargs="?", const=DEFAULT_MLX_MODEL, default=None, metavar="REPO",
                        help="run the real model locally on Apple Silicon via MLX — no Modal/CUDA. "
                             f"Bare --mlx uses the default pairing ({DEFAULT_MLX_MODEL} + its SAE); "
                             "pass a REPO to override.")
    parser.add_argument("--mlx-layer", type=int, default=DEFAULT_MLX_LAYER, help="probe/capture layer for --mlx")
    parser.add_argument("--mlx-sae", default=DEFAULT_MLX_SAE, metavar="REPO",
                        help="SAE repo for the --mlx SAE-feature path (default: the 2B W32K SAE). "
                             "Pass 'none' to run capture-only with no SAE download.")
    parser.add_argument("--mlx-d-sae", type=int, default=DEFAULT_MLX_D_SAE, help="SAE feature count for --mlx-sae")
    parser.add_argument("--recipes", default="recipes", help="recipe store root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args()
    app = build(args.config, args.dev, args.recipes, mlx=args.mlx, mlx_layer=args.mlx_layer,
                mlx_sae=args.mlx_sae, mlx_d_sae=args.mlx_d_sae)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
