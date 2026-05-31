"""Serve the Lab Bench web app over the FastAPI backend.

Dev (CPU, no downloads):   python serve_web.py --dev
Local 2B on Apple Silicon: python serve_web.py --mlx Qwen/Qwen3.5-2B   (detection paths; no Modal/CUDA)
Real 2B (needs CUDA):      python serve_web.py --config configs/qwen35_2b_dev_l0_100.yaml
Real 27B (needs CUDA):     python serve_web.py --config configs/qwen35_27b_l0_100.yaml
"""
from __future__ import annotations

import argparse

import uvicorn

from qwen_scope_steering_gui.web_api import create_app


def build(config: str | None, dev: bool, recipes: str, mlx: str | None = None, mlx_layer: int = 12):
    if mlx:
        from qwen_scope_steering_gui.mlx_backend import build_mlx_service

        service = build_mlx_service(mlx, default_layer=mlx_layer)
    elif dev or not config:
        from qwen_scope_steering_gui.dev_backend import build_dev_service

        service = build_dev_service()
    else:
        from qwen_scope_steering_gui.service import SteeringService

        service = SteeringService.from_config_path(config)
    return create_app(service, recipes_root=recipes, experiments_root="experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to a real SteeringConfig YAML (needs CUDA)")
    parser.add_argument("--dev", action="store_true", help="use the GPU-free dev backend")
    parser.add_argument("--mlx", default=None, metavar="REPO",
                        help="run a local model on Apple Silicon via MLX (e.g. Qwen/Qwen3.5-2B); "
                             "serves the detection paths + /demo with no Modal/CUDA")
    parser.add_argument("--mlx-layer", type=int, default=12, help="probe/capture layer for --mlx")
    parser.add_argument("--recipes", default="recipes", help="recipe store root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args()
    app = build(args.config, args.dev, args.recipes, mlx=args.mlx, mlx_layer=args.mlx_layer)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
