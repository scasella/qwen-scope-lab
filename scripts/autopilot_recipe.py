from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab_bench.autopilot import run_autopilot_from_files
from qwen_scope_lab_bench.benchmark import ServiceGenerationBackend
from qwen_scope_lab_bench.config import load_config
from qwen_scope_lab_bench.service import SteeringService


def parse_layers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-description", required=True)
    parser.add_argument("--positive-examples", required=True)
    parser.add_argument("--negative-examples", required=True)
    parser.add_argument("--validation-prompts", required=True)
    parser.add_argument("--candidate-layers", required=True)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--objective", default="maximize_rule_score")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-only-instruction", default="")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fake-backend", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    service = None
    backend = None
    if not args.fake_backend:
        service = SteeringService.from_config_path(args.config)
        backend = ServiceGenerationBackend(service)
    result = run_autopilot_from_files(
        config=cfg,
        config_path=args.config,
        target_name=args.target_name,
        target_description=args.target_description,
        positive_examples_path=args.positive_examples,
        negative_examples_path=args.negative_examples,
        validation_prompts_path=args.validation_prompts,
        candidate_layers=parse_layers(args.candidate_layers),
        candidate_count=args.candidate_count,
        objective=args.objective,
        output_dir=args.output_dir,
        backend=backend,
        service=service,
        prompt_only_instruction=args.prompt_only_instruction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps({"output_paths": result["output_paths"], "best_candidate": result["best_candidate"], "warning": result["warning"]}, indent=2))


if __name__ == "__main__":
    main()
