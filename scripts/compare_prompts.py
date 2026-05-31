from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_scope_lab_bench.service import SteeringService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--positive-prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    service = SteeringService.from_config_path(args.config)
    result = service.compare_prompts(args.positive_prompt, args.negative_prompt, layer=args.layer, limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
