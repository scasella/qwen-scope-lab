from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_scope_steering_gui.service import SteeringService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--feature-id", type=int)
    parser.add_argument("--auto-feature", action="store_true")
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--mode", default="all_positions")
    args = parser.parse_args()

    service = SteeringService.from_config_path(args.config)
    if args.auto_feature:
        result = service.auto_steer(args.prompt, args.layer, args.strength, args.max_new_tokens, args.temperature)
    else:
        if args.feature_id is None:
            raise SystemExit("--feature-id is required unless --auto-feature is set")
        result = service.steer(args.prompt, args.layer, args.feature_id, args.strength, args.max_new_tokens, args.temperature, args.mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
