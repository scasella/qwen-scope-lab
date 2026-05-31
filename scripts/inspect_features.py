from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_scope_lab.service import SteeringService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    service = SteeringService.from_config_path(args.config)
    result = service.inspect_prompt(args.prompt, layer=args.layer, top_k=args.top_k, max_seq_len=args.max_seq_len)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
