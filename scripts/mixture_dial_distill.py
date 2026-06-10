"""Mixture Dial Distill CLI.

Compile a candidate JSONL plus a mixture config into chat-format SFT JSONL and a
compact manifest. This is an offline data compiler; it does not load models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.experiments import mixture_dial_distill as md


def cmd_compile(args: argparse.Namespace) -> dict:
    return md.compile_to_dir(args.mixture, args.candidates, args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile mixture-slot SFT data from candidate JSONL.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="Compile sft.jsonl and mixture_manifest.json")
    c.add_argument("--mixture", required=True, help="Path to mixture YAML/JSON config")
    c.add_argument("--candidates", required=True, help="Path to candidate JSONL")
    c.add_argument("--out", required=True, help="Output directory")
    c.set_defaults(func=cmd_compile)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(args.func(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
