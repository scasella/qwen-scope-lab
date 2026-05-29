from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main(function_name: str, description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dry-run", action="store_true", help="Print the modal command without running it.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    command = ["modal", "run", f"modal_app.py::{function_name}"]
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=root, check=False).returncode
