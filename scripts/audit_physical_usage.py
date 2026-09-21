#!/usr/bin/env python3
"""Write a deterministic physical disk usage audit; never modify the tree."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.v2.storage.physical_usage import audit_physical_usage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--max-files", type=int, default=100_000)
    args = parser.parse_args()
    report = audit_physical_usage(args.root).as_dict()
    if len(report["files"]) > args.max_files:
        print("physical audit exceeds file bound; request a larger --max-files", file=sys.stderr)
        return 2
    content = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode()
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            raise SystemExit("preserve existing physical audit; choose a fresh path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(content)
    else:
        sys.stdout.buffer.write(content)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
