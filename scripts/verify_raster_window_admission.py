#!/usr/bin/env python3
"""Verify reviewed raster windows against private immutable source receipts."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.data.raster_windows import verify_raster_window_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_raster_window_output(args.admission)))


if __name__ == "__main__":
    main()
