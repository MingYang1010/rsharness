#!/usr/bin/env python3
"""Inspect or explicitly reconcile the private runtime storage ledger."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.storage.quota import StorageQuota


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runtime")
    parser.add_argument("--reconcile", type=Path, help="exact abandoned scope; releases unused reservation, never deletes files")
    args = parser.parse_args()
    quota = StorageQuota(args.root)
    print(json.dumps(quota.reconcile(args.reconcile) if args.reconcile else quota.status(), indent=2))


if __name__ == "__main__":
    main()
