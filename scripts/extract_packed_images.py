#!/usr/bin/env python3
"""Extract bounded reviewed image samples, excluding all question/answer columns."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.data.packed import extract_samples
from app.v2.storage.quota import CONTROL_ALLOWANCE, StorageQuota


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-shards", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=128)
    args = parser.parse_args()
    if not args.output.is_absolute() or "runtime" not in args.output.parts:
        parser.error("output must be an absolute ignored runtime directory")
    config = json.loads(args.config.read_text())
    spec = next((s for s in config["sources"] if s["dataset_id"] == args.dataset_id), None)
    if spec is None:
        parser.error("dataset has no reviewed image-column specification")
    # Includes bounded receipts/metadata and allocation overhead. Full source
    # shards stay outside this scope and are never copied by the adapter.
    payload_limit = 512 * 1024 * 1024
    quota = StorageQuota(ROOT / "runtime")
    with quota.hold(args.output, payload_limit + CONTROL_ALLOWANCE, "packed-image-samples"):
        report = extract_samples(spec, args.output, args.samples, args.max_shards, args.max_rows,
                                 max_output_bytes=payload_limit)
    print(json.dumps({k: report[k] for k in ("unique_images", "duplicate_images", "inspected_rows", "output_bytes", "shards_read", "label_columns_exported")}))


if __name__ == "__main__":
    main()
