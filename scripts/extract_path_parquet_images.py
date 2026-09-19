#!/usr/bin/env python3
"""Extract bounded samples from an operator-reviewed local Parquet path column."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.data.path_parquet import extract_path_samples
from app.v2.data.raster_windows import POLICY_ID, extract_raster_windows
from app.v2.storage.quota import CONTROL_ALLOWANCE, StorageQuota


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument(
        "--admission-mode",
        choices=("whole-image", "reviewed-window"),
        default="whole-image",
    )
    args = parser.parse_args()
    if not args.output.is_absolute() or "runtime" not in args.output.parts:
        parser.error("output must be an absolute ignored runtime directory")
    config = json.loads(args.config.read_text())
    spec = next((value for value in config["sources"] if value["dataset_id"] == args.dataset_id), None)
    if spec is None:
        parser.error("dataset has no reviewed local path-column specification")
    payload_limit = 512 * 1024 * 1024
    quota = StorageQuota(ROOT / "runtime")
    with quota.hold(args.output, payload_limit + CONTROL_ALLOWANCE, "path-parquet-image-samples"):
        if args.admission_mode == "whole-image":
            report = extract_path_samples(
                spec,
                args.output,
                sample_count=args.samples,
                max_rows=args.max_rows,
                max_output_bytes=payload_limit,
            )
            fields = (
                "unique_images", "duplicate_images", "inspected_rows", "output_bytes",
                "label_columns_exported",
            )
        else:
            if spec.get("window_policy") != POLICY_ID:
                parser.error("dataset has no reviewed raster-window policy")
            report = extract_raster_windows(
                spec,
                args.output,
                sample_count=args.samples,
                max_rows=args.max_rows,
                max_output_bytes=payload_limit,
            )
            fields = (
                "unique_windows", "unique_sources", "duplicate_source_rows",
                "ineligible_whole_images", "inspected_rows", "output_bytes",
                "label_columns_exported",
            )
    print(json.dumps({key: report[key] for key in fields}))


if __name__ == "__main__":
    main()
