#!/usr/bin/env python3
"""Audit bounded full-column image-path coverage without copying image data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.core.data.packed import MAX_RECEIPT_BYTES, AdmissionError
from app.core.data.path_coverage import (
    MAX_COVERAGE_ROWS,
    audit_path_coverage,
    validate_coverage_output,
)
from app.core.storage.quota import CONTROL_ALLOWANCE, StorageQuota


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=MAX_COVERAGE_ROWS)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if not args.output.is_absolute() or "runtime" not in args.output.parts:
        parser.error("output must be an absolute ignored runtime directory")
    if not 1 <= args.progress_every <= MAX_COVERAGE_ROWS:
        parser.error("progress-every out of range")
    config = json.loads(args.config.read_text())
    spec = next((value for value in config["sources"] if value["dataset_id"] == args.dataset_id), None)
    if spec is None:
        parser.error("dataset has no reviewed local path-column specification")
    validate_coverage_output(args.output, Path(spec["root"]))

    last_progress = 0

    def progress(rows: int, limit: int) -> None:
        nonlocal last_progress
        if rows == limit or rows - last_progress >= args.progress_every:
            print(json.dumps({"rows_scanned": rows, "scan_limit": limit}), file=sys.stderr, flush=True)
            last_progress = rows

    quota = StorageQuota(ROOT / "runtime")
    with quota.hold(args.output, 2 * MAX_RECEIPT_BYTES + CONTROL_ALLOWANCE, "path-parquet-header-coverage"):
        report, receipt = audit_path_coverage(spec, max_rows=args.max_rows, progress=progress)
        payloads = {
            "private/receipt.json": (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
            "coverage.json": (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
        }
        if any(len(value) > MAX_RECEIPT_BYTES for value in payloads.values()):
            raise AdmissionError("coverage_report_size_limit")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "private").mkdir(mode=0o700)
        for relative, content in payloads.items():
            path = args.output / relative
            with path.open("xb") as stream:
                stream.write(content)
        report_hash = hashlib.sha256(payloads["coverage.json"]).hexdigest()
    print(json.dumps({
        "dataset_id": report["dataset_id"],
        "rows_scanned": report["rows_scanned"],
        "scan_complete": report["scan_complete"],
        "report_sha256": report_hash,
    }))


if __name__ == "__main__":
    main()
