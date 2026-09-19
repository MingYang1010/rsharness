# Reviewed path-Parquet coverage audit

This operator-only audit measures the entire reviewed image-path column without
copying images or reading label columns. It closes the gap between three-sample
acceptance and dataset-wide coverage accounting, but it does not turn a dataset
into an admitted benchmark.

## Contract

- Project only the configured input-image path column from one reviewed Parquet.
- Apply the same normalized-relative-path, prefix, root-containment, symlink and
  128 MiB encoded-file checks as sample admission.
- Open only PNG/JPEG/TIFF headers with GDAL directory scans and PAM sidecars
  disabled. Do not read pixel blocks or calculate image content checksums.
- Count distinct path values by SHA-256 of the private normalized value. Reports
  expose only aggregates; raw paths and label values are never retained.
- Separate `whole_image_header_eligible` from
  `reviewed_window_header_candidate`. The latter means only that a source crosses
  the 20-million-pixel whole-image boundary; it still needs the immutable window
  admission policy and full pixel/content verification.
- Abort the report if the Parquet or an inspected image changes during the audit.
  A bounded `--max-rows` run is explicitly `scan_complete=false`.
- Require a fresh quota-managed output outside the configured source dataset;
  write the private receipt before publishing the public aggregate report.

The report says `pixel_content_read=false` and
`image_content_hashes_verified=false`. Header eligibility is not evidence of
decodability of every block, geographic semantics, licensing, labels, or model
performance.

## Command

Run from the A800 repository with the pinned PyArrow/rasterio dependencies:

```bash
PYTHONPATH=runtime/packed/python:harness_api \
python scripts/audit_path_parquet_coverage.py \
  --config config/path-parquet-sources.json \
  --dataset-id LRS-VQA-path-parquet \
  --output /sata/yangm/eo-harness/runtime/path-coverage-lrs-<run>
```

The output is quota-managed and bounded to two small JSON files:

- `coverage.json`: public aggregate counts and explicit verification limits;
- `private/receipt.json`: configured roots, Parquet checksum, versions and review
  identity, but no row path values.

Acceptance requires `scan_complete=true`, exact row accounting, no raw paths or
labels in outputs, a stable Parquet checksum and a separately documented source
mutation boundary. Full image admission and content hashing remain separate work.

## Real A800 header coverage (2026-09-19)

The final isolated reports scanned all 64,577 rows:

| Source | Rows | Distinct reviewed paths | Duplicate rows | Whole-image header eligible | Window header candidate | Over 128 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LRS-VQA | 7,333 | 1,228 | 6,105 | 726 | 354 | 148 |
| DOTA V2 processed | 51,083 | 51,083 | 0 | 51,083 | 0 | 0 |
| DOTA v1/v1.5/v2 | 6,161 | 2,423 | 3,738 | 2,272 | 145 | 6 |

Final report SHA-256 values are:

- LRS-VQA: `28308d4fe5f244193260cef7246aa31eea7f78235ed1e59ebe5467af447a388d`;
- DOTA V2 processed: `045496078fef5215e4982b9076d3027f4753d1fcc893279a8d1c4b160bbbab70`;
- DOTA v1/v1.5/v2: `31119eda844a65abe96071ea8cb8500228df55893af390859e1f3af59b182037`.

Every report passed independent row-accounting and public-payload scans. Each
private receipt's Parquet checksum matched a fresh host checksum. A repeated
LRS-VQA run produced byte-identical report and receipt. The accepted reports are
under `/tmp/eo-harness-coverage.j4Dz8G/runtime/path-coverage-*-20260919-02/`;
they are runtime evidence and must not be committed.

The byte-limit failures are not window candidates under the current contract:
`checked_local_image` rejects sources above 128 MiB before header inspection.
The next coverage step is content-hash/full-decode admission or an explicitly
reviewed larger-source policy, not silently relabeling these 154 files.
