# Reviewed local-path Parquet admission

This operator-side adapter handles real EO Parquets whose reviewed image column
contains a local relative path instead of embedded image bytes. It is separate
from the embedded Arrow/Parquet adapter: path dereference remains forbidden
there, and this adapter cannot accept arbitrary paths or URLs.

## Security and data boundary

- The source config fixes one absolute dataset root, one relative Parquet file,
  one relative image root, one projected input-image column and component-level
  path prefixes. The Parquet and image roots are operator configuration, never
  row-controlled.
- Row values must be normalized POSIX relative paths. Absolute paths, URLs,
  Windows paths/drives, backslashes, `.`/`..`, repeated separators and NUL are
  rejected. Every path component is checked for symlinks before resolution;
  the final regular file must remain inside the configured image root.
- PyArrow projects only the reviewed image-path column. Answer, annotation,
  bbox and label paths are neither decoded nor copied. Public coverage failures
  contain row numbers and stable error codes, never raw path values.
- The Parquet and every admitted image are checksumed independently. Inode,
  size and mtime are checked around reading. A changed source or staged checksum
  mismatch prevents manifest publication.
- Existing image limits remain: 128 MiB encoded bytes, 20 million pixels and up
  to four bands. PNG/JPEG/TIFF signatures are decoded before admission. Output
  images are copied to a fresh ignored runtime directory with checksum names;
  source data is never modified.

This is bounded sample admission, not dataset-wide approval, licensing review,
geographic correctness or semantic evaluation. Receipts remain private and the
original dataset roots must not be mounted into an Agent/provider container.

## Reviewed sources

`config/path-parquet-sources.json` pins three current A800 sources:

| Source | Rows | Projected column | Allowed prefixes |
| --- | ---: | --- | --- |
| LRS-VQA | 7,333 | `image` | `LRS_VQA/image` |
| DOTA V2 processed | 51,083 | `processed_image` | `images` |
| DOTA v1/v1.5/v2 | 6,161 | `image_path` | `train/images`, `val/images` |

All three columns are strings. Six other real Parquets under the same data root
contain embedded image bytes but belong to LaTeX OCR, so they are not EO-domain
acceptance evidence. No label values were used during schema review.

## Command and outputs

Run on A800 with the pinned packed-data dependency path and a fresh runtime name:

```bash
PYTHONPATH=runtime/packed/python:harness_api python scripts/extract_path_parquet_images.py \
  --config config/path-parquet-sources.json \
  --dataset-id LRS-VQA-path-parquet \
  --output /sata/yangm/eo-harness/runtime/path-parquet-lrs-vqa-<run>
```

The CLI reserves 512 MiB plus control allowance through the shared runtime
ledger before reading. Outputs use the existing provider-facing format:

- `inputs/` and `inputs.json`: copied input images and opaque allowlist only;
- `coverage.json`: public bounded sample coverage and sanitized failures;
- `private/receipt.json`: configured roots, Parquet checksum, image origins and
  checksums, review/license identifiers and PyArrow version.

Current acceptance requires three distinct readable inputs per reviewed source,
exact checksum verification, no label projection, process restart recovery and
at least one fresh-provider execution replay per source.

## Real A800 acceptance (2026-09-19)

The host-path-preserving extraction runs are:

| Source | Rows inspected | Duplicates | Stable failures | Images | Bytes |
| --- | ---: | ---: | --- | ---: | ---: |
| LRS-VQA | 8 | 1 | 4 dimension-limit | 3 | 160,097,138 |
| DOTA V2 processed | 3 | 0 | 0 | 3 | 1,159,359 |
| DOTA v1/v1.5/v2 | 3 | 0 | 0 | 3 | 25,426,162 |

Independent verification recomputed each Parquet, source-image and staged-image
SHA-256. All three reports contain `label_columns_exported=false`. Runtime paths
are `runtime/path-parquet-<source>-20260919-02/`; the earlier `-01` runs remain as
evidence of a container-path staging mistake and are not used by downstream
smokes.

Nine scripted Harness interactions (three per source) each completed real
`eo_gym.crop`, invalid-evidence rejection, evidence save, answer submission and
structural replay. After storage/provider/Harness force recreation, every run
retained the same episode, state, trace hash and artifact hash. LRS crops were
2000x2000; DOTA V2 crops were 256x256; DOTA v1-family crops were 1972x1918,
1133x975 and 635x540. Reports are under
`runtime/path-parquet-smoke-20260919-*/reports/`. All dedicated containers and
networks were removed. These are scripted interactions, not Qwen, semantic
evaluation or geographic validation.

One sample per source was then replayed against a distinct freshly started
provider. All three reports passed 4/4 recorded actions with one exact artifact
check, matched final state and semantic trace, unchanged original/task snapshots
and runtime fingerprint, and `original_artifact_bytes_read=false`:

- `runtime/execution-path-parquet-lrs-20260919-01/execution.json`
- `runtime/execution-path-parquet-dota-v2-20260919-01/execution.json`
- `runtime/execution-path-parquet-dota-v1v15v2-20260919-01/execution.json`

The three fresh providers and dedicated networks were removed after audit.
