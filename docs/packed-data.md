# Reviewed packed-image admission

The operator-side adapter extracts bounded image-only samples from local Arrow
IPC streams/files and Parquet shards. It does not expose raw tables or grant
dataset-wide Agent access. Existing answers, captions, categories, boxes and
segmentation labels stay in the original source directory.

## Dependency and deployment boundary

The A800 dependency is `pyarrow==21.0.0`, with the Linux CPython 3.12 wheel hash
pinned in `config/packed-requirements.txt`. Download on A800 through its existing
system proxy, then install offline into ignored `runtime/packed/python` with
`--no-index --no-deps --find-links runtime/packed/wheels --require-hashes`.
Do not install into system Python or the shared Qwen environment. Keep wheels and
installed packages out of Git. Tests require this path on PYTHONPATH, alongside
the base API image and isolated EO-Gym dependencies.

Run extraction in a read-only, no-egress container with bounded CPU/memory and
only the selected source roots mounted read-only. The runtime output is writable;
the Agent and provider must never receive the original packed files or private
receipts. PyArrow column projection is a data-selection boundary, not a sandbox
against arbitrary native parser vulnerabilities.

## Input review and limits

`config/packed-sources.json` records three inspected XLRS schemas: a sequence of
HF Image structs for lite, and a single Image struct for caption/grounding. Their
`image` field contains input imagery; question/answer and annotation fields are
excluded. These records authorize only local research samples. Dataset licensing
and redistribution approval remain separate work.

- Only the named reviewed input column is projected. Label-like column names are
  denied, even if a caller tries to mark them reviewed.
- Support embedded bytes, HF `{bytes, path}`, and bounded image sequences. Never
  dereference a row's path or URL; path-only rows are reported unsupported.
- Accept PNG/JPEG/TIFF signatures only, constrain raster drivers, inspect real
  dimensions, and decode all bands before admission. Preserve real CRS/bounds or
  explicitly report pixel coordinates; never fabricate geography.
- Current provider admission limits: 128 MiB encoded image, 20 million pixels,
  up to four channels. Reject oversize images without resizing. Per shard: 2 GiB.
  Default run: at most four shards, 128 rows each and 512 MiB of staged images.
- Selection follows sorted shard/row order until three different content hashes
  are admitted. This is a deterministic acceptance sample, not representative
  benchmark sampling. Duplicates share an input file but retain private origins.
- Hash full source shards and staged images. If source inode/size/mtime changes
  during reading, fail without publishing a provider manifest. Preserve existing
  output directories; retries need a fresh name. Incomplete staging is not ready.

The CLI now reserves its scope through the shared 3 TB runtime ledger before
extraction; combined metadata output is bounded to 8 MiB. See
[storage-quota.md](storage-quota.md). Other write paths still require integration
before whole-system quota enforcement can be claimed.

## Commands and outputs

Inside the isolated A800 container (with `/packed` containing the pinned package):

```bash
PYTHONPATH=/packed:/sata/yangm/eo-harness/harness_api python scripts/extract_packed_images.py \
  --config config/packed-sources.json --dataset-id XLRS-Bench-lite \
  --output /sata/yangm/eo-harness/runtime/<fresh-run-directory>
```

Outputs remain outside Git:

- `inputs/`: checksum-named original image bytes only, deduplicated within the run.
- `inputs.json`: provider allowlist of opaque asset IDs, basenames and hashes.
- `coverage.json`: admitted samples, examined-row counts, limits/failure reasons.
- `private/receipt.json`: shard checksums, selected column, row/image index,
  dataset/review identifiers and dependency version. Not an Agent observation.

`prepare_eo_gym_smoke.py` consumes this coverage format, stages one approved image
and creates a separate task/store. It carries the original packed-source checksum
into `AssetRef.source_snapshot_hash`. Task identity includes dataset, image and
source snapshot so equal image bytes from different sources do not silently
replace a task version. Use separate smoke names and the `EO_SMOKE_ROOT` override.

## Verification scope

Tests cover Arrow stream/file, Parquet, image sequences, projection, label/path
denial, source changes, corrupted containers, deduplication and byte/row limits.
Real A800 extraction has been exercised on the three configured Arrow sources;
Parquet support currently has generated-fixture acceptance, not real-dataset
coverage. Unsupported formats, unreviewed columns and oversized images remain
explicit coverage gaps. No semantic labels/evaluator, Agent catalog search,
Qwen inference or dataset-wide admission is implied by image extraction.

### Real Arrow acceptance (2026-09-17)

| Source | Rows examined | Images admitted | Rejected by dimension limit | Image bytes |
| --- | ---: | ---: | ---: | ---: |
| XLRS-Bench-lite | 15 | 3 | 12 | 8,052,330 |
| XLRS-Bench_caption_en | 3 | 3 | 0 | 9,176,037 |
| XLRS-Bench_visual_grounding_en | 23 | 3 | 20 | 12,955,974 |

Each source used one shard; the combined staged image payload is 30,184,341
bytes. These ordered samples are not an unbiased estimate of dataset coverage.
All nine independent Harness HTTP smokes passed real upstream crop, idempotent
retry, evidence validation and submission. All nine Harness recreation checks
preserved state, trace hash and artifact hash. Crops were 2048x2048 or 2049x2048.
Only the selected image and its allowlist were mounted into each provider.
Source labels/tables and private receipts were not mounted.

Receipts and coverage remain in `runtime/packed-samples-20260917-<dataset>/`;
HTTP/restart reports remain in
`runtime/packed-smoke-20260917-<group>-<sample>/reports/` (groups 0/1/2 follow
the table order, samples 0/1/2). The nine dedicated projects were removed after
testing; these reports do not imply an online service. Original sources and
shared model environments were not modified.
