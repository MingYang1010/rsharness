# Reviewed raster window admission

`reviewed-raster-window-v1` admits bounded, immutable GeoTIFF windows from an operator-reviewed local Parquet path column. It is opt-in and preprocessing-only.

It does not raise the 20,000,000-pixel whole-image provider limit. The policy reads a fixed 2048 x 2048 grid through Rasterio, clips edge windows without padding or resampling, and stages only outputs that pass exact pixel, mask, CRS, transform, nodata and checksum verification.

## Fixed boundaries

- Source paths still use the reviewed dataset root, image root, component-prefix allowlist and no-symlink checks.
- Only the reviewed image-path column is projected. Labels and annotations are never loaded.
- The source encoded file remains bounded to 128 MiB.
- Each decoded window is bounded to 128 MiB; the extraction output has an independently reserved total-byte limit.
- Source drivers are restricted to GTiff, PNG and JPEG. Outputs are classic GeoTIFF, never BigTIFF or sidecar-mask output.
- Sources at or below the existing whole-image pixel limit are skipped by this policy and remain eligible for the existing whole-image admission path.
- A common dataset mask is preserved exactly. Sources whose band masks cannot be represented losslessly by the fixed profile are rejected.

## Identity

`AssetRef.sha256` is the staged GeoTIFF content checksum. `AssetRef.source_snapshot_hash` is a separate canonical derivation hash binding:

- policy and output-profile versions;
- Parquet and source-image checksums;
- source dimensions, dtype, CRS, affine transform, nodata, color interpretation and mask flags;
- the exact half-open pixel window.

The frozen `AssetRef` schema is unchanged. The task identity already includes `source_snapshot_hash`; task metadata additionally projects the fixed admission profile and pixel window.

## Operator usage

The dataset specification must contain:

```json
"window_policy": "reviewed-raster-window-v1"
```

Run the existing extractor with the explicit mode:

```bash
python scripts/extract_path_parquet_images.py \
  --config config/path-parquet-sources.json \
  --dataset-id LRS-VQA-path-parquet \
  --output /absolute/project/runtime/new-window-run \
  --admission-mode reviewed-window \
  --samples 3 \
  --max-rows 128
```

The default mode remains `whole-image`; existing callers do not silently change behavior.

## Replay boundary

Execution replay starts from the admitted immutable asset and re-executes Agent tool actions. It does not re-run preprocessing admission. Admission reproducibility is checked separately by extracting into a fresh output directory and comparing the canonical derivation hashes plus independently decoded pixels and headers.

Run the independent source verifier against an admission directory that still has access to the reviewed read-only source root:

```bash
python scripts/verify_raster_window_admission.py \
  --admission /absolute/project/runtime/new-window-run
```

## Non-claims

- This is not dataset-wide admission.
- This is not reprojection or resampling.
- This is not redistribution approval.
- Passing interaction/replay checks is not semantic-answer accuracy.
