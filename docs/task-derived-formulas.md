# Task-derived raster formulas

## Decision

The next raster-formula slice is a fixed Sentinel-2 normalized difference
moisture index (NDMI) path, not an arbitrary expression interpreter.

The motivating task is narrow and observable: after aligning reviewed B11
SWIR16 reflectance to the same-scene B08 NIR grid, calculate the normalized
difference raster and report its independently verified value over a pinned
zone. This reuses the accepted continuous-grid and zonal-statistics contracts
while adding one new scientific transformation at a time.

This document selects the next implementation slice. It is not evidence that
NDMI is already implemented, semantically calibrated or accepted.

## Proposed contract

Use an opt-in `raster.band_math@1.2.0` operation with Agent-visible arguments:

```json
{
  "operation": "ndmi",
  "nir_asset_id": "asset-...",
  "swir_artifact_id": "art-..."
}
```

The Agent cannot provide a formula string, constants, paths, grids, masks or
geometry. The immutable task privately binds:

- one reviewed Sentinel-2 B08 NIR asset and its DN scale/offset/nodata profile;
- one expected B11 SWIR16 source and B08 reference profile;
- the exact `raster.resample@1.1.0` bilinear lineage policy;
- formula ID `sentinel-2-ndmi-nir-swir16-v1`;
- output dtype, nodata, validity policy and numeric bounds.

The Harness must accept only a current-episode SWIR16 continuous artifact whose
source/ref IDs, scene, acquisition, grid, checksum and parameters hash match the
task. It then reads the authorized artifact after releasing the episode write
transaction and sends only the bounded aligned SWIR TIFF plus the reviewed NIR
input to the isolated raster provider.

For each common valid pixel:

```text
nir  = B08_DN * B08_scale + B08_offset
swir = aligned B11 physical reflectance
ndmi = (nir - swir) / (nir + swir)
```

The fixed validity policy should require both source masks, finite values,
non-negative physical reflectance and denominator greater than `1e-6`. Invalid
pixels receive float32 `-9999` plus an internal invalid mask. Computation uses
float64 and serializes the final raster to float32. The typed result records the
formula/policy IDs, exact input lineage, grid, valid counts/fraction and
min/max/mean. Output must remain within `[-1, 1]` for valid pixels.

The result is a new derivation-identified raster artifact. The existing
`raster.zonal_stats@1.0.0` contract should accept it only after an explicit
versioned extension; it must not silently broaden its current requirement for a
`raster.resample@1.1.0` artifact.

## Acceptance benchmark

Start from `runtime/zonal-stats-nanjing-20260920-01` inputs, but create a fresh
immutable task and episode. The scripted acceptance sequence should be:

1. search and inspect the reviewed B08/B11 inputs;
2. align B11 physical reflectance to the B08 grid;
3. compute fixed-policy NDMI;
4. compute statistics for the same pinned zone through an explicitly extended
   zonal contract;
5. save evidence on the NDMI artifact and submit the exact numeric claim.

Independent reference code must not import production band-math or polygon
membership functions. It should manually reproduce bilinear B11 alignment,
physical B08 conversion, validity intersection, float64 NDMI, float32 output,
mask and zonal statistics. Required comparisons are exact grid/mask/counts and
at most `1e-7` maximum absolute error for valid float32 pixels.

The runtime acceptance must separately show:

- five-service restart returns all cached actions and artifact hashes with zero
  new raster calls;
- fresh execution replay reruns alignment, NDMI and zonal statistics and matches
  artifact content, final state and semantic trace;
- disabling only the NDMI worker fails at the formula action with
  `tool_unavailable` after alignment succeeds;
- disabling only the extended zonal capability fails after NDMI succeeds;
- original episode DB and immutable task bytes remain unchanged;
- runtime/data/reports/artifacts/credentials are absent from the Git payload.

This proves deterministic computation and grounded interaction, not that NDMI
is a calibrated soil-moisture, drought, crop-stress or change-detection label.
A later semantic benchmark needs public observations and explicit labels or a
falsifiable proxy before making those claims.

## Why this formula is next

- It answers a concrete EO task and uses bands already present in the accepted
  A800 runtime; no new acquisition is required for the first acceptance.
- It exercises artifact-to-formula-to-zonal chaining, which is a more important
  Harness capability than adding another isolated calculator.
- Its formula, grid and validity semantics can be independently reproduced and
  fail closed under capability removal.
- It preserves the research boundary: the benchmark can score tool selection,
  evidence grounding and exact numeric claims without treating an index as
  ground-truth environmental truth.

## Deferred formulas

- NDWI is deferred because multiple incompatible definitions use different
  green/NIR/SWIR combinations; a task must name the intended observable first.
- NBR is deferred until a reviewed B12 SWIR22 input and an explicit burn task
  with public labels are admitted.
- EVI is deferred until its blue/red/NIR inputs, constants and atmospheric
  assumptions are pinned and justified by a task.
- General arithmetic, Python, SQL, GDAL expressions and user-supplied formulas
  remain out of scope because they expand the attack surface, weaken lineage and
  make evaluator semantics ambiguous.

## Implementation checkpoints

The next code pass should use three checkpoints after the current five-commit
zonal series:

1. implement the fixed NDMI contract, artifact authorization and isolated
   provider path;
2. add unit/failure tests and real Sentinel-2 independent acceptance/replay;
3. record exact evidence and subtract the completed NDMI slice from the
   remaining-work plan.

Qwen3.5-9B remains a separate acceptance class and must wait for an authorized,
healthy GPU allocation. Scripted oracle success must not be reported as model
reasoning success.
