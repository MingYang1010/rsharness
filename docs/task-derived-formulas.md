# Task-derived raster formulas

## Current status

The first task-derived formula slice is complete. `raster.band_math@1.2.0`
implements only fixed Sentinel-2 NDMI under formula ID
`sentinel-2-ndmi-nir-swir16-v1`. Its A800 interaction, independent numeric
reference, restart, execution replay and capability-specific failure evidence
are recorded in [fixed-ndmi-acceptance.md](fixed-ndmi-acceptance.md).

The Agent-visible request remains deliberately narrow:

```json
{
  "operation": "ndmi",
  "nir_asset_id": "asset-...",
  "swir_artifact_id": "art-..."
}
```

The Agent cannot provide formulas, constants, paths, grids, masks or geometry.
The immutable task binds the reviewed B08 input, expected B11-to-B08 alignment,
formula and validity policies. Only a current-episode
`raster.resample@1.1.0` B11 artifact with exact task-derived lineage is
accepted. `raster.zonal_stats@1.1.0` is the explicit NDMI-aware extension;
`raster.zonal_stats@1.0.0` remains limited to continuous reflectance artifacts.

## Selection rule for another formula

No additional formula is selected yet. Add one only when a concrete task and
evaluator require an observable that the current tools cannot produce. Each
candidate must pin its bands, scaling, grid, mask, constants, units, output
bounds and lineage, then pass an independent numerical reference, restart,
fresh execution replay and capability-specific fail-closed replay.

The next formula must not be chosen merely to expand a calculator library.
Before implementation it needs:

- a public or license-reviewed benchmark task with an explicit target claim;
- reviewed sensor/band semantics and bounded inputs already admitted by the
  Harness;
- an evaluator that distinguishes correct tool use and grounded evidence from
  a plausible final answer;
- a statement of what the derived index does and does not measure.

## Deferred formulas

- NDWI remains deferred because incompatible definitions use different
  green/NIR/SWIR combinations; the task must name the intended observable.
- NBR remains deferred until a reviewed B12 SWIR22 input and a burn task with
  public labels are admitted.
- EVI remains deferred until blue/red/NIR inputs, constants and atmospheric
  assumptions are pinned and justified by a task.
- General arithmetic, Python, SQL, GDAL expressions and Agent-supplied formulas
  remain out of scope because they expand the attack surface, weaken lineage
  and make evaluator semantics ambiguous.

Qwen3.5-9B is a separate acceptance class. Scripted oracle success is not model
reasoning success; real model interaction still requires an authorized healthy
GPU allocation.
