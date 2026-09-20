# Continuous-band grid alignment acceptance

## Accepted contract

`raster.resample@1.1.0` is the opt-in continuous-band counterpart to the frozen
categorical `raster.resample@1.0.0` path. Version 1.1.0 accepts a reviewed
single-band source plus an explicit same-scene reference asset and currently
allows only `method: bilinear`.

The source DN values are converted to physical values with
`DN * scale + offset` before interpolation. The output is a single-band float32
GeoTIFF on the exact reference CRS, transform and shape, with nodata `-9999`, an
internal mask, bounded statistics and complete source/reference lineage. The
reference contributes geometry only: its values and mask do not become a
science mask. A separate `ContinuousGridResult` keeps these semantics distinct
from the categorical histogram result.

The existing `raster.resample@1.0.0` nearest-neighbour behavior is unchanged.
Its categorical fixture remains byte-identical with SHA-256
`bff667a343302e30763d03b013220f934a984d2372766d9e59c2e2c6467a38c7`.

## Real Sentinel-2 acceptance

The accepted A800 run is
`runtime/continuous-grid-nanjing-20260920-01`. Runtime imagery, receipts,
episode state, reports and artifacts are ignored by Git.

- Pinned item: `S2A_50SPA_20240405_0_L2A`, acquired
  `2024-04-05T02:58:52.954000Z`.
- Source: B11/SWIR16, EPSG:32650, 97x113 at 20 m, uint16, scale `0.0001`,
  offset `-0.1`; source window SHA-256
  `0347dee43fc5c2ce54692b8baf6853504b2445a50002a5f527e170265bac7c65`,
  19,228 bytes.
- Reference: B08/NIR, 194x226 at 10 m. The task uses its grid only.
- Admission: 9 bounded HTTP requests and 2,097,152 transferred payload bytes
  through the existing A800 system proxy. Task manifest SHA-256 is
  `38683288a09ecaeff7017dd881ac95d80183a8b4767bc0689482ccc273b0bcc6`.
- Episode: `ep2-8a8a8f0eabf04c07adc22cf90e614b75`; 6 accepted actions
  (search, two inspections, resample, evidence, submit).
- Output artifact:
  `art-22fce6788fa013e2ba1647d3500a62ce1a19754d2ed6369211354c5556fc85c8`,
  SHA-256
  `f2ff7b33ddb2f93e5e75dea4c697864f07a9b63b3ba23b78a93f361509752219`,
  149,872 bytes.
- Result: 43,844/43,844 valid pixels, minimum
  `-0.07959999889135361`, maximum `0.4171749949455261`, and mean
  `0.07146705592380573`.

`scripts/verify_continuous_grid_reference.py` implements pixel-centre bilinear
interpolation independently and does not call the production reprojection
function. It matched the target grid and mask exactly, with maximum absolute
error `0.0` over all valid pixels.

The split-network Agent could not connect directly to Harness, EO-Gym provider,
raster provider, storage broker or the Harness backend IP. Recreating all five
services preserved the episode and returned all 6 cached actions and the same
artifact content; the recreated raster provider received 0 new resample calls.

## Execution replay and failure semantics

The original Harness stack was stopped before replay. A separate internal
Compose project started fresh EO-Gym and raster providers. The inspected fresh
container IDs were:

- EO-Gym provider:
  `ceafd09ba038745c79729215fcdb9f5b291ef4380edf571c211c8445c8005846`.
- Raster provider:
  `e72525991c289f9f7e13d9b280d6c5c5a85076bc6f47d60582bcf59809425712`.

`runtime/execution-continuous-grid-20260920-01/execution.json` passed 6/6
actions with exactly one fresh `/resample-grid` call. Artifact metadata/content,
final state and semantic trace matched; task, original snapshot and runtime
fingerprint stayed unchanged; no original artifact bytes were read. Report
SHA-256 is
`2583064160ba6dc01919c8af91a2bf3e640e8311eb95ec717e7925e7e025a195`.

After stopping only the fresh raster provider,
`runtime/execution-continuous-grid-offline-20260920-01/execution.json` matched
the first three catalog actions and failed closed on action 4,
`raster.resample`, with `tool_unavailable`. It executed no old-output fallback
and read no original artifact bytes. Report SHA-256 is
`8c2c759724936693b7fec3bcf1f81ac55fb7c0b58156c42b9172b89b7d9f7e09`.
The original episode DB remained byte-identical before and after both replays:
`73cd67770ec93de0a3b0a811415ad6a3b3c4710a7ed1a9688e71827bd3ef8a2f`.
The dedicated replay containers and network were removed after verification.

## Verification and limits

Final gates passed on A800: 41 focused tests; 335 full Python tests with 34
environment-dependent skips; 4/4 renderer tests; compileall; Git whitespace,
payload and credential checks.

This acceptance proves one bounded, deterministic B11-to-B08 numerical
alignment and its restart/replay/failure behavior. It does not validate NDMI or
another band formula, cloud masking, land-cover semantics, geographic
generalization, multiple sensors/providers, historical container identity or
Qwen/model reasoning. Zonal statistics, selected formula evaluation and broader
licensed imagery are separate future checkpoints.
