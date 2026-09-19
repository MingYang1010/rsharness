# Native raster tools

`raster.band_math@1.0.0` currently implements NDVI only. It is a scientific CPU
provider alongside EO-Gym, not an EO-Gym simulator, Python evaluator or arbitrary
expression interpreter. The Harness still owns episodes, budgets and evidence.

`raster.resample@1.0.0` implements one reviewed categorical operation: align a
Sentinel-2 SCL asset to an explicit same-scene reference-asset grid. It accepts
only `method: nearest`; bilinear/cubic interpolation, arbitrary target grids and
implicit alignment inside band math remain forbidden.

`raster.band_math@1.1.0` is an opt-in extension for episode-local artifact
chaining. It consumes a prior `raster.resample@1.0.0` SCL artifact plus the
reviewed red/NIR assets and applies the fixed policy
`sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1`. Version 1.0.0 remains
unchanged for tasks that do not opt in.

## Computation and output semantics

Arguments are `operation: ndvi`, `red_asset_id`, `nir_asset_id`. Inputs must be
explicitly reviewed reflectance assets, visible to the task and episode. The
task's private `raster_inputs` binds each asset to its checksum, band, scene,
acquisition time, native CRS/transform/shape, DN scale/offset and nodata value.
Red/NIR must share the exact date/scene/grid. No silent resampling is performed.

For each valid pixel, apply `reflectance = DN * scale + offset`, then
`NDVI = (nir - red) / (nir + red)` in float64 before serialization to float32.
Invalid pixels are source-mask/nodata pixels, nonfinite or negative reflectance,
or denominator <= 1e-6. They receive -9999 and an internal invalid-data mask.
All-invalid results retain null statistics, not a fabricated zero. Cloud masking
is **not applied**; scene cloud metadata does not establish local cloud-free data.

For version1.1.0, SCL classes1,3,8,9,10 and11 are excluded; classes2,4,5,6 and7
remain eligible before ordinary reflectance/mask checks. SCL nodata/invalid
pixels are also excluded. The result reports mask-valid, policy-clear,
policy-excluded and final NDVI-valid pixel counts separately. This is a fixed,
versioned filtering policy, not a claim that SCL or the resulting cloud mask is
semantically correct for every scene.

The result is a numeric single-band GeoTIFF, not a rendered PNG. It preserves the
native grid/CRS and acquisition time. Typed observation metadata includes native
CRS/transform, WGS84 bounds, dimensions, invalid policy, valid/total pixel counts
and numeric min/max/mean. Existing ArtifactRef stores WGS84 extent, shape/time,
checksum and lineage without changing frozen schemas. Source hashes/profiles,
arguments, policy and version determine lineage/derivation identity. Pixel-window,
bbox and time evidence selectors remain validated by the Harness.

For SCL alignment, private `grid_inputs` binds the SCL source and red/NIR
reference to checksums, scene/time and actual native profiles. The reference
provides CRS/transform/shape only; its values and mask do not invalidate SCL.
Source mask, SCL class 0 and pixels outside the source are invalid. Valid classes
must be 1--11 and are serialized unchanged as uint8; invalid output is 255 plus
an internal mask. The typed result reports a 12-bin histogram and valid coverage.
It does not classify cloud/non-cloud pixels and `cloud_mask_applied` stays false.

## Isolation and integrity

- `EO_HARNESS_RASTER_URL` explicitly enables the provider. Existing EO-Gym and
  catalog behavior stays unchanged when absent; old tasks are not rewritten.
- Provider only mounts approved native inputs/manifest and worker source read-only.
  A grid task may mount its reviewed SCL/reference files; it never mounts evaluator
  labels, whole dataset roots, broker credentials or writable SATA.
  Internal network, no published ports or egress; 1 worker, 20s process timeout,
  2 CPU/1GiB container bound,128MiB tmpfs. There is no persistent output cache.
- Artifact ownership and lineage are checked inside the episode reservation
  transaction. The Harness releases the SQLite write lock before reading the
  bounded artifact content. It verifies checksum/size and then sends only that
  TIFF as an internal request body; the raster provider receives no broker
  credential and no writable artifact/SATA mount.
- Inputs <=16MiB each and1024x1024, single-band reviewed GTiff: uint16 reflectance
  or uint8 SCL with exact scale/offset/nodata. Reject paths/URLs, symlinks, foreign
  formats, checksum or actual profile mismatch.
- Output <=8MiB. The provider and Harness decode float32 NDVI or uint8 SCL TIFF and
  validate grid, masks/nodata, numeric range/histogram, statistics and checksum. Permanent storage
  uses the existing quota broker; budget reserves both input sizes and8MiB output.
- Gateway validates both input IDs and explicitly projects the tool-specific fields. Scientific
  downloads require allowed tool/version/lineage/scope and size, with content
  checksum verification. This is not blanket admission of arbitrary TIFF files.
- Failed calls use existing durable failure/idempotency accounting. Physical
  failed-I/O metering and whole-system DB/log quotas remain separate work.

## Reproducible local-data workflow

Use the existing CPU project image, no network and no new dependencies. Under a
128MiB quota reservation, the preparer hashes six reviewed red/NIR windows from
the prior Sentinel admission and creates a fresh immutable task plus native input
mount. The original RGB task and source windows are preserved. A separate grid
preparer admits only the three reviewed SCL/red pairs to a new immutable task.

```sh
python scripts/prepare_raster_smoke.py \
  --source runtime/stac-sentinel-nanjing-20260917-01 \
  --output-name <fresh-native-run>

python scripts/prepare_grid_smoke.py \
  --source runtime/stac-sentinel-nanjing-20260917-01 \
  --output-name <fresh-grid-run>
```

Combine `compose.eo-gym-smoke.yaml`, `compose.agent-smoke.yaml` and
`compose.raster-smoke.yaml`, using a dedicated project and
`EO_SMOKE_ROOT=./runtime/<fresh-native-run>`, `EO_AGENT_RUN=<fresh-native-run>`,
`EO_HARNESS_CATALOG_ENABLED=1`. Start storage/provider/raster/harness, issue-agent,
agent-gateway, then agent-check. The EO-Gym provider has an empty image manifest
and does not process native bands; native calculations go only to raster.
Recreate the five services and run agent-check with `python /verify.py --resume`.
Cleanup requires `--profile agent down`; preserve runtime reports and credentials.

The trusted independent audit runs `scripts/verify_raster_reference.py --run
runtime/<fresh-native-run>`. It does not import production math: it reads original
DN/masks/scales, recomputes float32 NDVI and compares every pixel/grid/mask to the
broker blob. Reports are quota-reserved and refuse overwrite.

For the grid task set `EO_RASTER_VERIFY_SCRIPT=./scripts/verify_grid_smoke.py`.
`scripts/verify_grid_reference.py --run runtime/<fresh-grid-run>` independently
maps every target pixel centre back to its source cell without importing the
production resampler, then compares exact pixels, masks, grid and class counts.

For execution replay, start fresh EO-Gym/raster providers on a separate internal
project; keep the original Harness stopped. Add `compose.execution-replay.yaml`
and pass `--raster-provider http://raster:8084 --raster-provider-instance <actual
container-ID>` to the existing replay CLI along with its ordinary DB/tasks/report
arguments. Record both actual provider IDs. The replay reruns the calculations;
old artifact blobs are not read as execution output. It does not reconstruct
historical environments or model reasoning.

## Verified acceptance (2026-09-17)

- Runtime `native-ndvi-nanjing-20260917-01`, task hash
  `2a01faa049f72096b8be06331a6859c64c5fa228899278e663a3e7b830eaeb17`, episode
  `ep2-1ee12c96f75a443a9ae70839a3e3ac2c`: six reviewed inputs, three dates,
  14 actions, three real NDVI jobs and three persisted scientific evidence refs.
- Split-network Agent could not reach Harness/EO-Gym/raster/storage or backend IP.
  Five-service recreation preserved all14 cached responses and three content
  hashes with zero new calculations; `reports/native-resume.json` passed.
- Independent reference has exact float32 equality for all three rasters,
  `reference/numeric.json`; this proves numerical implementation, not vegetation
  change or cloud/land-cover accuracy.
- Fresh-provider replay `execution-native-ndvi-20260917-01/execution.json`
  passed14/14 with three new calculations, full normalized trace/state and hashes.
- Offline-raster negative `execution-native-ndvi-offline-20260917-01/execution.json`
  failed at action8 (first calculation), as required; no old-output fallback.
  Original DB checksum stayed `d0702bea444a19e23907e282b06c0a7da720603d5461af1e8a791e39b5821893`.
- Valid pixels are24,515/43,844,25,975/43,844 and25,913/43,844 by date. Means
  exclude invalid/negative-reflectance pixels, so they are conditional summaries,
  not whole-window vegetation estimates. Cloud contamination remains possible.
- Full203 Python tests (13 new numerical/isolation/integration tests), renderer4/4,
  compilation and Git payload checks pass. Data/runtime remains outside Git.

### Categorical grid acceptance (2026-09-19)

- Runtime `scl-grid-nanjing-20260919-01`, task hash
  `b3bc3715812f735d00c46d4fe6c045a560ac07c7259e511dd4735477809f9812`, episode
  `ep2-a5058fb50a424794b27ed8863a1bb544`: six reviewed inputs,14 Agent actions,
  three real `/resample-grid` jobs and three persisted evidence refs.
- All three97x113 SCL inputs aligned to194x226 red grids with43,844/43,844 valid
  pixels. Independent pixel-centre reference matched every uint8 pixel and mask.
  Output hashes/sizes are `d8a2dd1e...fff8`/2,997, `7ecfdea0...d85d`/2,461 and
  `5dc899d9...ff93`/3,467 bytes; full values remain in the runtime audit report.
- Five-service recreation returned all14 cached actions with zero new grid calls.
  Fresh-provider replay passed14/14 and matched final state, normalized trace and
  artifacts. With raster stopped, replay failed at action8 with `tool_unavailable`.
  Original DB SHA-256 remained
  `0c17bca663d39146b08861537691a5d998cba70f7ae7700a857ca9d3519ef4d0`.
- Current combined regression is223 Python and4 renderer tests. This proves the
  bounded categorical alignment path, not SCL semantic accuracy, cloud masking,
  temporal change correctness or general-purpose reprojection.

### Episode-artifact chaining acceptance (2026-09-19)

- `raster.band_math@1.1.0` accepts only a derivation-identified, current-episode
  SCL-grid artifact whose tool/version, two source refs, task profiles and exact
  resample parameters match the immutable task. Cross-episode artifacts and
  wrong policy/version/grid/date/lineage are rejected before provider execution.
- Tests cover release-before-content-read planning, content checksum tampering,
  aggregate input budget reservation, provider offline, cached restart and fresh
  execution replay. The isolated provider also runs the bounded binary-mask
  path through the real subprocess worker.
- Runtime `cloud-chain-nanjing-20260919-01`, task hash
  `c59a64b2495e5db3a0cb51fa88a80fb8e9b882b23e78b7aa14b028c921110663`,
  episode `ep2-6a88c02d8c554b6aa6678fa49dd34047`: nine reviewed inputs, 20 Agent
  actions, three SCL-grid artifacts, three policy-masked NDVI artifacts and
  three evidence refs. Split-network denial covered all four private services
  and the backend IP.
- Five-service recreation returned all 20 cached actions and six checksum-
  verified artifacts with zero new raster/provider POSTs. Independent
  pixel-centre, SCL-policy and scaled-NDVI reference code matched all six
  rasters exactly; maximum absolute NDVI error was 0 for all three dates.
- Fresh-provider replay
  `runtime/execution-cloud-chain-20260919-01/execution.json` passed 20/20,
  verified all six artifact contents, and matched final state and semantic
  trace after six fresh raster calls. With the replay raster stopped,
  `runtime/execution-cloud-chain-offline-20260919-01/execution.json` failed
  after 11 executed actions with `action_execution_mismatch`; no cached or old
  artifact fallback succeeded. The original DB SHA-256 remained
  `bed8e63dbc1d6dd9aaefafcf6586c68eba59873ace4edb445dd6a48c960cb0e8`.
- Full 234 Python tests and 4 renderer tests pass. This establishes the deployed
  fixed-policy computation and recovery/replay path, not SCL cloud semantic
  accuracy, cloud ground truth or temporal-change correctness.

Remaining tools include general continuous-band reprojection/resampling, zonal
statistics, more reviewed band formulas and temporal stacks/change evaluation.
The observed masked-NDVI means are conditional summaries under the fixed SCL
policy, not a scientifically validated change-detection result.
