# Fixed Sentinel-2 NDMI acceptance

## Accepted contract

`raster.band_math@1.2.0` adds one opt-in operation: fixed Sentinel-2 NDMI with
formula ID `sentinel-2-ndmi-nir-swir16-v1`. The Agent may provide only
`operation: ndmi`, one reviewed B08 NIR asset ID and one current-episode aligned
B11 artifact ID. Formula strings, constants, paths, grids, masks and geometry
are rejected.

The Harness reconstructs and verifies the immutable task's B08 profile and the
full `raster.resample@1.1.0` B11-to-B08 lineage. It reads the authorized B11
artifact only after the episode transaction is released, checksum-validates the
content, and sends the bounded TIFF plus reviewed B08 input to the isolated
raster provider. The provider has no broker credential, whole-dataset mount or
writable artifact/SATA mount.

For pixels valid in both masks, the fixed computation is:

```text
nir  = B08_DN * B08_scale + B08_offset
swir = aligned B11 physical reflectance
ndmi = (nir - swir) / (nir + swir)
```

Both reflectances must be finite and non-negative, and the denominator must be
greater than `1e-6`. Computation uses float64 and writes float32; invalid pixels
use nodata `-9999` plus an internal mask. Output lineage binds the B08 asset and
the exact aligned B11 artifact.

`raster.zonal_stats@1.1.0` is the explicit NDMI-aware extension.
`raster.zonal_stats@1.0.0` is unchanged and continues to accept only
`raster.resample@1.1.0` continuous-reflectance artifacts. The NDMI provider has
its own `raster_ndmi_worker.py`; removing that capability does not disable
continuous alignment or zonal execution.

Implementation checkpoints:

- `8e4ddc9` — fixed formula contract, authorization, isolated worker and replay
  routing;
- `08cc54a` — explicit NDMI-aware zonal version;
- `5987340` — formula, mask, lineage, budget, provider and replay tests;
- `dff7325` — real acceptance scripts plus fail-closed public NDMI artifact
  download after gateway restart.

## Real Sentinel-2 interaction

The accepted A800 runtime is
`runtime/fixed-ndmi-nanjing-20260920-01`. It creates a fresh immutable task and
episode while reusing the already reviewed B08/B11 native files as read-only
same-filesystem hard links. It reuses no old artifact and makes no network
request. Runtime imagery, task payloads, state, reports, credentials and
artifacts remain outside Git.

- task manifest SHA-256:
  `2e357da1eeaa8187f366a98d2bd2371629055e7f19a2d918f30954aa79b0c19b`;
- episode: `ep2-eb080c8bb9034b6eb1099b89aedff71f`;
- scene: `S2A_50SPA_20240405_0_L2A`, acquired
  `2024-04-05T02:58:52.954000Z`;
- B08 SHA-256:
  `30ea0903a799d02e4bbad0b03e3404b198567ac220e992ea49ce0506c4f12bc8`,
  73,975 bytes;
- B11 SHA-256:
  `0347dee43fc5c2ce54692b8baf6853504b2445a50002a5f527e170265bac7c65`,
  19,228 bytes;
- aligned B11 artifact:
  `art-22fce6788fa013e2ba1647d3500a62ce1a19754d2ed6369211354c5556fc85c8`,
  content SHA-256
  `f2ff7b33ddb2f93e5e75dea4c697864f07a9b63b3ba23b78a93f361509752219`,
  149,872 bytes;
- NDMI artifact:
  `art-76839cc9dd9e751d4801cf77b7d3fd880aa138643594312df282672c1e5f3008`,
  content SHA-256
  `d0c531dc68f7f2dfb21faa68fd3a5592e95878b3a005968a5e06c61a02ab4cc1`,
  158,827 bytes.

The split-network scripted Agent completed eight actions: catalog search, two
asset inspections, continuous B11 alignment, fixed NDMI, zonal statistics,
evidence save and answer submission. It could not connect directly to Harness,
EO-Gym provider, raster provider, storage broker or the backend IP.

The full 194 by 226 NDMI raster contains 40,599 valid pixels out of 43,844.
Its valid-pixel minimum, maximum and mean are `-1.0`,
`0.9976691007614136` and `0.10177850098454595`.

For `zone-central-pixel-centres`, the accepted claim is:

| Field | Value |
| --- | ---: |
| Zone pixels | 11,172 |
| Valid pixels | 10,406 |
| Invalid pixels | 766 |
| Valid fraction | 0.9314357321876119 |
| Minimum | -1.0 |
| Maximum | 0.9963136315345764 |
| Mean | 0.12779076340103457 |

## Independent numeric reference

`scripts/verify_ndmi_reference.py` does not import production alignment,
band-math or polygon-membership functions. It independently performs
pixel-centre bilinear B11 alignment, B08 physical conversion, mask and validity
intersection, float64 NDMI, float32 serialization and a scalar
boundary-inclusive point-in-polygon zonal calculation.

The target grid, aligned mask and NDMI mask match exactly. Maximum absolute
error is `0.0` for both aligned B11 and NDMI values, and the zonal float64
statistics match exactly. The durable report is
`runtime/fixed-ndmi-nanjing-20260920-01/reference/fixed-ndmi.json`, SHA-256
`d0f30f05024da5cb51b21889fc6bcc99e1e5bf2bdc65b01603bb26552b475dc1`.

## Recovery, replay and failure evidence

Recreating storage, EO-Gym provider, raster provider, Harness and Agent gateway
produced new container IDs. Reissuing all eight requests returned cached
responses and checksum-verified content for both artifacts; the recreated raster
provider received zero resample, NDMI and zonal requests. The final resume
report SHA-256 is
`e23d18bb660e153aab5bf80508c0288e7e9389fbc23bab870e3c3ca556aa0251`.

This recovery test first exposed a missing gateway allowlist branch: the NDMI
metadata was visible but its public content endpoint returned 422. The final
implementation admits only `raster.band_math@1.2.0` NDMI artifacts whose first
lineage input is a task input and whose second input is a valid artifact ID. A
focused regression now downloads and checksum-checks NDMI content through the
Agent gateway; the complete five-service recovery then passed.

Fresh execution replay is recorded at
`runtime/execution-fixed-ndmi-20260920-01/execution.json`, SHA-256
`4a8a69bd01207a6677c17021ba9672c6ea2561dcb7f05641368a22a7b4642cf0`.
It passed 8/8 actions, made exactly one resample, one NDMI and one zonal request,
regenerated both exact artifacts, and matched final state and semantic trace.
Task/original/runtime snapshots stayed unchanged and no old artifact bytes were
read as execution output.

Two independent negative replays preserve the other capabilities:

- with only `EO_RASTER_NDMI_WORKER` empty,
  `runtime/execution-fixed-ndmi-worker-offline-20260920-01/execution.json`
  matched the first four actions, then returned `tool_unavailable` at
  `raster.band_math`; report SHA-256
  `dec3419818eb0cdb5fc44b285f1b25cf41879fb175ef592918bb5d2d9f6c3b18`;
- with only `EO_RASTER_ZONAL_WORKER` empty,
  `runtime/execution-fixed-ndmi-zonal-offline-20260920-01/execution.json`
  matched alignment and NDMI, then returned `tool_unavailable` at
  `raster.zonal_stats`; report SHA-256
  `5896799fe45bc67574ef2cdeb67491b91e68b6b18254983ed802a30ff2181200`.

Both negative replays failed closed, read no old artifact bytes and preserved
the task, original snapshot and runtime identity. The original episode database
remained byte-identical across all replays, SHA-256
`3c8fb90d28e9cd999a2c7763fe7fd3acc4f4b0438366185130e9b68d8a1edfdd`.

## Verification and limits

- focused NDMI tests: 7/7;
- full Python regression: 348/348, with 34 existing environment-dependent
  skips;
- renderer regression: 4/4;
- compileall, Compose config, whitespace, payload and credential guards pass;
- commit `dff7325` contains five source/test/acceptance files and 85,847 bytes;
  no dataset, runtime, database, artifact, report, model, bulk metadata or
  credential payload is committed.

This establishes deterministic computation and grounded interaction for one
reviewed Sentinel-2 scene and one pinned zone. It does not establish NDMI as a
soil-moisture, drought, crop-stress or change-detection label; it is not a
multi-sensor, multi-region or dataset-wide benchmark; and it is not Qwen/model
reasoning acceptance.
