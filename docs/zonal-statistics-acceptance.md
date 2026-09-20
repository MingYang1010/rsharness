# Bounded zonal-statistics acceptance

## Accepted contract

`raster.zonal_stats@1.0.0` is an opt-in metadata-only tool over one
episode-local continuous raster artifact. The Agent may provide only
`raster_artifact_id` and `zone_id`. The immutable task privately binds the zone
polygon, CRS, WGS84 bounds and inclusion policy; arbitrary Agent geometry is not
accepted.

The source must be the current episode's derivation-identified output from
`raster.resample@1.1.0`. The Harness reconstructs and verifies its two reviewed
input profiles, bilinear parameters hash, grid, date, checksum and task scope.
The accepted policies are:

- polygon CRS: the same reviewed UTM CRS as the raster;
- inclusion: `pixel-centre-in-polygon-boundary-inclusive`;
- validity: `source-mask-and-finite-and-not-nodata`;
- result: bounded JSON containing zone/valid/invalid pixel counts, valid
  fraction, minimum, maximum and float64 mean;
- no second raster artifact and no arbitrary expression evaluation.

The Harness checksum-validates the authorized artifact and sends only its
bounded TIFF bytes to the isolated provider. The provider receives no broker
credential, whole dataset mount or writable artifact/SATA mount. Both sides
validate the source profile, polygon bounds and result identity. Provider JSON
is streamed with a 16 KiB limit. `raster.resample@1.0.0`/`1.1.0` and existing
band-math behavior are unchanged.

Implementation checkpoints:

- `e0261d68d81280827e90da0190a8deedd3a9478f` — bounded tool, provider and
  execution-replay adapter;
- `36af755ed2f9b77092dd4c5e091c051aed56d81a` — contract, policy, isolation,
  restart and replay regression tests;
- `09b45ac910ce727fabb52d18ae8bd097b4cb9c18` — real Sentinel-2 preparer,
  Agent verifier and independent reference.

## Real Sentinel-2 acceptance

The accepted A800 runtime is
`runtime/zonal-stats-nanjing-20260920-01`. It derives a fresh immutable task from
the accepted B11-to-B08 continuous-grid run without a network request. The two
reviewed native inputs are same-filesystem hard links mounted read-only; no
previous artifact is supplied as tool input. Runtime imagery, task payloads,
receipts, database, reports, credentials and artifacts remain ignored by Git.

- task manifest SHA-256:
  `12a95b705ff7a35589f17482f80a498076d6730cfd311bfe701bba7be3b27294`;
- episode: `ep2-f05853775bf54afab426271447e4a831`;
- zone: `zone-central-pixel-centres`, `EPSG:32650`;
- UTM polygon: `(669545,3542415)`, `(670515,3542415)`,
  `(670515,3541285)`, `(669545,3541285)`, closed at the first point;
- WGS84 bounds: `[118.79474741845826, 31.994820686936443,
  118.8052108648105, 32.005155951777475]`;
- aligned artifact ID:
  `art-22fce6788fa013e2ba1647d3500a62ce1a19754d2ed6369211354c5556fc85c8`;
- artifact SHA-256:
  `f2ff7b33ddb2f93e5e75dea4c697864f07a9b63b3ba23b78a93f361509752219`,
  149,872 bytes.

The scripted Agent completed seven actions: catalog search, two asset
inspections, continuous resampling, zonal statistics, evidence save and answer
submission. It could not connect directly to Harness, EO-Gym provider, raster,
storage or the backend IP. The result is:

| Field | Value |
| --- | ---: |
| Zone pixels | 11,172 |
| Valid pixels | 11,172 |
| Invalid pixels | 0 |
| Valid fraction | 1.0 |
| Minimum | -0.06274999678134918 |
| Maximum | 0.21735624969005585 |
| Mean | 0.060896887867167494 |

`scripts/verify_zonal_stats_reference.py` deliberately does not import the
production point-in-polygon code. It loops over every raster pixel centre,
applies a scalar boundary-inclusive even-odd test and computes statistics from
the source mask and float32 values promoted to float64. Membership, the expected
98 by 114 boundary-inclusive rectangle, counts and all statistics match exactly.
The report is
`runtime/zonal-stats-nanjing-20260920-01/reference/zonal-statistics.json`,
SHA-256
`de6106d43980dc93f71f70dcb0f17eef264f55f6b287846c55a62fd698f08676`.

## Recovery, replay and fail-closed evidence

Before five-service recreation the raster provider received one resample and
one zonal request. Storage, EO-Gym provider, raster, Harness and Agent gateway
were all recreated with different container IDs. Reissuing all seven recorded
Agent requests returned cached responses and the checksum-verified raster; the
new raster provider received zero resample and zero zonal requests.

Fresh execution replay is recorded at
`runtime/execution-zonal-stats-20260920-01/execution.json`, SHA-256
`b2d5267eb697b2a84965f9850af8e19bed95b5ec2eca62bbb2af1546b32baf0c`.
It passed 7/7 actions against raster container
`de059d211727ba615818bc333a7381e018d04e9c24757075ef9cdc048d475c2d`,
made exactly one resample and one zonal request, regenerated and verified the
149,872-byte artifact, and matched final state and semantic trace. It did not
read old artifact bytes as execution output.

A temporary Compose override then kept continuous resampling available but set
`EO_RASTER_ZONAL_WORKER` empty. The negative report
`runtime/execution-zonal-stats-offline-20260920-01/execution.json`, SHA-256
`0822190d9e3e0c3d046b6237c821ab79dd8e0dce30d5deff1a2ab5b57e35b239`,
matched search, both inspections and a fresh resample, then failed action 5 at
`raster.zonal_stats` with `tool_unavailable`. No cached or old-output fallback
made the replay pass.

The original episode database SHA-256 remained
`a3d20825a09e51827057f4ba2e9a8107df7b8844f79f89a355efb9dabd9646dc`
before and after both replays. All three owned Compose projects ended with zero
containers and zero networks.

## Verification and limits

- focused zonal tests: 6/6;
- full Python regression: 341/341, with 34 existing environment-dependent skips;
- renderer regression: 4/4;
- compileall, Compose config, whitespace and Git payload guards pass;
- committed files contain no runtime, raster, database, credential, report,
  model or bulk metadata payload.

This establishes one bounded, pinned-polygon interaction over one reviewed
Sentinel-2 continuous artifact. It is not arbitrary GIS overlay, arbitrary
geometry upload, multi-feature aggregation, semantic land-cover/cloud/change
accuracy, multi-CRS coverage, dataset-wide admission or Qwen/model reasoning.
Broader geometry and formulas require their own task-derived contract,
independent reference and evaluator-backed acceptance.
