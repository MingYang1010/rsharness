# Bounded Sentinel-2 STAC admission

This is an operator-only adapter, not an Agent Internet tool or global catalog.
Discovery yields candidates; only reviewed, verified local windows enter a task.
Data, STAC snapshots, legal/source receipts and generated tasks stay in ignored
`runtime/`. No public imagery or bulk metadata belongs in Git.

## Supported profile

`config/stac-sentinel-nanjing.json` pins a small Nanjing AOI, three 2024 Sentinel-2
L2A items, acquisition interval/cutoff and visual/red/nir/scl assets from Earth
Search. The operator explicitly reviews licensing before admission. The collection
reports `proprietary`; that original value is retained, not relabeled CC-BY.
The official [terms](https://dataspace.copernicus.eu/terms-and-conditions) and
[Sentinel legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)
allow lawful use/modification with attribution, no warranty and stated exceptions.
The reviewed legal PDF hash is pinned; a change requires a fresh review. This
workflow only prepares local research inputs, not an upstream redistribution.

The current receipt proves acquisition-time cutoff, not historical product
availability. Scene cloud percentage is not an AOI cloud mask. RGB is the display
TCI product, not calibrated reflectance. Red/NIR DN values remain unchanged with
STAC scale/offset metadata; SCL is categorical. These nine native-band/SCL windows
remain private in this RGB task. Only three visual inputs are mounted into the
crop provider and exposed through its episode catalog. A separate explicit
[native NDVI admission](native-raster-tools.md) uses six red/NIR windows without
changing this task; SCL/cloud tooling remains pending.

## Resource and integrity boundaries

- Explicit existing A800 HTTPS proxy; missing proxy fails. No tunnels, proxy
  configuration changes, redirects, asset query credentials or arbitrary hosts.
- Default HTTP application payload128MiB,512 requests,600s; JSON1MiB. These are
  application limits, not exact physical wire accounting. A rejected streaming
  response can consume one additional64KiB chunk before rejection.
- HEAD pins object size (<=2GiB) and strong ETag. Exact206 ranges use If-Match;
  range/length/ETag/encoding mismatch fails. Each response has a checksum receipt.
- Rasterio uses a custom seekable Python opener, never GDAL independent network
  reads. Range blocks256KiB, shared LRU2MiB; each read<=8MiB. Sidecars denied.
- AOI<=0.05 degrees per side, one to three pinned items, UTM CRS only. Validate
  actual CRS/transform/type/bands/shape, full AOI coverage and <=1024x1024 output.
- Native-resolution outward-rounded windows are not resampled. Pixels, CRS,
  transform, valid-data mask and scale/offset tags are reopened and checked before
  exclusive publication. Output<=16MiB per raster. Public spatial bbox is WGS84;
  the GeoTIFF's native grid/CRS is retained in private receipts.
- A256MiB shared ledger reservation precedes output/network. Failed jobs preserve
  outputs/receipts and conservative reservation. Existing output names are refused;
  automatic resume across changed metadata is not implemented. This does not
  close unrelated DB/log/installer hard-quota gaps.

## Operator workflow on A800

Run in the existing project image, CPU-only, read-only source and explicit existing
HTTPS proxy environment. Admission alone needs writable project `runtime/` and
network egress via that proxy. Do not expose the acquisition CLI to the Agent.

```sh
python scripts/admit_stac_windows.py --discover-only
python scripts/admit_stac_windows.py \
  --output-name <fresh-run-name> --reviewed-license
```

Then use `compose.eo-gym-smoke.yaml` plus `compose.agent-smoke.yaml` with a unique
project name and `EO_SMOKE_ROOT=./runtime/<fresh-run-name>`,
`EO_AGENT_RUN=<fresh-run-name>`, `EO_HARNESS_CATALOG_ENABLED=1`.
Start storage/provider/harness, run issue-agent, start agent-gateway, run
agent-check. Recreate only these four services, rerun issue-agent (preserves the
binding), then agent-check with `python /verify.py --resume`. No published ports;
Agent sees only the front network. The generated task enables derivation identity.
The smoke checks interaction, not land-cover/change accuracy or Qwen inference.

## Verified public-data slice, 2026-09-17

`runtime/stac-sentinel-nanjing-20260917-01` holds12 windows from
`S2A_50SPA_20240405_0_L2A`, `S2A_50SPA_20240415_0_L2A`,
`S2B_50SPA_20240510_0_L2A`, AOI `[118.79,31.99,118.81,32.01]`.
HTTP payload23,869,170B; no full-scene download. All12 raster outputs passed the
pixel/grid/mask/scaling check; three visual inputs were admitted.
Episode `ep2-5e59858853af47128171b06401d9fb6c` passed12 scripted actions with3 real
crop calls, denied private DNS/IP access, and recovered12 cached actions with0 new
crop calls after four-service recreation. Reports remain under the run's `reports/`.
Task hash `ed1ad7cdafdab3f86ecd37165234bba2f052b2bc85acf697deaef27c5de78ac9`.
Fresh-provider execution replay also passed12/12 actions, regenerated all three
crop hashes and preserved the original DB checksum; report:
`runtime/execution-stac-20260917-01/execution.json`.
At this admission checkpoint the A800 tree passed190 Python tests and4 renderer tests, including11
STAC admission/range/pixel/privacy regressions.

Still absent: other STAC providers/collections, signed-assets credentials, generic
historical freshness, full-source checksum downloads, resumable window acquisition,
window-specific cloud policies, native-band tools beyond NDVI and semantic tasks.
Planetary Computer was unreachable through the current proxy during discovery;
this is not evidence of general unavailability. Do not infer dataset-wide coverage
from these three date-selected smoke samples.
