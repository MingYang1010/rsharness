# Sentinel-2 temporal benchmark v1

## Acceptance status

The frozen A800 acceptance pack at
`runtime/temporal-benchmark-20260920-06` passed on 2026-09-20. Runtime imagery,
task instances, credentials, SQLite state, artifacts and reports are ignored and
must not be committed. The versioned repository contains only the preparation,
execution, evaluation and replay contracts needed to reproduce a fresh pack from
the separately reviewed local inputs.

This acceptance proves deterministic temporal selection, aligned artifact
construction, abstention scoring and replay for one bounded three-date
Sentinel-2 sample. It is an operator-oracle run, not a Qwen/model run and not a
claim of general temporal change understanding.

## Frozen data and policy

`config/temporal-benchmark-v1.json` pins three Sentinel-2 L2A items, each with a
red window and an SCL window. The preparer verifies the source manifest,
native-input manifest and license-review SHA-256 values before creating output.
It stages six bounded TIFF inputs under an ignored runtime scope and records:

- attribution: `Contains modified Copernicus Sentinel data (2024)`;
- scope: local research only;
- redistribution authorization: false;
- cloud policy: `sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1`;
- excluded SCL classes: 1, 3, 8, 9, 10 and 11.

Cloud fractions are recomputed from the frozen SCL pixels in each admitted
window, not copied from STAC item-level cloud-cover metadata:

| Item date | Window-level cloud fraction |
| --- | ---: |
| 2024-04-05 | 0.0 |
| 2024-04-15 | 0.0 |
| 2024-05-10 | 0.003558069519204452 |

The cloudy case sets `maximum_cloud_fraction=0.001`, so its rejection is backed
by the pinned SCL pixels. This remains policy-derived truth from the same SCL
source used by the tool. The separate
[CloudSEN12 policy benchmark](cloud-policy-benchmark-v1.md) now checks that
policy against high-quality manual labels and records two false accepts at the
normal 0.2 threshold; it does not retroactively turn this task's SCL-derived
expectation into independent truth.

## Frozen cases

The preparer emits four immutable task cases and one counterfactual acceptance
run. Each task exposes only `temporal.select_align@1.0.0`, allows abstention and
keeps the evaluator configuration behind the operator API.

| Case | Expected outcome | Task manifest SHA-256 |
| --- | --- | --- |
| `valid-pair` | select 2024-04-05 and 2024-05-10, save full-stack evidence, submit | `beaac875c0a46c4a888da5c93bfcf9a60fcc7c8b157e95f904560de97be14e38` |
| `cloudy-rejection` | reject as `cloudy`, then abstain | `92060917a40abe73844e0ccde6e511d20711b46f27b23be28d2eaf6cda9d80f4` |
| `insufficient-coverage` | reject as `insufficient_coverage`, then abstain | `99d91865b005aa0e10f03699fce851298f6c5f4119d5e0d67f1e05c1617cc2ae` |
| `wrong-date` | reject as `wrong_date`, then abstain | `d7ae84b73a20c89ddf04d93242cc159fdca53082c3695577738ac6d491a9231a` |
| `cloudy-rejection-false-confidence` | reuse the cloudy task but submit a confident pair claim | same as `cloudy-rejection` |

Pair selection applies date availability, AOI coverage, window-level cloud
fraction and matching platform/instrument in that order. Remaining candidates
are deterministically ranked by combined cloud fraction, coverage, acquisition
times and item IDs.

## Artifact and evidence contract

A selected pair produces a four-band GeoTIFF in this fixed order:
`before_red`, `before_scl`, `after_red`, `after_scl`. Red grids must match
exactly. SCL is aligned to each red grid with nearest-neighbour resampling, and
all bands share the intersection mask of both dates. The typed temporal artifact
records both items, acquisition times, input asset IDs and hashes, CRS,
transform, footprint, band order, per-date/aligned coverage, per-date cloud
fractions, alignment method, cloud policy and derivation lineage.

The successful task saves evidence over the full artifact bbox and time range.
Rejected tasks produce no artifact and must abstain without asserting a pair.

## Hidden evaluator

`temporal-selection-v1@1.0.0` scores five metrics:

| Metric | Weight |
| --- | ---: |
| `temporal.validity` | 0.3 |
| `spatial.coverage` | 0.2 |
| `answer.abstention_correctness` | 0.2 |
| `evidence.faithfulness` | 0.2 |
| `process.efficiency` | 0.1 |

The scoped Agent gateway does not expose hidden truth or evaluation state. The
acceptance driver injects an operator-known action sequence and records
`oracle_answer_injected=true`; therefore these scores validate Harness semantics,
not autonomous model reasoning. The counterfactual submits a confident answer
after a cloudy rejection and must receive zero for abstention correctness and
evidence faithfulness, aggregate reward approximately 0.6 and
`false_confidence=true`.

## Replay and isolation checks

Every run verifies idempotent gateway actions and denies the Agent direct access
to Harness, provider, raster and storage services. After recreating those
services, all cached actions must be byte-equivalent; the valid-pair artifact is
also re-read and checksum-verified.

Execution replay opens the original episode snapshot read-only, invokes a fresh
raster provider, regenerates actions/artifacts and re-runs the semantic evaluator.
It requires matching normalized state, trace, evaluation and artifact hashes,
while asserting that original artifact bytes were not read and the original task
and episode snapshots were unchanged. The valid-pair run additionally stops the
raster provider and verifies fail-closed replay with
`reason=action_execution_mismatch`.

## Accepted results

The final `summary.json` reports:

- correct run success: 4/4;
- false confidence among the three correct rejection runs: 0/3;
- counterfactual false-confidence detection: 1/1;
- structural replay, post-recreation cached access and fresh execution replay:
  passed for all five runs;
- valid-pair offline negative replay: failed closed as expected;
- original snapshot unchanged: true.

The five accepted episode IDs are:

- `valid-pair`: `ep2-fb661cebc80140dfba6699f0be1d78ed`;
- `cloudy-rejection`: `ep2-2d2cc4f50f964c76901b55e96b5f0dcf`;
- `insufficient-coverage`: `ep2-155e5fc73f9a4d64af3af00d3c3aae1b`;
- `wrong-date`: `ep2-904338a7dc3c4cb5983a6e8f3dc5a980`;
- counterfactual: `ep2-cc4f4935838a4e878c1572efb5f43b1f`.

## Reproduction outline

Create a fresh ignored runtime path; the preparer and summarizer refuse to
overwrite existing evidence.

```bash
python scripts/prepare_temporal_benchmark.py \
  --source runtime/<reviewed-native-source> \
  --license-review runtime/<license-review.json> \
  --output runtime/<fresh-temporal-pack>
```

For each run, combine the normal deployment Compose file with
`compose.temporal-smoke.yaml`, issue a fresh scoped Agent session, run the gateway
oracle, run the hidden backend verifier, recreate services, recheck cached
actions and run `compose.temporal-replay.yaml` against a fresh raster provider.
Finally run:

```bash
python scripts/summarize_temporal_benchmark.py \
  --pack runtime/<fresh-temporal-pack> \
  --output runtime/<fresh-temporal-pack>/reports/summary.json
```

## Remaining research boundary

This pack does not establish sensor-general temporal selection,
change-detection accuracy, cross-task evidence memory, public/multitenant
security or real Qwen3.5-9B interaction. Independent cloud-policy validation is
reported separately and remains limited to two ROIs and eight windows; none of
those broader claims may be inferred from the 4/4 scripted result.
