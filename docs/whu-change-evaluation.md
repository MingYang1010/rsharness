# WHU temporal change evaluation

This checkpoint adds a real-label temporal evaluator to the headless EO Harness.
It does not add a learned change-detection model or claim general temporal-tool
coverage.

## Contract

whu-building-change-v1@1.0.0 evaluates two public 512 x 512 RGB inputs against
two hidden human-edited WHU building masks. The public assets carry
temporal_before and temporal_after roles. The labels carry evaluator,
building_label and the matching temporal role, but are not task inputs.

The Harness computes XOR change from the two binary masks and derives the
change class, change direction and changed-pixel fraction. The five metrics are:

- task.change_class_accuracy
- task.direction_accuracy
- task.changed_fraction_score
- evidence.faithfulness
- process.efficiency

Faithfulness requires submitted, frozen, full-image PNG crop evidence for both
dates. Each artifact must exist in the content-addressed store and have
eo_gym.crop lineage pointing to exactly one corresponding public input.

Evaluation runs only when task metadata explicitly contains
evaluation_profile: whu-building-change-v1. Existing headless tasks that retain
placeholder WorldCover evaluator metadata remain unevaluated.

## Fail-closed data boundary

The committed file config/whu-change-samples.json contains only three reviewed
sample receipts, source hashes, dimensions and expected counts. It contains no
image or label bytes.

The preparation script verifies every source file's SHA-256, size, dimensions,
band count, dtype, lack of georeferencing and label value domain. It copies six
public RGB tiles into the ignored provider runtime and six labels into a
separate ignored evaluator runtime. The provider manifest contains only the RGB
asset IDs. Compose mounts the label directory into the Harness and replay
containers read-only; it never mounts labels into the provider or Agent
checker.

The official download page did not state an explicit redistribution license
when reviewed. Treat these files as internal research data. Do not commit,
publish, sync or redistribute the copied RGB tiles or labels.

## Three-sample acceptance

The gateway checker is an operator oracle, not an autonomous Agent: it reads the
frozen expected answer from the operator-mounted acceptance job and submits it
through the same scoped gateway surface. This proves network isolation, action,
evidence, hidden-label scoring and replay behavior; it does not measure model
reasoning or task accuracy. Reports record `oracle_answer_injected: true`.

| Sample | Class | Direction | Changed fraction |
| --- | --- | --- | ---: |
| 0_224.tif | no_change | no_change | 0 |
| 0_137.tif | minor_change | reduction | 0.02618408203125 |
| 0_255.tif | major_change | expansion | 0.6843147277832031 |

Prepare a fresh ignored runtime with the source directory mounted read-only:

~~~bash
export EO_SMOKE_ROOT=./runtime/whu-change-accept-20260920
export WHU_SOURCE='/sata/yangm/datasets/source_datasets/WHU_Building_Change/extracted/Building change detection dataset_add/1. The two-period image data'

docker run --rm --read-only \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/runtime:/workspace/runtime:rw" \
  -v "$WHU_SOURCE:/source:ro" \
  --tmpfs /tmp:size=256m -w /workspace \
  -e PYTHONPATH=/workspace/harness_api \
  eo-harness/harness-api:0.4.0 \
  python scripts/prepare_whu_change_smoke.py \
    --source /source --output-name whu-change-accept-20260920
~~~

Start the isolated storage, provider and Harness:

~~~bash
docker compose \
  -f compose.eo-gym-smoke.yaml \
  -f compose.agent-smoke.yaml \
  -f compose.whu-change-smoke.yaml \
  up -d storage provider harness
~~~

Issue the three scoped credentials sequentially so the shared registry update is
atomic, then start the gateway:

~~~bash
for service in issue-whu-0-224 issue-whu-0-137 issue-whu-0-255; do
  docker compose \
    -f compose.eo-gym-smoke.yaml \
    -f compose.agent-smoke.yaml \
    -f compose.whu-change-smoke.yaml \
    --profile whu-agent-setup run -T --rm "$service"
done

docker compose \
  -f compose.eo-gym-smoke.yaml \
  -f compose.agent-smoke.yaml \
  -f compose.whu-change-smoke.yaml \
  --profile agent up -d agent-gateway
~~~

Run the three Agent-isolated interactions, then verify the hidden evaluations
from the operator network:

~~~bash
for service in check-whu-0-224 check-whu-0-137 check-whu-0-255; do
  docker compose \
    -f compose.eo-gym-smoke.yaml \
    -f compose.agent-smoke.yaml \
    -f compose.whu-change-smoke.yaml \
    --profile agent --profile whu-agent run -T --rm "$service"
done

for service in verify-whu-0-224 verify-whu-0-137 verify-whu-0-255; do
  docker compose \
    -f compose.eo-gym-smoke.yaml \
    -f compose.agent-smoke.yaml \
    -f compose.whu-change-smoke.yaml \
    --profile whu-verify run -T --rm "$service"
done
~~~

Each accepted report must show all five metrics and aggregate reward equal to
1.0, a passing structural replay, and two labels absent from the provider
manifest.

## Fresh execution replay

For each accepted episode, create a fresh ignored report directory and run the
replay service with all four Compose files:

~~~bash
export EO_REPLAY_PROVIDER_ID="$(docker compose \
  -f compose.eo-gym-smoke.yaml \
  -f compose.agent-smoke.yaml \
  -f compose.whu-change-smoke.yaml ps -q provider)"

export EO_REPLAY_EPISODE='<episode ID from accepted report>'
export EO_REPLAY_RUN='whu-change-replay-0-224'

docker compose \
  -f compose.eo-gym-smoke.yaml \
  -f compose.execution-replay.yaml \
  -f compose.whu-change-replay.yaml \
  run -T --rm replay
~~~

A valid execution replay reports status: passed,
semantic_evaluator_replayed: true, evaluation_matched: true, and exact artifact
metadata/content matches. It reuses only the reviewed public inputs and the
read-only hidden evaluator directory; it does not read original artifact bytes.

## Boundaries

## Answerability development package

`config/whu-answerability-dev-v1.json` freezes a 40-episode development package from the reviewed WHU test split: 30 expected submissions and 10 expected abstentions. The expected-submission strata cover all seven configured class/direction combinations (7 no-change, 20 minor-change, and 3 major-change cases by class). Abstention cases are metadata-controlled variants of 10 distinct source tiles, with after-image public coverage between 0.25 and 0.70 while the minimum remains 0.80.

Generate the package with `scripts/prepare_whu_change_smoke.py` and the config override. The preparer copies 80 public image inputs and 80 hidden evaluator labels, verifies source bytes against SHA-256 receipts, and writes 40 task manifests and jobs. It never mounts evaluator labels in the provider dataset and retains `redistribution_allowed: false`.

This package supports development/debugging for action, answerability, and cost analysis. It is not a held-out statistical benchmark until scene/event grouping and a fixed test split are declared.

This checkpoint proves a deterministic, evidence-grounded evaluator for three
real WHU pairs. It does not yet cover cloudy or insufficient-coverage examples,
wrong-date task construction, calibrated abstention, aligned temporal-stack
artifacts, general change masks, or Qwen3.5-9B behavior. Those remain separate
roadmap items.
