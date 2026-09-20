#!/usr/bin/env python3
"""Replay a terminal episode against a fresh isolated provider; write audit only."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.eo_gym_bridge import REVISION, SOURCE_FILES_HASH
from app.v2.capabilities import TaskRegistry
from app.v2.evaluation import EvaluatorRegistry
from app.v2.execution_replay import ReplayError, read_snapshot, replay_episode
from app.v2.renderer.terriamap import TerriaMapRenderer
from app.v2.schemas import V2EpisodeState
from app.v2.storage.quota import StorageQuota
from app.v2.tools.eo_gym import EOGymExecutor
from app.v2.tools.raster import RasterExecutor
from app.v2.tools.raster_grid import RasterGridExecutor
from app.v2.tools.raster_zonal import RasterZonalExecutor
from app.v2.tools.temporal import TemporalExecutor
from app.v2.tools.runtime import ToolRouter


def runtime_fingerprint(renderer_config=None):
    files = ["harness_api/app/eo_gym_bridge.py", "scripts/eo_gym_crop_worker.py",
             "harness_api/app/v2/tools/eo_gym.py", "harness_api/app/v2/tools/catalog.py",
             "harness_api/app/v2/tools/runtime.py", "harness_api/app/v2/tool_execution.py",
             "harness_api/app/v2/store.py", "harness_api/app/v2/domain.py", "harness_api/app/v2/schemas.py",
             "harness_api/app/v2/artifacts.py", "harness_api/app/v2/artifact_identity.py",
             "harness_api/app/v2/execution_replay.py", "harness_api/app/raster_bridge.py",
             "harness_api/app/v2/raster_math.py", "harness_api/app/v2/tools/raster.py", "scripts/raster_worker.py",
             "scripts/raster_ndmi_worker.py",
             "harness_api/app/v2/raster_grid.py", "harness_api/app/v2/tools/raster_grid.py", "scripts/raster_grid_worker.py",
             "harness_api/app/v2/raster_zonal.py", "harness_api/app/v2/tools/raster_zonal.py", "scripts/raster_zonal_worker.py",
             "harness_api/app/v2/temporal.py", "harness_api/app/v2/tools/temporal.py", "scripts/temporal_stack_worker.py",
             "harness_api/app/v2/evaluation.py", "harness_api/app/v2/renderer/terriamap.py"]
    versions = {}
    for name in ("Pillow", "requests", "numpy", "rasterio", "pydantic", "fastapi", "httpx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    fingerprint = {"python": platform.python_version(), "packages": versions, "upstream_revision": REVISION,
            "upstream_source_manifest_sha256": SOURCE_FILES_HASH,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}}
    if renderer_config is not None:
        fingerprint["renderer_config_sha256"] = hashlib.sha256(
            renderer_config.read_bytes()
        ).hexdigest()
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--provider", default="http://provider:8081")
    parser.add_argument("--raster-provider", help="fresh isolated native-band provider origin")
    parser.add_argument("--raster-provider-instance", help="operator-inspected raster container ID")
    parser.add_argument("--provider-instance", help="operator-inspected fresh crop-provider container ID")
    parser.add_argument("--renderer-url", help="fresh deterministic renderer origin")
    parser.add_argument("--renderer-instance", help="operator-inspected renderer container ID")
    parser.add_argument("--datasets", type=Path, help="read-only evaluator dataset root")
    parser.add_argument(
        "--renderer-config",
        type=Path,
        default=ROOT / "config" / "v2" / "renderer.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("/managed"))
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("preserve existing replay report; choose a fresh path")
    if not args.report.resolve().is_relative_to(args.runtime_root.resolve()):
        raise SystemExit("report must be quota-managed")
    import re
    if args.provider_instance and not re.fullmatch(r"[a-f0-9]{64}", args.provider_instance):
        raise SystemExit("provider-instance must be a real container ID")
    if args.raster_provider and not re.fullmatch(r"[a-f0-9]{64}", args.raster_provider_instance or ""):
        raise SystemExit("raster-provider-instance must be a real container ID")
    if args.renderer_url and not re.fullmatch(r"[a-f0-9]{64}", args.renderer_instance or ""):
        raise SystemExit("renderer-instance must be a real container ID")
    with StorageQuota(args.runtime_root).hold(args.report.parent, 2 * 1024 * 1024, "execution-replay-report"):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {"mode": "execution", "status": "incomplete", "original_episode_id": args.episode_id}
        runtime_renderer_config = None
        before_runtime = runtime_fingerprint()
        try:
            snapshot = read_snapshot(args.database, args.episode_id)
            registry = TaskRegistry(args.tasks)
            state = V2EpisodeState.model_validate_json(snapshot.episode["state_json"])
            manifest = registry.get(
                snapshot.episode["task_id"], snapshot.episode["task_version"]
            )
            rendered_profile = (
                manifest.task.metadata.get("observation_profile")
                == "rendered-worldcover-v1"
            )
            crop_recorded = any(
                row["tool_id"] == "eo_gym.crop" for row in snapshot.tool_runs
            )
            if crop_recorded and not args.provider_instance:
                raise ReplayError("provider_instance_required")
            if rendered_profile and not args.renderer_url:
                raise ReplayError("renderer_url_required")
            if rendered_profile and not args.renderer_instance:
                raise ReplayError("renderer_instance_required")
            if state.evaluation is not None and args.datasets is None:
                raise ReplayError("evaluator_dataset_root_required")
            renderer_config = None
            if rendered_profile:
                if (
                    args.renderer_config.is_symlink()
                    or not args.renderer_config.is_file()
                    or args.renderer_config.stat().st_size > 64 * 1024
                ):
                    raise ReplayError("renderer_config_unavailable")
                try:
                    renderer_config = json.loads(args.renderer_config.read_text())
                except (OSError, ValueError):
                    raise ReplayError("renderer_config_invalid")
                if not isinstance(renderer_config, dict):
                    raise ReplayError("renderer_config_invalid")
                runtime_renderer_config = args.renderer_config
                before_runtime = runtime_fingerprint(runtime_renderer_config)

            def executor_factory(artifacts):
                return ToolRouter(EOGymExecutor(args.provider, artifacts),
                    RasterExecutor(args.raster_provider, artifacts) if args.raster_provider else None,
                    RasterGridExecutor(args.raster_provider, artifacts) if args.raster_provider else None,
                    RasterZonalExecutor(args.raster_provider, artifacts) if args.raster_provider else None,
                    TemporalExecutor(args.raster_provider, artifacts) if args.raster_provider else None)

            with tempfile.TemporaryDirectory(prefix="execution-replay-") as directory:
                report = replay_episode(
                    snapshot,
                    registry,
                    Path(directory),
                    executor_factory if snapshot.tool_runs else None,
                    renderer_factory=(
                        lambda artifacts: TerriaMapRenderer(
                            args.renderer_url,
                            artifacts,
                            renderer_config,
                        )
                        if rendered_profile
                        else None
                    ),
                    evaluator_factory=(
                        lambda artifacts: EvaluatorRegistry(
                            str(args.datasets), artifacts
                        )
                        if state.evaluation is not None
                        else None
                    ),
                    renderer_config=renderer_config,
                )
            after = read_snapshot(args.database, args.episode_id)
            report["original_snapshot_unchanged"] = snapshot.fingerprint == after.fingerprint
            if not report["original_snapshot_unchanged"]:
                report.update(status="failed", reason="original_snapshot_changed_during_replay")
            manifest_after = TaskRegistry(args.tasks).get(snapshot.episode["task_id"], snapshot.episode["task_version"])
            report["task_snapshot_unchanged"] = manifest_after.task_manifest_hash == snapshot.episode["task_manifest_hash"]
            if not report["task_snapshot_unchanged"]:
                report.update(status="failed", reason="task_snapshot_changed_during_replay")
        except Exception as error:
            report.update(status="incomplete", reason=error.code if isinstance(error, ReplayError) else "invalid_or_unavailable_replay_input")
        report["replay_runtime"] = runtime_fingerprint(runtime_renderer_config)
        report["replay_runtime_unchanged"] = report["replay_runtime"] == before_runtime
        if not report["replay_runtime_unchanged"]:
            report.update(status="failed", reason="replay_runtime_changed")
        report["fresh_provider_container_id"] = args.provider_instance
        report["fresh_raster_provider_container_id"] = args.raster_provider_instance
        report["fresh_renderer_container_id"] = args.renderer_instance
        report["provider_instance_source"] = "operator-inspected; no historical container identity available"
        report["renderer_instance_source"] = "operator-inspected; no historical container identity available"
        report["original_artifact_bytes_read"] = False
        content = json.dumps(report, indent=2, ensure_ascii=False).encode()
        if len(content) > 1024 * 1024:
            raise SystemExit("replay report exceeds bound")
        with args.report.open("xb") as output:
            output.write(content)
        print(json.dumps({key: report.get(key) for key in ("mode", "status", "reason", "recorded_actions", "executed_actions", "original_snapshot_unchanged")}))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
