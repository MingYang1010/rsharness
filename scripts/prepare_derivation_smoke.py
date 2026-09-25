#!/usr/bin/env python3
"""Create a new reviewed single-image task to test equal bytes, distinct lineage."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.core.artifact_identity import DERIVATION_SCHEME
from app.core.capabilities import TaskRegistry
from app.core.storage.quota import StorageQuota
from prepare_catalog_smoke import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-run", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    # Existing preparer validates paths, bounds, reviewed hash/dimensions and quota.
    # It refuses any preexisting destination; do not resume partial preparation.
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_catalog_smoke.py"),
                    "--sample-run", str(args.sample_run), "--output-name", args.output_name], check=True)
    out = ROOT / "runtime" / args.output_name
    with StorageQuota(ROOT / "runtime").hold(out, 2 * 1024 * 1024, "derivation-task-configuration"):
        task_path = out / "tasks/catalog-smoke/task.json"
        task = read_json(task_path)
        task["task_id"] += "-derivation"
        task["metadata"].update(artifact_identity=DERIVATION_SCHEME, acceptance="scripted-multi-derivation-only")
        task["prompt"] = "Inspect the image and request two central crops with slightly different normalized bounds. Cite both distinct derivations even if their pixel contents are identical."
        task_path.write_text(json.dumps(task, indent=2) + "\n")
        job = read_json(out / "job.json")
        job["task_ref"]["task_id"] = task["task_id"]
        (out / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        manifest = TaskRegistry(out / "tasks").get(task["task_id"], task["task_version"])
        print(json.dumps({"task_ref": job["task_ref"], "task_manifest_hash": manifest.task_manifest_hash,
                          "artifact_identity": DERIVATION_SCHEME}))


if __name__ == "__main__":
    main()
