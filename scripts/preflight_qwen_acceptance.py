#!/usr/bin/env python3
"""Machine-check all non-GPU prerequisites for real Qwen acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "config" / "qwen-dataset-matrix-v1.json"
MODEL_FILES = [
    "config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja",
]
RUNTIME_FILES = [
    "compose.qwen-runner.yaml", "scripts/run_qwen_agent.py",
    "scripts/issue_agent_session.py", "scripts/manage_agent_registry.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: list[str]) -> tuple[int, str]:
    completed = subprocess.run(args, text=True, capture_output=True, check=False, timeout=30)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def matrix_status() -> dict:
    value = json.loads(MATRIX.read_text())
    datasets = value["datasets"]
    ready = [item for item in datasets if item.get("status", "ready") == "ready"]
    samples = sum(len(item.get("samples", [])) for item in ready)
    pending = [item["dataset_id"] for item in datasets if item.get("status", "ready") != "ready"]
    return {"ready_datasets": len(ready), "total_samples": samples,
            "pending_datasets": pending, "matrix_sha256": sha256(MATRIX)}


def git_status() -> dict:
    code, branch = command(["git", "-C", str(ROOT), "branch", "--show-current"])
    code, head = command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    code, dirty = command(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {"branch": branch, "commit": head, "clean": not dirty}


def model_status(model_root: Path) -> dict:
    files = {}
    for name in MODEL_FILES:
        path = model_root / name
        files[name] = {"exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None}
    weights = sorted(model_root.glob("model.safetensors-*.safetensors"))
    expected = json.loads((model_root / "model.safetensors.index.json").read_text()).get("weight_map", {})
    expected_files = sorted(set(expected.values()))
    return {"model_root": str(model_root), "model_name": "Qwen3.5-9B",
            "files": files, "weight_file_count": len(weights),
            "weight_files_complete": [path.name for path in weights] == expected_files}


def environment_status(python: Path) -> dict:
    if not python.is_file():
        return {"available": False}
    code, output = command([str(python), "-c",
        "import sys,vllm,torch,transformers; print(sys.version.split()[0]); print(vllm.__version__); print(torch.__version__); print(transformers.__version__)"])
    if code != 0:
        return {"available": False, "error": output}
    python_version, vllm, torch, transformers = output.splitlines()
    return {"available": True, "python": str(python), "python_version": python_version,
            "vllm": vllm, "torch": torch, "transformers": transformers}


def gpu_status() -> dict:
    code, output = command(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])
    if code != 0:
        return {"available": False, "error": output}
    gpus = []
    for line in output.splitlines():
        index, used, total, utilization = [part.strip() for part in line.split(",")]
        gpus.append({"index": int(index), "used_mib": int(used.split()[0]), "total_mib": int(total.split()[0]),
                     "utilization_percent": int(utilization.split()[0])})
    return {"available": True, "gpus": gpus}


def slurm_status() -> dict:
    code, output = command(["sinfo", "-N", "-o", "%N %T %E"])
    nodes = []
    for line in output.splitlines()[1:]:
        name, state, reason = line.strip().split(maxsplit=2)
        nodes.append({"node": name, "state": state, "reason": reason})
    code, queue = command(["squeue", "-h"])
    return {"nodes": nodes, "queue_output": queue, "allocation_ready": bool(nodes) and all(node["state"].startswith("idle") for node in nodes)}


def execution_status(mode: str, gpu_index: int) -> dict:
    if mode not in {"slurm", "direct-admin"}:
        raise SystemExit("execution mode must be slurm or direct-admin")
    result: dict = {"mode": mode}
    if mode == "direct-admin":
        code, state_output = command([
            "systemctl", "is-active", "slurmd.service",
            "slurm-ssh-tunnel.service", "slurmstepd.scope",
        ])
        states = [line.strip() for line in state_output.splitlines() if line.strip()]
        result["slurm"] = {"service_states": states, "allocation_ready": False}
        result["slurm_inactive"] = code != 0 and bool(states) and all(
            state in {"inactive", "failed"} for state in states
        )
        gpu = gpu_status()
        selected = next((item for item in gpu.get("gpus", []) if item["index"] == gpu_index), None)
        result["gpu_index"] = gpu_index
        result["selected_gpu"] = selected
        result["selected_gpu_idle"] = bool(
            selected and selected["used_mib"] == 0 and selected["utilization_percent"] == 0
        )
        result["ready"] = result["slurm_inactive"] and result["selected_gpu_idle"]
        return result
    result["slurm"] = slurm_status()
    result["ready"] = result["slurm"]["allocation_ready"]
    return result


def runtime_file_status() -> dict:
    return {name: {"exists": (ROOT / name).is_file(), "sha256": sha256(ROOT / name) if (ROOT / name).is_file() else None}
            for name in RUNTIME_FILES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/sata/yangm/models/Qwen3.5-9B"))
    parser.add_argument("--qwen-python", type=Path, default=Path("/sata/yangm/miniconda3/envs/qwen35vllm/bin/python"))
    parser.add_argument("--execution-mode", choices=["slurm", "direct-admin"], default="slurm")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": "qwen-acceptance-preflight-v1",
        "matrix": matrix_status(),
        "git": git_status(),
        "model": model_status(args.model_root),
        "environment": environment_status(args.qwen_python),
        "gpu": gpu_status(),
        "execution": execution_status(args.execution_mode, args.gpu_index),
        "runtime_files": runtime_file_status(),
        "proxy": {"configured": bool(os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY"))},
        "real_model_acceptance_ready": False,
    }
    report["real_model_acceptance_ready"] = (
        not report["matrix"]["pending_datasets"] and report["matrix"]["total_samples"] == 22
        and report["git"]["clean"] and report["model"]["weight_files_complete"]
        and report["environment"]["available"] and report["execution"]["ready"]
    )
    content = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
