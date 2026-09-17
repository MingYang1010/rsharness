#!/usr/bin/env python3
"""Reject data/runtime payloads in the staged index, including forced adds."""
from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_STAGE_BYTES = 10 * 1024 * 1024
BLOCKED_PARTS = frozenset({
    "datasets", "models", "weights", "checkpoints", "runtime", "artifacts",
    "state", "logs", "downloads", "metadata", "indexes", "index_files",
    "node_modules", ".venv", "venv", "__pycache__",
})
BLOCKED_SUFFIXES = (
    ".tif", ".tiff", ".jp2", ".h5", ".hdf5", ".nc", ".npy", ".npz",
    ".parquet", ".arrow", ".safetensors", ".pth", ".pt", ".ckpt", ".bin",
    ".zip", ".7z", ".tar", ".tar.gz", ".tgz", ".zst", ".sqlite",
    ".sqlite3", ".db", ".jsonl", ".log",
)


def check_entry(path: str, size: int, mode: str = "100644") -> list[str]:
    parts = PurePosixPath(path).parts
    errors = []
    if set(parts) & BLOCKED_PARTS:
        errors.append("data/runtime directory")
    if path.lower().endswith(BLOCKED_SUFFIXES):
        errors.append("data/archive/model format")
    if any(p == ".env" or p.startswith(".env.") and p != ".env.example" for p in parts):
        errors.append("environment/secret file")
    if size > MAX_FILE_BYTES:
        errors.append("file exceeds 2 MiB")
    if mode != "100644" and mode != "100755":
        errors.append("symlinks and submodules require explicit review")
    return errors


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def main() -> int:
    changed = set(git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split(b"\0"))
    failures = []
    total = 0
    count = 0
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        if raw_path not in changed:
            continue
        mode, oid, stage = header.decode("ascii").split()
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if stage != "0":
            failures.append(f"{path}: unresolved index stage")
            continue
        size = int(git("cat-file", "-s", oid))
        count += 1
        total += size
        failures.extend(f"{path}: {reason}" for reason in check_entry(path, size, mode))
    if total > MAX_STAGE_BYTES:
        failures.append("staged payload exceeds 10 MiB")
    if failures:
        print("Git payload rejected:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print(f"Git payload OK: {count} files, {total} bytes; no data/runtime payload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
