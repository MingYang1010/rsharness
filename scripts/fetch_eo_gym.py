#!/usr/bin/env python3
"""Download pinned EO-Gym source, without datasets, using a selected network mode.

Upstream files remain external runtime dependencies, never vendored into Git.
The separate archive option is resumable and verifies SHA-256 before renaming.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "harness_api"))
from app.v2.storage.quota import CONTROL_ALLOWANCE, StorageQuota, checked_path
LOCK = REPO_ROOT / "config" / "eo-gym-source.json"


def safe_relative(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if not path or value.is_absolute() or ".." in value.parts or "\\" in path:
        raise ValueError("unsafe upstream path")
    return value


def source_file(path: str) -> bool:
    safe_relative(path)
    return (
        path in {"pyproject.toml", "README.md", "LICENSE", "LICENSE.txt", "uv.lock"}
        or path.startswith("src/") and path.endswith(".py")
        or path.startswith("config/") and path.endswith(".toml")
    )


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def network_opener(network: str) -> urllib.request.OpenerDirector:
    if network not in {"direct", "system-proxy"}:
        raise ValueError("unknown network mode")
    handler = urllib.request.ProxyHandler({}) if network == "direct" else urllib.request.ProxyHandler()
    return urllib.request.build_opener(handler)


def fetch_source(destination: Path, lock: dict, network: str = "system-proxy") -> dict:
    opener = network_opener(network)
    root_url = "https://huggingface.co"
    repo, revision = lock["repository"], lock["revision"]
    url = f"{root_url}/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000"
    with opener.open(url, timeout=30) as response:
        if 'rel="next"' in response.headers.get("Link", ""):
            raise RuntimeError("upstream tree is paginated; refuse incomplete acquisition")
        tree = json.loads(response.read(4 * 1024 * 1024))
    selected = [e for e in tree if e["type"] == "file" and source_file(e["path"])]
    size = sum(e["size"] for e in selected)
    if not selected or size > lock["source_budget_bytes"]:
        raise ValueError("source acquisition exceeds pinned budget or is empty")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in selected:
        if entry.get("lfs"):
            raise ValueError("unexpected large file in source selection")
        relative = safe_relative(entry["path"])
        target = destination.joinpath(*relative.parts)
        checked_path(destination.resolve(), target)
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("source path escapes destination")
        data = target.read_bytes() if target.is_file() else None
        expected = entry["oid"]
        def valid(value: bytes | None) -> bool:
            return value is not None and len(value) == entry["size"] and hashlib.sha1(
                f"blob {len(value)}\0".encode() + value
            ).hexdigest() == expected
        if not valid(data):
            path = urllib.parse.quote(str(relative), safe="/")
            with opener.open(f"{root_url}/datasets/{repo}/resolve/{revision}/{path}", timeout=30) as response:
                data = response.read(entry["size"] + 1)
            if not valid(data):
                raise ValueError(f"upstream Git blob verification failed: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            checked_path(destination.resolve(), temporary)
            temporary.write_bytes(data)
            temporary.replace(target)
        records.append({"path": str(relative), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    report = {"revision": revision, "files": records, "bytes": size, "network": network,
              "software_license": lock["software_license"]}
    receipt = destination / "acquisition.json"
    checked_path(destination.resolve(), receipt)
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return report


def fetch_archive(destination: Path, name: str, lock: dict, network: str = "system-proxy") -> Path:
    entry = lock["archives"][name]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    partial = destination / (name + ".partial")
    checked_path(destination.resolve(), target)
    checked_path(destination.resolve(), partial)
    if target.is_file():
        if target.stat().st_size == entry["size"] and digest(target) == entry["sha256"]:
            return target
        raise ValueError("existing archive checksum differs; preserve it for inspection")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry["size"]:
        raise ValueError("oversized partial archive")
    # Compressed files plus a conservative 5x extraction reserve. Extraction is separate.
    needed = entry["size"] * 6 - offset
    if needed > lock["new_storage_budget_bytes"] or shutil.disk_usage(destination).free < needed:
        raise ValueError("insufficient reserved download/extraction space")
    if offset < entry["size"]:
        url = f'https://huggingface.co/datasets/{lock["repository"]}/resolve/{lock["revision"]}/{name}'
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"})
        with network_opener(network).open(request, timeout=45) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("server does not honor resume; partial preserved")
            with partial.open("ab") as stream:
                copied = offset
                checkpoint = copied
                while chunk := response.read(8 * 1024 * 1024):
                    copied += len(chunk)
                    if copied > entry["size"]:
                        raise ValueError("archive exceeds pinned size")
                    stream.write(chunk)
                    if copied - checkpoint >= 512 * 1024 * 1024:
                        print(json.dumps({"archive": name, "bytes": copied, "total": entry["size"]}), flush=True)
                        checkpoint = copied
    if partial.stat().st_size != entry["size"] or digest(partial) != entry["sha256"]:
        raise ValueError("archive checksum/size mismatch; partial preserved")
    os.replace(partial, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", choices=("EO_GYM_DATA.zip", "image_cache.zip"))
    parser.add_argument("--network", choices=("direct", "system-proxy"), default="system-proxy")
    args = parser.parse_args()
    if not args.destination.is_absolute():
        parser.error("destination must be absolute and outside tracked source paths")
    lock = json.loads(LOCK.read_text())
    result = acquire(args.destination, lock, args.archive, args.network,
                     StorageQuota(REPO_ROOT / "runtime"))
    if args.archive:
        print(result)
    else:
        print(json.dumps({"revision": result["revision"], "files": len(result["files"]), "bytes": result["bytes"], "network": args.network}))


def acquire(destination: Path, lock: dict, archive: str | None,
            network: str, quota: StorageQuota):
    """The CLI's mandatory reservation boundary, before network or payload writes."""
    checked_path(quota.root, destination)
    if archive:
        # One dedicated directory per acquisition. Existing partial files count
        # within—not in addition to—the maximum total directory reservation.
        capacity = lock["archives"][archive]["size"] * 6 + CONTROL_ALLOWANCE
    else:
        capacity = lock["source_budget_bytes"] * 2 + CONTROL_ALLOWANCE
    with quota.hold(destination, capacity, "eo-gym-archive" if archive else "eo-gym-source"):
        if archive:
            return fetch_archive(destination, archive, lock, network)
        return fetch_source(destination, lock, network)


if __name__ == "__main__":
    main()
