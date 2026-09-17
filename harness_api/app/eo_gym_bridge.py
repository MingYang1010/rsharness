"""Restricted stateless EO-Gym provider; the Harness retains episode ownership.

Run only inside the documented no-egress, read-only container. It must receive
an image-only input mount, never the original dataset root or annotation index.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

REVISION = "d168c28e870ccbf75f63ce60d087420a6babed9e"
SOURCE_FILES_HASH = "0c139f2ba5ffa81a9a3490448863bf5988645f98980b252888aefdcfa555b968"
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class CropArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    asset_id: str = Field(pattern=r"^asset-[a-zA-Z0-9_-]{1,100}$")
    aoi: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def valid_aoi(self):
        x0, y0, x1, y1 = self.aoi
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("aoi must be finite normalized [x0,y0,x1,y1]")
        return self


class CropCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool_name: Literal["crop_optical_or_sar_image"]
    arguments: CropArguments


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream(source: Path) -> None:
    receipt = json.loads((source / "acquisition.json").read_text())
    records = receipt["files"]
    signature = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if receipt["revision"] != REVISION or signature != SOURCE_FILES_HASH:
        raise ValueError("unrecognized upstream source snapshot")
    for record in records:
        path = (source / record["path"]).resolve()
        if not path.is_relative_to(source.resolve()) or hash_file(path) != record["sha256"]:
            raise ValueError("upstream source checksum mismatch")


class CropBridge:
    def __init__(self, source: Path, inputs: Path, outputs: Path, manifest: dict, worker: Path):
        verify_upstream(source)
        self.source = source.resolve()
        self.inputs = inputs.resolve()
        self.outputs = outputs.resolve()
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.worker = worker.resolve()
        self.slot = threading.BoundedSemaphore(1)

    def resolve(self, asset_id: str) -> tuple[Path, str]:
        entry = self.manifest.get(asset_id)
        if not isinstance(entry, dict) or entry.get("role") != "input_image":
            raise HTTPException(403, "asset is not an approved image input")
        name = entry.get("filename", "")
        if not name or Path(name).name != name:
            raise HTTPException(403, "invalid image manifest path")
        path = self.inputs / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.inputs) or not path.is_file():
            raise HTTPException(403, "image input unavailable")
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise HTTPException(413, "input exceeds provider byte limit")
        expected = entry.get("sha256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or hash_file(path) != expected:
            raise HTTPException(409, "input checksum mismatch")
        return path, expected

    def crop(self, call: CropCall) -> dict:
        if not self.slot.acquire(blocking=False):
            raise HTTPException(429, "provider busy; retry later")
        try:
            path, input_hash = self.resolve(call.arguments.asset_id)
            with tempfile.TemporaryDirectory(prefix="crop-", dir=self.outputs) as work:
                command = [sys.executable, str(self.worker), str(self.source), str(path), work, json.dumps(call.arguments.aoi)]
                try:
                    run = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
                except subprocess.TimeoutExpired:
                    raise HTTPException(504, "upstream crop timed out") from None
                if run.returncode or len(run.stdout) > 8192:
                    raise HTTPException(502, "upstream crop failed")
                try:
                    result = json.loads(run.stdout)
                    if not isinstance(result, dict) or set(result) != {"width", "height", "bbox_px", "aoi_norm"}:
                        raise ValueError("invalid crop metadata")
                    if any(type(result[key]) is not int or result[key] <= 0 for key in ("width", "height")):
                        raise ValueError("invalid dimensions")
                except (ValueError, TypeError):
                    raise HTTPException(502, "invalid upstream crop result") from None
                temporary = Path(work) / "result.png"
                if not temporary.is_file() or temporary.stat().st_size > MAX_OUTPUT_BYTES:
                    raise HTTPException(502, "invalid or oversized upstream artifact")
                digest = hash_file(temporary)
                destination = self.outputs / (digest + ".png")
                size = temporary.stat().st_size
                temporary.replace(destination)
                return {"output": {**result, "artifact_id": "art-" + digest, "sha256": digest,
                                   "size_bytes": size, "media_type": "image/png",
                                   "input_asset_id": call.arguments.asset_id, "input_sha256": input_hash,
                                   "provider": "eo-gym", "upstream_revision": REVISION,
                                   "simulation": False}}
        finally:
            self.slot.release()


def create_app(bridge: CropBridge | None = None) -> FastAPI:
    if bridge is None:
        bridge = CropBridge(
            Path(os.environ["EO_BRIDGE_SOURCE"]), Path(os.environ["EO_BRIDGE_INPUTS"]),
            Path(os.environ["EO_BRIDGE_OUTPUTS"]),
            json.loads(Path(os.environ["EO_BRIDGE_MANIFEST"]).read_text()),
            Path(os.environ["EO_BRIDGE_WORKER"]),
        )
    app = FastAPI(title="EO Harness restricted EO-Gym provider", version="0.1.0")

    @app.get("/healthz")
    def health():
        return {"status": "ok", "upstream_revision": REVISION,
                "tools": ["crop_optical_or_sar_image"], "episode_owner": "harness"}

    @app.post("/execute")
    def execute(request: CropCall):
        return bridge.crop(request)

    @app.get("/artifacts/{digest}")
    def artifact(digest: str):
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise HTTPException(404, "artifact not found")
        path = bridge.outputs / (digest + ".png")
        if not path.is_file():
            raise HTTPException(404, "artifact not found")
        if hash_file(path) != digest:
            raise HTTPException(409, "artifact checksum mismatch")
        return FileResponse(path, media_type="image/png")

    return app
