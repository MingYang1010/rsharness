"""Trusted internal content store. Never publish its port or token to an Agent."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from .core.storage.quota import QuotaError, StorageQuota, checked_path

MAX_OBJECT_BYTES = 64 * 1024 * 1024
OBJECT_OVERHEAD = 64 * 1024


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_app(quota: StorageQuota | None = None, token: str | None = None) -> FastAPI:
    if quota is None:
        quota = StorageQuota(Path(os.environ["EO_STORAGE_RUNTIME"]))
    if token is None:
        token = Path(os.environ["EO_STORAGE_TOKEN_FILE"]).read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("storage token must be 32 random bytes encoded as hex")
    app = FastAPI(title="EO Harness private storage broker", docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(request: Request) -> None:
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + token).encode()):
            raise HTTPException(401, "storage credential required")

    def object_path(digest: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise HTTPException(404, "object not found")
        path = quota.root / "managed-artifacts" / digest[:2] / digest / "content"
        checked_path(quota.root, path)
        return path

    def verify_existing(path: Path, digest: str) -> None:
        if path.stat().st_size > MAX_OBJECT_BYTES or file_hash(path) != digest:
            raise HTTPException(409, "stored object corrupt; operator review required")

    async def receive(request: Request, size: int, digest: str, stream=None) -> None:
        received = 0
        checksum = hashlib.sha256()
        async with asyncio.timeout(30):
            async for block in request.stream():
                received += len(block)
                if received > size:
                    raise HTTPException(413, "request exceeds declared size")
                checksum.update(block)
                if stream is not None:
                    stream.write(block)
        if received != size or checksum.hexdigest() != digest:
            raise HTTPException(422, "content length or checksum mismatch")

    @app.get("/healthz")
    def health():
        return {"status": "ok", "storage": "content-addressed-with-runtime-quota"}

    @app.put("/objects/{digest}")
    async def put(digest: str, request: Request):
        authorize(request)
        try:
            size_text = request.headers.get("content-length", "")
            if not re.fullmatch(r"[0-9]{1,10}", size_text):
                raise HTTPException(411, "bounded content length required")
            size = int(size_text)
            if size > MAX_OBJECT_BYTES:
                raise HTTPException(413, "object exceeds byte limit")
            destination = object_path(digest)
            if destination.is_file():
                verify_existing(destination, digest)
                await receive(request, size, digest)
                return {"sha256": digest, "size_bytes": size, "created": False}
            # Include a crash-left partial plus this attempt; retries never publish
            # a partial object. Additional abandoned files can deny future grants.
            with quota.hold(destination.parent, 2 * size + OBJECT_OVERHEAD, "artifact-upload"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(prefix="upload-", dir=destination.parent)
                temporary = Path(temporary)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        await receive(request, size, digest, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, destination)
                    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    # Only this request's unique temporary file; never user inputs.
                    temporary.unlink(missing_ok=True)
            return {"sha256": digest, "size_bytes": size, "created": True}
        except QuotaError as error:
            status = 503 if str(error) == "scope_writer_active" else 507
            raise HTTPException(status, "storage reservation unavailable") from None
        except TimeoutError:
            raise HTTPException(408, "upload deadline exceeded") from None

    @app.get("/objects/{digest}")
    def get(digest: str, request: Request):
        authorize(request)
        try:
            path = object_path(digest)
        except QuotaError:
            raise HTTPException(409, "invalid stored object path") from None
        if not path.is_file():
            raise HTTPException(404, "object not found")
        verify_existing(path, digest)
        return FileResponse(path, media_type="application/octet-stream")

    return app
