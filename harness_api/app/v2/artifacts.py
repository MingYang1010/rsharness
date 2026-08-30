import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

from .schemas import (
    ArtifactLineage,
    ArtifactRef,
    SpatialExtent,
    TemporalExtent,
)


class ArtifactStore:
    """Content-addressed artifact storage scaffold used by M2 workers."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def content_path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return self.root / "sha256" / sha256[:2] / sha256

    def put_bytes(
        self,
        content: bytes,
        kind: str,
        media_type: str,
        lineage: ArtifactLineage,
        spatial: Optional[SpatialExtent] = None,
        temporal: Optional[TemporalExtent] = None,
    ) -> ArtifactRef:
        digest = hashlib.sha256(content).hexdigest()
        destination = self.content_path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".%s." % digest,
                dir=str(destination.parent),
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, destination)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
        return ArtifactRef(
            artifact_id="art-%s" % digest,
            kind=kind,
            media_type=media_type,
            uri="artifact://sha256/%s/%s" % (digest[:2], digest),
            sha256=digest,
            size_bytes=len(content),
            spatial=spatial,
            temporal=temporal,
            lineage=lineage,
        )

    def audit_exists(self, artifact: ArtifactRef) -> bool:
        path = self.content_path(artifact.sha256)
        if not path.is_file() or path.stat().st_size != artifact.size_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == artifact.sha256
