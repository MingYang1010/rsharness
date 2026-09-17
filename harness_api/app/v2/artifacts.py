import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schemas import (
    ArtifactLineage,
    Artifact,
    ArtifactRef,
    PixelArtifactRef,
    PixelExtent,
    SpatialExtent,
    TemporalExtent,
)


@dataclass(frozen=True)
class ArtifactContent:
    content: bytes
    end: int
    partial: bool
    start: int
    total: int


class ArtifactStoreError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


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
        pixel: Optional[PixelExtent] = None,
    ) -> Artifact:
        if pixel is not None:
            self._validate_pixels(content, kind, media_type, pixel)
        digest = hashlib.sha256(content).hexdigest()
        artifact_type = PixelArtifactRef if pixel is not None else ArtifactRef
        artifact = artifact_type(
            artifact_id="art-%s" % digest,
            kind=kind,
            media_type=media_type,
            uri="artifact://sha256/%s/%s" % (digest[:2], digest),
            sha256=digest,
            size_bytes=len(content),
            spatial=spatial,
            temporal=temporal,
            lineage=lineage,
            **({"pixel": pixel} if pixel is not None else {}),
        )
        destination = self.content_path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_required = not destination.is_file()
        if not write_required:
            existing_digest = hashlib.sha256()
            with destination.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    existing_digest.update(chunk)
            write_required = (
                destination.stat().st_size != len(content)
                or existing_digest.hexdigest() != digest
            )
        if write_required:
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
        return artifact

    @staticmethod
    def _validate_pixels(content: bytes, kind: str, media_type: str, pixel: PixelExtent) -> None:
        # The first typed provider emits bounded PNGs only; do not let GDAL
        # interpret arbitrary VRT/remote references through this interface.
        if kind not in {"image", "raster"} or media_type != "image/png" or not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ArtifactStoreError("invalid_pixel_artifact", "typed pixel artifacts currently require PNG image content")
        if len(content) > 64 * 1024 * 1024:
            raise ArtifactStoreError("invalid_pixel_artifact", "pixel artifact exceeds byte limit")
        from rasterio.io import MemoryFile
        from rasterio.errors import RasterioError
        try:
            with MemoryFile(content) as memory, memory.open(driver="PNG") as image:
                if image.width * image.height > 20_000_000 or image.count > 4:
                    raise ArtifactStoreError("invalid_pixel_artifact", "pixel artifact exceeds decode limit")
                if (image.width, image.height, image.count) != (pixel.width, pixel.height, pixel.channels):
                    raise ArtifactStoreError("invalid_pixel_artifact", "pixel dimensions do not match decoded content")
                image.read()  # Validate the payload, not just a claimed PNG header.
        except RasterioError:
            raise ArtifactStoreError("invalid_pixel_artifact", "pixel artifact could not be decoded") from None

    def audit_exists(self, artifact: ArtifactRef) -> bool:
        path = self.content_path(artifact.sha256)
        if not path.is_file() or path.stat().st_size != artifact.size_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == artifact.sha256

    @staticmethod
    def _range_bounds(range_header: Optional[str], total: int) -> tuple[int, int, bool]:
        if range_header is None:
            return 0, max(total - 1, 0), False
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if match is None or (not match.group(1) and not match.group(2)):
            raise ArtifactStoreError(
                "invalid_range",
                "Range must use one bytes=start-end interval",
            )
        if total == 0:
            raise ArtifactStoreError(
                "range_not_satisfiable",
                "artifact content is empty",
            )
        start_text, end_text = match.groups()
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ArtifactStoreError(
                    "range_not_satisfiable",
                    "suffix byte range must be positive",
                )
            start = max(total - suffix_length, 0)
            end = total - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else total - 1
            if start >= total or end < start:
                raise ArtifactStoreError(
                    "range_not_satisfiable",
                    "requested byte range is outside artifact content",
                )
            end = min(end, total - 1)
        return start, end, True

    def read_content(
        self,
        artifact: ArtifactRef,
        range_header: Optional[str] = None,
    ) -> ArtifactContent:
        path = self.content_path(artifact.sha256)
        if not path.is_file():
            raise ArtifactStoreError(
                "artifact_content_missing",
                "artifact metadata exists but content is missing",
            )
        if path.stat().st_size != artifact.size_bytes:
            raise ArtifactStoreError(
                "artifact_content_corrupt",
                "artifact content size does not match metadata",
            )
        if not self.audit_exists(artifact):
            raise ArtifactStoreError(
                "artifact_content_corrupt",
                "artifact content checksum does not match metadata",
            )
        start, end, partial = self._range_bounds(range_header, artifact.size_bytes)
        with path.open("rb") as stream:
            stream.seek(start)
            content = stream.read(end - start + 1)
        return ArtifactContent(
            content=content,
            end=end,
            partial=partial,
            start=start,
            total=artifact.size_bytes,
        )
