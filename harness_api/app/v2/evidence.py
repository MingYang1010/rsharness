from typing import Dict, Union

from .schemas import Artifact, EvidenceRef, PixelArtifactRef, PixelAssetRef, SpatialBoundingBox, TaskAsset


EvidenceSource = Union[TaskAsset, Artifact]


def _contains(
    outer: SpatialBoundingBox,
    inner: SpatialBoundingBox,
) -> bool:
    return (
        outer.west <= inner.west
        and outer.south <= inner.south
        and outer.east >= inner.east
        and outer.north >= inner.north
    )


def validate_evidence(
    evidence: EvidenceRef,
    sources: Dict[str, EvidenceSource],
) -> None:
    source = sources.get(evidence.source_ref)
    if source is None:
        raise ValueError("evidence source_ref is not an accessible source")
    if evidence.frozen_sha256 != source.sha256:
        raise ValueError("evidence frozen_sha256 does not match the source")
    if evidence.selector.geometry is not None and source.spatial is None:
        raise ValueError("geographic geometry requires a georeferenced source")
    if evidence.selector.pixel_window is not None:
        if isinstance(source, (PixelAssetRef, PixelArtifactRef)):
            source_width, source_height = source.pixel.width, source.pixel.height
        elif source.spatial is not None and source.spatial.shape is not None:
            source_height, source_width = source.spatial.shape[:2]
        else:
            raise ValueError("pixel_window requires verified source dimensions")
        x, y, width, height = evidence.selector.pixel_window
        if x + width > source_width or y + height > source_height:
            raise ValueError("evidence pixel_window is outside the source image")
    if isinstance(source, PixelAssetRef) and evidence.selector.bands:
        if not set(evidence.selector.bands).issubset(source.bands):
            raise ValueError("evidence bands are not declared by the source")
    if isinstance(source, PixelArtifactRef) and evidence.selector.bands:
        raise ValueError("pixel artifact has no declared named spectral bands")
    if evidence.selector.bbox is not None:
        if source.spatial is None or not _contains(
            source.spatial.bbox,
            evidence.selector.bbox,
        ):
            raise ValueError("evidence bbox is outside the source extent")
    if evidence.selector.time_range is not None:
        if source.temporal is None:
            raise ValueError("evidence time_range requires a temporal source extent")
        if (
            evidence.selector.time_range.start < source.temporal.start
            or evidence.selector.time_range.end > source.temporal.end
        ):
            raise ValueError("evidence time_range is outside the source extent")
