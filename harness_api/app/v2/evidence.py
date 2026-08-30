from typing import Dict

from .schemas import AssetRef, EvidenceRef, SpatialBoundingBox


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
    assets: Dict[str, AssetRef],
) -> None:
    asset = assets.get(evidence.source_ref)
    if asset is None:
        raise ValueError("evidence source_ref is not an accessible M1 asset")
    if evidence.frozen_sha256 != asset.sha256:
        raise ValueError("evidence frozen_sha256 does not match the source asset")
    if evidence.selector.bbox is not None and not _contains(
        asset.spatial.bbox,
        evidence.selector.bbox,
    ):
        raise ValueError("evidence bbox is outside the source asset extent")
    if evidence.selector.time_range is not None:
        if asset.temporal is None:
            raise ValueError("evidence time_range requires a temporal source extent")
        if (
            evidence.selector.time_range.start < asset.temporal.start
            or evidence.selector.time_range.end > asset.temporal.end
        ):
            raise ValueError("evidence time_range is outside the source asset extent")
