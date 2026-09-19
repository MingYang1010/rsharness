"""Reviewed native-grid NDVI contract. No eval, reprojection or network IO."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .schemas import ArtifactId, Identifier, Sha256, UtcTimestamp, V2RequestModel

VERSION = "1.0.0"
MASKED_VERSION = "1.1.0"
TOOL_ID = "raster.band_math"
MAX_INPUT = 16 * 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024
NODATA = -9999.0
EPSILON = 1e-6
MEDIA_TYPE = "image/tiff"
CLOUD_POLICY = "sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"
CLOUD_EXCLUDED_CLASSES = (1, 3, 8, 9, 10, 11)


class BandMathArguments(V2RequestModel):
    operation: Literal["ndvi"]
    red_asset_id: Identifier
    nir_asset_id: Identifier
    mask_artifact_id: ArtifactId | None = None
    cloud_policy: Literal["sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"] | None = None

    @model_validator(mode="after")
    def distinct(self):
        if self.red_asset_id == self.nir_asset_id:
            raise ValueError("red and nir must be distinct inputs")
        if (self.mask_artifact_id is None) != (self.cloud_policy is None):
            raise ValueError("mask artifact and cloud policy must be provided together")
        return self

    @property
    def masked(self) -> bool:
        return self.mask_artifact_id is not None


class MaskArtifactInput(V2RequestModel):
    artifact_id: ArtifactId
    sha256: Sha256
    size_bytes: int = Field(gt=0, le=MAX_OUTPUT)
    cloud_policy: Literal["sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"]


class NativeBand(V2RequestModel):
    asset_id: Identifier
    sha256: Sha256
    item_id: str = Field(min_length=1, max_length=100)
    band: Literal["red", "nir"]
    acquired: UtcTimestamp
    crs: str = Field(pattern=r"^EPSG:32[67][0-9]{2}$")
    transform: list[float] = Field(min_length=6, max_length=6)
    width: int = Field(gt=0, le=1024)
    height: int = Field(gt=0, le=1024)
    dtype: Literal["uint16"]
    scale: float = Field(gt=0, le=1, allow_inf_nan=False)
    offset: float = Field(ge=-1, le=1, allow_inf_nan=False)
    nodata: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def grid(self):
        a,b,c,d,e,f = self.transform
        if not all(math.isfinite(v) for v in self.transform) or b != 0 or d != 0 or a <= 0 or e >= 0:
            raise ValueError("native north-up finite grid required")
        if not 1 <= int(self.crs[-2:]) <= 60:
            raise ValueError("invalid UTM zone")
        return self


class NDVIResult(V2RequestModel):
    operation: Literal["ndvi"]
    input_asset_ids: list[Identifier] = Field(min_length=2, max_length=2)
    input_sha256: list[Sha256] = Field(min_length=2, max_length=2)
    acquired: UtcTimestamp
    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    width: int = Field(gt=0, le=1024)
    height: int = Field(gt=0, le=1024)
    dtype: Literal["float32"]
    nodata: Literal[-9999.0]
    valid_pixels: int = Field(ge=0)
    total_pixels: int = Field(gt=0, le=1024*1024)
    minimum: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    maximum: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    mean: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    cloud_mask_applied: Literal[False]
    invalid_policy: Literal["source-mask-or-negative-reflectance-or-denominator-le-1e-6"]
    scaling: Literal["DN*scale+offset-before-ratio"]

    @model_validator(mode="after")
    def consistent(self):
        if self.total_pixels != self.width*self.height or self.valid_pixels > self.total_pixels:
            raise ValueError("pixel counts disagree")
        if (self.valid_pixels == 0) != (self.mean is None):
            raise ValueError("empty values require null statistics")
        if self.valid_pixels and (self.minimum is None or self.maximum is None or not self.minimum <= self.mean <= self.maximum):
            raise ValueError("invalid statistics")
        if not self.valid_pixels and (self.minimum is not None or self.maximum is not None):
            raise ValueError("empty statistics required")
        if not all(math.isfinite(v) for v in self.transform+self.bbox_wgs84):
            raise ValueError("finite grid required")
        return self


class MaskedNDVIResult(V2RequestModel):
    operation: Literal["ndvi"]
    input_asset_ids: list[Identifier] = Field(min_length=2, max_length=2)
    input_sha256: list[Sha256] = Field(min_length=2, max_length=2)
    mask_artifact_id: ArtifactId
    mask_sha256: Sha256
    acquired: UtcTimestamp
    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    width: int = Field(gt=0, le=1024)
    height: int = Field(gt=0, le=1024)
    dtype: Literal["float32"]
    nodata: Literal[-9999.0]
    valid_pixels: int = Field(ge=0)
    total_pixels: int = Field(gt=0, le=1024*1024)
    minimum: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    maximum: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    mean: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    mask_valid_pixels: int = Field(ge=0)
    clear_mask_pixels: int = Field(ge=0)
    cloud_excluded_pixels: int = Field(ge=0)
    cloud_mask_applied: Literal[True]
    cloud_policy_version: Literal["sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"]
    invalid_policy: Literal["source-mask-or-negative-reflectance-or-denominator-le-1e-6-or-scl-policy"]
    scaling: Literal["DN*scale+offset-before-ratio"]

    @model_validator(mode="after")
    def consistent(self):
        if (self.total_pixels != self.width*self.height or self.valid_pixels > self.clear_mask_pixels
                or self.clear_mask_pixels + self.cloud_excluded_pixels != self.mask_valid_pixels
                or self.mask_valid_pixels > self.total_pixels):
            raise ValueError("pixel counts disagree")
        if (self.valid_pixels == 0) != (self.mean is None):
            raise ValueError("empty values require null statistics")
        if self.valid_pixels and (self.minimum is None or self.maximum is None
                                  or not self.minimum <= self.mean <= self.maximum):
            raise ValueError("invalid statistics")
        if not self.valid_pixels and (self.minimum is not None or self.maximum is not None):
            raise ValueError("empty statistics required")
        if not all(math.isfinite(v) for v in self.transform+self.bbox_wgs84):
            raise ValueError("finite grid required")
        return self


def checked_pair(red: NativeBand, nir: NativeBand) -> None:
    if red.band != "red" or nir.band != "nir" or red.asset_id == nir.asset_id:
        raise ValueError("wrong band mapping")
    for name in ("item_id", "acquired", "crs", "transform", "width", "height"):
        if getattr(red, name) != getattr(nir, name):
            raise ValueError("same acquisition and exact grid required; no implicit alignment")


def statistics(values, valid) -> dict:
    import numpy as np
    selected = values[valid].astype(np.float64)
    return {"valid_pixels": int(valid.sum()), "total_pixels": int(valid.size),
            "minimum": float(selected.min()) if selected.size else None,
            "maximum": float(selected.max()) if selected.size else None,
            "mean": float(selected.mean()) if selected.size else None}


def _read_native(path: Path, band: NativeBand):
    import numpy as np
    from rasterio.io import MemoryFile

    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT:
        raise ValueError("bounded regular input required")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != band.sha256 or content[:4] not in {b"II*\x00", b"MM\x00*"}:
        raise ValueError("input checksum or TIFF mismatch")
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        if ((image.width, image.height, image.count, image.dtypes) != (band.width,band.height,1,(band.dtype,))
                or image.crs is None or image.crs.to_string() != band.crs
                or list(image.transform)[:6] != band.transform or image.nodata != band.nodata
                or image.scales != (band.scale,) or image.offsets != (band.offset,)):
            raise ValueError("input differs from reviewed native metadata")
        return (image.read(1).astype(np.float64)*band.scale+band.offset,
                image.dataset_mask()>0, image.transform, image.crs)


def read_cloud_mask(content: bytes, mask: MaskArtifactInput, reference: NativeBand):
    import numpy as np
    from rasterio.io import MemoryFile

    if (len(content) != mask.size_bytes or len(content) > MAX_OUTPUT
            or hashlib.sha256(content).hexdigest() != mask.sha256
            or content[:4] not in {b"II*\x00", b"MM\x00*"}):
        raise ValueError("mask content checksum, size or TIFF type mismatch")
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        if ((image.width, image.height, image.count, image.dtypes) !=
                (reference.width, reference.height, 1, ("uint8",))
                or image.crs is None or image.crs.to_string() != reference.crs
                or list(image.transform)[:6] != reference.transform or image.nodata != 255
                or image.scales != (1.,) or image.offsets != (0.,)):
            raise ValueError("mask grid or categorical profile mismatch")
        values, valid = image.read(1), image.dataset_mask() > 0
    if np.any(values[~valid] != 255) or np.any((values[valid] < 1) | (values[valid] > 11)):
        raise ValueError("mask classes or nodata representation invalid")
    excluded = valid & np.isin(values, CLOUD_EXCLUDED_CLASSES)
    clear = valid & ~excluded
    return clear, {"mask_valid_pixels": int(valid.sum()),
                   "clear_mask_pixels": int(clear.sum()),
                   "cloud_excluded_pixels": int(excluded.sum())}


def validate_ndvi(content: bytes, result: NDVIResult) -> None:
    """Decode bounded TIFF, not arbitrary GDAL formats, and verify public metadata."""
    result = NDVIResult.model_validate(result.model_dump(mode="json"))
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    if len(content) > MAX_OUTPUT or content[:4] not in {b"II*\x00", b"MM\x00*"}:
        raise ValueError("bounded classic TIFF required")
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), MemoryFile(content) as memory:
        with memory.open(driver="GTiff") as image:
            if (image.width, image.height, image.count, image.dtypes) != (result.width, result.height, 1, ("float32",)):
                raise ValueError("scientific raster shape/type mismatch")
            if image.crs is None or image.crs.to_string() != result.crs or list(image.transform)[:6] != result.transform:
                raise ValueError("scientific raster grid mismatch")
            if image.nodata != NODATA or image.scales != (1.,) or image.offsets != (0.,):
                raise ValueError("scientific raster nodata/scaling mismatch")
            bbox = list(transform_bounds(image.crs, "EPSG:4326", *image.bounds, densify_pts=21))
            if not np.allclose(bbox, result.bbox_wgs84, rtol=0, atol=1e-10):
                raise ValueError("scientific raster bbox mismatch")
            values = image.read(1)
            valid = image.dataset_mask() > 0
            if not np.isfinite(values).all() or not np.all(values[~valid] == NODATA):
                raise ValueError("invalid nodata representation")
            if np.any(values[valid] < -1) or np.any(values[valid] > 1):
                raise ValueError("NDVI out of range")
            if statistics(values, valid) != {key: getattr(result,key) for key in statistics(values,valid)}:
                raise ValueError("statistics do not match scientific pixels")


def validate_masked_ndvi(content: bytes, result: MaskedNDVIResult) -> None:
    """Verify the derived TIFF independently of provider response headers."""
    result = MaskedNDVIResult.model_validate(result.model_dump(mode="json"))
    import numpy as np
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    if len(content) > MAX_OUTPUT or content[:4] not in {b"II*\x00", b"MM\x00*"}:
        raise ValueError("bounded classic TIFF required")
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        if ((image.width, image.height, image.count, image.dtypes) !=
                (result.width, result.height, 1, ("float32",))
                or image.crs is None or image.crs.to_string() != result.crs
                or list(image.transform)[:6] != result.transform
                or image.nodata != NODATA or image.scales != (1.,) or image.offsets != (0.,)):
            raise ValueError("masked scientific raster profile mismatch")
        bbox = list(transform_bounds(image.crs, "EPSG:4326", *image.bounds, densify_pts=21))
        if not np.allclose(bbox, result.bbox_wgs84, rtol=0, atol=1e-10):
            raise ValueError("masked scientific raster bbox mismatch")
        values, valid = image.read(1), image.dataset_mask() > 0
        tags = image.tags()
        if (tags.get("operation") != "ndvi" or tags.get("version") != MASKED_VERSION
                or tags.get("cloud_mask_applied") != "true"
                or tags.get("cloud_policy_version") != result.cloud_policy_version
                or tags.get("mask_artifact_id") != result.mask_artifact_id):
            raise ValueError("masked scientific provenance tags mismatch")
        if (not np.isfinite(values).all() or not np.all(values[~valid] == NODATA)
                or np.any(values[valid] < -1) or np.any(values[valid] > 1)):
            raise ValueError("masked NDVI values invalid")
        if statistics(values, valid) != {key: getattr(result,key) for key in statistics(values,valid)}:
            raise ValueError("masked statistics do not match pixels")


def compute_ndvi(red_path: Path, nir_path: Path, red: NativeBand, nir: NativeBand) -> tuple[bytes, NDVIResult]:
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    checked_pair(red, nir)

    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_NUM_THREADS="1", GDAL_TIFF_INTERNAL_MASK=True):
        r,rm,transform,crs = _read_native(red_path, red)
        n,nm,_,_ = _read_native(nir_path, nir)
        denominator = n+r
        valid = rm & nm & np.isfinite(r) & np.isfinite(n) & (r>=0) & (n>=0) & (denominator>EPSILON)
        values = np.full(r.shape, NODATA, dtype="float32")
        values[valid] = ((n[valid]-r[valid])/denominator[valid]).astype("float32")
        from rasterio.transform import array_bounds
        bbox = list(transform_bounds(crs,"EPSG:4326",*array_bounds(red.height,red.width,transform),densify_pts=21))
        result = NDVIResult(operation="ndvi",input_asset_ids=[red.asset_id,nir.asset_id],input_sha256=[red.sha256,nir.sha256],
            acquired=red.acquired,crs=red.crs,transform=red.transform,bbox_wgs84=bbox,width=red.width,height=red.height,
            dtype="float32",nodata=NODATA,cloud_mask_applied=False,scaling="DN*scale+offset-before-ratio",
            invalid_policy="source-mask-or-negative-reflectance-or-denominator-le-1e-6",**statistics(values,valid))
        with MemoryFile() as memory:
            with memory.open(driver="GTiff",width=red.width,height=red.height,count=1,dtype="float32",crs=crs,
                             transform=transform,nodata=NODATA,compress="deflate") as output:
                output.write(values,1)
                output.write_mask(valid.astype("uint8")*255)
                output.update_tags(operation="ndvi",version=VERSION,cloud_mask_applied="false",
                    attribution=f"Contains modified Copernicus Sentinel data ({red.acquired[:4]})")
            content = memory.read()
    validate_ndvi(content,result)
    return content,result


def compute_masked_ndvi(red_path: Path, nir_path: Path, mask_path: Path,
                        red: NativeBand, nir: NativeBand,
                        mask: MaskArtifactInput) -> tuple[bytes, MaskedNDVIResult]:
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    checked_pair(red, nir)
    if mask.cloud_policy != CLOUD_POLICY or mask_path.is_symlink() or not mask_path.is_file():
        raise ValueError("unsupported or unsafe cloud mask")
    mask_content = mask_path.read_bytes()

    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      GDAL_NUM_THREADS="1", GDAL_TIFF_INTERNAL_MASK=True):
        r,rm,transform,crs = _read_native(red_path, red)
        n,nm,_,_ = _read_native(nir_path, nir)
        clear, mask_counts = read_cloud_mask(mask_content, mask, red)
        denominator = n+r
        valid = (rm & nm & clear & np.isfinite(r) & np.isfinite(n)
                 & (r>=0) & (n>=0) & (denominator>EPSILON))
        values = np.full(r.shape, NODATA, dtype="float32")
        values[valid] = ((n[valid]-r[valid])/denominator[valid]).astype("float32")
        from rasterio.transform import array_bounds
        bbox = list(transform_bounds(crs,"EPSG:4326",
                    *array_bounds(red.height,red.width,transform),densify_pts=21))
        result = MaskedNDVIResult(operation="ndvi",input_asset_ids=[red.asset_id,nir.asset_id],
            input_sha256=[red.sha256,nir.sha256],mask_artifact_id=mask.artifact_id,
            mask_sha256=mask.sha256,acquired=red.acquired,crs=red.crs,transform=red.transform,
            bbox_wgs84=bbox,width=red.width,height=red.height,dtype="float32",nodata=NODATA,
            cloud_mask_applied=True,cloud_policy_version=CLOUD_POLICY,
            scaling="DN*scale+offset-before-ratio",
            invalid_policy="source-mask-or-negative-reflectance-or-denominator-le-1e-6-or-scl-policy",
            **mask_counts,**statistics(values,valid))
        with MemoryFile() as memory:
            with memory.open(driver="GTiff",width=red.width,height=red.height,count=1,dtype="float32",
                             crs=crs,transform=transform,nodata=NODATA,compress="deflate") as output:
                output.write(values,1)
                output.write_mask(valid.astype("uint8")*255)
                output.update_tags(operation="ndvi",version=MASKED_VERSION,cloud_mask_applied="true",
                    cloud_policy_version=CLOUD_POLICY,mask_artifact_id=mask.artifact_id,
                    attribution=f"Contains modified Copernicus Sentinel data ({red.acquired[:4]})")
            content = memory.read()
    validate_masked_ndvi(content,result)
    return content,result
