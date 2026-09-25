"""Native-resolution, bounded COG windows with offline georeferencing receipts."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .http_range import BoundedHTTP, RangeSource


def extract_window(http: BoundedHTTP, item: dict, key: str, bbox: list[float], destination: Path) -> dict:
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    if destination.exists() or destination.is_symlink():
        raise ValueError("preserve existing raster window")
    asset = item["assets"][key]
    source = RangeSource(http, asset["href"])
    before = http.bytes
    # File opener cannot access sidecars or any URL other than the pinned object.
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO",
                      GDAL_GEOREF_SOURCES="INTERNAL", GDAL_NUM_THREADS="1", GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open("asset.tif", opener=source.open, driver="GTiff") as dataset:
            expected_count, expected_dtype = (3, "uint8") if key == "visual" else (1, "uint8" if key == "scl" else "uint16")
            if (dataset.crs is None or dataset.crs.to_epsg() != item["epsg"] or
                    [dataset.height, dataset.width] != asset["proj:shape"] or
                    dataset.count != expected_count or set(dataset.dtypes) != {expected_dtype} or
                    dataset.transform.b != 0 or dataset.transform.d != 0 or dataset.transform.a <= 0 or dataset.transform.e >= 0):
                raise ValueError("source raster does not match reviewed STAC geometry/type")
            declared = asset.get("proj:transform")
            if declared is None or not np.allclose(list(dataset.transform)[:6], declared[:6], rtol=0, atol=1e-7):
                raise ValueError("source transform differs from pinned STAC metadata")
            bounds = transform_bounds("EPSG:4326", dataset.crs, *bbox, densify_pts=21)
            fractional = from_bounds(*bounds, transform=dataset.transform)
            col, row = math.floor(fractional.col_off), math.floor(fractional.row_off)
            right, bottom = math.ceil(fractional.col_off+fractional.width), math.ceil(fractional.row_off+fractional.height)
            width, height = right-col, bottom-row
            if not (0 <= col < right <= dataset.width and 0 <= row < bottom <= dataset.height and
                    0 < width <= 1024 and 0 < height <= 1024):
                raise ValueError("AOI not fully covered or window exceeds decode bound")
            window = Window(col, row, width, height)
            pixels = dataset.read(window=window)
            mask = dataset.dataset_mask(window=window)
            nodata_fraction = float((mask == 0).mean())
            if nodata_fraction == 1:
                raise ValueError("window has no valid source pixels")
            transform = dataset.window_transform(window)
            native_crs = dataset.crs.to_string()
            native_bounds = rasterio.windows.bounds(window, dataset.transform)
            actual_bbox = list(transform_bounds(dataset.crs, "EPSG:4326", *native_bounds, densify_pts=21))
            profile = {"driver": "GTiff", "width": width, "height": height, "count": expected_count,
                       "dtype": expected_dtype, "crs": dataset.crs, "transform": transform,
                       "nodata": dataset.nodata, "compress": "deflate"}
            bands = asset.get("raster:bands", [])
            scales = [float(b.get("scale", 1)) for b in bands] if bands else [1.] * expected_count
            offsets = [float(b.get("offset", 0)) for b in bands] if bands else [0.] * expected_count
            if len(scales) != expected_count or len(offsets) != expected_count or not all(math.isfinite(v) for v in scales+offsets):
                raise ValueError("invalid declared raster scaling")
            with MemoryFile() as memory:
                with memory.open(**profile) as output:
                    output.write(pixels)
                    output.write_mask(mask)
                    output.scales, output.offsets = scales, offsets
                    output.update_tags(source_item=item["id"], source_asset=key, source_etag=source.etag,
                                       attribution=f"Contains modified Copernicus Sentinel data ({item['acquired'][:4]})")
                content = memory.read()
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("raster output exceeds bound")
    # Independent reopen verifies serialized pixels, transform and scaling before publication.
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as decoded:
        if (not np.array_equal(decoded.read(), pixels) or decoded.crs.to_string() != native_crs or
                decoded.transform != transform or list(decoded.scales) != scales or list(decoded.offsets) != offsets
                or not np.array_equal(decoded.dataset_mask(), mask)):
            raise ValueError("serialized window verification failed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(content)
    return {"item_id": item["id"], "asset_key": key, "filename": destination.name,
            "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
            "source_url": asset["href"], "source_etag": source.etag, "source_object_bytes": source.size,
            "transferred_payload_bytes": http.bytes-before, "window": [col, row, width, height],
            "crs": native_crs, "transform": list(transform)[:6], "bbox_wgs84": actual_bbox,
            "width": width, "height": height, "channels": expected_count, "dtype": expected_dtype,
            "scales": scales, "offsets": offsets, "nodata_fraction": nodata_fraction,
            "acquired": item["acquired"], "scene_cloud_cover_percent": item["scene_cloud_cover_percent"],
            "source_snapshot_hash": item["snapshot_sha256"],
            "agent_admitted": key == "visual", "processing": "native-resolution window; no resampling"}
