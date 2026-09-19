import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.raster_bridge import RasterBridge, create_app
from app.v2.raster_grid import NativeSCL
from app.v2.raster_math import CLOUD_POLICY, NativeBand
from app.v2.temporal import (
    STACK_BANDS,
    STACK_NODATA,
    TemporalAlignRequest,
    checked_temporal_inputs,
    compute_temporal_stack,
    validate_temporal_stack,
)


ROOT = Path(__file__).resolve().parents[2]


def fixture(path, values, band, item, acquired, transform=None, mask=None):
    transform = transform or from_origin(
        669060, 3542980, 20 if band == "scl" else 10, 20 if band == "scl" else 10
    )
    dtype = "uint8" if band == "scl" else "uint16"
    scale, offset = (1.0, 0.0) if band == "scl" else (0.0001, -0.1)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype=dtype,
            crs="EPSG:32650",
            transform=transform,
            nodata=0,
        ) as image:
            image.write(values.astype(dtype), 1)
            image.scales = (scale,)
            image.offsets = (offset,)
            if mask is not None:
                image.write_mask(mask)
    model = NativeSCL if band == "scl" else NativeBand
    return model(
        asset_id=f"asset-{item}-{band}",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        item_id=item,
        band=band,
        acquired=acquired,
        crs="EPSG:32650",
        transform=list(transform)[:6],
        width=values.shape[1],
        height=values.shape[0],
        dtype=dtype,
        scale=scale,
        offset=offset,
        nodata=0.0,
    )


class TemporalStackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = (
            self.root / "before-red.tif",
            self.root / "before-scl.tif",
            self.root / "after-red.tif",
            self.root / "after-scl.tif",
        )
        before = "2024-04-05T02:58:52.954000Z"
        after = "2024-05-10T02:58:56.417000Z"
        self.profiles = (
            fixture(self.paths[0], np.full((4, 4), 1000), "red", "before", before),
            fixture(
                self.paths[1], np.array([[4, 4], [4, 8]]), "scl", "before", before
            ),
            fixture(self.paths[2], np.full((4, 4), 2000), "red", "after", after),
            fixture(
                self.paths[3], np.array([[5, 5], [5, 5]]), "scl", "after", after
            ),
        )

    def compute(self):
        return compute_temporal_stack(self.paths, self.profiles)

    def test_stack_is_deterministic_and_preserves_fixed_band_semantics(self):
        content, result = self.compute()
        repeated, repeated_result = self.compute()
        self.assertEqual(content, repeated)
        self.assertEqual(result, repeated_result)
        self.assertEqual(result.band_order, STACK_BANDS)
        self.assertEqual(result.before_cloud_fraction, 0.25)
        self.assertEqual(result.after_cloud_fraction, 0.0)
        self.assertEqual(result.aligned_coverage_fraction, 1.0)
        with MemoryFile(content) as memory, memory.open() as image:
            self.assertEqual(list(image.descriptions), STACK_BANDS)
            np.testing.assert_array_equal(image.read(1), np.full((4, 4), 1000))
            np.testing.assert_array_equal(
                image.read(2),
                [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 8, 8], [4, 4, 8, 8]],
            )
            np.testing.assert_array_equal(image.read(3), np.full((4, 4), 2000))
        validate_temporal_stack(content, result)

    def test_cross_date_gap_is_explicit_nodata(self):
        mask = np.full((4, 4), 255, dtype="uint8")
        mask[0, 0] = 0
        self.profiles = (
            self.profiles[0],
            self.profiles[1],
            fixture(
                self.paths[2], np.full((4, 4), 2000), "red", "after",
                "2024-05-10T02:58:56.417000Z", mask=mask
            ),
            self.profiles[3],
        )
        content, result = self.compute()
        self.assertEqual(result.aligned_valid_pixels, 15)
        with MemoryFile(content) as memory, memory.open() as image:
            self.assertTrue(np.all(image.read()[:, 0, 0] == STACK_NODATA))
            self.assertEqual(image.dataset_mask()[0, 0], 0)

    def test_profile_payload_and_metadata_tamper_fail_closed(self):
        content, result = self.compute()
        damaged = bytearray(content)
        damaged[0] = 0
        with self.assertRaises(ValueError):
            validate_temporal_stack(bytes(damaged), result)
        with self.assertRaises(ValueError):
            validate_temporal_stack(content, result.model_copy(update={"width": 3}))

    def test_inputs_require_ordered_exact_reflectance_grid(self):
        with self.assertRaisesRegex(ValueError, "exact grid"):
            checked_temporal_inputs(
                self.profiles[0],
                self.profiles[1],
                self.profiles[2].model_copy(update={"width": 5}),
                self.profiles[3],
            )
        with self.assertRaisesRegex(ValueError, "precede"):
            checked_temporal_inputs(
                self.profiles[2], self.profiles[3], self.profiles[0], self.profiles[1]
            )

    def test_provider_subprocess_revalidates_inputs_and_output(self):
        manifest = {
            profile.asset_id: {
                "filename": path.name,
                "native": profile.model_dump(mode="json"),
            }
            for path, profile in zip(self.paths, self.profiles)
        }
        bridge = RasterBridge(
            self.root,
            manifest,
            ROOT / "scripts/raster_worker.py",
            temporal_worker=ROOT / "scripts/temporal_stack_worker.py",
        )
        request = TemporalAlignRequest(
            before_red_asset_id=self.profiles[0].asset_id,
            before_scl_asset_id=self.profiles[1].asset_id,
            after_red_asset_id=self.profiles[2].asset_id,
            after_scl_asset_id=self.profiles[3].asset_id,
            cloud_policy=CLOUD_POLICY,
        )
        with TestClient(create_app(bridge)) as client:
            health = client.get("/healthz")
            self.assertEqual(health.json()["temporal_tool_version"], "1.0.0")
            response = client.post("/temporal-align", json=request.model_dump(mode="json"))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                hashlib.sha256(response.content).hexdigest(),
                response.headers["X-Content-SHA256"],
            )
            self.assertEqual(response.headers["X-Temporal-Version"], "1.0.0")
            invalid = request.model_copy(
                update={"after_scl_asset_id": "asset-not-reviewed"}
            )
            self.assertEqual(
                client.post("/temporal-align", json=invalid.model_dump(mode="json")).status_code,
                422,
            )


if __name__ == "__main__":
    unittest.main()
