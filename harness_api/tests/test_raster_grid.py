import copy
import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_coordinates

from v2.test_tool_execution import make_tool_tasks
from app.main import create_app
from app.agent_gateway import build_binding, create_app as gateway
from app.raster_bridge import RasterBridge, create_app as provider_app
from app.core.artifacts import ArtifactStore
from app.core.capabilities import TaskRegistry
from app.core.domain import V2DomainError
from app.core.execution_replay import read_snapshot, replay_episode
from app.core.raster_math import MAX_INPUT, MAX_OUTPUT, NativeBand
from app.core.raster_grid import (CONTINUOUS_NODATA, CONTINUOUS_VERSION,
                                ContinuousBand, ContinuousGridResult,
                                GridArguments, GridResult, NativeSCL, NODATA,
                                TOOL_ID, VERSION, checked_continuous_grids,
                                checked_grids, compute_continuous_grid,
                                compute_grid, validate_continuous_grid,
                                validate_grid)
from app.core.schemas import ToolInvokeAction, V2EpisodeState
from app.core.tools.raster_grid import RasterGridExecutor
from app.core.tools.runtime import ToolRouter

ROOT = Path(__file__).resolve().parents[2]


def fixture(path, values, band, transform=None, mask=None, crs="EPSG:32650"):
    transform = transform if transform is not None else from_origin(660000, 3550000, 20 if band == "scl" else 10, 20 if band == "scl" else 10)
    dtype = "uint8" if band == "scl" else "uint16"
    scale, offset = (1., 0.) if band == "scl" else (.0001, -.1)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                           count=1, dtype=dtype, crs=crs, transform=transform, nodata=0) as image:
            image.write(values.astype(dtype), 1)
            image.scales = (scale,)
            image.offsets = (offset,)
            if mask is not None:
                image.write_mask(mask)
    cls = NativeSCL if band == "scl" else ContinuousBand
    return cls(asset_id="asset-" + band, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               item_id="scene-one", band=band, acquired="2024-04-05T00:00:00Z", crs=crs,
               transform=list(transform)[:6], width=values.shape[1], height=values.shape[0],
               dtype=dtype, scale=scale, offset=offset, nodata=0.)


def independent_reference(source_path, target_path):
    """Pixel-centre inverse mapping; deliberately no rasterio reproject/resample."""
    with rasterio.open(source_path) as source, rasterio.open(target_path) as target:
        values, source_mask = source.read(1), source.read_masks(1) > 0
        rr, cc = np.indices((target.height, target.width))
        xx, yy = target.transform * (cc.flatten() + .5, rr.flatten() + .5)
        if source.crs != target.crs:
            xx, yy = transform_coordinates(target.crs, source.crs, xx.tolist(), yy.tolist())
        sx, sy = ~source.transform * (np.asarray(xx), np.asarray(yy))
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        inside = (ix >= 0) & (ix < source.width) & (iy >= 0) & (iy < source.height)
        expected = np.full(target.width * target.height, 255, dtype="uint8")
        indices = np.flatnonzero(inside)
        valid = source_mask[iy[indices], ix[indices]] & (values[iy[indices], ix[indices]] != 0)
        indices = indices[valid]
        expected[indices] = values[iy[indices], ix[indices]]
        return expected.reshape(target.height, target.width)


def independent_bilinear_reference(source_path, target_path):
    """Direct pixel-centre bilinear interpolation; no production warp call."""
    with rasterio.open(source_path) as source, rasterio.open(target_path) as target:
        raw = source.read(1)
        physical = raw.astype("float64") * source.scales[0] + source.offsets[0]
        source_valid = source.read_masks(1) > 0
        if source.nodata is not None:
            source_valid &= raw != source.nodata
        expected = np.full((target.height, target.width), CONTINUOUS_NODATA,
                           dtype="float32")
        expected_valid = np.zeros((target.height, target.width), dtype=bool)
        for row in range(target.height):
            for column in range(target.width):
                x, y = target.transform * (column + .5, row + .5)
                if source.crs != target.crs:
                    x, y = transform_coordinates(target.crs, source.crs, [x], [y])
                    x, y = x[0], y[0]
                source_column, source_row = ~source.transform * (x, y)
                if not (0 <= source_column <= source.width
                        and 0 <= source_row <= source.height):
                    continue
                centred_column, centred_row = source_column - .5, source_row - .5
                left, top = int(np.floor(centred_column)), int(np.floor(centred_row))
                right, bottom = left + 1, top + 1
                dx, dy = centred_column - left, centred_row - top
                samples = [
                    (top, left, (1 - dx) * (1 - dy)),
                    (top, right, dx * (1 - dy)),
                    (bottom, left, (1 - dx) * dy),
                    (bottom, right, dx * dy),
                ]
                value = 0.
                valid_weight = 0.
                for sample_row, sample_column, weight in samples:
                    sample_row = min(max(sample_row, 0), source.height - 1)
                    sample_column = min(max(sample_column, 0), source.width - 1)
                    if source_valid[sample_row, sample_column]:
                        value += physical[sample_row, sample_column] * weight
                        valid_weight += weight
                if valid_weight >= 1 - 1e-6:
                    expected[row, column] = np.float32(value / valid_weight)
                    expected_valid[row, column] = True
        return expected, expected_valid


class GridTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sp, self.rp = self.root / "scl.tif", self.root / "red.tif"
        self.source = fixture(self.sp, np.array([[4, 5], [6, 9]]), "scl")
        self.reference = fixture(self.rp, np.full((4, 4), 2000), "red")

    def compute(self):
        return compute_grid(self.sp, self.rp, self.source, self.reference)

    def assert_reference(self):
        content, result = self.compute()
        with MemoryFile(content) as memory, memory.open() as image:
            expected = independent_reference(self.sp, self.rp)
            np.testing.assert_array_equal(image.read(1), expected)
            np.testing.assert_array_equal(image.read_masks(1) > 0, expected != 255)
            self.assertEqual(list(image.transform)[:6], self.reference.transform)
        validate_grid(content, result)
        return content, result

    def test_nearest_upsample_preserves_classes_and_cloud(self):
        content, result = self.assert_reference()
        self.assertEqual(hashlib.sha256(content).hexdigest(),
                         "bff667a343302e30763d03b013220f934a984d2372766d9e59c2e2c6467a38c7")
        with MemoryFile(content) as memory, memory.open() as image:
            np.testing.assert_array_equal(image.read(1), [[4, 4, 5, 5], [4, 4, 5, 5], [6, 6, 9, 9], [6, 6, 9, 9]])
        self.assertEqual(result.class_counts[9], 4)
        self.assertEqual(result.valid_fraction, 1.)
        self.assertFalse(result.cloud_mask_applied)

    def test_shifted_grid_marks_outside_source_invalid(self):
        self.reference = fixture(self.rp, np.full((5, 6), 2000), "red", from_origin(659990, 3550010, 10, 10))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 16)
        self.assertEqual(result.total_pixels, 30)

    def test_downsample_is_nearest_not_class_average(self):
        self.source = fixture(self.sp, np.array([[1, 4, 2, 5], [6, 7, 8, 9], [4, 5, 6, 7], [8, 9, 10, 11]]), "scl")
        self.reference = fixture(self.rp, np.ones((2, 2)), "red", from_origin(660001, 3549999, 40, 40))
        self.assert_reference()

    def test_zero_class_and_explicit_mask_both_invalid(self):
        self.source = fixture(self.sp, np.array([[0, 4], [5, 6]]), "scl", mask=np.array([[255, 0], [255, 255]], dtype="uint8"))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 8)
        self.assertEqual(result.class_counts[0], 0)

    def test_reference_mask_and_values_are_not_science_mask(self):
        self.reference = fixture(self.rp, np.zeros((4, 4)), "red", mask=np.zeros((4, 4), dtype="uint8"))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 16)

    def test_all_invalid_is_explicit_zero_coverage(self):
        self.source = fixture(self.sp, np.zeros((2, 2)), "scl")
        _, result = self.assert_reference()
        self.assertEqual(result.class_counts, [0] * 12)
        self.assertEqual(result.valid_fraction, 0.)

    def test_disjoint_grid_has_zero_coverage(self):
        self.reference = fixture(self.rp, np.ones((4, 4)), "red", from_origin(670000, 3550000, 10, 10))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 0)

    def test_cross_crs_reprojection_matches_inverse_mapping(self):
        x, y = transform_coordinates("EPSG:32650", "EPSG:32651", [660000.], [3550000.])
        self.reference = fixture(self.rp, np.ones((5, 5)), "red", from_origin(x[0] + 1, y[0] - 1, 9, 9), crs="EPSG:32651")
        _, result = self.assert_reference()
        self.assertGreater(result.valid_pixels, 0)

    def test_dates_scene_and_alias_rejected(self):
        for change in ({"acquired": "2024-04-15T00:00:00Z"}, {"item_id": "other"}, {"asset_id": self.source.asset_id}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checked_grids(self.source, self.reference.model_copy(update=change))

    def test_unreviewed_class_codes_rejected_even_masked(self):
        for value in (12, 255):
            self.source = fixture(self.sp, np.full((2, 2), value), "scl", mask=np.zeros((2, 2), dtype="uint8"))
            with self.assertRaises(ValueError):
                self.compute()

    def test_source_and_reference_hash_profile_tamper_rejected(self):
        for position in (0, 1):
            for change in ({"sha256": "f" * 64}, {"width": 3}, {"crs": "EPSG:32651"}):
                profiles = [self.source, self.reference]
                profiles[position] = profiles[position].model_copy(update=change)
                with self.subTest(position=position, change=change), self.assertRaises(ValueError):
                    compute_grid(self.sp, self.rp, *profiles)

    def test_schema_rejects_invalid_grid_scaling_and_huge_shape(self):
        for change in ({"scale": .01}, {"offset": -.1}, {"dtype": "float32"}, {"nodata": None},
                       {"transform": [0, 0, 0, 0, -20, 0]}, {"transform": [20, 1, 0, 0, -20, 0]},
                       {"transform": [20, 0, float("inf"), 0, -20, 0]}, {"crs": "EPSG:32661"}, {"width": 1025}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                compute_grid(self.sp, self.rp, self.source.model_copy(update=change), self.reference)

    def test_paths_non_tiff_and_input_bounds_rejected(self):
        link = self.root / "alias.tif"
        link.symlink_to(self.sp)
        with self.assertRaises(ValueError):
            compute_grid(link, self.rp, self.source, self.reference)
        self.sp.write_bytes(b"<VRTDataset/>")
        with self.assertRaises(ValueError):
            self.compute()
        with self.sp.open("wb") as stream:
            stream.truncate(MAX_INPUT + 1)
        with self.assertRaises(ValueError):
            self.compute()

    def test_arguments_admit_bilinear_but_reject_other_methods_urls_and_alias(self):
        args = {"source_asset_id": "asset-scl", "reference_asset_id": "asset-red", "method": "nearest"}
        GridArguments.model_validate(args)
        GridArguments.model_validate({**args, "method": "bilinear"})
        for change in ({"method": "cubic"}, {"source_asset_id": "https://private"},
                       {"expression": "eval"}, {"reference_asset_id": "asset-scl"}):
            with self.assertRaises(ValueError):
                GridArguments.model_validate({**args, **change})

    def test_result_grid_histogram_coverage_and_payload_tamper_rejected(self):
        content, result = self.compute()
        for change in ({"valid_fraction": .5}, {"class_counts": [0] * 12}, {"crs": "EPSG:32651"},
                       {"width": 3}, {"nodata": 0}, {"cloud_mask_applied": True},
                       {"input_asset_ids": ["asset-scl", "asset-scl"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_grid(content, result.model_copy(update=change))
        with self.assertRaises(ValueError):
            validate_grid(b"<VRTDataset/>", result)

    def test_repeated_computation_has_same_bytes_and_result(self):
        self.assertEqual(self.compute(), self.compute())

    def test_real_subprocess_provider_and_missing_worker(self):
        manifest = {profile.asset_id: {"filename": path.name, "native": profile.model_dump(mode="json")}
                    for path, profile in ((self.sp, self.source), (self.rp, self.reference))}
        args = GridArguments(source_asset_id=self.source.asset_id,
                             reference_asset_id=self.reference.asset_id, method="nearest")
        bridge = RasterBridge(self.root, manifest, ROOT / "scripts/raster_worker.py",
                              ROOT / "scripts/raster_grid_worker.py")
        content, result = bridge.execute_grid(args)
        validate_grid(content, result)
        with TestClient(provider_app(bridge)) as client:
            response = client.post("/resample-grid", json=args.model_dump())
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["X-Raster-Grid-Version"], VERSION)
            self.assertEqual(hashlib.sha256(response.content).hexdigest(), response.headers["X-Content-SHA256"])
        unavailable = RasterBridge(self.root, manifest, ROOT / "scripts/raster_worker.py")
        with TestClient(provider_app(unavailable)) as client:
            self.assertEqual(client.post("/resample-grid", json=args.model_dump()).status_code, 503)


class ContinuousGridTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sp, self.rp = self.root / "swir16.tif", self.root / "nir.tif"
        self.source = fixture(
            self.sp, np.array([[2000, 4000], [6000, 8000]]), "swir16",
            from_origin(660000, 3550000, 20, 20))
        self.reference = fixture(
            self.rp, np.full((4, 4), 3000), "nir",
            from_origin(660000, 3550000, 10, 10))

    def compute(self):
        return compute_continuous_grid(
            self.sp, self.rp, self.source, self.reference)

    def assert_reference(self):
        content, result = self.compute()
        expected, expected_valid = independent_bilinear_reference(self.sp, self.rp)
        with MemoryFile(content) as memory, memory.open() as image:
            np.testing.assert_allclose(image.read(1), expected, rtol=0, atol=1e-7)
            np.testing.assert_array_equal(image.read_masks(1) > 0, expected_valid)
            self.assertEqual(list(image.transform)[:6], self.reference.transform)
        validate_continuous_grid(content, result)
        return content, result

    def test_bilinear_physical_reflectance_matches_independent_reference(self):
        _, result = self.assert_reference()
        self.assertEqual(result.source_band, "swir16")
        self.assertAlmostEqual(result.minimum, .1, places=7)
        self.assertAlmostEqual(result.maximum, .7, places=7)
        self.assertEqual(result.valid_fraction, 1.)
        self.assertEqual(result.source_scale, .0001)
        self.assertEqual(result.source_offset, -.1)

    def test_reference_pixels_and_mask_are_geometry_only(self):
        first, first_result = self.compute()
        self.reference = fixture(
            self.rp, np.zeros((4, 4)), "nir",
            from_origin(660000, 3550000, 10, 10),
            mask=np.zeros((4, 4), dtype="uint8"))
        second, second_result = self.compute()
        self.assertEqual(first, second)
        self.assertEqual(first_result.model_copy(update={
            "input_sha256": second_result.input_sha256}), second_result)

    def test_source_mask_and_nodata_require_complete_neighborhood(self):
        self.source = fixture(
            self.sp, np.array([[0, 4000], [6000, 8000]]), "swir16",
            from_origin(660000, 3550000, 20, 20),
            mask=np.array([[0, 255], [255, 255]], dtype="uint8"))
        _, result = self.assert_reference()
        self.assertGreater(result.valid_pixels, 0)
        self.assertLess(result.valid_pixels, result.total_pixels)

    def test_disjoint_and_all_invalid_outputs_are_explicit(self):
        self.source = fixture(
            self.sp, np.zeros((2, 2)), "swir16",
            from_origin(660000, 3550000, 20, 20))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 0)
        self.assertIsNone(result.mean)
        self.reference = fixture(
            self.rp, np.ones((4, 4)), "nir",
            from_origin(670000, 3550000, 10, 10))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 0)

    def test_scene_date_alias_and_profile_tamper_are_rejected(self):
        for change in ({"acquired": "2024-04-15T00:00:00Z"},
                       {"item_id": "other"}, {"asset_id": self.source.asset_id}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checked_continuous_grids(
                    self.source, self.reference.model_copy(update=change))
        for position in (0, 1):
            for change in ({"sha256": "f" * 64}, {"width": 3},
                           {"scale": .001}, {"crs": "EPSG:32651"}):
                profiles = [self.source, self.reference]
                profiles[position] = profiles[position].model_copy(update=change)
                with self.subTest(position=position, change=change), self.assertRaises(ValueError):
                    compute_continuous_grid(self.sp, self.rp, *profiles)

    def test_result_profile_statistics_and_provenance_tamper_are_rejected(self):
        content, result = self.compute()
        for change in ({"valid_fraction": .5}, {"mean": 99.},
                       {"crs": "EPSG:32651"}, {"width": 3}, {"nodata": 0.},
                       {"source_band": "red"},
                       {"input_asset_ids": ["asset-swir16", "asset-swir16"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_continuous_grid(content, result.model_copy(update=change))
        with self.assertRaises(ValueError):
            validate_continuous_grid(b"<VRTDataset/>", result)

    def test_repeated_computation_is_byte_deterministic(self):
        self.assertEqual(self.compute(), self.compute())

    def test_real_subprocess_provider_returns_continuous_version(self):
        manifest = {
            profile.asset_id: {"filename": path.name,
                               "native": profile.model_dump(mode="json")}
            for path, profile in ((self.sp, self.source), (self.rp, self.reference))
        }
        args = GridArguments(source_asset_id=self.source.asset_id,
                             reference_asset_id=self.reference.asset_id,
                             method="bilinear")
        bridge = RasterBridge(self.root, manifest, ROOT / "scripts/raster_worker.py",
                              ROOT / "scripts/raster_grid_worker.py")
        content, result = bridge.execute_grid(args)
        validate_continuous_grid(content, result)
        with TestClient(provider_app(bridge)) as client:
            response = client.post("/resample-grid", json=args.model_dump())
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["X-Raster-Grid-Version"],
                             CONTINUOUS_VERSION)
            self.assertEqual(hashlib.sha256(response.content).hexdigest(),
                             response.headers["X-Content-SHA256"])


class GridHarnessTests(unittest.TestCase):
    compute = GridTests.compute

    def setUp(self):
        GridTests.setUp(self)
        self.content, self.result = self.compute()
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        task.update(inputs=[self.source.asset_id, self.reference.asset_id])
        task["metadata"].update(artifact_identity="derivation-sha256-v1",
                                grid_inputs={profile.asset_id: profile.model_dump(mode="json")
                                             for profile in (self.source, self.reference)})
        (directory / "task.json").write_text(json.dumps(task))
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"] = [TOOL_ID, "catalog.search", "catalog.inspect_asset"]
        (directory / "scenario.json").write_text(json.dumps(scenario))
        template = json.loads((directory / "assets.json").read_text())[0]
        west, south, east, north = self.result.bbox_wgs84
        assets = []
        for profile, path, roles in ((self.source, self.sp, ["input_image", "scene_classification"]),
                                     (self.reference, self.rp, ["input_image", "reflectance"])):
            asset = copy.deepcopy(template)
            asset.update(asset_id=profile.asset_id, sha256=profile.sha256, size_bytes=path.stat().st_size,
                         roles=roles, bands=[profile.band],
                         temporal={"start": profile.acquired, "end": profile.acquired},
                         spatial={"crs": "EPSG:4326",
                                  "bbox": dict(west=west, south=south, east=east, north=north),
                                  "gsd_meters": profile.transform[0],
                                  "shape": [profile.height, profile.width, 1]})
            assets.append(asset)
        (directory / "assets.json").write_text(json.dumps(assets))
        self.registry = TaskRegistry(self.tasks)
        self.manifest = self.registry.get("crop-smoke", "1.0.0")
        self.calls, self.bad = 0, None

        def handle(request):
            self.calls += 1
            if self.bad == "timeout":
                raise httpx.ReadTimeout("fixture")
            data = self.content if self.bad != "bytes" else b"wrong"
            metadata = self.result.model_dump(mode="json")
            if self.bad == "source":
                metadata["input_sha256"] = ["f" * 64] * 2
            return httpx.Response(200, content=data, headers={
                "X-Raster-Grid-Version": VERSION,
                "X-Raster-Grid-Metadata": json.dumps(metadata),
                "X-Content-SHA256": hashlib.sha256(self.content).hexdigest()})

        self.transport = httpx.MockTransport(handle)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.executor = RasterGridExecutor("http://raster", self.artifacts, self.transport)
        self.action = ToolInvokeAction(type="tool.invoke", tool_id=TOOL_ID,
            arguments={"source_asset_id": self.source.asset_id,
                       "reference_asset_id": self.reference.asset_id, "method": "nearest"})

    def backend(self):
        return create_app(database_path=str(self.root / "state.db"),
            v2_tasks_path=str(self.tasks), v2_artifacts_path=str(self.artifacts.root),
            v2_tool_executor=ToolRouter(grid=self.executor))

    def test_scope_roles_profiles_and_output_budget(self):
        with self.assertRaises(V2DomainError):
            self.executor.plan(self.action, self.manifest, [self.source.asset_id])
        broken = self.manifest.model_copy(deep=True)
        broken.assets[0].roles = ["label"]
        with self.assertRaises(V2DomainError):
            self.executor.plan(self.action, broken, broken.task.inputs)
        prepared = self.executor.plan(self.action, self.manifest, self.manifest.task.inputs)
        self.assertEqual(prepared.input_bytes, sum(a.size_bytes for a in self.manifest.assets))
        self.assertEqual(prepared.max_output_bytes, MAX_OUTPUT)
        self.assertEqual(self.calls, 0)
        task_path = self.tasks / "crop-smoke/task.json"
        task = json.loads(task_path.read_text())
        task["budget"]["max_artifact_bytes"] = MAX_OUTPUT - 1
        task_path.write_text(json.dumps(task))
        with TestClient(self.backend()) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            response = operator.post(f"/v2/episodes/{initial['episode_id']}/step", json={
                "client_action_id": "over-budget", "expected_state_version": 0,
                "action": self.action.model_dump()})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["error"]["code"], "tool_budget_exceeded")
            self.assertEqual(self.calls, 0)

    def test_provider_tamper_and_timeout_fail_without_artifact(self):
        for bad in ("bytes", "source", "timeout"):
            self.bad = bad
            with self.assertRaises(V2DomainError):
                self.executor.plan(self.action, self.manifest, self.manifest.task.inputs).invoke()
        self.assertFalse(self.artifacts.root.exists())

    def test_gateway_evidence_restart_and_execution_replay(self):
        backend = self.backend()
        with TestClient(backend) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            binding = build_binding(self.manifest, V2EpisodeState.model_validate(initial["state"]),
                operator.get("/v2/capabilities").json()["data"], hashlib.sha256(b"2" * 64).hexdigest())
            with TestClient(gateway(binding, "http://operator", httpx.ASGITransport(app=backend)),
                            headers={"Authorization": "Bearer " + "2" * 64}) as client:
                request = {"client_action_id": "grid-1", "expected_state_version": 0,
                           "action": self.action.model_dump()}
                wrong = copy.deepcopy(request)
                wrong["action"]["arguments"]["reference_asset_id"] = "asset-other"
                self.assertEqual(client.post("/agent/step", json=wrong).status_code, 403)
                response = client.post("/agent/step", json=request)
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(client.post("/agent/step", json=request).json(), data)
                self.assertEqual(self.calls, 1)
                inline = data["observation"]["items"][0]["inline"]
                self.assertEqual(inline["method"], "nearest")
                self.assertEqual(inline["class_counts"], self.result.class_counts)
                ref = data["observation"]["items"][1]["artifact_ref"]
                artifact = client.get("/agent/artifacts/" + ref).json()["artifact"]
                content = client.get("/agent/artifacts/" + ref + "/content")
                self.assertEqual(content.status_code, 200, content.text[:100])
                self.assertEqual(content.content, self.content)
                evidence = {"type": "memory.save_evidence", "evidence": {
                    "evidence_id": "ev-grid", "claim_id": "grid", "source_ref": ref,
                    "selector": {"pixel_window": [0, 0, 4, 4]},
                    "description": "Nearest SCL categorical grid", "frozen_sha256": artifact["sha256"]}}
                response = client.post("/agent/step", json={"client_action_id": "evidence",
                    "expected_state_version": 1, "action": evidence})
                self.assertEqual(response.status_code, 200, response.text)
                response = client.post("/agent/step", json={"client_action_id": "submit",
                    "expected_state_version": 2, "action": {"type": "answer.submit",
                    "answer": {"label": "grid", "confidence": 1., "claims": []},
                    "confidence": 1., "evidence_ids": ["ev-grid"]}})
                self.assertEqual(response.status_code, 200, response.text)
            with TestClient(self.backend()) as restarted:
                cached = restarted.post(f"/v2/episodes/{episode}/step", json=request).json()["data"]
                self.assertEqual(cached["state"]["state_version"], 1)
            self.assertEqual(self.calls, 1)
        gc.collect()
        before = hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest()
        report = replay_episode(read_snapshot(self.root / "state.db", episode), self.registry,
            self.root / "replay", lambda artifacts: ToolRouter(grid=RasterGridExecutor(
                "http://raster", artifacts, self.transport)))
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(self.calls, 2)
        self.assertEqual(hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest(), before)


class ContinuousGridHarnessTests(unittest.TestCase):
    compute = ContinuousGridTests.compute

    def setUp(self):
        ContinuousGridTests.setUp(self)
        self.content, self.result = self.compute()
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        task.update(inputs=[self.source.asset_id, self.reference.asset_id])
        task["metadata"].update(
            artifact_identity="derivation-sha256-v1",
            grid_inputs={profile.asset_id: profile.model_dump(mode="json")
                         for profile in (self.source, self.reference)})
        (directory / "task.json").write_text(json.dumps(task))
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"] = [TOOL_ID]
        (directory / "scenario.json").write_text(json.dumps(scenario))
        template = json.loads((directory / "assets.json").read_text())[0]
        west, south, east, north = self.result.bbox_wgs84
        assets = []
        for profile, path in ((self.source, self.sp), (self.reference, self.rp)):
            asset = copy.deepcopy(template)
            asset.update(
                asset_id=profile.asset_id, sha256=profile.sha256,
                size_bytes=path.stat().st_size,
                roles=["input_image", "reflectance"], bands=[profile.band],
                temporal={"start": profile.acquired, "end": profile.acquired},
                spatial={"crs": "EPSG:4326",
                         "bbox": dict(west=west, south=south, east=east, north=north),
                         "gsd_meters": profile.transform[0],
                         "shape": [profile.height, profile.width, 1]})
            assets.append(asset)
        (directory / "assets.json").write_text(json.dumps(assets))
        self.registry = TaskRegistry(self.tasks)
        self.manifest = self.registry.get("crop-smoke", "1.0.0")
        self.calls = 0

        def handle(request):
            self.calls += 1
            return httpx.Response(200, content=self.content, headers={
                "X-Raster-Grid-Version": CONTINUOUS_VERSION,
                "X-Raster-Grid-Metadata": self.result.model_dump_json(),
                "X-Content-SHA256": hashlib.sha256(self.content).hexdigest()})

        self.transport = httpx.MockTransport(handle)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.executor = RasterGridExecutor(
            "http://raster", self.artifacts, self.transport)
        self.action = ToolInvokeAction(
            type="tool.invoke", tool_id=TOOL_ID,
            arguments={"source_asset_id": self.source.asset_id,
                       "reference_asset_id": self.reference.asset_id,
                       "method": "bilinear"})

    def backend(self):
        return create_app(
            database_path=str(self.root / "state.db"),
            v2_tasks_path=str(self.tasks),
            v2_artifacts_path=str(self.artifacts.root),
            v2_tool_executor=ToolRouter(grid=self.executor))

    def test_continuous_policy_gateway_restart_and_execution_replay(self):
        prepared = self.executor.plan(
            self.action, self.manifest, self.manifest.task.inputs)
        self.assertEqual(prepared.tool_version, CONTINUOUS_VERSION)
        broken = self.manifest.model_copy(deep=True)
        broken.assets[0].roles = ["label"]
        with self.assertRaises(V2DomainError):
            self.executor.plan(self.action, broken, broken.task.inputs)
        backend = self.backend()
        with TestClient(backend) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            binding = build_binding(
                self.manifest, V2EpisodeState.model_validate(initial["state"]),
                operator.get("/v2/capabilities").json()["data"],
                hashlib.sha256(b"3" * 64).hexdigest())
            with TestClient(
                    gateway(binding, "http://operator",
                            httpx.ASGITransport(app=backend)),
                    headers={"Authorization": "Bearer " + "3" * 64}) as client:
                request = {"client_action_id": "continuous-grid-1",
                           "expected_state_version": 0,
                           "action": self.action.model_dump()}
                response = client.post("/agent/step", json=request)
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(client.post("/agent/step", json=request).json(), data)
                inline = data["observation"]["items"][0]["inline"]
                self.assertEqual(inline["tool_version"], CONTINUOUS_VERSION)
                self.assertEqual(inline["method"], "bilinear")
                self.assertEqual(inline["source_band"], "swir16")
                ref = data["observation"]["items"][1]["artifact_ref"]
                artifact = client.get("/agent/artifacts/" + ref).json()["artifact"]
                content = client.get("/agent/artifacts/" + ref + "/content")
                self.assertEqual(content.status_code, 200, content.text[:100])
                self.assertEqual(content.content, self.content)
                evidence = {"type": "memory.save_evidence", "evidence": {
                    "evidence_id": "ev-continuous-grid", "claim_id": "grid",
                    "source_ref": ref, "selector": {"pixel_window": [0, 0, 4, 4]},
                    "description": "Bilinear physical reflectance grid",
                    "frozen_sha256": artifact["sha256"]}}
                response = client.post("/agent/step", json={
                    "client_action_id": "continuous-evidence",
                    "expected_state_version": 1, "action": evidence})
                self.assertEqual(response.status_code, 200, response.text)
                response = client.post("/agent/step", json={
                    "client_action_id": "continuous-submit",
                    "expected_state_version": 2,
                    "action": {"type": "answer.submit",
                               "answer": {"label": "continuous-grid",
                                          "confidence": 1., "claims": []},
                               "confidence": 1.,
                               "evidence_ids": ["ev-continuous-grid"]}})
                self.assertEqual(response.status_code, 200, response.text)
            with TestClient(self.backend()) as restarted:
                cached = restarted.post(
                    f"/v2/episodes/{episode}/step", json=request).json()["data"]
                self.assertEqual(cached["state"]["state_version"], 1)
        self.assertEqual(self.calls, 1)
        gc.collect()
        before = hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest()
        report = replay_episode(
            read_snapshot(self.root / "state.db", episode), self.registry,
            self.root / "replay",
            lambda artifacts: ToolRouter(grid=RasterGridExecutor(
                "http://raster", artifacts, self.transport)))
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(self.calls, 2)
        self.assertEqual(hashlib.sha256(
            (self.root / "state.db").read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
