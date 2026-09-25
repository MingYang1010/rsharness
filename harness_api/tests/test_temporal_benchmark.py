import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy
import rasterio
from rasterio.transform import from_origin

from app.core.capabilities import TaskRegistry
from app.core.temporal import (
    TemporalInputProfile,
    TemporalSelectAlignArguments,
    select_temporal_pair,
)


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_temporal_benchmark",
    PROJECT / "scripts" / "prepare_temporal_benchmark.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class TemporalBenchmarkPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.config = json.loads(
            (PROJECT / "config" / "temporal-benchmark-v1.json").read_text()
        )
        timestamps = [
            "2024-04-05T02:58:52.954000Z",
            "2024-04-15T02:58:58.108000Z",
            "2024-05-10T02:58:56.417000Z",
        ]
        cloud_percent = [14.473213, 29.402331, 0.189687]
        assets = []
        native_inputs = {}
        bbox = {
            "west": 118.78951655559106,
            "south": 31.98965277990787,
            "east": 118.81044344860592,
            "north": 32.01032330934839,
        }
        native_root = self.source / "native-inputs"
        native_root.mkdir()
        for index, item in enumerate(self.config["items"]):
            for band, asset_id in (
                ("red", item["red_asset_id"]),
                ("scl", item["scl_asset_id"]),
            ):
                filename = "%s-%s.tif" % (item["item_id"], band)
                path = native_root / filename
                is_red = band == "red"
                width = height = 40 if is_red else 20
                transform = from_origin(
                    669060.0,
                    3542980.0,
                    10.0 if is_red else 20.0,
                    10.0 if is_red else 20.0,
                )
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=width,
                    height=height,
                    count=1,
                    dtype="uint16" if is_red else "uint8",
                    crs="EPSG:32650",
                    transform=transform,
                    nodata=0,
                ) as image:
                    values = numpy.full(
                        (height, width),
                        1000 if is_red else 4,
                        dtype=numpy.uint16 if is_red else numpy.uint8,
                    )
                    if not is_red and index == 2:
                        values[0, 0] = 9
                    image.write(values, 1)
                digest = _sha256(path)
                native = {
                    "asset_id": asset_id,
                    "sha256": digest,
                    "item_id": item["item_id"],
                    "band": band,
                    "acquired": timestamps[index],
                    "crs": "EPSG:32650",
                    "transform": list(transform)[:6],
                    "width": width,
                    "height": height,
                    "dtype": "uint16" if is_red else "uint8",
                    "scale": 0.0001 if is_red else 1.0,
                    "offset": -0.1 if is_red else 0.0,
                    "nodata": 0.0,
                }
                native_inputs[asset_id] = {"filename": filename, "native": native}
                assets.append(
                    {
                        "asset_id": asset_id,
                        "uri": "local://approved-cloud-chain/" + filename,
                        "media_type": "image/tiff",
                        "roles": [
                            "input_image",
                            "reflectance" if is_red else "scene_classification",
                        ],
                        "sha256": digest,
                        "size_bytes": path.stat().st_size,
                        "spatial": {
                            "crs": "EPSG:4326",
                            "bbox": bbox,
                            "gsd_meters": 10.0 if is_red else 20.0,
                            "shape": [height, width, 1],
                        },
                        "temporal": {
                            "start": timestamps[index],
                            "end": timestamps[index],
                        },
                        "platform": "sentinel-2",
                        "bands": [band],
                        "quality": {
                            "cloud_cover_percent": cloud_percent[index],
                            "nodata_fraction": 0.0,
                        },
                        "license": "Contains modified Copernicus Sentinel data (2024)",
                        "source": "reviewed-test-window",
                        "source_snapshot_hash": hashlib.sha256(
                            item["item_id"].encode()
                        ).hexdigest(),
                    }
                )
        assets_path = (
            self.source
            / "tasks"
            / "cloud-masked-ndvi"
            / "assets.json"
        )
        native_path = self.source / "native-inputs.json"
        _write_json(assets_path, assets)
        _write_json(native_path, native_inputs)
        self.license_path = self.root / "license-review.json"
        _write_json(
            self.license_path,
            {
                "attribution": self.config["source"]["attribution"],
                "scope": "local-research",
                "not_redistribution_authorization": True,
            },
        )
        self.config["source"].update(
            assets_sha256=_sha256(assets_path),
            native_inputs_sha256=_sha256(native_path),
            license_review_sha256=_sha256(self.license_path),
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_freezes_four_tasks_and_one_false_confidence_run(self):
        output = self.root / "pack"
        result = MODULE.prepare(
            self.source,
            self.license_path,
            output,
            self.config,
        )
        self.assertEqual(result["task_case_count"], 4)
        self.assertEqual(result["acceptance_run_count"], 5)
        self.assertFalse(result["data_policy"]["imagery_in_git"])
        for run in result["runs"]:
            case_root = output / run["case_id"]
            manifest = TaskRegistry(case_root / "tasks").get(
                run["task_id"], "1.0.0"
            )
            self.assertEqual(manifest.task_manifest_hash, run["task_manifest_hash"])
            self.assertEqual(manifest.scenario.allowed_tools, ["temporal.select_align"])
            self.assertEqual(len(manifest.task.inputs), 6)
            selected = run["case_id"] == "valid-pair"
            self.assertEqual(
                manifest.evaluator.config["efficiency"][
                    "expected_renderer_calls"
                ],
                1 if selected else 0,
            )
            self.assertTrue((case_root / "inputs.json").is_file())
            self.assertFalse((case_root / "agent").exists())
            self.assertEqual(len(list((case_root / "native-inputs").iterdir())), 6)
            job = json.loads((case_root / "job.json").read_text())
            profiles = [
                TemporalInputProfile.model_validate(value)
                for value in manifest.task.metadata["temporal_inputs"]
            ]
            self.assertEqual(
                [profile.cloud_fraction for profile in profiles],
                [0.0, 0.0, 0.0025],
            )
            self.assertEqual(
                [
                    asset.quality.cloud_cover_percent
                    for asset in manifest.assets
                    if asset.bands == ["scl"]
                ],
                [0.0, 0.0, 0.25],
            )
            selection = select_temporal_pair(
                TemporalSelectAlignArguments.model_validate(job["arguments"]),
                profiles,
            )
            self.assertEqual(selection.status, job["truth"]["expected_selection_status"])
            self.assertEqual(selection.reason, job["truth"]["expected_selection_reason"])
            self.assertFalse(any("nir" in asset.bands for asset in manifest.assets))

    def test_license_receipt_drift_is_rejected_before_output(self):
        self.config["source"]["license_review_sha256"] = "f" * 64
        output = self.root / "refused"
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            MODULE.prepare(
                self.source,
                self.license_path,
                output,
                self.config,
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
