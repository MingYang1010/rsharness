import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from app.v2.data.http_range import BoundedHTTP, RangeSource, checked_url, BLOCK_SIZE
from app.v2.data.stac import (ASSETS, checked_config, discover, validate_item,
                              validate_swir16_asset)
from app.v2.data.stac_windows import extract_window

URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/test/B04.tif"
PROJECT = Path(__file__).resolve().parents[2]


def make_http(content, change=None, **limits):
    def handle(request):
        headers = {"etag": '"fixture-v1"'}
        if request.method == "HEAD":
            headers["content-length"] = str(len(content))
            return httpx.Response(200, headers=headers, stream=httpx.ByteStream(b""))
        start, end = map(int, request.headers["range"][6:].split("-"))
        payload = content[start:end+1]
        headers.update({"content-range": f"bytes {start}-{end}/{len(content)}", "content-length": str(len(payload))})
        status = 206
        if change:
            status, headers, payload = change(status, headers, payload)
        return httpx.Response(status, headers=headers, stream=httpx.ByteStream(payload))
    client = httpx.Client(transport=httpx.MockTransport(handle), trust_env=False)
    return BoundedHTTP(client=client, **limits), client


def item_fixture(config):
    name = config["items"][0]
    assets = {key: {"href": f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/test/{name}/{filename}",
                        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                        "roles": ["visual"] if key == "visual" else ["data", "reflectance"],
                        "proj:shape": [10980, 10980]} for key, filename in ASSETS.items()}
    return {"type": "Feature", "collection": "sentinel-2-l2a", "id": name,
            "bbox": [118, 31, 120, 33], "properties": {"datetime": "2024-04-05T02:58:52.954000Z", "eo:cloud_cover": 14,
                                                     "proj:epsg": 32650}, "assets": assets}


class BoundedRangeTests(unittest.TestCase):
    def test_url_policy_denies_credentials_private_hosts_queries_and_traversal(self):
        for url in ("http://earth-search.aws.element84.com/v1/search", "https://localhost/v1/search",
                    "https://earth-search.aws.element84.com.evil.test/v1/search", "https://user:pass@earth-search.aws.element84.com/v1/search",
                    URL + "?sig=secret", URL + "#fragment", URL.replace("/test/", "/%2e%2e/"),
                    "https://earth-search.aws.element84.com:444/v1/search",
                    "https://dataspace.copernicus.eu/terms-and-conditions-evil"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                checked_url(url)
        self.assertEqual(checked_url(URL), URL)

    def test_proxy_is_mandatory_for_real_clients(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            BoundedHTTP()

    def test_seek_read_cache_and_eof(self):
        content = bytes(range(256)) * 2200
        http, client = make_http(content)
        self.addCleanup(client.close)
        source = RangeSource(http, URL)
        with source.open("asset.tif") as stream:
            stream.seek(BLOCK_SIZE-10)
            self.assertEqual(stream.read(30), content[BLOCK_SIZE-10:BLOCK_SIZE+20])
            requests = http.requests
            stream.seek(BLOCK_SIZE-10)
            self.assertEqual(stream.read(30), content[BLOCK_SIZE-10:BLOCK_SIZE+20])
            self.assertEqual(http.requests, requests)
            stream.seek(-5, 2)
            self.assertEqual(stream.read(), content[-5:])
            self.assertEqual(stream.read(9), b"")
            with self.assertRaises(ValueError):
                stream.seek(-1)
        with self.assertRaises(FileNotFoundError):
            source.open("asset.tif.aux.xml")

    def test_changed_etag_ignored_range_and_bad_content_range_fail(self):
        for transform in (lambda s,h,b:(200,h,b), lambda s,h,b:(s,{**h,"etag":'"changed"'},b),
                          lambda s,h,b:(s,{**h,"content-range":"bytes 1-9/10"},b),
                          lambda s,h,b:(s,{**h,"content-encoding":"gzip"},b)):
            http, client = make_http(b"data", transform)
            with client, self.assertRaises(ValueError):
                RangeSource(http, URL).open("asset.tif").read(1)

    def test_request_byte_and_deadline_limits(self):
        http, client = make_http(b"x" * 512, max_bytes=128)
        with client, self.assertRaises(ValueError):
            RangeSource(http, URL).open("asset.tif").read(1)
        http, client = make_http(b"x", max_requests=1)
        with client, self.assertRaises(ValueError):
            RangeSource(http, URL).open("asset.tif").read(1)
        http, client = make_http(b"x")
        http.deadline = 0
        with client, self.assertRaises(ValueError):
            RangeSource(http, URL)

    def test_redirect_is_never_followed(self):
        calls = []
        def redirect(request):
            calls.append(str(request.url))
            return httpx.Response(302, headers={"location":"https://127.0.0.1/private"})
        with httpx.Client(transport=httpx.MockTransport(redirect)) as client:
            http = BoundedHTTP(client=client)
            with self.assertRaises(ValueError):
                http.fetch(URL, 100)
        self.assertEqual(calls, [URL])


class STACAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PROJECT / "config/stac-sentinel-nanjing.json").read_text())

    def test_config_bounds_and_unknown_fields(self):
        self.assertEqual(checked_config(self.config), self.config)
        for patch_value in ({"bbox":[0,0,1,1]}, {"items":self.config["items"]*2}, {"cutoff":"2024-01-01T00:00:00Z"},
                            {"assets":["thumbnail"]}, {"extra":1}):
            with self.subTest(patch=patch_value), self.assertRaises(ValueError):
                checked_config({**self.config, **patch_value})

    def test_pinned_item_time_cloud_bbox_and_role_validation(self):
        item = item_fixture(self.config)
        selected = validate_item(item, self.config)
        self.assertEqual(set(selected["assets"]), set(ASSETS))
        with self.assertRaises(ValueError):
            validate_item(item, self.config, expected_id=self.config["items"][1])
        variants = []
        for key, value in (("datetime", "2025-01-01T00:00:00Z"), ("eo:cloud_cover",31), ("proj:epsg",4326)):
            trial = copy.deepcopy(item); trial["properties"][key]=value; variants.append(trial)
        trial = copy.deepcopy(item); trial["assets"]["visual"]["roles"]=["label"]; variants.append(trial)
        trial = copy.deepcopy(item); trial["assets"]["red"]["href"]=URL; variants.append(trial)
        trial = copy.deepcopy(item); trial["bbox"]=[0,0,1,1]; variants.append(trial)
        for trial in variants:
            with self.assertRaises(ValueError):
                validate_item(trial, self.config)

    def test_swir16_requires_validated_snapshot_band_grid_and_radiometry(self):
        item = item_fixture(self.config)
        item["assets"]["swir16"] = {
            "href": ("https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
                     "sentinel-s2-l2a-cogs/test/" + item["id"] + "/B11.tif"),
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "roles": ["data", "reflectance"], "proj:shape": [5490, 5490],
            "proj:transform": [20, 0, 600000, 0, -20, 3600000],
            "raster:bands": [{"data_type": "uint16", "nodata": 0,
                              "scale": .0001, "offset": -.1,
                              "spatial_resolution": 20}],
            "eo:bands": [{"name": "swir16", "common_name": "swir16"}]}
        selected = validate_item(item, self.config)
        self.assertIs(validate_swir16_asset(item, selected),
                      item["assets"]["swir16"])
        changes = [
            ("roles", ["data"]),
            ("proj:transform", [10, 0, 600000, 0, -10, 3600000]),
            ("raster:bands", [{"data_type": "uint16", "nodata": 0,
                               "scale": .001, "offset": -.1,
                               "spatial_resolution": 20}]),
            ("eo:bands", [{"name": "nir", "common_name": "nir"}]),
        ]
        for key, value in changes:
            trial = copy.deepcopy(item)
            trial["assets"]["swir16"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_swir16_asset(trial, validate_item(trial, self.config))
        with self.assertRaises(ValueError):
            validate_swir16_asset(item, {**selected, "snapshot_sha256": "f" * 64})

    def test_discovery_stays_candidate_and_does_not_follow_pagination(self):
        value = {"type":"FeatureCollection", "features":[item_fixture(self.config)],
                 "links":[{"rel":"next","href":"https://127.0.0.1/private"}]}
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200,stream=httpx.ByteStream(json.dumps(value).encode())))) as client:
            http = BoundedHTTP(client=client)
            result = discover(http, self.config)
            self.assertFalse(result["agent_admitted"])
            self.assertTrue(result["has_more"])
            self.assertEqual(http.requests, 1)

    def test_prepared_task_exposes_only_visual_windows_and_freezes_cutoff(self):
        spec = importlib.util.spec_from_file_location("admit_stac_windows", PROJECT / "scripts/admit_stac_windows.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = []
        for item_id in self.config["items"]:
            for key in ASSETS:
                records.append({"item_id": item_id, "asset_key": key, "filename": item_id + "-" + key + ".tif",
                    "sha256": hashlib.sha256((item_id + key).encode()).hexdigest(), "size_bytes": 1024,
                    "bbox_wgs84": self.config["bbox"], "width": 194, "height": 226,
                    "acquired": "2024-04-05T02:58:52.954000Z", "scene_cloud_cover_percent": 14,
                    "nodata_fraction": 0, "source_snapshot_hash": "a" * 64, "agent_admitted": key == "visual"})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            digest = module.prepare_task(out, self.config, records)
            job = json.loads((out / "job.json").read_text())
            manifest = module.TaskRegistry(out / "tasks").get(**job["task_ref"])
            self.assertEqual(manifest.task_manifest_hash, digest)
            self.assertEqual(len(manifest.assets), 3)
            self.assertEqual(manifest.scenario.data_cutoff, self.config["cutoff"])
            self.assertEqual(manifest.task.metadata["artifact_identity"], "derivation-sha256-v1")
            self.assertTrue(all(asset.uri.endswith("-visual.tif") for asset in manifest.assets))
            self.assertEqual(set(json.loads((out / "inputs.json").read_text())), set(manifest.task.inputs))

    def test_native_window_preserves_pixels_crs_scaling_and_refuses_overwrite(self):
        transform = from_origin(660000, 3550000, 10, 10)
        pixels = np.arange(1024*1024, dtype=np.uint16).reshape(1,1024,1024)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=1024,height=1024,count=1,dtype="uint16",crs="EPSG:32650",
                             transform=transform, tiled=True,blockxsize=256,blockysize=256,compress="deflate",nodata=0) as dataset:
                dataset.write(pixels)
                mask = np.full((1024, 1024), 255, dtype=np.uint8)
                mask[110:120,110:120] = 0
                dataset.write_mask(mask)
            content = memory.read()
        http, client = make_http(content)
        self.addCleanup(client.close)
        selected = {"id":"fixture", "epsg":32650,"acquired":"2024-04-05T00:00:00Z", "scene_cloud_cover_percent":14,
            "snapshot_sha256":"a"*64, "assets":{"red":{"href":URL,"proj:shape":[1024,1024],"proj:transform":list(transform)[:6],
                                            "raster:bands":[{"scale":.0001,"offset":-.1}]}}}
        bbox = list(transform_bounds("EPSG:32650","EPSG:4326",661000,3548500,661500,3549000))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"window.tif"
            result = extract_window(http, selected,"red",bbox,path)
            self.assertFalse(result["agent_admitted"])
            self.assertEqual(result["sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
            col,row,width,height = result["window"]
            with MemoryFile(path.read_bytes()) as memory, memory.open() as dataset:
                self.assertTrue(np.array_equal(dataset.read(),pixels[:,row:row+height,col:col+width]))
                self.assertEqual(dataset.crs.to_epsg(),32650)
                self.assertEqual(dataset.scales,(.0001,));self.assertEqual(dataset.offsets,(-.1,))
                self.assertTrue(np.array_equal(dataset.dataset_mask(),mask[row:row+height,col:col+width]))
            with self.assertRaises(ValueError):
                extract_window(http, selected,"red",bbox,path)
            self.assertLess(http.bytes,len(content))
