"""Reviewed Sentinel-2 STAC metadata admission; discovery never grants access."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from urllib.parse import urlencode

from .http_range import BoundedHTTP, checked_url

ENDPOINT = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
ASSETS = {"visual": "TCI.tif", "red": "B04.tif", "nir": "B08.tif", "scl": "SCL.tif"}


def utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("explicit UTC timestamp required")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def checked_config(config: dict) -> dict:
    if set(config) != {"collection", "bbox", "start", "end", "cutoff", "max_cloud_cover", "items", "assets", "license_review"}:
        raise ValueError("configuration fields are not the reviewed schema")
    if config["collection"] != COLLECTION or config["assets"] != list(ASSETS):
        raise ValueError("only reviewed Sentinel-2 asset profile supported")
    bbox = config["bbox"]
    if (not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) not in (float, int) or not math.isfinite(v) for v in bbox)
            or not -180 <= bbox[0] < bbox[2] <= 180 or not -90 <= bbox[1] < bbox[3] <= 90
            or bbox[2]-bbox[0] > .05 or bbox[3]-bbox[1] > .05):
        raise ValueError("use a bounded non-antimeridian WGS84 AOI")
    if not utc(config["start"]) <= utc(config["end"]) <= utc(config["cutoff"]) <= datetime.now(timezone.utc):
        raise ValueError("invalid acquisition interval or cutoff")
    cloud = config["max_cloud_cover"]
    if type(cloud) not in (float, int) or not 0 <= cloud <= 100:
        raise ValueError("invalid cloud limit")
    ids = config["items"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids):
        raise ValueError("pin one to three distinct item IDs")
    if any(not re.fullmatch(r"S2[ABC]_[0-9]{2}[A-Z]{3}_[0-9]{8}_[0-9]+_L2A", name) for name in ids):
        raise ValueError("invalid pinned item ID")
    review = config["license_review"]
    if (set(review) != {"collection_license", "terms_url", "legal_notice_url", "attribution", "scope"}
            or review["collection_license"] != "proprietary"
            or review["terms_url"] != "https://dataspace.copernicus.eu/terms-and-conditions"
            or review["legal_notice_url"] != "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice"
            or review["scope"] != "local-research" or not 1 <= len(review["attribution"]) <= 256):
        raise ValueError("license review does not match approved source")
    return config


def fetch_json(http: BoundedHTTP, url: str) -> tuple[dict, bytes]:
    content, _ = http.fetch(url, 1024 * 1024)
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("STAC response must be an object")
    return value, content


def validate_item(item: dict, config: dict, expected_id: str | None = None) -> dict:
    if item.get("type") != "Feature" or item.get("collection") != COLLECTION or item.get("id") not in config["items"]:
        raise ValueError("STAC item identity mismatch")
    if expected_id is not None and item["id"] != expected_id:
        raise ValueError("STAC response does not match requested item")
    properties = item["properties"]
    acquired = utc(properties["datetime"])
    if not utc(config["start"]) <= acquired <= utc(config["end"]) or acquired > utc(config["cutoff"]):
        raise ValueError("item acquisition time outside cutoff")
    cloud = properties.get("eo:cloud_cover")
    if type(cloud) not in (int, float) or not 0 <= cloud <= config["max_cloud_cover"]:
        raise ValueError("item cloud metadata exceeds reviewed limit")
    bbox, aoi = item["bbox"], config["bbox"]
    if len(bbox) != 4 or any(type(v) not in (float, int) or not math.isfinite(v) for v in bbox):
        raise ValueError("invalid item footprint bounds")
    if not (bbox[0] <= aoi[0] < aoi[2] <= bbox[2] and bbox[1] <= aoi[1] < aoi[3] <= bbox[3]):
        raise ValueError("item bbox does not contain AOI; raster coverage still requires verification")
    epsg = properties.get("proj:epsg")
    if type(epsg) is not int or not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        raise ValueError("unsupported source CRS")
    selected = {}
    for key, filename in ASSETS.items():
        asset = item["assets"][key]
        url = checked_url(asset["href"])
        if not url.startswith("https://sentinel-cogs.s3.us-west-2.amazonaws.com/") or not url.endswith("/" + item["id"] + "/" + filename):
            raise ValueError("asset path does not bind pinned item/band")
        if not asset.get("type", "").startswith("image/tiff"):
            raise ValueError("only reviewed COG TIFF assets supported")
        expected_roles = ["visual"] if key == "visual" else ["data", "reflectance"]
        if asset.get("roles") != expected_roles:
            raise ValueError("unexpected asset roles; manual review required")
        shape = asset.get("proj:shape")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or not 0 < v <= 20000 for v in shape):
            raise ValueError("invalid source shape")
        selected[key] = asset
    return {"id": item["id"], "acquired": properties["datetime"], "epsg": epsg,
            "scene_cloud_cover_percent": cloud, "assets": selected,
            "snapshot_sha256": hashlib.sha256(json_bytes(item)).hexdigest()}


def discover(http: BoundedHTTP, config: dict) -> dict:
    checked_config(config)
    query = urlencode({"collections": COLLECTION, "bbox": ",".join(map(str, config["bbox"])),
        "datetime": config["start"] + "/" + config["end"], "limit": 3,
        "query": json.dumps({"eo:cloud_cover": {"lte": config["max_cloud_cover"]}}), "sortby": "+properties.datetime"})
    value, raw = fetch_json(http, ENDPOINT + "/search?" + query)
    if value.get("type") != "FeatureCollection" or len(value.get("features", [])) > 3:
        raise ValueError("unexpected discovery response")
    return {"candidates": [{"id": i["id"], "datetime": i["properties"].get("datetime"),
              "cloud_cover": i["properties"].get("eo:cloud_cover")} for i in value["features"]],
            "has_more": any(link.get("rel") == "next" for link in value.get("links", [])),
            "response_sha256": hashlib.sha256(raw).hexdigest(), "agent_admitted": False}
