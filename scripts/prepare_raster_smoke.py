#!/usr/bin/env python3
"""Admit reviewed native bands from an existing Sentinel window receipt."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"harness_api"))
from app.v2.raster_math import NativeBand, checked_pair, MAX_INPUT
from app.v2.storage.quota import StorageQuota
from app.v2.capabilities import TaskRegistry
from app.v2.data.stac import json_bytes


def write(path,value):
    content=json_bytes(value)
    if len(content)>1024*1024:raise ValueError("metadata too large")
    with path.open("xb") as stream:stream.write(content)


def prepare(source: Path,out: Path):
    import rasterio
    if source.is_symlink() or not source.resolve().is_relative_to((ROOT/"runtime").resolve()) or out.exists():
        raise ValueError("existing reviewed runtime source and fresh output required")
    receipt_path=source/"receipts/admission.json"
    if receipt_path.is_symlink() or receipt_path.stat().st_size>1024*1024:raise ValueError("receipt exceeds bound")
    receipt=json.loads(receipt_path.read_text())
    if receipt["status"]!="admitted" or len(receipt["windows"])!=12:raise ValueError("three reviewed dates required")
    records=[r for r in receipt["windows"] if r["asset_key"] in {"red","nir"}]
    if len(records)!=6:raise ValueError("six reviewed native inputs required")
    with StorageQuota(ROOT/"runtime").hold(out,128*1024*1024,"native-band-task-admission"):
        for name in ("inputs","native-inputs","state","reports","tasks/native-ndvi"):(out/name).mkdir(parents=True)
        template=ROOT/"tasks/worldcover-grounded-vqa"
        assets=[];native={};provider={};pairs={}
        for record in records:
            filename=record["filename"]
            if Path(filename).name!=filename:raise ValueError("invalid approved filename")
            path=source/"native-bands"/filename
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or path.stat().st_size>MAX_INPUT:
                raise ValueError("bounded native file required")
            content=path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=record["sha256"] or len(content)!=record["size_bytes"]:
                raise ValueError("approved content changed")
            with rasterio.open(path,driver="GTiff") as image:
                if (image.count!=1 or image.dtypes!=("uint16",) or image.crs.to_string()!=record["crs"]
                        or list(image.transform)[:6]!=record["transform"] or [image.width,image.height]!=[record["width"],record["height"]]
                        or list(image.scales)!=record["scales"] or list(image.offsets)!=record["offsets"]):
                    raise ValueError("approved native metadata changed")
                asset_id="asset-"+hashlib.sha256(json_bytes([record["item_id"],record["asset_key"],record["sha256"]])).hexdigest()
                band=NativeBand(asset_id=asset_id,sha256=record["sha256"],item_id=record["item_id"],band=record["asset_key"],
                    acquired=record["acquired"],crs=record["crs"],transform=record["transform"],width=image.width,height=image.height,
                    dtype="uint16",scale=image.scales[0],offset=image.offsets[0],nodata=image.nodata)
            with (out/"native-inputs"/filename).open("xb") as stream:stream.write(content)
            native[asset_id]=band.model_dump(mode="json")
            provider[asset_id]={"filename":filename,"native":native[asset_id]}
            pairs.setdefault(band.item_id,{})[band.band]=band
            west,south,east,north=record["bbox_wgs84"]
            assets.append({"asset_id":asset_id,"uri":"local://approved-native/"+filename,"media_type":"image/tiff",
                "roles":["input_image","reflectance"],"sha256":record["sha256"],"size_bytes":len(content),
                "spatial":{"crs":"EPSG:4326","bbox":dict(west=west,south=south,east=east,north=north),
                           "gsd_meters":band.transform[0],"shape":[band.height,band.width,1]},
                "temporal":{"start":band.acquired,"end":band.acquired},"platform":"sentinel-2","bands":[band.band],
                "quality":{"cloud_cover_percent":record["scene_cloud_cover_percent"],"nodata_fraction":record["nodata_fraction"]},
                "license":receipt["config"]["license_review"]["attribution"],"source":"Reviewed Sentinel-2 native reflectance DN window",
                "source_snapshot_hash":record["source_snapshot_hash"]})
        for pair in pairs.values():checked_pair(pair["red"],pair["nir"])
        if len(pairs)!=3:raise ValueError("three unique pairs required")
        assets.sort(key=lambda a:a["asset_id"])
        task=json.loads((template/"task.json").read_text())
        identity=hashlib.sha256(json_bytes(native)).hexdigest()[:16]
        task.update(task_id="native-ndvi-"+identity,inputs=[a["asset_id"] for a in assets],
            prompt="Find same-date red/NIR pairs and compute native NDVI for all three dates. Cite numeric raster evidence and report valid-pixel means. Values apply DN scale/offset first. Cloud masking is NOT applied; these means do not establish vegetation change or ground-truth accuracy.",
            metadata={"observation_profile":"headless-tools-v1","artifact_identity":"derivation-sha256-v1",
                      "acceptance":"scripted-native-ndvi-numeric-check","raster_inputs":native})
        task["budget"].update(max_steps=40,max_tool_calls=25,max_wall_time_ms=600000,max_input_bytes=256*1024*1024,max_artifact_bytes=64*1024*1024)
        scenario=json.loads((template/"scenario.json").read_text())
        scenario.update(data_cutoff=receipt["config"]["cutoff"],freshness_max_age_seconds=None,
            allowed_actions=["tool.invoke","memory.save_evidence","answer.*"],allowed_tools=["catalog.search","catalog.inspect_asset","raster.band_math"])
        directory=out/"tasks/native-ndvi"
        for name,value in (("task.json",task),("assets.json",assets),("scenario.json",scenario),
                           ("evaluator.json",json.loads((template/"evaluator.json").read_text()))):write(directory/name,value)
        write(out/"inputs.json",{})
        write(out/"native-inputs.json",provider)
        write(out/"job.json",{"seed":42,"task_ref":{"task_id":task["task_id"],"task_version":task["task_version"]}})
        manifest=TaskRegistry(out/"tasks").get(task["task_id"],task["task_version"])
        write(out/"admission.json",{"source_receipt_sha256":hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "task_manifest_hash":manifest.task_manifest_hash,"inputs":6,"date_pairs":3,"cloud_mask_applied":False})
        return manifest.task_manifest_hash


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--output-name",required=True)
    args=parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}",args.output_name):raise SystemExit("simple fresh output name required")
    digest=prepare(args.source,ROOT/"runtime"/args.output_name)
    print(json.dumps({"task_manifest_hash":digest,"inputs":6,"date_pairs":3}))
