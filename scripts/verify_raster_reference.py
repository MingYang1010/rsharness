#!/usr/bin/env python3
"""Operator-only numeric reference, independent from raster_math implementation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"harness_api"))
from app.v2.storage.quota import StorageQuota


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True,type=Path);args=parser.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to((ROOT/"runtime").resolve()):raise SystemExit("runtime run required")
    manifest=json.loads((run/"native-inputs.json").read_text())
    saved=json.loads((run/"reports/native-checkpoint.json").read_text())
    report_path=run/"reference/numeric.json"
    if report_path.exists():raise SystemExit("preserve prior report")
    results=[]
    for artifact,result in zip(saved["artifacts"],saved["results"]):
        bands=[];masks=[]
        for key in result["input_asset_ids"]:
            entry=manifest[key];source=run/"native-inputs"/entry["filename"]
            assert hashlib.sha256(source.read_bytes()).hexdigest()==entry["native"]["sha256"]
            with rasterio.open(source,driver="GTiff") as image:
                bands.append(image.read(1).astype(np.float64)*image.scales[0]+image.offsets[0]);masks.append(image.read_masks(1)>0)
                grid=image.transform;crs=image.crs
        r,n=bands;valid=masks[0]&masks[1]&(r>=0)&(n>=0)&np.isfinite(r)&np.isfinite(n)&((n+r)>1e-6)
        expected=np.full(r.shape,-9999.,dtype=np.float32)
        ratio=np.divide(n-r,n+r,out=np.zeros(r.shape),where=valid)  # No production math import.
        expected[valid]=ratio[valid].astype(np.float32)
        digest=artifact["sha256"];path=ROOT/"runtime/managed-artifacts"/digest[:2]/digest/"content"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        with rasterio.open(path,driver="GTiff") as output:
            actual=output.read(1)
            assert output.transform==grid and output.crs==crs
            assert np.array_equal(output.read_masks(1)>0,valid)
            assert np.array_equal(actual,expected)
        results.append({"artifact_id":artifact["artifact_id"],"sha256":digest,"valid_pixels":int(valid.sum()),
                        "total_pixels":int(valid.size),"maximum_absolute_error":float(np.max(np.abs(actual-expected)))})
    assert len(results)==3
    with StorageQuota(ROOT/"runtime").hold(report_path.parent,1024*1024,"native-reference-report"):
        report_path.parent.mkdir()
        with report_path.open("x") as stream:json.dump({"status":"passed","rasters":results,"scope":"numeric reference only; not vegetation-change ground truth"},stream,indent=2)
    print(json.dumps({"status":"passed","rasters":len(results),"exact_pixel_equality":True}))

if __name__=="__main__":main()
