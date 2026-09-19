#!/usr/bin/env python3
"""One bounded CPU scientific job; invoked only by the isolated provider."""
import json
import sys
from pathlib import Path
from app.v2.raster_math import MaskArtifactInput, NativeBand, compute_masked_ndvi, compute_ndvi

if __name__ == "__main__":
    values=sys.argv[1:]
    if len(values)==4:
        red_path,nir_path,output,metadata = values
        red,nir = [NativeBand.model_validate(value) for value in json.loads(metadata)]
        content,result = compute_ndvi(Path(red_path),Path(nir_path),red,nir)
    elif len(values)==5:
        red_path,nir_path,mask_path,output,metadata = values
        raw=json.loads(metadata)
        red,nir = [NativeBand.model_validate(value) for value in raw[:2]]
        mask=MaskArtifactInput.model_validate(raw[2])
        content,result = compute_masked_ndvi(Path(red_path),Path(nir_path),Path(mask_path),red,nir,mask)
    else:
        raise SystemExit("invalid raster worker arguments")
    with Path(output).open("xb") as stream:
        stream.write(content)
    print(result.model_dump_json())
