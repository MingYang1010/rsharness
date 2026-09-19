#!/usr/bin/env python3
"""One bounded categorical grid job; invoked only by the isolated provider."""
import json
import sys
from pathlib import Path

from app.v2.raster_grid import NativeSCL, compute_grid
from app.v2.raster_math import NativeBand


if __name__ == "__main__":
    source_path, reference_path, output, metadata = sys.argv[1:]
    source_raw, reference_raw = json.loads(metadata)
    source = NativeSCL.model_validate(source_raw)
    reference = NativeBand.model_validate(reference_raw)
    content, result = compute_grid(Path(source_path), Path(reference_path), source, reference)
    with Path(output).open("xb") as stream:
        stream.write(content)
    print(result.model_dump_json())
