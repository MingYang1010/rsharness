#!/usr/bin/env python3
"""One bounded reviewed grid job; invoked only by the isolated provider."""
import json
import sys
from pathlib import Path

from app.core.raster_grid import (ContinuousBand, GridArguments, NativeSCL,
                                compute_continuous_grid, compute_grid)
from app.core.raster_math import NativeBand


if __name__ == "__main__":
    source_path, reference_path, output, metadata = sys.argv[1:]
    request = json.loads(metadata)
    args = GridArguments.model_validate(request["arguments"])
    source_raw, reference_raw = request["profiles"]
    if args.method == "nearest":
        source = NativeSCL.model_validate(source_raw)
        reference = NativeBand.model_validate(reference_raw)
        content, result = compute_grid(Path(source_path), Path(reference_path), source, reference)
    else:
        source = ContinuousBand.model_validate(source_raw)
        reference = ContinuousBand.model_validate(reference_raw)
        content, result = compute_continuous_grid(
            Path(source_path), Path(reference_path), source, reference)
    with Path(output).open("xb") as stream:
        stream.write(content)
    print(result.model_dump_json())
