#!/usr/bin/env python3
"""One bounded two-date stack job; invoked only by the isolated provider."""
import json
import sys
from pathlib import Path

from app.core.raster_grid import NativeSCL
from app.core.raster_math import NativeBand
from app.core.temporal import compute_temporal_stack


if __name__ == "__main__":
    before_red, before_scl, after_red, after_scl, output, metadata = sys.argv[1:]
    raw = json.loads(metadata)
    profiles = (
        NativeBand.model_validate(raw[0]),
        NativeSCL.model_validate(raw[1]),
        NativeBand.model_validate(raw[2]),
        NativeSCL.model_validate(raw[3]),
    )
    content, result = compute_temporal_stack(
        tuple(Path(value) for value in (before_red, before_scl, after_red, after_scl)),
        profiles,
    )
    with Path(output).open("xb") as stream:
        stream.write(content)
    print(result.model_dump_json())
