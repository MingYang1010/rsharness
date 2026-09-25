#!/usr/bin/env python3
"""One fixed NDMI job over reviewed B08 and aligned B11 inputs."""
import json
import sys
from pathlib import Path

from app.core.raster_math import (AlignedSWIRInput, BandMathArguments, NativeBand,
                                compute_ndmi)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("invalid NDMI worker arguments")
    nir_path, swir_path, output, raw_arguments, raw_profiles = sys.argv[1:]
    arguments = BandMathArguments.model_validate_json(raw_arguments)
    profiles = json.loads(raw_profiles)
    nir = NativeBand.model_validate(profiles[0])
    swir = AlignedSWIRInput.model_validate(profiles[1])
    content, result = compute_ndmi(
        Path(nir_path), Path(swir_path), arguments, nir, swir)
    with Path(output).open("xb") as stream:
        stream.write(content)
    print(result.model_dump_json())
