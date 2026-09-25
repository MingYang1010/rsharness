#!/usr/bin/env python3
"""One bounded zonal-statistics job for the isolated raster provider."""
import sys
from pathlib import Path

from app.core.raster_zonal import ZonalRequest, compute_zonal_stats


if __name__ == "__main__":
    source_path, metadata = sys.argv[1:]
    request = ZonalRequest.model_validate_json(metadata)
    result = compute_zonal_stats(Path(source_path), request)
    print(result.model_dump_json())
