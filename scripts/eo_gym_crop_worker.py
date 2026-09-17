#!/usr/bin/env python3
"""Execute the pinned upstream CPU crop class without eager GPU tool imports."""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import resource
import sys
import types
from pathlib import Path


def main() -> None:
    source, input_path, work, aoi_json = sys.argv[1:]
    source_root, work_root = Path(source), Path(work)
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    cache = work_root / "cache"
    cache.mkdir()
    os.environ.update({
        "EO_GYM_CONFIG_FILE": str(work_root / "absent.toml"),
        "EO_GYM_RUN_WITH_MASKING": "false", "EO_GYM_RUN_ON_LOCAL": "false",
        "EO_GYM_TOOL_RUN_ON_LOCAL": "true", "EO_GYM_IMAGE_CACHE_DIR": str(cache),
        "EO_GYM_VQA_OUTPUTS_DIR": str(work_root / "maps"),
        "EO_GYM_DATA_DIR": str(work_root / "absent-data"),
        "EO_GYM_PUBLIC_DATA_BASE_URL": "", "HF_HUB_OFFLINE": "1",
    })
    sys.path.insert(0, str(source_root / "src"))
    import eo_gym.runtime.rs_tools
    # Upstream tools/__init__.py eagerly imports SAM3/VLM models. This namespace
    # shim loads original, hash-verified CPU modules only; no source is rewritten.
    name = "eo_gym.runtime.rs_tools.tools"
    package = types.ModuleType(name)
    package.__path__ = [str(source_root / "src/eo_gym/runtime/rs_tools/tools")]
    sys.modules[name] = package
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 20_000_000
    with Image.open(input_path) as input_image:
        if input_image.width * input_image.height > 20_000_000:
            raise ValueError("input exceeds CPU crop pixel limit; use a windowed raster tool")
    module = importlib.import_module(name + ".load_image_with_crop")
    if module.UPDATE_CROP_GT is not False:
        raise RuntimeError("ground-truth crop propagation must remain disabled")
    with contextlib.redirect_stdout(sys.stderr):
        result = module.CropOpticalOrSarImage(device="cpu")(input_path, json.loads(aoi_json))
    result_path = Path(result["cropped_path"]).resolve()
    if not result_path.is_relative_to(cache.resolve()):
        raise RuntimeError("upstream output escaped isolated cache")
    output = work_root / "result.png"
    with Image.open(result_path) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("crop pixel limit exceeded")
        image.convert("RGB").save(output, format="PNG")
    print(json.dumps({"width": result["width"], "height": result["height"],
                      "bbox_px": result["bbox_px"], "aoi_norm": result["aoi_norm"]}))


if __name__ == "__main__":
    main()
