"""CPU raster provider. Mount approved native inputs only; no egress or credentials."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from .v2.raster_grid import (CONTINUOUS_VERSION as CONTINUOUS_GRID_VERSION,
                             ContinuousBand, ContinuousGridResult, GridArguments,
                             GridResult, NativeSCL, VERSION as GRID_VERSION,
                             checked_continuous_grids, checked_grids,
                             validate_continuous_grid, validate_grid)
from .v2.raster_math import (BandMathArguments, MASKED_VERSION, MaskArtifactInput,
                             MaskedNDVIResult, NativeBand, NDVIResult, MAX_INPUT,
                             MAX_OUTPUT, VERSION, checked_pair,
                             validate_masked_ndvi, validate_ndvi)
from .v2.temporal import (TemporalAlignRequest, TemporalStackResult,
                          VERSION as TEMPORAL_VERSION, checked_temporal_inputs,
                          validate_temporal_stack)


class RasterBridge:
    def __init__(self, inputs: Path, manifest: dict, worker: Path, grid_worker: Path | None = None,
                 temporal_worker: Path | None = None):
        self.inputs,self.manifest,self.worker = inputs.resolve(),manifest,worker.resolve()
        self.grid_worker = grid_worker.resolve() if grid_worker is not None else None
        self.temporal_worker = temporal_worker.resolve() if temporal_worker is not None else None
        self.slot = threading.BoundedSemaphore(1)

    def resolve(self, asset_id: str, profile_type=NativeBand):
        entry = self.manifest.get(asset_id)
        if not isinstance(entry,dict) or set(entry) != {"filename","native"}:
            raise ValueError("input not approved")
        name = entry["filename"]
        if not name or Path(name).name != name:
            raise ValueError("invalid input path")
        band = profile_type.model_validate(entry["native"])
        path = self.inputs/name
        if band.asset_id != asset_id or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.inputs):
            raise ValueError("invalid input identity/path")
        if path.stat().st_size > MAX_INPUT:
            raise ValueError("input exceeds limit")
        return path,band

    def execute(self, args: BandMathArguments) -> tuple[bytes,NDVIResult]:
        if args.masked:
            raise HTTPException(422,"mask requires the bounded binary endpoint")
        if not self.slot.acquire(blocking=False):
            raise HTTPException(429,"raster provider busy")
        try:
            red_path,red = self.resolve(args.red_asset_id)
            nir_path,nir = self.resolve(args.nir_asset_id)
            checked_pair(red,nir)
            with tempfile.TemporaryDirectory(prefix="ndvi-") as directory:
                output = Path(directory)/"ndvi.tif"
                run = subprocess.run([sys.executable,str(self.worker),str(red_path),str(nir_path),str(output),
                    json.dumps([red.model_dump(mode="json"),nir.model_dump(mode="json")])],capture_output=True,timeout=20,check=False)
                if run.returncode or len(run.stdout)>16384 or not output.is_file() or output.stat().st_size>MAX_OUTPUT:
                    raise ValueError("invalid raster worker output")
                result = NDVIResult.model_validate_json(run.stdout)
                content = output.read_bytes()
                validate_ndvi(content,result)
                if result.input_asset_ids != [red.asset_id,nir.asset_id] or result.input_sha256 != [red.sha256,nir.sha256]:
                    raise ValueError("worker source mismatch")
                return content,result
        except subprocess.TimeoutExpired:
            raise HTTPException(504,"raster worker timed out") from None
        except (ValueError,OSError):
            raise HTTPException(422,"reviewed raster inputs or output invalid") from None
        finally:
            self.slot.release()

    def execute_masked(self, args: BandMathArguments, mask: MaskArtifactInput,
                       mask_content: bytes) -> tuple[bytes,MaskedNDVIResult]:
        if (not args.masked or args.mask_artifact_id!=mask.artifact_id
                or args.cloud_policy!=mask.cloud_policy or len(mask_content)!=mask.size_bytes
                or hashlib.sha256(mask_content).hexdigest()!=mask.sha256):
            raise HTTPException(422,"mask request identity invalid")
        if not self.slot.acquire(blocking=False):
            raise HTTPException(429,"raster provider busy")
        try:
            red_path,red = self.resolve(args.red_asset_id)
            nir_path,nir = self.resolve(args.nir_asset_id)
            checked_pair(red,nir)
            with tempfile.TemporaryDirectory(prefix="masked-ndvi-") as directory:
                mask_path,output = Path(directory)/"mask.tif",Path(directory)/"ndvi.tif"
                with mask_path.open("xb") as stream:
                    stream.write(mask_content)
                run = subprocess.run([sys.executable,str(self.worker),str(red_path),str(nir_path),
                    str(mask_path),str(output),json.dumps([red.model_dump(mode="json"),
                    nir.model_dump(mode="json"),mask.model_dump(mode="json")])],
                    capture_output=True,timeout=20,check=False)
                if run.returncode or len(run.stdout)>16384 or not output.is_file() or output.stat().st_size>MAX_OUTPUT:
                    raise ValueError("invalid masked raster worker output")
                result = MaskedNDVIResult.model_validate_json(run.stdout)
                content = output.read_bytes()
                validate_masked_ndvi(content,result)
                if (result.input_asset_ids != [red.asset_id,nir.asset_id]
                        or result.input_sha256 != [red.sha256,nir.sha256]
                        or result.mask_artifact_id!=mask.artifact_id or result.mask_sha256!=mask.sha256):
                    raise ValueError("masked worker source mismatch")
                return content,result
        except subprocess.TimeoutExpired:
            raise HTTPException(504,"raster worker timed out") from None
        except (ValueError,OSError):
            raise HTTPException(422,"reviewed masked inputs or output invalid") from None
        finally:
            self.slot.release()

    def execute_grid(self, args: GridArguments) -> tuple[bytes, GridResult | ContinuousGridResult]:
        if self.grid_worker is None:
            raise HTTPException(503,"raster grid worker unavailable")
        if not self.slot.acquire(blocking=False):
            raise HTTPException(429,"raster provider busy")
        try:
            if args.method == "nearest":
                source_path,source = self.resolve(args.source_asset_id,NativeSCL)
                reference_path,reference = self.resolve(args.reference_asset_id,NativeBand)
                checked_grids(source,reference)
                result_type = GridResult
            else:
                source_path,source = self.resolve(args.source_asset_id,ContinuousBand)
                reference_path,reference = self.resolve(args.reference_asset_id,ContinuousBand)
                checked_continuous_grids(source,reference)
                result_type = ContinuousGridResult
            with tempfile.TemporaryDirectory(prefix="grid-") as directory:
                output = Path(directory)/"aligned-grid.tif"
                run = subprocess.run([sys.executable,str(self.grid_worker),str(source_path),str(reference_path),str(output),
                    json.dumps({"arguments":args.model_dump(mode="json"),
                                "profiles":[source.model_dump(mode="json"),
                                            reference.model_dump(mode="json")]})],
                    capture_output=True,timeout=20,check=False)
                if run.returncode or len(run.stdout)>16384 or not output.is_file() or output.stat().st_size>MAX_OUTPUT:
                    raise ValueError("invalid raster grid worker output")
                result = result_type.model_validate_json(run.stdout)
                content = output.read_bytes()
                if args.method == "nearest":
                    validate_grid(content,result)
                else:
                    validate_continuous_grid(content,result)
                if result.input_asset_ids != [source.asset_id,reference.asset_id] or result.input_sha256 != [source.sha256,reference.sha256]:
                    raise ValueError("worker grid source mismatch")
                return content,result
        except subprocess.TimeoutExpired:
            raise HTTPException(504,"raster grid worker timed out") from None
        except (ValueError,OSError):
            raise HTTPException(422,"reviewed raster grid inputs or output invalid") from None
        finally:
            self.slot.release()

    def execute_temporal(self, args: TemporalAlignRequest) -> tuple[bytes, TemporalStackResult]:
        if self.temporal_worker is None:
            raise HTTPException(503, "temporal worker unavailable")
        if not self.slot.acquire(blocking=False):
            raise HTTPException(429, "raster provider busy")
        try:
            identifiers = [args.before_red_asset_id, args.before_scl_asset_id,
                           args.after_red_asset_id, args.after_scl_asset_id]
            types = [NativeBand, NativeSCL, NativeBand, NativeSCL]
            resolved = [self.resolve(asset_id, profile_type)
                        for asset_id, profile_type in zip(identifiers, types)]
            paths = [value[0] for value in resolved]
            profiles = [value[1] for value in resolved]
            checked_temporal_inputs(*profiles)
            with tempfile.TemporaryDirectory(prefix="temporal-stack-") as directory:
                output = Path(directory) / "temporal-stack.tif"
                run = subprocess.run([sys.executable, str(self.temporal_worker),
                    *[str(path) for path in paths], str(output),
                    json.dumps([profile.model_dump(mode="json") for profile in profiles])],
                    capture_output=True, timeout=30, check=False)
                if (run.returncode or len(run.stdout) > 32768 or not output.is_file()
                        or output.stat().st_size > 64 * 1024 * 1024):
                    raise ValueError("invalid temporal worker output")
                result = TemporalStackResult.model_validate_json(run.stdout)
                content = output.read_bytes()
                validate_temporal_stack(content, result)
                if (result.input_asset_ids != identifiers
                        or result.input_sha256 != [profile.sha256 for profile in profiles]
                        or result.cloud_policy != args.cloud_policy):
                    raise ValueError("temporal worker source mismatch")
                return content, result
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "temporal worker timed out") from None
        except (ValueError, OSError):
            raise HTTPException(422, "reviewed temporal inputs or output invalid") from None
        finally:
            self.slot.release()


def create_app(bridge: RasterBridge | None = None):
    if bridge is None:
        manifest = Path(os.environ["EO_RASTER_MANIFEST"])
        if manifest.is_symlink() or manifest.stat().st_size>1024*1024:
            raise ValueError("bounded manifest required")
        grid_worker = Path(os.environ["EO_RASTER_GRID_WORKER"]) if os.environ.get("EO_RASTER_GRID_WORKER") else None
        temporal_worker = Path(os.environ["EO_RASTER_TEMPORAL_WORKER"]) if os.environ.get("EO_RASTER_TEMPORAL_WORKER") else None
        bridge = RasterBridge(Path(os.environ["EO_RASTER_INPUTS"]),json.loads(manifest.read_text()),
                              Path(os.environ["EO_RASTER_WORKER"]),grid_worker,temporal_worker)
    app = FastAPI(docs_url=None,redoc_url=None,openapi_url=None)

    @app.get("/healthz")
    def health():
        return {"status":"ok","tool_version":VERSION,"grid_tool_version":GRID_VERSION if bridge.grid_worker else None,
                "grid_tool_versions":[GRID_VERSION,CONTINUOUS_GRID_VERSION] if bridge.grid_worker else [],
                "temporal_tool_version":TEMPORAL_VERSION if bridge.temporal_worker else None}

    @app.post("/band-math")
    def band_math(args: BandMathArguments):
        content,result = bridge.execute(args)
        # One bounded response; no public cache or output-fetch URL namespace.
        return Response(content,media_type="image/tiff",headers={
            "X-Raster-Metadata":result.model_dump_json(),"X-Raster-Version":VERSION,
            "X-Content-SHA256":hashlib.sha256(content).hexdigest()})

    @app.post("/masked-band-math")
    async def masked_band_math(request: Request):
        raw_args=request.headers.get("X-Raster-Arguments","")
        raw_mask=request.headers.get("X-Raster-Masked-Request","")
        if len(raw_args)>4096 or len(raw_mask)>4096:
            raise HTTPException(422,"masked request metadata too large")
        try:
            args=BandMathArguments.model_validate_json(raw_args)
            mask=MaskArtifactInput.model_validate_json(raw_mask)
        except ValueError:
            raise HTTPException(422,"masked request metadata invalid") from None
        content=bytearray()
        async for chunk in request.stream():
            if len(content)+len(chunk)>MAX_OUTPUT:
                raise HTTPException(413,"mask body too large")
            content.extend(chunk)
        output,result=bridge.execute_masked(args,mask,bytes(content))
        return Response(output,media_type="image/tiff",headers={
            "X-Raster-Metadata":result.model_dump_json(),"X-Raster-Version":MASKED_VERSION,
            "X-Content-SHA256":hashlib.sha256(output).hexdigest()})

    @app.post("/resample-grid")
    def resample_grid(args: GridArguments):
        content,result = bridge.execute_grid(args)
        version = GRID_VERSION if args.method == "nearest" else CONTINUOUS_GRID_VERSION
        return Response(content,media_type="image/tiff",headers={
            "X-Raster-Grid-Metadata":result.model_dump_json(),"X-Raster-Grid-Version":version,
            "X-Content-SHA256":hashlib.sha256(content).hexdigest()})

    @app.post("/temporal-align")
    def temporal_align(args: TemporalAlignRequest):
        content, result = bridge.execute_temporal(args)
        return Response(content, media_type="image/tiff", headers={
            "X-Temporal-Metadata": result.model_dump_json(),
            "X-Temporal-Version": TEMPORAL_VERSION,
            "X-Content-SHA256": hashlib.sha256(content).hexdigest()})

    return app
