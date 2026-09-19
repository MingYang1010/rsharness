"""Harness policy and lineage for an isolated native-band provider."""
from __future__ import annotations

import hashlib
import json

import httpx

from ..artifacts import ArtifactStoreError
from ..artifact_identity import DERIVATION_SCHEME, validate_derivation_metadata
from ..domain import V2DomainError
from ..events import sha256_json
from ..raster_grid import (GridArguments, NativeSCL, TOOL_ID as GRID_TOOL,
                           VERSION as GRID_VERSION)
from ..raster_math import (BandMathArguments, CLOUD_POLICY, MASKED_VERSION,
                           MaskArtifactInput, MaskedNDVIResult, NativeBand,
                           NDVIResult, TOOL_ID, VERSION, MAX_OUTPUT, MAX_INPUT,
                           MEDIA_TYPE, checked_pair, read_cloud_mask,
                           validate_masked_ndvi, validate_ndvi)
from ..schemas import ArtifactLineage, SpatialExtent, SpatialBoundingBox, TemporalExtent
from .catalog import _is_public_image
from .runtime import PreparedTool, ToolOutput


class RasterExecutor:
    tool_id = TOOL_ID
    tool_version = VERSION
    tool_versions = {VERSION, MASKED_VERSION}

    def __init__(self,base_url,artifacts,transport=None):
        self.base_url,self.artifacts,self.transport = base_url,artifacts,transport

    def plan(self,action,manifest,accessible_asset_refs,episode_artifacts=None):
        if action.tool_id != TOOL_ID or TOOL_ID not in manifest.scenario.allowed_tools:
            raise V2DomainError("policy_rejected","raster tool not allowed",403,phase="policy")
        try:
            args = BandMathArguments.model_validate(action.arguments)
            ids = [args.red_asset_id,args.nir_asset_id]
            if not set(ids).issubset(set(manifest.task.inputs).intersection(accessible_asset_refs)):
                raise V2DomainError("policy_rejected","native input not accessible",403,phase="policy")
            bands = [NativeBand.model_validate(manifest.task.metadata["raster_inputs"][key]) for key in ids]
            checked_pair(*bands)
            assets = [next(a for a in manifest.assets if a.asset_id==key) for key in ids]
            for asset,band in zip(assets,bands):
                if (not _is_public_image(asset) or "reflectance" not in asset.roles or asset.bands != [band.band]
                        or asset.sha256!=band.sha256 or asset.asset_id!=band.asset_id or asset.size_bytes>MAX_INPUT
                        or asset.spatial is None or asset.temporal is None or asset.temporal.start!=band.acquired
                        or asset.temporal.end!=band.acquired):
                    raise ValueError("native profile/asset mismatch")
        except V2DomainError:
            raise
        except (ValueError,KeyError,TypeError,StopIteration):
            raise V2DomainError("invalid_tool_arguments","reviewed same-grid red/NIR pair required",phase="request") from None
        size = sum(a.size_bytes for a in assets)
        if not args.masked:
            return PreparedTool(VERSION,size,MAX_OUTPUT,lambda:self.invoke(args,bands,size))
        try:
            if (manifest.task.metadata.get("artifact_identity") != DERIVATION_SCHEME
                    or manifest.task.metadata.get("cloud_mask_policy") != CLOUD_POLICY):
                raise ValueError("task does not opt into reviewed cloud masking")
            mask = (episode_artifacts or {}).get(args.mask_artifact_id)
            if mask is None:
                raise V2DomainError("policy_rejected","mask artifact is not accessible in this episode",403,phase="policy")
            validate_derivation_metadata(mask.model_dump(mode="json"))
            if (mask.kind!="raster" or mask.media_type!=MEDIA_TYPE or mask.size_bytes>MAX_OUTPUT
                    or mask.temporal is None or mask.temporal.start!=bands[0].acquired
                    or mask.temporal.end!=bands[0].acquired or mask.spatial is None
                    or mask.spatial.shape!=[bands[0].height,bands[0].width,1]
                    or mask.lineage.tool_id!=GRID_TOOL or mask.lineage.tool_version!=GRID_VERSION
                    or len(mask.lineage.input_refs)!=2 or mask.lineage.input_refs[1] not in ids):
                raise ValueError("mask metadata or lineage mismatch")
            source_id,reference_id=mask.lineage.input_refs
            grid_inputs=manifest.task.metadata["grid_inputs"]
            source=NativeSCL.model_validate(grid_inputs[source_id])
            reference=NativeBand.model_validate(grid_inputs[reference_id])
            if (source.item_id!=bands[0].item_id or source.acquired!=bands[0].acquired
                    or reference.model_dump(mode="json")!=next(b for b in bands if b.asset_id==reference_id).model_dump(mode="json")):
                raise ValueError("mask task source or target mismatch")
            expected=sha256_json({"arguments":GridArguments(source_asset_id=source_id,
                reference_asset_id=reference_id,method="nearest").model_dump(),
                "grid_inputs":[source.model_dump(mode="json"),reference.model_dump(mode="json")],
                "invalid_policy":"source-mask-or-scl-zero-or-outside-source",
                "reference_policy":"geometry-only-ignore-reference-values-and-mask"})
            if mask.lineage.parameters_hash!=expected:
                raise ValueError("mask derivation policy mismatch")
            mask_input=MaskArtifactInput(artifact_id=mask.artifact_id,sha256=mask.sha256,
                size_bytes=mask.size_bytes,cloud_policy=args.cloud_policy)
        except V2DomainError:
            raise
        except (ValueError,KeyError,TypeError,StopIteration):
            raise V2DomainError("invalid_tool_arguments","reviewed episode-local SCL mask required",phase="request") from None
        size += mask.size_bytes
        return PreparedTool(MASKED_VERSION,size,MAX_OUTPUT,
                            lambda:self.invoke(args,bands,size,mask,mask_input))

    def invoke(self,args,bands,size,mask=None,mask_input=None):
        try:
            mask_content=None
            if mask is not None:
                try:
                    mask_content=self.artifacts.read_content(mask).content
                except ArtifactStoreError as error:
                    raise V2DomainError(error.code,"cloud mask artifact unavailable",409,
                                        phase="artifact") from None
                _,mask_counts=read_cloud_mask(mask_content,mask_input,bands[0])
            with httpx.Client(base_url=self.base_url,timeout=30,transport=self.transport,trust_env=False,follow_redirects=False) as client:
                request_kwargs = ({"json":args.model_dump(exclude_none=True)} if mask is None else {
                    "content":mask_content,
                    "headers":{"Content-Type":MEDIA_TYPE,
                               "X-Raster-Masked-Request":mask_input.model_dump_json(),
                               "X-Raster-Arguments":args.model_dump_json(exclude_none=True)}})
                endpoint="/band-math" if mask is None else "/masked-band-math"
                with client.stream("POST",endpoint,**request_kwargs) as response:
                    response.raise_for_status()
                    expected_version=VERSION if mask is None else MASKED_VERSION
                    if response.status_code!=200 or response.headers.get("X-Raster-Version")!=expected_version:
                        raise ValueError("unsupported raster provider")
                    raw = response.headers.get("X-Raster-Metadata","")
                    if len(raw)>16384:
                        raise ValueError("oversized metadata")
                    result = (NDVIResult if mask is None else MaskedNDVIResult).model_validate_json(raw)
                    content = bytearray()
                    for block in response.iter_bytes():
                        if len(content)+len(block)>MAX_OUTPUT:
                            raise ValueError("oversized output")
                        content.extend(block)
                    content=bytes(content)
                    if hashlib.sha256(content).hexdigest()!=response.headers.get("X-Content-SHA256"):
                        raise ValueError("raster content checksum mismatch")
            red,nir = bands
            if (result.input_asset_ids!=[red.asset_id,nir.asset_id] or result.input_sha256!=[red.sha256,nir.sha256]
                    or result.acquired!=red.acquired or result.crs!=red.crs or result.transform!=red.transform
                    or result.width!=red.width or result.height!=red.height):
                raise ValueError("raster source or geometry mismatch")
            if mask is None:
                validate_ndvi(content,result)
            else:
                if (result.mask_artifact_id!=mask.artifact_id or result.mask_sha256!=mask.sha256
                        or result.cloud_policy_version!=CLOUD_POLICY
                        or any(getattr(result,key)!=value for key,value in mask_counts.items())):
                    raise ValueError("cloud mask identity or summary mismatch")
                validate_masked_ndvi(content,result)
        except httpx.TimeoutException:
            raise V2DomainError("tool_timeout","raster provider timed out",504,True,"tool") from None
        except httpx.HTTPError:
            raise V2DomainError("tool_unavailable","raster provider failed",502,True,"tool") from None
        except (ValueError,KeyError,TypeError):
            raise V2DomainError("invalid_tool_output","scientific output validation failed",502,phase="tool") from None
        west,south,east,north=result.bbox_wgs84
        spatial=SpatialExtent(crs="EPSG:4326",bbox=SpatialBoundingBox(west=west,south=south,east=east,north=north),
                              gsd_meters=red.transform[0],shape=[red.height,red.width,1])
        try:
            artifact=self.artifacts.put_bytes(content,kind="raster",media_type=MEDIA_TYPE,spatial=spatial,
                temporal=TemporalExtent(start=red.acquired,end=red.acquired),lineage=ArtifactLineage(
                tool_id=TOOL_ID,tool_version=VERSION if mask is None else MASKED_VERSION,
                input_refs=result.input_asset_ids+([] if mask is None else [mask.artifact_id]),
                parameters_hash=sha256_json({"arguments":args.model_dump(exclude_none=True),"native_inputs":[b.model_dump(mode="json") for b in bands],
                    "invalid_policy":result.invalid_policy,"cloud_mask_applied":mask is not None,
                    **({} if mask is None else {"mask":{"artifact_id":mask.artifact_id,"sha256":mask.sha256,
                        "lineage":mask.lineage.model_dump(mode="json")},"cloud_policy":CLOUD_POLICY})})))
        except ArtifactStoreError as error:
            raise V2DomainError(error.code,"scientific artifact storage failed",503,True,"artifact") from None
        return ToolOutput(artifact,result.model_dump(mode="json"),size)
