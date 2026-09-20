"""Harness policy and lineage for an isolated native-band provider."""
from __future__ import annotations

import hashlib
import json

import httpx

from ..artifacts import ArtifactStoreError
from ..artifact_identity import DERIVATION_SCHEME, validate_derivation_metadata
from ..domain import V2DomainError
from ..events import sha256_json
from ..raster_grid import (CONTINUOUS_VERSION, ContinuousBand, GridArguments,
                           NativeSCL, TOOL_ID as GRID_TOOL,
                           VERSION as GRID_VERSION,
                           checked_continuous_grids)
from ..raster_math import (BandMathArguments, CLOUD_POLICY, MASKED_VERSION,
                           NDMI_FORMULA, NDMI_VERSION, AlignedSWIRInput,
                           MaskArtifactInput, MaskedNDVIResult, NativeBand,
                           NDMIResult, NDVIResult, TOOL_ID, VERSION, MAX_OUTPUT,
                           MAX_INPUT, MEDIA_TYPE, checked_ndmi_inputs,
                           checked_pair, ndmi_lineage_payload, read_cloud_mask,
                           validate_aligned_swir, validate_masked_ndvi,
                           validate_ndmi, validate_ndvi)
from ..schemas import ArtifactLineage, SpatialExtent, SpatialBoundingBox, TemporalExtent
from .catalog import _is_public_image
from .runtime import PreparedTool, ToolOutput


class RasterExecutor:
    tool_id = TOOL_ID
    tool_version = VERSION
    tool_versions = {VERSION, MASKED_VERSION, NDMI_VERSION}

    def __init__(self,base_url,artifacts,transport=None):
        self.base_url,self.artifacts,self.transport = base_url,artifacts,transport

    def plan(self,action,manifest,accessible_asset_refs,episode_artifacts=None):
        if action.tool_id != TOOL_ID or TOOL_ID not in manifest.scenario.allowed_tools:
            raise V2DomainError("policy_rejected","raster tool not allowed",403,phase="policy")
        try:
            args = BandMathArguments.model_validate(action.arguments)
            if args.operation == "ndmi":
                return self.plan_ndmi(
                    args, manifest, accessible_asset_refs,
                    episode_artifacts or {})
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

    def plan_ndmi(self, args, manifest, accessible_asset_refs, episode_artifacts):
        try:
            if (manifest.task.metadata.get("artifact_identity") != DERIVATION_SCHEME
                    or manifest.task.metadata.get("band_math_formula") != NDMI_FORMULA):
                raise ValueError("task does not opt into fixed NDMI")
            allowed = set(manifest.task.inputs).intersection(accessible_asset_refs)
            artifact = episode_artifacts.get(args.swir_artifact_id)
            if artifact is None:
                raise V2DomainError(
                    "policy_rejected",
                    "aligned SWIR artifact is not accessible in this episode",
                    403,
                    phase="policy",
                )
            validate_derivation_metadata(artifact.model_dump(mode="json"))
            if (artifact.kind != "raster" or artifact.media_type != MEDIA_TYPE
                    or artifact.size_bytes > MAX_OUTPUT
                    or artifact.spatial is None or artifact.temporal is None
                    or artifact.spatial.crs != "EPSG:4326"
                    or artifact.lineage.tool_id != GRID_TOOL
                    or artifact.lineage.tool_version != CONTINUOUS_VERSION
                    or len(artifact.lineage.input_refs) != 2):
                raise ValueError("current-episode continuous SWIR artifact required")
            source_id, reference_id = artifact.lineage.input_refs
            if (args.nir_asset_id != reference_id
                    or not {source_id, reference_id}.issubset(allowed)):
                raise ValueError("NDMI inputs are outside the reviewed task")
            raw = manifest.task.metadata["grid_inputs"]
            source = ContinuousBand.model_validate(raw[source_id])
            nir = ContinuousBand.model_validate(raw[reference_id])
            checked_continuous_grids(source, nir)
            if source.band != "swir16" or nir.band != "nir":
                raise ValueError("fixed B08 NIR and B11 SWIR16 inputs required")
            nir_asset = next(a for a in manifest.assets
                             if a.asset_id == nir.asset_id)
            if (not _is_public_image(nir_asset)
                    or "reflectance" not in nir_asset.roles
                    or nir_asset.bands != ["nir"]
                    or nir_asset.sha256 != nir.sha256
                    or nir_asset.size_bytes > MAX_INPUT
                    or nir_asset.spatial is None or nir_asset.temporal is None
                    or nir_asset.temporal.start != nir.acquired
                    or nir_asset.temporal.end != nir.acquired):
                raise ValueError("reviewed B08 asset required")
            expected_grid_lineage = sha256_json({
                "arguments": GridArguments(
                    source_asset_id=source_id,
                    reference_asset_id=reference_id,
                    method="bilinear",
                ).model_dump(),
                "grid_inputs": [source.model_dump(mode="json"),
                                nir.model_dump(mode="json")],
                "invalid_policy": (
                    "source-mask-or-nodata-or-nonfinite-or-outside-source-"
                    "or-incomplete-bilinear-neighborhood"
                ),
                "reference_policy": "geometry-only-ignore-reference-values-and-mask",
            })
            if artifact.lineage.parameters_hash != expected_grid_lineage:
                raise ValueError("aligned SWIR lineage policy mismatch")
            west, south, east, north = (
                artifact.spatial.bbox.west, artifact.spatial.bbox.south,
                artifact.spatial.bbox.east, artifact.spatial.bbox.north)
            if (artifact.spatial.shape != [nir.height, nir.width, 1]
                    or artifact.spatial.gsd_meters != nir.transform[0]
                    or artifact.temporal.start != nir.acquired
                    or artifact.temporal.end != nir.acquired
                    or not -180 <= west < east <= 180
                    or not -90 <= south < north <= 90):
                raise ValueError("aligned SWIR metadata differs from B08 grid")
            swir = AlignedSWIRInput(
                artifact_id=artifact.artifact_id,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                source_asset_id=source.asset_id,
                source_sha256=source.sha256,
                reference_asset_id=nir.asset_id,
                reference_sha256=nir.sha256,
                acquired=nir.acquired,
                crs=nir.crs,
                transform=nir.transform,
                width=nir.width,
                height=nir.height,
                dtype="float32",
                nodata=-9999.,
                lineage_parameters_hash=expected_grid_lineage,
            )
            checked_ndmi_inputs(args, nir, swir)
        except V2DomainError:
            raise
        except (ValueError, KeyError, TypeError, StopIteration):
            raise V2DomainError(
                "invalid_tool_arguments",
                "reviewed B08 and current-episode aligned B11 artifact required",
                phase="request",
            ) from None
        size = nir_asset.size_bytes + artifact.size_bytes
        return PreparedTool(
            NDMI_VERSION, size, MAX_OUTPUT,
            lambda: self.invoke_ndmi(args, nir, swir, artifact, size))

    def invoke_ndmi(self, args, nir, swir, artifact, size):
        try:
            try:
                swir_content = self.artifacts.read_content(artifact).content
            except ArtifactStoreError as error:
                raise V2DomainError(
                    error.code, "aligned SWIR artifact unavailable", 409,
                    phase="artifact") from None
            validate_aligned_swir(swir_content, swir)
            with httpx.Client(
                    base_url=self.base_url, timeout=30, transport=self.transport,
                    trust_env=False, follow_redirects=False) as client:
                with client.stream(
                        "POST", "/ndmi-band-math", content=swir_content,
                        headers={
                            "Content-Type": MEDIA_TYPE,
                            "X-Raster-Arguments": args.model_dump_json(exclude_none=True),
                            "X-Raster-NDMI-Input": swir.model_dump_json(),
                        }) as response:
                    response.raise_for_status()
                    if (response.status_code != 200
                            or response.headers.get("X-Raster-Version") != NDMI_VERSION):
                        raise ValueError("unsupported NDMI provider response")
                    raw = response.headers.get("X-Raster-Metadata", "")
                    if len(raw) > 16384:
                        raise ValueError("oversized NDMI metadata")
                    result = NDMIResult.model_validate_json(raw)
                    content = bytearray()
                    for block in response.iter_bytes():
                        if len(content) + len(block) > MAX_OUTPUT:
                            raise ValueError("oversized NDMI output")
                        content.extend(block)
                    content = bytes(content)
                    if hashlib.sha256(content).hexdigest() != response.headers.get(
                            "X-Content-SHA256"):
                        raise ValueError("NDMI content checksum mismatch")
            if (result.input_asset_ids != [nir.asset_id, swir.source_asset_id]
                    or result.input_sha256 != [nir.sha256, swir.source_sha256]
                    or result.swir_artifact_id != artifact.artifact_id
                    or result.swir_artifact_sha256 != artifact.sha256
                    or result.acquired != nir.acquired or result.crs != nir.crs
                    or result.transform != nir.transform
                    or result.width != nir.width or result.height != nir.height):
                raise ValueError("NDMI source, artifact or grid mismatch")
            validate_ndmi(content, result)
        except httpx.TimeoutException:
            raise V2DomainError("tool_timeout", "raster provider timed out", 504,
                                True, "tool") from None
        except httpx.HTTPError:
            raise V2DomainError("tool_unavailable", "raster provider failed", 502,
                                True, "tool") from None
        except (ValueError, KeyError, TypeError):
            raise V2DomainError("invalid_tool_output", "NDMI output validation failed",
                                502, phase="tool") from None
        west, south, east, north = result.bbox_wgs84
        spatial = SpatialExtent(
            crs="EPSG:4326",
            bbox=SpatialBoundingBox(west=west, south=south, east=east, north=north),
            gsd_meters=nir.transform[0], shape=[nir.height, nir.width, 1])
        try:
            derived = self.artifacts.put_bytes(
                content, kind="raster", media_type=MEDIA_TYPE,
                spatial=spatial,
                temporal=TemporalExtent(start=nir.acquired, end=nir.acquired),
                lineage=ArtifactLineage(
                    tool_id=TOOL_ID, tool_version=NDMI_VERSION,
                    input_refs=[nir.asset_id, artifact.artifact_id],
                    parameters_hash=sha256_json(
                        ndmi_lineage_payload(args, nir, swir)),
                ))
        except ArtifactStoreError as error:
            raise V2DomainError(
                error.code, "NDMI artifact storage failed", 503, True,
                "artifact") from None
        return ToolOutput(derived, result.model_dump(mode="json"), size)

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
