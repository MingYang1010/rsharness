import copy
import gc
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from rasterio.io import MemoryFile

from v2.test_tool_execution import make_tool_tasks
from app.main import create_app
from app.agent_gateway import build_binding, create_app as gateway, public_observation
from app.raster_bridge import RasterBridge, create_app as provider_app
from app.core.raster_math import (NativeBand, BandMathArguments, NDVIResult, compute_ndvi,
    checked_pair, validate_ndvi, VERSION, TOOL_ID, NODATA, MAX_OUTPUT)
from app.core.tools.raster import RasterExecutor
from app.core.tools.runtime import ToolRouter
from app.core.artifacts import ArtifactStore
from app.core.capabilities import TaskRegistry
from app.core.domain import V2DomainError
from app.core.schemas import ToolInvokeAction, V2EpisodeState
from app.core.execution_replay import read_snapshot, replay_episode

ROOT=Path(__file__).resolve().parents[2]


def fixture(path,band,values,mask=None,transform=None,scale=.0001,offset=-.1):
    transform=transform or from_origin(660000,3550000,10,10)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path,"w",driver="GTiff",width=values.shape[1],height=values.shape[0],count=1,dtype="uint16",
                           crs="EPSG:32650",transform=transform,nodata=0) as image:
            image.write(values.astype("uint16"),1)
            image.scales=(scale,);image.offsets=(offset,)
            if mask is not None:image.write_mask(mask)
    return NativeBand(asset_id="asset-"+band,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),item_id="scene-one",
        band=band,acquired="2024-04-05T00:00:00Z",crs="EPSG:32650",transform=list(transform)[:6],width=values.shape[1],
        height=values.shape[0],dtype="uint16",scale=scale,offset=offset,nodata=0.)


class RasterMathTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.rp=self.root/"red.tif";self.np=self.root/"nir.tif"
        self.red=fixture(self.rp,"red",np.array([[2000,1000,500],[0,2000,3000]]))
        self.nir=fixture(self.np,"nir",np.array([[4000,1000,4000],[4000,2000,2000]]))

    def compute(self):return compute_ndvi(self.rp,self.np,self.red,self.nir)

    def test_scaled_ndvi_known_values_invalid_reflectance_and_zero_denominator(self):
        content,result=self.compute()
        with MemoryFile(content) as mem,mem.open() as image:
            values=image.read(1)
            self.assertTrue(np.allclose(values,[[.5,NODATA,NODATA],[NODATA,0.,-1/3]],atol=1e-7))
            self.assertEqual(image.crs.to_epsg(),32650)
        self.assertEqual(result.valid_pixels,3)
        self.assertFalse(result.cloud_mask_applied)
        self.assertNotAlmostEqual(float(values[0,0]),(4000-2000)/(4000+2000))

    def test_explicit_source_masks_are_preserved(self):
        self.red=fixture(self.rp,"red",np.full((2,3),2000),np.array([[0,255,255],[255,255,255]],dtype="uint8"))
        content,result=self.compute()
        with MemoryFile(content) as mem,mem.open() as image:self.assertEqual(image.read(1)[0,0],NODATA)

    def test_all_invalid_has_null_statistics_not_fake_zero(self):
        self.red=fixture(self.rp,"red",np.zeros((2,3),dtype="uint16"))
        content,result=self.compute()
        self.assertEqual(result.valid_pixels,0);self.assertIsNone(result.mean)
        validate_ndvi(content,result)

    def test_pair_rejects_dates_grids_bands_and_alias(self):
        for changes in ({"acquired":"2024-04-15T00:00:00Z"},{"item_id":"other"},{"width":4},
                        {"crs":"EPSG:32651"},{"transform":[20.,0.,660000.,0.,-10.,3550000.]},
                        {"band":"red"},{"asset_id":self.red.asset_id}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                checked_pair(self.red,self.nir.model_copy(update=changes))

    def test_actual_input_checksum_and_scaling_are_not_trusted(self):
        with self.assertRaises(ValueError):compute_ndvi(self.rp,self.np,self.red.model_copy(update={"sha256":"a"*64}),self.nir)
        with self.assertRaises(ValueError):compute_ndvi(self.rp,self.np,self.red.model_copy(update={"scale":.001}),self.nir)

    def test_symlink_and_non_tiff_input_rejected(self):
        link=self.root/"alias.tif";link.symlink_to(self.rp)
        with self.assertRaises(ValueError):compute_ndvi(link,self.np,self.red,self.nir)
        self.rp.write_bytes(b'<VRTDataset><SourceFilename>http://private</SourceFilename></VRTDataset>')
        with self.assertRaises(ValueError):self.compute()

    def test_output_shape_grid_stats_and_payload_tamper_rejected(self):
        content,result=self.compute()
        for changes in ({"width":4},{"crs":"EPSG:32651"},{"mean":.9},{"nodata":0.}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_ndvi(content,result.model_copy(update=changes))
        with self.assertRaises(ValueError):validate_ndvi(b"<VRTDataset/>",result)

    def test_arguments_reject_expression_url_extra_and_duplicate(self):
        good={"operation":"ndvi","red_asset_id":"asset-red","nir_asset_id":"asset-nir"}
        for change in ({"operation":"eval"},{"expression":"__import__('os')"},{"red_asset_id":"https://private"},
                       {"nir_asset_id":"asset-red"}):
            with self.assertRaises(ValueError):BandMathArguments.model_validate({**good,**change})

    def test_real_subprocess_provider_and_invalid_path_busy_timeout(self):
        manifest={b.asset_id:{"filename":p.name,"native":b.model_dump(mode="json")} for p,b in ((self.rp,self.red),(self.np,self.nir))}
        bridge=RasterBridge(self.root,manifest,ROOT/"scripts/raster_worker.py")
        args=BandMathArguments(operation="ndvi",red_asset_id="asset-red",nir_asset_id="asset-nir")
        content,result=bridge.execute(args);validate_ndvi(content,result)
        with TestClient(provider_app(bridge)) as client:
            bridge.slot.acquire()
            try:self.assertEqual(client.post("/band-math",json=args.model_dump()).status_code,429)
            finally:bridge.slot.release()
            with patch("app.raster_bridge.subprocess.run",side_effect=subprocess.TimeoutExpired("worker",20)):
                self.assertEqual(client.post("/band-math",json=args.model_dump()).status_code,504)
            manifest["asset-red"]["filename"]="../red.tif"
            self.assertEqual(client.post("/band-math",json=args.model_dump()).status_code,422)


class RasterHarnessTests(unittest.TestCase):
    compute = RasterMathTests.compute

    def setUp(self):
        RasterMathTests.setUp(self)
        self.content,self.result=self.compute()
        self.tasks=make_tool_tasks(self.root);directory=self.tasks/"crop-smoke"
        task=json.loads((directory/"task.json").read_text());task.update(inputs=["asset-red","asset-nir"])
        task["metadata"].update(artifact_identity="derivation-sha256-v1",raster_inputs={b.asset_id:b.model_dump(mode="json") for b in (self.red,self.nir)})
        (directory/"task.json").write_text(json.dumps(task))
        scenario=json.loads((directory/"scenario.json").read_text());scenario["allowed_tools"]=[TOOL_ID,"catalog.search","catalog.inspect_asset"]
        (directory/"scenario.json").write_text(json.dumps(scenario))
        template=json.loads((directory/"assets.json").read_text())[0];assets=[]
        west,south,east,north=self.result.bbox_wgs84
        for band,path in ((self.red,self.rp),(self.nir,self.np)):
            asset=copy.deepcopy(template);asset.update(asset_id=band.asset_id,sha256=band.sha256,size_bytes=path.stat().st_size,
                roles=["input_image","reflectance"],bands=[band.band],temporal={"start":band.acquired,"end":band.acquired},
                spatial={"crs":"EPSG:4326","bbox":dict(west=west,south=south,east=east,north=north),"shape":[2,3,1]})
            assets.append(asset)
        (directory/"assets.json").write_text(json.dumps(assets))
        self.registry=TaskRegistry(self.tasks);self.manifest=self.registry.get("crop-smoke","1.0.0")
        self.calls=0;self.bad=None
        def handle(request):
            self.calls+=1
            if self.bad=="timeout":raise httpx.ReadTimeout("fixture")
            data=self.content if self.bad!="bytes" else b"wrong"
            metadata=self.result.model_dump(mode="json")
            if self.bad=="source":metadata["input_sha256"]=["f"*64]*2
            return httpx.Response(200,content=data,headers={"X-Raster-Version":VERSION,"X-Raster-Metadata":json.dumps(metadata),
                "X-Content-SHA256":hashlib.sha256(self.content).hexdigest()})
        self.transport=httpx.MockTransport(handle)
        self.artifacts=ArtifactStore(str(self.root/"artifacts"));self.executor=RasterExecutor("http://raster",self.artifacts,self.transport)
        self.action=ToolInvokeAction(type="tool.invoke",tool_id=TOOL_ID,arguments={"operation":"ndvi","red_asset_id":"asset-red","nir_asset_id":"asset-nir"})

    def backend(self):
        return create_app(database_path=str(self.root/"state.db"),v2_tasks_path=str(self.tasks),
            v2_artifacts_path=str(self.artifacts.root),v2_tool_executor=ToolRouter(raster=self.executor))

    def test_scope_role_and_budget_preflight(self):
        with self.assertRaises(V2DomainError):self.executor.plan(self.action,self.manifest,["asset-red"])
        broken=self.manifest.model_copy(deep=True);broken.assets[0].roles=["label"]
        with self.assertRaises(V2DomainError):self.executor.plan(self.action,broken,broken.task.inputs)
        prepared=self.executor.plan(self.action,self.manifest,self.manifest.task.inputs)
        self.assertEqual(prepared.input_bytes,sum(a.size_bytes for a in self.manifest.assets))
        self.assertEqual(prepared.max_output_bytes,MAX_OUTPUT);self.assertEqual(self.calls,0)

    def test_provider_tamper_and_timeout_fail_without_artifact(self):
        for bad in ("bytes","source","timeout"):
            self.bad=bad
            with self.assertRaises(V2DomainError):self.executor.plan(self.action,self.manifest,self.manifest.task.inputs).invoke()
        self.assertFalse(self.artifacts.root.exists())

    def test_output_budget_refusal_precedes_provider_execution(self):
        path=self.tasks/"crop-smoke/task.json"
        task=json.loads(path.read_text());task["budget"]["max_artifact_bytes"]=MAX_OUTPUT-1;path.write_text(json.dumps(task))
        with TestClient(self.backend()) as operator:
            state=operator.post("/v2/reset",json={"task_ref":{"task_id":"crop-smoke","task_version":"1.0.0"}}).json()["data"]
            response=operator.post(f"/v2/episodes/{state['episode_id']}/step",json={"client_action_id":"over-budget",
                "expected_state_version":0,"action":self.action.model_dump()})
            self.assertEqual(response.status_code,422,response.text)
            self.assertEqual(response.json()["error"]["code"],"tool_budget_exceeded");self.assertEqual(self.calls,0)

    def test_gateway_evidence_restart_and_execution_replay(self):
        backend=self.backend()
        with TestClient(backend) as operator:
            initial=operator.post("/v2/reset",json={"task_ref":{"task_id":"crop-smoke","task_version":"1.0.0"}}).json()["data"]
            episode=initial["episode_id"]
            binding=build_binding(self.manifest,V2EpisodeState.model_validate(initial["state"]),
                operator.get("/v2/capabilities").json()["data"],hashlib.sha256(b"1"*64).hexdigest())
            with TestClient(gateway(binding,"http://operator",httpx.ASGITransport(app=backend)),headers={"Authorization":"Bearer "+"1"*64}) as client:
                request={"client_action_id":"ndvi-1","expected_state_version":0,"action":self.action.model_dump()}
                wrong=copy.deepcopy(request);wrong["action"]["arguments"]["nir_asset_id"]="asset-other"
                self.assertEqual(client.post("/agent/step",json=wrong).status_code,403)
                response=client.post("/agent/step",json=request)
                self.assertEqual(response.status_code,200,response.text);data=response.json()
                self.assertEqual(client.post("/agent/step",json=request).json(),data);self.assertEqual(self.calls,1)
                ref=data["observation"]["items"][1]["artifact_ref"]
                art=client.get("/agent/artifacts/"+ref).json()["artifact"]
                foreign=operator.post("/v2/reset",json={"task_ref":{"task_id":"crop-smoke","task_version":"1.0.0"}}).json()["data"]["episode_id"]
                denied=operator.get("/v2/artifacts/"+ref,params={"episode_id":foreign})
                self.assertIn(denied.status_code,(403,404))
                content=client.get("/agent/artifacts/"+ref+"/content")
                self.assertEqual(content.status_code,200,content.text[:100]);self.assertEqual(content.content,self.content)
                self.assertEqual(art["spatial"]["shape"],[2,3,1]);self.assertNotIn("uri",art)
                evidence={"type":"memory.save_evidence","evidence":{"evidence_id":"ev-ndvi","claim_id":"ndvi",
                    "source_ref":ref,"selector":{"pixel_window":[0,0,3,2]},"description":"NDVI numeric raster","frozen_sha256":art["sha256"]}}
                response=client.post("/agent/step",json={"client_action_id":"evidence","expected_state_version":1,"action":evidence})
                self.assertEqual(response.status_code,200,response.text)
                response=client.post("/agent/step",json={"client_action_id":"submit","expected_state_version":2,
                    "action":{"type":"answer.submit","answer":{"label":"ndvi","confidence":1.,"claims":[]},"confidence":1.,"evidence_ids":["ev-ndvi"]}})
                self.assertEqual(response.status_code,200,response.text)
            with TestClient(self.backend()) as restarted:
                self.assertEqual(restarted.post(f"/v2/episodes/{episode}/step",json=request).json()["data"]["state"]["state_version"],1)
            self.assertEqual(self.calls,1)
        gc.collect()
        before=hashlib.sha256((self.root/"state.db").read_bytes()).hexdigest()
        report=replay_episode(read_snapshot(self.root/"state.db",episode),self.registry,self.root/"replay",
            lambda artifacts:ToolRouter(raster=RasterExecutor("http://raster",artifacts,self.transport)))
        self.assertEqual(report["status"],"passed",report);self.assertEqual(self.calls,2)
        self.assertEqual(hashlib.sha256((self.root/"state.db").read_bytes()).hexdigest(),before)
