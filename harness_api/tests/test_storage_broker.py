import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.storage_broker import create_app, MAX_OBJECT_BYTES
from app.core.storage.quota import CONTROL_ALLOWANCE, StorageQuota
from app.core.storage.client import BrokerClient
from app.core.artifacts import ArtifactStore, ArtifactStoreError
from app.core.schemas import ArtifactLineage
from app.eo_gym_bridge import CropBridge

MIB = 1024 * 1024


class StorageBrokerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.quota = StorageQuota(self.root / "runtime", CONTROL_ALLOWANCE + 8 * MIB)
        self.token = "a" * 64  # Deterministic test-only credential, not deployed.
        self.client = TestClient(create_app(self.quota, self.token))
        self.auth = {"Authorization": "Bearer " + self.token}
        self.content = b"verified artifact content"
        self.digest = hashlib.sha256(self.content).hexdigest()
        self.url = "/objects/" + self.digest

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def target(self):
        return self.quota.root / "managed-artifacts" / self.digest[:2] / self.digest / "content"

    def put(self, content=None, headers=None):
        return self.client.put(self.url, content=self.content if content is None else content,
                               headers=self.auth if headers is None else headers)

    def test_authorization_does_not_reveal_or_write_objects(self):
        for method in ("GET", "PUT"):
            response = self.client.request(method, self.url, content=self.content)
            self.assertEqual(response.status_code, 401)
            self.assertNotIn(self.token, response.text)
        self.assertFalse(self.target().exists())
        self.assertEqual(self.client.get("/objects", headers=self.auth).status_code, 404)

    def test_put_repeat_restart_and_verified_read(self):
        self.assertTrue(self.put().json()["created"])
        before = self.quota.status()["data_bytes"]
        self.assertFalse(self.put().json()["created"])
        self.assertEqual(self.quota.status()["data_bytes"], before)
        with TestClient(create_app(StorageQuota(self.quota.root, self.quota.limit), self.token)) as restarted:
            result = restarted.get(self.url, headers=self.auth)
            self.assertEqual(result.content, self.content)
        self.assertEqual(self.quota.status()["reserved_headroom_bytes"], 0)
        self.assertEqual(len(list(self.quota.root.rglob("content"))), 1)

    def test_invalid_checksum_never_publishes_partial(self):
        response = self.put(b"wrong")
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.target().exists())
        self.assertEqual(list(self.quota.root.rglob("upload-*")), [])
        self.assertGreater(self.quota.status()["reserved_headroom_bytes"], 0)
        self.assertEqual(self.put().status_code, 200)

    def test_size_and_stream_length_guard(self):
        self.assertEqual(self.put(headers={**self.auth, "Content-Length": str(MAX_OBJECT_BYTES + 1)}).status_code, 413)
        self.assertEqual(self.put(headers={**self.auth, "Content-Length": "2"}).status_code, 413)
        self.assertEqual(self.put(headers={**self.auth, "Content-Length": "bad"}).status_code, 411)
        self.assertFalse(self.target().exists())

    def test_overquota_denied_before_object_write(self):
        with self.quota.hold(self.quota.root / "other", 8 * MIB, "test-pressure"):
            response = self.put()
        self.assertEqual(response.status_code, 507)
        self.assertFalse(self.target().exists())

    def test_active_upload_scope_refuses_second_writer(self):
        with self.quota.hold(self.target().parent, MIB, "test-live-upload"):
            self.assertEqual(self.put().status_code, 503)
        self.assertFalse(self.target().exists())

    def test_corrupt_object_is_not_served_or_overwritten(self):
        self.assertEqual(self.put().status_code, 200)
        self.target().write_bytes(b"corrupt")
        self.assertEqual(self.client.get(self.url, headers=self.auth).status_code, 409)
        self.assertEqual(self.put().status_code, 409)
        self.assertEqual(self.target().read_bytes(), b"corrupt")

    def test_symlink_object_cannot_escape_managed_root(self):
        self.target().parent.mkdir(parents=True)
        outside = self.root / "private"
        outside.write_bytes(self.content)
        self.target().symlink_to(outside)
        self.assertEqual(self.client.get(self.url, headers=self.auth).status_code, 409)
        self.assertEqual(self.put().status_code, 507)
        self.assertEqual(outside.read_bytes(), self.content)

    def test_artifact_store_broker_range_and_no_local_fallback(self):
        def transport(request):
            response = self.client.request(request.method, request.url.path, content=request.content, headers=dict(request.headers))
            return httpx.Response(response.status_code, content=response.content, headers=response.headers)
        broker = BrokerClient("http://storage", self.token, httpx.MockTransport(transport))
        store = ArtifactStore(str(self.root / "must-remain-absent"), broker)
        lineage = ArtifactLineage(tool_id="test.storage", tool_version="1.0.0", input_refs=[], parameters_hash="a" * 64)
        artifact = store.put_bytes(self.content, "text", "text/plain", lineage)
        self.assertTrue(store.audit_exists(artifact))
        result = store.read_content(artifact, "bytes=2-7")
        self.assertTrue(result.partial)
        self.assertEqual(result.content, self.content[2:8])
        self.assertFalse(store.root.exists())
        def unavailable(request):
            return httpx.Response(503)
        store.broker.transport = httpx.MockTransport(unavailable)
        with self.assertRaises(ArtifactStoreError):
            store.put_bytes(b"other", "text", "text/plain", lineage)
        self.assertFalse(store.root.exists())

    def test_client_rejects_corrupt_download_and_credential_redirect(self):
        for response in (httpx.Response(200, content=b"wrong"), httpx.Response(302, headers={"Location": "http://outside/"})):
            from app.core.storage.client import BrokerError
            broker = BrokerClient("http://storage", self.token, httpx.MockTransport(lambda request: response))
            with self.assertRaises(BrokerError):
                broker.get(self.digest)

    def test_client_byte_bounds_apply_to_download_and_upload(self):
        from app.core.storage.client import BrokerError
        data = b"abcdef"
        digest = hashlib.sha256(data).hexdigest()
        broker = BrokerClient("http://storage", self.token,
                              httpx.MockTransport(lambda request: httpx.Response(200, content=data)))
        with patch("app.core.storage.client.MAX_OBJECT_BYTES", 4):
            with self.assertRaises(BrokerError) as read_error:
                broker.get(digest)
            self.assertEqual(read_error.exception.code, "artifact_content_corrupt")
            with self.assertRaises(BrokerError) as write_error:
                broker.put(digest, data)
            self.assertEqual(write_error.exception.code, "artifact_storage_capacity")


class ProviderCacheTests(unittest.TestCase):
    def test_provider_capacity_is_not_reported_as_transient_network_failure(self):
        from app.core.tools.eo_gym import EOGymExecutor
        from app.core.domain import V2DomainError
        with httpx.Client(base_url="http://provider", transport=httpx.MockTransport(lambda request: httpx.Response(507))) as client:
            with self.assertRaises(V2DomainError) as raised:
                EOGymExecutor._bounded_response(client, "POST", "/execute", 100)
        self.assertEqual(raised.exception.code, "provider_storage_capacity")
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(raised.exception.status_code, 507)

    def test_cache_capacity_and_same_content_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("app.eo_gym_bridge.verify_upstream"):
                bridge = CropBridge(root, root, root / "outputs", {}, root / "worker")
            source = root / "result.png"
            source.write_bytes(b"x" * 32768)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch("app.eo_gym_bridge.MAX_OUTPUT_STORE_BYTES", 49152):
                bridge.publish(source, digest)
                bridge.publish(source, digest)
                source.write_bytes(b"y" * 32768)
                with self.assertRaises(HTTPException) as raised:
                    bridge.publish(source, hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertEqual(raised.exception.status_code, 507)
            self.assertEqual(len(list(bridge.outputs.glob("*.png"))), 1)
            self.assertEqual(list(bridge.outputs.glob("publish-*")), [])


if __name__ == "__main__":
    unittest.main()
