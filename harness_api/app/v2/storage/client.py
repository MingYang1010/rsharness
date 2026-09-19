"""Bounded internal broker client; no filesystem path or label index access."""
import hashlib
import re

import httpx

MAX_OBJECT_BYTES = 64 * 1024 * 1024


class BrokerError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class BrokerClient:
    def __init__(self, url: str, token: str, transport=None):
        parsed = httpx.URL(url)
        if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("invalid internal storage URL")
        if not re.fullmatch(r"[0-9a-f]{64}", token):
            raise ValueError("invalid storage credential")
        self.url = url.rstrip("/")
        self.token = token
        self.transport = transport

    @staticmethod
    def check_status(response: httpx.Response) -> None:
        if response.status_code in {413, 507}:
            raise BrokerError("artifact_storage_capacity")
        if response.status_code == 404:
            raise BrokerError("artifact_content_missing")
        if response.status_code in {409, 422}:
            raise BrokerError("artifact_content_corrupt")
        if response.status_code >= 300:
            raise BrokerError("artifact_store_unavailable")

    def request(self, method: str, digest: str, content: bytes | None = None) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BrokerError("artifact_content_corrupt")
        if content is not None and len(content) > MAX_OBJECT_BYTES:
            raise BrokerError("artifact_storage_capacity")
        try:
            with httpx.Client(base_url=self.url, timeout=35, trust_env=False, follow_redirects=False,
                              transport=self.transport, headers={"Authorization": "Bearer " + self.token}) as client:
                with client.stream(method, "/objects/" + digest, content=content) as response:
                    self.check_status(response)
                    limit = MAX_OBJECT_BYTES if method == "GET" else 4096
                    result = bytearray()
                    for block in response.iter_bytes(chunk_size=64 * 1024):
                        if len(result) + len(block) > limit:
                            raise BrokerError("artifact_content_corrupt")
                        result.extend(block)
            return bytes(result)
        except httpx.HTTPError:
            raise BrokerError("artifact_store_unavailable") from None

    def put(self, digest: str, content: bytes) -> None:
        import json
        if hashlib.sha256(content).hexdigest() != digest:
            raise BrokerError("artifact_content_corrupt")
        try:
            receipt = json.loads(self.request("PUT", digest, content))
            if receipt["sha256"] != digest or receipt["size_bytes"] != len(content):
                raise ValueError("invalid receipt")
        except (ValueError, KeyError, TypeError):
            raise BrokerError("artifact_content_corrupt") from None

    def get(self, digest: str) -> bytes:
        content = self.request("GET", digest)
        if hashlib.sha256(content).hexdigest() != digest:
            raise BrokerError("artifact_content_corrupt")
        return content
