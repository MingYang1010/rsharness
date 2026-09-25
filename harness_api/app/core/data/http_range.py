"""Operator-only allowlisted HTTPS access with payload/request/deadline bounds.

COG access goes through Python file handles, never GDAL's independent network IO.
"""
from __future__ import annotations

import hashlib
import io
import os
import time
from collections import OrderedDict
from urllib.parse import unquote, urlsplit

import httpx

HOST_PATHS = {
    "earth-search.aws.element84.com": "/v1/",
    "sentinel-cogs.s3.us-west-2.amazonaws.com": "/sentinel-s2-l2a-cogs/",
    "dataspace.copernicus.eu": "/terms-and-conditions",
    "sentinels.copernicus.eu": "/documents/247904/690755/Sentinel_Data_Legal_Notice",
}
MAX_RANGE = 8 * 1024 * 1024
BLOCK_SIZE = 256 * 1024


def checked_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in HOST_PATHS or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.fragment or "\\" in url
            or any(ord(c) < 33 for c in url)):
        raise ValueError("URL is outside approved public endpoints")
    path = unquote(parsed.path)
    if ".." in path.split("/") or not path.startswith(HOST_PATHS[parsed.hostname]):
        raise ValueError("URL path is outside approved source")
    if parsed.hostname in {"dataspace.copernicus.eu", "sentinels.copernicus.eu"} and path != HOST_PATHS[parsed.hostname]:
        raise ValueError("only the reviewed legal document is allowed")
    if parsed.hostname != "earth-search.aws.element84.com" and parsed.query:
        raise ValueError("signed or arbitrary asset query URLs are not admitted")
    return url


class BoundedHTTP:
    def __init__(self, max_bytes: int = 128 * 1024 * 1024, max_requests: int = 512,
                 seconds: int = 600, client: httpx.Client | None = None):
        if not (0 < max_bytes <= 256 * 1024 * 1024 and 0 < max_requests <= 1024 and 0 < seconds <= 1800):
            raise ValueError("invalid acquisition bounds")
        self.max_bytes, self.max_requests = max_bytes, max_requests
        self.bytes = self.requests = 0
        self.deadline = time.monotonic() + seconds
        self.receipts: list[dict] = []
        self.owns_client = client is None
        if client is None:
            proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
            if not proxy:
                raise ValueError("A800 system HTTPS proxy is required")
            client = httpx.Client(proxy=proxy, trust_env=False, follow_redirects=False, timeout=20)
        self.client = client

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def _check(self) -> None:
        if time.monotonic() > self.deadline:
            raise ValueError("acquisition deadline exceeded")
        if self.requests >= self.max_requests:
            raise ValueError("acquisition request limit exceeded")

    def fetch(self, url: str, limit: int, *, method: str = "GET", headers: dict | None = None,
              expected_status: int = 200, expected_range: str | None = None,
              expected_etag: str | None = None) -> tuple[bytes, dict]:
        checked_url(url)
        self._check()
        if method not in {"GET", "HEAD"} or not 0 <= limit <= MAX_RANGE:
            raise ValueError("request size or method not permitted")
        if limit > self.max_bytes - self.bytes:
            raise ValueError("acquisition byte limit would be exceeded")
        self.requests += 1
        with self.client.stream(method, url, headers={"Accept-Encoding": "identity", **(headers or {})},
                                follow_redirects=False, timeout=min(20, max(.1, self.deadline-time.monotonic()))) as response:
            if response.status_code != expected_status or response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("unexpected HTTP status/encoding; redirects not followed")
            if expected_range is not None and response.headers.get("content-range") != expected_range:
                raise ValueError("server did not honor exact byte range")
            if expected_etag is not None and response.headers.get("etag") != expected_etag:
                raise ValueError("source object changed during range acquisition")
            result = bytearray()
            if method != "HEAD":
                if "content-length" in response.headers and int(response.headers["content-length"]) > limit:
                    raise ValueError("response exceeds byte bound")
                for block in response.iter_raw(chunk_size=65536):
                    self.bytes += len(block)
                    if len(result) + len(block) > limit or self.bytes > self.max_bytes or time.monotonic() > self.deadline:
                        raise ValueError("response exceeds payload/deadline bound")
                    result.extend(block)
            record = {"url": url, "method": method, "status": response.status_code,
                      "bytes": len(result), "sha256": hashlib.sha256(result).hexdigest(),
                      "etag": response.headers.get("etag"), "content_range": response.headers.get("content-range")}
            self.receipts.append(record)
            return bytes(result), dict(response.headers)


class RangeSource:
    """Pinned HTTP object with a 2 MiB shared LRU cache and seekable handles."""
    def __init__(self, http: BoundedHTTP, url: str):
        self.http, self.url = http, checked_url(url)
        _, headers = http.fetch(url, 0, method="HEAD")
        self.size = int(headers.get("content-length", "-1"))
        self.etag = headers.get("etag", "")
        if not 0 < self.size <= 2 * 1024**3 or not self.etag or self.etag.startswith("W/"):
            raise ValueError("bounded object length and strong ETag are required")
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * BLOCK_SIZE
        end = min(self.size - 1, start + BLOCK_SIZE - 1)
        if start > end:
            return b""
        data, _ = self.http.fetch(self.url, end-start+1, headers={"Range": f"bytes={start}-{end}", "If-Match": self.etag},
            expected_status=206, expected_range=f"bytes {start}-{end}/{self.size}", expected_etag=self.etag)
        if len(data) != end-start+1:
            raise ValueError("truncated byte range")
        self.cache[index] = data
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return data

    def open(self, path: str, mode: str = "rb") -> RangeFile:
        if str(path) != "asset.tif" or mode not in {"r", "rb"}:
            raise FileNotFoundError("only the pinned COG is available")
        return RangeFile(self)


class RangeFile(io.RawIOBase):
    def __init__(self, source: RangeSource):
        self.source, self.position = source, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self._checkClosed()
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek mode")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.source.size)
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        length = max(0, self.source.size-self.position) if size < 0 else min(size, max(0, self.source.size-self.position))
        if length > MAX_RANGE:
            raise ValueError("unbounded/full-scene read refused")
        result = bytearray()
        while len(result) < length:
            block = self.source.block(self.position // BLOCK_SIZE)
            offset = self.position % BLOCK_SIZE
            take = min(length-len(result), len(block)-offset)
            if take <= 0:
                raise ValueError("incomplete pinned source")
            result.extend(block[offset:offset+take])
            self.position += take
        return bytes(result)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
