import hashlib
import tempfile
import unittest
from pathlib import Path

from app.v2.artifacts import ArtifactStore
from app.v2.schemas import ArtifactLineage


class V2ArtifactTests(unittest.TestCase):
    def test_content_addressed_atomic_write_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            content = b"deterministic EO artifact\n"
            lineage = ArtifactLineage(
                tool_id="test.identity",
                tool_version="1.0.0",
                input_refs=["asset-worldcover-n30e120"],
                parameters_hash=hashlib.sha256(b"{}").hexdigest(),
            )
            first = store.put_bytes(
                content,
                kind="text",
                media_type="text/plain",
                lineage=lineage,
            )
            second = store.put_bytes(
                content,
                kind="text",
                media_type="text/plain",
                lineage=lineage,
            )
            self.assertEqual(first.artifact_id, second.artifact_id)
            self.assertEqual(first.sha256, hashlib.sha256(content).hexdigest())
            self.assertTrue(store.audit_exists(first))
            self.assertEqual(
                store.content_path(first.sha256).read_bytes(),
                content,
            )
            self.assertEqual(len(list(Path(directory).rglob(first.sha256))), 1)

    def test_invalid_content_hash_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaises(ValueError):
                store.content_path("../escape")


if __name__ == "__main__":
    unittest.main()
