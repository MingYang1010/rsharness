import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.v2.storage.physical_usage import (
    PhysicalUsageError,
    audit_physical_usage,
)


ROOT = Path(__file__).resolve().parents[2]


class PhysicalUsageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "runtime"
        self.root.mkdir()

    def write(self, relative, content=b"x"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_counts_allocated_blocks_not_sparse_logical_size(self):
        sparse = self.write("large.sparse", b"")
        with sparse.open("wb") as stream:
            stream.truncate(2 * 1024 * 1024)
        report = audit_physical_usage(self.root)
        self.assertTrue(report.complete)
        self.assertEqual(report.physical_bytes, sparse.stat().st_blocks * 512)
        self.assertLess(report.physical_bytes, 2 * 1024 * 1024)

    def test_hardlinks_are_deduplicated_by_inode(self):
        content = bytearray(16 * 1024)
        first = self.write("first.bin", content)
        second = self.root / "second.bin"
        os.link(first, second)
        report = audit_physical_usage(self.root)
        first_entry = next(item for item in report.files if item.path == "first.bin")
        second_entry = next(item for item in report.files if item.path == "second.bin")
        self.assertEqual(first_entry.object_key, second_entry.object_key)
        self.assertTrue(first_entry.counted != second_entry.counted)
        self.assertEqual(report.hardlink_aliases_omitted, 1)
        self.assertEqual(report.file_count, 2)

    def test_symlink_target_bytes_are_not_followed_or_counted(self):
        outside = Path(self.temp.name) / "outside.sparse"
        with outside.open("wb") as stream:
            stream.truncate(2 * 1024 * 1024)
        (self.root / "inside.link").symlink_to(outside)
        report = audit_physical_usage(self.root)
        link = next(item for item in report.files if item.path == "inside.link")
        self.assertEqual(report.physical_bytes, link.physical_bytes)
        self.assertEqual(link.physical_bytes, outside.lstat().st_blocks * 512)
        self.assertEqual(report.symlink_entries_counted, 1)

    def test_category_breakdown_and_report_serialization(self):
        files = [
            "state/episodes.sqlite3",
            "state/episodes.sqlite3-wal",
            "state/episodes.sqlite3-shm",
            "reports/replay/execution.json",
            "logs/api.log",
            "logs/docker-json.log",
            "checkpoints/model.safetensors",
            "artifacts/render.png",
        ]
        paths = [self.write(item) for item in files]
        report = audit_physical_usage(self.root).as_dict()
        self.assertEqual(
            set(report["category_bytes"]),
            {
                "sqlite_main", "sqlite_wal", "sqlite_shm", "reports", "logs",
                "docker_logs", "checkpoints", "runtime_artifacts", "uncategorized",
            },
        )
        for category, path in zip(
            ("sqlite_main", "sqlite_wal", "sqlite_shm", "reports", "logs",
             "docker_logs", "checkpoints", "runtime_artifacts"),
            paths,
        ):
            self.assertGreater(report["category_bytes"][category], 0)
        # dataclass encoding is deterministic JSON for operator receipts.
        decoded = json.loads(json.dumps(report))
        self.assertEqual(decoded["physical_bytes"], report["physical_bytes"])

    def test_scan_error_is_explicit_and_report_is_incomplete(self):
        self.write("visible", b"x")
        with patch(
            "app.v2.storage.physical_usage.os.scandir",
            side_effect=PermissionError(13, "Permission denied"),
        ):
            report = audit_physical_usage(self.root)
        self.assertFalse(report.complete)
        self.assertEqual(report.errors[0].operation, "scan_directory")
        self.assertEqual(report.errors[0].error, "PermissionError")
        self.assertEqual(report.errors[0].path, ".")
        self.assertEqual(report.physical_bytes, 0)

    def test_count_error_is_explicit_not_silently_ignored(self):
        self.write("counted", b"x")
        with patch(
            "app.v2.storage.physical_usage._allocated_bytes",
            side_effect=PhysicalUsageError("invalid_block_count"),
        ):
            report = audit_physical_usage(self.root)
        self.assertFalse(report.complete)
        self.assertEqual(report.errors[0].operation, "count_blocks")
        self.assertEqual(report.errors[0].error, "invalid_block_count")

    def test_invalid_roots_fail_before_scan(self):
        outside = Path(self.temp.name) / "link"
        outside.symlink_to(self.root, target_is_directory=True)
        for root in (self.root / ".." / "escape", outside):
            with self.subTest(root=root), self.assertRaisesRegex(
                PhysicalUsageError, "absolute_canonical_audit_root_required"
            ):
                audit_physical_usage(root)
        with self.assertRaisesRegex(PhysicalUsageError, "audit_root_unavailable"):
            audit_physical_usage(self.root / "missing")

    def test_operator_cli_writes_fresh_report(self):
        self.write("state/episodes.sqlite3", b"database")
        output = Path(self.temp.name) / "physical.json"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/audit_physical_usage.py"), str(self.root), "--output", str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(json.loads(output.read_text())["complete"])
        with self.assertRaises(FileExistsError):
            output.open("xb")


if __name__ == "__main__":
    unittest.main()
