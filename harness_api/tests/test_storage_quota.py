import importlib.util
import io
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.storage.quota import CONTROL_ALLOWANCE, MAX_BYTES, QuotaError, StorageQuota

MIB = 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("quota_fetch", ROOT / "scripts/fetch_eo_gym.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def crash_writer(root, limit, pipe):
    quota = StorageQuota(Path(root), limit)
    with quota.hold(Path(root) / "crashed", 2 * MIB, "test-crash"):
        pipe.send("reserved")
        pipe.close()
        os._exit(17)


def concurrent_writer(root, limit, name, start, result, release):
    quota = StorageQuota(Path(root), limit)
    start.wait(10)
    try:
        with quota.hold(Path(root) / name, 3 * MIB, "concurrent-test"):
            result.put("granted")
            release.wait(10)
    except QuotaError as error:
        result.put(str(error))


class StorageQuotaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve() / "runtime"
        self.limit = CONTROL_ALLOWANCE + 8 * MIB
        self.quota = StorageQuota(self.root, self.limit)

    def tearDown(self):
        self.temp.cleanup()

    def test_decimal_limit_and_policy_is_persistent(self):
        self.assertEqual(MAX_BYTES, 3_000_000_000_000)
        self.assertEqual(StorageQuota(self.root, self.limit).status()["limit_bytes"], self.limit)
        with self.assertRaisesRegex(QuotaError, "policy_mismatch"):
            StorageQuota(self.root, self.limit + 1)
        with self.assertRaises(QuotaError):
            StorageQuota(self.root, MAX_BYTES + 1)

    def test_unmanaged_and_partial_files_are_charged(self):
        (self.root / "unmanaged.partial").write_bytes(b"x" * MIB)
        with self.quota.hold(self.root / "output", 2 * MIB, "test"):
            (self.root / "output").mkdir()
            (self.root / "output/part").write_bytes(b"x" * MIB)
            status = self.quota.status()
            self.assertGreaterEqual(status["data_bytes"], 2 * MIB)
            self.assertLessEqual(status["reserved_headroom_bytes"], MIB)
            self.assertGreaterEqual(status["charged_bytes"], CONTROL_ALLOWANCE + 3 * MIB)
        self.assertEqual(self.quota.status()["reserved_headroom_bytes"], 0)

    def test_denial_happens_before_output_creation(self):
        with self.assertRaisesRegex(QuotaError, "storage_quota_exceeded"):
            with self.quota.hold(self.root / "denied", 9 * MIB, "test"):
                self.fail("writer must not run")
        self.assertFalse((self.root / "denied").exists())
        self.assertEqual(self.quota.status()["reservations"], [])

    def test_failure_retains_headroom_and_retry_does_not_double_count(self):
        target = self.root / "resume"
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with self.quota.hold(target, 3 * MIB, "test"):
                target.mkdir()
                (target / "partial").write_bytes(b"x" * MIB)
                raise RuntimeError("interrupted")
        before = self.quota.status()
        self.assertEqual(before["reservations"][0]["status"], "failed")
        self.assertGreater(before["reserved_headroom_bytes"], MIB)
        with StorageQuota(self.root, self.limit).hold(target, 3 * MIB, "test-resume"):
            self.assertEqual(self.quota.status()["charged_bytes"], before["charged_bytes"])
        self.assertEqual(self.quota.status()["reserved_headroom_bytes"], 0)

    def test_reconciliation_keeps_files_and_releases_only_headroom(self):
        target = self.root / "failed"
        with self.assertRaises(RuntimeError):
            with self.quota.hold(target, 3 * MIB, "test"):
                target.mkdir()
                (target / "partial").write_bytes(b"kept")
                raise RuntimeError()
        status = self.quota.reconcile(target)
        self.assertEqual((target / "partial").read_bytes(), b"kept")
        self.assertEqual(status["reserved_headroom_bytes"], 0)
        self.assertGreater(status["data_bytes"], 0)

    def test_live_scope_cannot_be_reconciled_or_reentered(self):
        target = self.root / "live"
        with self.quota.hold(target, MIB, "test"):
            with self.assertRaisesRegex(QuotaError, "scope_writer_active"):
                self.quota.reconcile(target)
            with self.assertRaisesRegex(QuotaError, "scope_writer_active"):
                with self.quota.hold(target, MIB, "test"):
                    self.fail("second writer ran")

    def test_nested_scopes_cannot_double_reserve(self):
        with self.quota.hold(self.root / "parent", 2 * MIB, "test"):
            with self.assertRaisesRegex(QuotaError, "overlapping_reservation"):
                with self.quota.hold(self.root / "parent/child", MIB, "test"):
                    self.fail()

    def test_outside_root_symlink_and_control_scopes_denied(self):
        outside = Path(self.temp.name).resolve() / "outside"
        outside.mkdir()
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        for target in (outside / "data", self.root, self.root / "link/data", self.root / ".quota/data", self.root / "../escape"):
            with self.subTest(target=target), self.assertRaises(QuotaError):
                with self.quota.hold(target, MIB, "test"):
                    self.fail()

    def test_overrun_is_retained_and_blocks_further_grants(self):
        with self.assertRaisesRegex(QuotaError, "writer_exceeded_reservation"):
            with self.quota.hold(self.root / "overrun", MIB, "test"):
                (self.root / "overrun").write_bytes(b"x" * (9 * MIB))
        self.assertEqual(self.quota.status()["reservations"][0]["status"], "failed")
        with self.assertRaisesRegex(QuotaError, "storage_quota_exceeded"):
            with self.quota.hold(self.root / "later", MIB, "test"):
                self.fail()

    def test_free_disk_space_is_checked(self):
        with patch("app.core.storage.quota.shutil.disk_usage", return_value=type("Disk", (), {"free": 0})()):
            with self.assertRaisesRegex(QuotaError, "insufficient_filesystem_space"):
                with self.quota.hold(self.root / "disk", MIB, "test"):
                    self.fail()

    def test_symlink_partial_is_not_followed(self):
        target = self.root / "download"
        target.mkdir()
        outside = Path(self.temp.name) / "original"
        outside.write_bytes(b"preserved")
        (target / "test.zip.partial").symlink_to(outside)
        lock = {"archives": {"test.zip": {"size": 100}}}
        with self.assertRaisesRegex(QuotaError, "symlink_scope_forbidden"):
            fetch.fetch_archive(target, "test.zip", lock)
        self.assertEqual(outside.read_bytes(), b"preserved")

    def test_sparse_files_charge_logical_size_and_links_do_not_escape(self):
        with (self.root / "sparse").open("wb") as stream:
            stream.truncate(2 * MIB)
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "not-managed").write_bytes(b"x" * MIB)
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        status = self.quota.status()
        self.assertGreaterEqual(status["data_bytes"], 2 * MIB)
        self.assertLess(status["data_bytes"], 3 * MIB)

    def test_crashed_process_leaves_reservation_and_can_resume(self):
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=crash_writer, args=(str(self.root), self.limit, send))
        process.start()
        try:
            self.assertTrue(receive.poll(10))
            self.assertEqual(receive.recv(), "reserved")
            process.join(10)
            self.assertEqual(process.exitcode, 17)
            self.assertEqual(self.quota.status()["reserved_headroom_bytes"], 2 * MIB)
            with self.quota.hold(self.root / "crashed", 2 * MIB, "resume"):
                pass
            self.assertEqual(self.quota.status()["reserved_headroom_bytes"], 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            receive.close()
            send.close()

    def test_concurrent_processes_cannot_oversubscribe(self):
        # Both 3 MiB requests cannot fit in the same 4 MiB remaining pool.
        other_root = Path(self.temp.name).resolve() / "concurrent-runtime"
        limit = CONTROL_ALLOWANCE + 4 * MIB
        context = multiprocessing.get_context("spawn")
        start, release, results = context.Event(), context.Event(), context.Queue()
        processes = [context.Process(target=concurrent_writer, args=(str(other_root), limit, str(i), start, results, release)) for i in range(2)]
        for process in processes:
            process.start()
        start.set()
        try:
            observed = [results.get(timeout=15), results.get(timeout=15)]
            self.assertCountEqual(observed, ["granted", "storage_quota_exceeded"])
        finally:
            release.set()
            for process in processes:
                process.join(15)
                if process.is_alive():
                    process.terminate()
                    process.join()
            results.close()

    def test_archive_denial_precedes_network_and_download(self):
        lock = {"archives": {"test.zip": {"size": 100 * MIB}}}
        with patch.object(fetch, "network_opener") as opener:
            with self.assertRaises(QuotaError):
                fetch.acquire(self.root / "download", lock, "test.zip", "system-proxy", self.quota)
            opener.assert_not_called()
        self.assertFalse((self.root / "download").exists())

    def test_partial_archive_resume_retains_bytes_without_duplication(self):
        import hashlib
        target = self.root / "download"
        target.mkdir()
        content = b"archive content for deterministic resume"
        partial = target / "test.zip.partial"
        partial.write_bytes(content[:7])
        lock = {"archives": {"test.zip": {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}},
                "repository": "fixture", "revision": "fixed", "new_storage_budget_bytes": MAX_BYTES}
        response = io.BytesIO(content[7:])
        response.status = 206
        response.headers = {"Content-Range": f"bytes 7-{len(content)-1}/{len(content)}"}
        # Acquisition includes 16 MiB overhead, so give this fixture a larger pool.
        larger_root = Path(self.temp.name).resolve() / "archive-runtime"
        larger_quota = StorageQuota(larger_root, CONTROL_ALLOWANCE + 32 * MIB)
        target = larger_root / "download"
        target.mkdir()
        partial = target / "test.zip.partial"
        partial.write_bytes(content[:7])
        with patch.object(fetch, "network_opener") as opener:
            opener.return_value.open.return_value = response
            result = fetch.acquire(target, lock, "test.zip", "system-proxy", larger_quota)
            self.assertEqual(opener.call_args.args, ("system-proxy",))
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header("Range"), "bytes=7-")
        self.assertEqual(result.read_bytes(), content)
        self.assertFalse(partial.exists())
        self.assertEqual(larger_quota.status()["reserved_headroom_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
