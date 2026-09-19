"""Cooperative, crash-conservative reservations for a single runtime filesystem.

All writers must use the same root and bound their own output. This is not a
kernel quota or a sandbox for arbitrary writers. Unmanaged bytes are scanned and
charged before each grant; unfinished reservation headroom is never timed out.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import sqlite3
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


MAX_BYTES = 3_000_000_000_000  # Decimal TB, matching the acquisition lock.
CONTROL_ALLOWANCE = 16 * 1024 * 1024


class QuotaError(RuntimeError):
    """A stable error code suitable for operator receipts."""


def checked_path(root: Path, path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise QuotaError("absolute_canonical_path_required")
    if not path.is_relative_to(root) or path == root or path.parts[len(root.parts)] == ".quota":
        raise QuotaError("scope_outside_managed_runtime")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise QuotaError("symlink_scope_forbidden")
    return path


def charge(info: os.stat_result) -> int:
    # Charge sparse files by logical length and tiny files by allocated blocks.
    return max(info.st_size, getattr(info, "st_blocks", 0) * 512)


def scan(root: Path, scopes: list[str]) -> tuple[int, dict[str, int], int]:
    """One no-follow scan; hardlinks count per pathname (conservative)."""
    totals = {scope: 0 for scope in scopes}
    data_bytes = control_bytes = 0
    pending = [root]
    device = root.stat().st_dev
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if info.st_dev != device:
                    raise QuotaError("cross_filesystem_entry")
                path = Path(entry.path)
                relative = str(path.relative_to(root))
                if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                    raise QuotaError("unsupported_runtime_file_type")
                size = charge(info)
                if relative == ".quota" or relative.startswith(".quota/"):
                    control_bytes += size
                else:
                    data_bytes += size
                    for scope in scopes:
                        if relative == scope or relative.startswith(scope + "/"):
                            totals[scope] += size
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
    return data_bytes, totals, max(CONTROL_ALLOWANCE, control_bytes)


class StorageQuota:
    def __init__(self, root: Path, limit_bytes: int = MAX_BYTES):
        raw = Path(root)
        if not raw.is_absolute() or ".." in raw.parts or any(p.is_symlink() for p in (raw, *raw.parents)):
            raise QuotaError("invalid_runtime_root")
        if type(limit_bytes) is not int or not CONTROL_ALLOWANCE < limit_bytes <= MAX_BYTES:
            raise QuotaError("invalid_storage_limit")
        raw.mkdir(parents=True, exist_ok=True)
        self.root = raw.resolve(strict=True)
        self.control = self.root / ".quota"
        if self.control.is_symlink():
            raise QuotaError("symlink_control_forbidden")
        self.control.mkdir(mode=0o700, exist_ok=True)
        self.database = self.control / "ledger.sqlite3"
        if self.database.is_symlink():
            raise QuotaError("symlink_control_forbidden")
        # SQLite serializes policy initialization as well as reservations.
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL, limit_bytes INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS reservations (scope TEXT PRIMARY KEY, capacity INTEGER NOT NULL, purpose TEXT NOT NULL, status TEXT NOT NULL, updated REAL NOT NULL)")
            row = db.execute("SELECT version, limit_bytes FROM policy WHERE id=1").fetchone()
            if row is None:
                db.execute("INSERT INTO policy VALUES(1, 1, ?)", (limit_bytes,))
            elif tuple(row) != (1, limit_bytes):
                raise QuotaError("quota_policy_mismatch")
        self.limit = limit_bytes

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _snapshot(self, db: sqlite3.Connection) -> dict:
        rows = [dict(row) for row in db.execute("SELECT * FROM reservations ORDER BY scope")]
        active = [row for row in rows if row["status"] in {"active", "failed"}]
        data, sizes, control = scan(self.root, [row["scope"] for row in active])
        headroom = sum(max(0, row["capacity"] - sizes[row["scope"]]) for row in active)
        return {"limit_bytes": self.limit, "data_bytes": data, "control_bytes": control,
                "reserved_headroom_bytes": headroom, "charged_bytes": data + control + headroom,
                "available_bytes": max(0, self.limit - data - control - headroom),
                "scope_bytes": sizes, "reservations": rows}

    def status(self) -> dict:
        with self._transaction() as db:
            return self._snapshot(db)

    def _scope(self, path: Path) -> str:
        return str(checked_path(self.root, Path(path)).relative_to(self.root))

    @contextmanager
    def _writer_lock(self, scope: str) -> Iterator[None]:
        # Crash releases the OS lock, not the persisted reservation. A live writer
        # cannot be resumed/reconciled even by another process with the same key.
        name = hashlib.sha256(scope.encode()).hexdigest() + ".lock"
        descriptor = os.open(self.control / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise QuotaError("scope_writer_active") from None
            yield
        finally:
            os.close(descriptor)

    @contextmanager
    def hold(self, path: Path, capacity_bytes: int, purpose: str) -> Iterator[None]:
        """Reserve maximum total scope size, including existing partial files.

        The caller must enforce its byte bound before writes and include filesystem
        and metadata overhead. Failed operations retain full headroom. Retrying
        the same scope takes the kernel lock and replaces—not adds—a reservation.
        """
        scope = self._scope(path)
        if type(capacity_bytes) is not int or not 0 < capacity_bytes <= self.limit:
            raise QuotaError("invalid_reservation_size")
        if not purpose or len(purpose) > 128:
            raise QuotaError("invalid_reservation_purpose")
        with self._writer_lock(scope):
            with self._transaction() as db:
                for row in db.execute("SELECT scope FROM reservations WHERE status IN ('active','failed')"):
                    other = row[0]
                    if other != scope and (other.startswith(scope + "/") or scope.startswith(other + "/")):
                        raise QuotaError("overlapping_reservation")
                db.execute("INSERT INTO reservations VALUES(?, ?, ?, 'active', ?) ON CONFLICT(scope) DO UPDATE SET capacity=excluded.capacity, purpose=excluded.purpose, status='active', updated=excluded.updated",
                           (scope, capacity_bytes, purpose, time.time()))
                snapshot = self._snapshot(db)
                if snapshot["scope_bytes"][scope] > capacity_bytes:
                    raise QuotaError("existing_scope_exceeds_reservation")
                if snapshot["charged_bytes"] > self.limit:
                    raise QuotaError("storage_quota_exceeded")
                if shutil.disk_usage(self.root).free < snapshot["reserved_headroom_bytes"] + CONTROL_ALLOWANCE:
                    raise QuotaError("insufficient_filesystem_space")
            try:
                yield
            except BaseException:
                with self._transaction() as db:
                    db.execute("UPDATE reservations SET status='failed', updated=? WHERE scope=?", (time.time(), scope))
                raise
            else:
                with self._transaction() as db:
                    snapshot = self._snapshot(db)
                    oversize = snapshot["scope_bytes"][scope] > capacity_bytes or snapshot["charged_bytes"] > self.limit
                    db.execute("UPDATE reservations SET status=?, updated=? WHERE scope=?",
                               ("failed" if oversize else "complete", time.time(), scope))
                if oversize:
                    raise QuotaError("writer_exceeded_reservation")

    def reconcile(self, path: Path) -> dict:
        """Explicitly close an abandoned operation, never delete its files.

        Requires no live cooperative writer. All current files remain charged;
        only unused headroom is released. A new writer must reserve again.
        """
        scope = self._scope(path)
        with self._writer_lock(scope), self._transaction() as db:
            if db.execute("SELECT 1 FROM reservations WHERE scope=?", (scope,)).fetchone() is None:
                raise QuotaError("unknown_reservation")
            self._snapshot(db)  # Fail closed if the filesystem cannot be audited.
            db.execute("UPDATE reservations SET status='reconciled', updated=? WHERE scope=?", (time.time(), scope))
            return self._snapshot(db)
