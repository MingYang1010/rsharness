"""Audit physical bytes in an operator-owned filesystem subtree.

This report is independent from StorageQuota reservations. It measures allocated
disk blocks, does not follow symlinks, and returns scan failures explicitly.
"""
from __future__ import annotations

import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Set, Tuple


class PhysicalUsageError(RuntimeError):
    """A stable error code for an invalid audit request."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PhysicalFile:
    path: str
    category: str
    physical_bytes: int
    counted: bool
    object_key: Tuple[int, int] | None


@dataclass(frozen=True)
class PhysicalUsageErrorRecord:
    path: str
    operation: str
    error: str


@dataclass(frozen=True)
class PhysicalUsageReport:
    root: str
    complete: bool
    physical_bytes: int
    category_bytes: dict[str, int]
    file_count: int
    hardlink_aliases_omitted: int
    symlink_entries_counted: int
    files: List[PhysicalFile]
    errors: List[PhysicalUsageErrorRecord]

    def as_dict(self) -> dict:
        return asdict(self)


def category_for(relative: Path) -> str:
    """Map one no-follow entry to a stable operator reporting category."""
    name = relative.name.lower()
    parts = tuple(part.lower() for part in relative.parts)
    if name.endswith(("-wal", "-wal.sqlite3", ".sqlite3-wal", ".sqlite-wal")):
        return "sqlite_wal"
    if name.endswith(("-shm", "-shm.sqlite3", ".sqlite3-shm", ".sqlite-shm")):
        return "sqlite_shm"
    if name.endswith((".sqlite3", ".sqlite", ".db")):
        return "sqlite_main"
    if "reports" in parts or name in {"execution.json"}:
        return "reports"
    if "logs" in parts or name.endswith((".log", ".jsonl")):
        return "docker_logs" if "docker" in name or "docker" in parts else "logs"
    if "checkpoints" in parts or name.endswith((".ckpt", ".pt", ".safetensors")):
        return "checkpoints"
    if "artifacts" in parts:
        return "runtime_artifacts"
    return "uncategorized"


def _allocated_bytes(info: os.stat_result) -> int:
    blocks = getattr(info, "st_blocks", 0)
    if type(blocks) is not int or blocks < 0:
        raise PhysicalUsageError("invalid_block_count")
    return blocks * 512


def _entries(directory: Path) -> Iterator[os.DirEntry]:
    with os.scandir(directory) as entries:
        yield from entries


def audit_physical_usage(root: Path) -> PhysicalUsageReport:
    """Scan one local filesystem without following links or deleting data."""
    raw = Path(root)
    if not raw.is_absolute() or ".." in raw.parts or raw.is_symlink():
        raise PhysicalUsageError("absolute_canonical_audit_root_required")
    try:
        resolved = raw.resolve(strict=True)
        root_info = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise PhysicalUsageError("audit_root_unavailable") from error
    if not stat.S_ISDIR(root_info.st_mode):
        raise PhysicalUsageError("audit_root_not_directory")

    categories = [
        "sqlite_main", "sqlite_wal", "sqlite_shm", "reports", "logs",
        "docker_logs", "checkpoints", "runtime_artifacts", "uncategorized",
    ]
    totals = {category: 0 for category in categories}
    files: List[PhysicalFile] = []
    errors: List[PhysicalUsageErrorRecord] = []
    seen_objects: Set[Tuple[int, int]] = set()
    aliases = 0
    symlink_count = 0
    complete = True
    pending = [resolved]

    while pending:
        directory = pending.pop()
        try:
            entries = list(_entries(directory))
        except OSError as error:
            errors.append(PhysicalUsageErrorRecord(
                str(directory.relative_to(resolved)), "scan_directory", type(error).__name__
            ))
            complete = False
            continue
        for entry in entries:
            relative = Path(entry.path).relative_to(resolved)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                errors.append(PhysicalUsageErrorRecord(
                    str(relative), "stat_entry", type(error).__name__
                ))
                complete = False
                continue
            if info.st_dev != root_info.st_dev:
                errors.append(PhysicalUsageErrorRecord(
                    str(relative), "stat_entry", "cross_filesystem_entry"
                ))
                complete = False
                continue
            try:
                physical = _allocated_bytes(info)
            except PhysicalUsageError as error:
                errors.append(PhysicalUsageErrorRecord(
                    str(relative), "count_blocks", error.code
                ))
                complete = False
                continue

            is_link = stat.S_ISLNK(info.st_mode)
            object_key = None
            counted = True
            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                object_key = (info.st_dev, info.st_ino)
                if object_key in seen_objects:
                    counted = False
                    aliases += 1
                else:
                    seen_objects.add(object_key)
            if is_link:
                symlink_count += 1
            category = category_for(relative)
            if counted:
                totals[category] += physical
            files.append(PhysicalFile(
                path=str(relative), category=category, physical_bytes=physical,
                counted=counted, object_key=object_key,
            ))
            if stat.S_ISDIR(info.st_mode):
                pending.append(Path(entry.path))

    files.sort(key=lambda item: item.path)
    errors.sort(key=lambda item: (item.path, item.operation, item.error))
    return PhysicalUsageReport(
        root=str(resolved),
        complete=complete,
        physical_bytes=sum(totals.values()),
        category_bytes=totals,
        file_count=len(files),
        hardlink_aliases_omitted=aliases,
        symlink_entries_counted=symlink_count,
        files=files,
        errors=errors,
    )
