#!/usr/bin/env python3
"""Initialize private storage directories and an owner-only persistent credential."""
import os
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.storage.quota import StorageQuota


def main():
    quota = StorageQuota(ROOT / "runtime")
    token_file = quota.control / "broker-token"
    if token_file.is_symlink():
        raise ValueError("invalid credential path")
    # No token is printed or written to the repository or process records.
    if not token_file.exists():
        descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_hex(32) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    if token_file.stat().st_mode & 0o077 or not re.fullmatch(r"[0-9a-f]{64}", token_file.read_text().strip()):
        raise ValueError("storage credential invalid or not owner-only")
    target = quota.root / "managed-artifacts"
    if not target.exists():
        with quota.hold(target, 64 * 1024, "storage-bootstrap"):
            target.mkdir()
    print("storage bootstrap ready; credential retained privately")


if __name__ == "__main__":
    main()
