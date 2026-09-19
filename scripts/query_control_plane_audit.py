#!/usr/bin/env python3
"""Verify and query the private hash-chained Agent control-plane audit."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.control_plane import verify_control_audit  # noqa: E402


def query(
    path: Path,
    *,
    operation_id: str | None = None,
    episode_id: str | None = None,
    subject_id: str | None = None,
    event_type: str | None = None,
    policy_sha256: str | None = None,
    limit: int = 100,
) -> dict:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("query limit must be in [1, 1000]")
    events = verify_control_audit(path)
    if policy_sha256 is not None and any(
        item.policy_sha256 != policy_sha256 for item in events
    ):
        raise ValueError("control-plane audit contains an unexpected policy pin")
    selected = [
        item
        for item in events
        if (operation_id is None or item.operation_id == operation_id)
        and (episode_id is None or item.episode_id == episode_id)
        and (subject_id is None or item.subject_id == subject_id)
        and (event_type is None or item.event_type == event_type)
    ]
    counts = Counter(item.event_type for item in events)
    return {
        "schema_version": "1.0.0",
        "chain_valid": True,
        "event_count": len(events),
        "event_counts": dict(sorted(counts.items())),
        "first_sequence": events[0].sequence if events else None,
        "last_sequence": events[-1].sequence if events else None,
        "head_event_sha256": events[-1].event_sha256 if events else None,
        "matched_count": len(selected),
        "returned_count": min(len(selected), limit),
        "truncated": len(selected) > limit,
        "events": [
            item.model_dump(mode="json") for item in selected[:limit]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-log", required=True, type=Path)
    parser.add_argument("--operation-id")
    parser.add_argument("--episode-id")
    parser.add_argument("--subject-id")
    parser.add_argument("--event-type")
    parser.add_argument("--policy-sha256")
    parser.add_argument("--limit", type=int, default=100)
    arguments = parser.parse_args()
    runtime_root = (ROOT / "runtime").resolve()
    audit_path = arguments.audit_log.resolve()
    if (
        arguments.audit_log.is_symlink()
        or not audit_path.is_relative_to(runtime_root)
        or audit_path.parent == runtime_root
    ):
        raise SystemExit("audit log must use a dedicated project runtime directory")
    try:
        result = query(
            audit_path,
            operation_id=arguments.operation_id,
            episode_id=arguments.episode_id,
            subject_id=arguments.subject_id,
            event_type=arguments.event_type,
            policy_sha256=arguments.policy_sha256,
            limit=arguments.limit,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
