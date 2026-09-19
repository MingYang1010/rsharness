"""Versioned derivation identity; content storage remains SHA-256 addressed.

No DB lookup or first-writer-dependent choice participates in this identity.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import Artifact

DERIVATION_SCHEME = "derivation-sha256-v1"
LEGACY_SCHEME = "content-sha256-v1"


def derivation_artifact_id(metadata: dict[str, Any]) -> str:
    """Bind every serialized artifact field except its own ID, with a domain tag."""
    body = {key: value for key, value in metadata.items() if key != "artifact_id"}
    encoded = json.dumps({"domain": "eo-harness/artifact-derivation/v1", "artifact": body},
                         sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return "art-" + hashlib.sha256(encoded).hexdigest()


def validate_derivation_metadata(metadata: dict[str, Any]) -> None:
    digest = metadata["sha256"]
    if metadata["uri"] != f"artifact://sha256/{digest[:2]}/{digest}":
        raise ValueError("content URI does not match checksum")
    if metadata["artifact_id"] != derivation_artifact_id(metadata):
        raise ValueError("artifact derivation identity does not match metadata")


def with_derivation_identity(artifact: Artifact) -> Artifact:
    """Return a new typed reference; never mutate original metadata or bytes."""
    from .schemas import Artifact
    from pydantic import TypeAdapter
    body = artifact.model_dump(mode="json")
    body["artifact_id"] = derivation_artifact_id(body)
    validate_derivation_metadata(body)
    return TypeAdapter(Artifact).validate_python(body)
