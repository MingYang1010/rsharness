"""Operator-owned persistent storage accounting; never an Agent observation."""

from .physical_usage import audit_physical_usage

__all__ = ["audit_physical_usage"]
