"""
Source handling for Schemify.

Implements URL/domain helpers and value-merging primitives.
Source quality tiers were removed in favour of the simpler
``Evidence`` model (see ``models.py``).
"""

from dataclasses import dataclass
from typing import Optional
import re

from .models import Citation, AttributeValue


def extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    # Remove protocol
    url = re.sub(r'^https?://', '', url.lower())
    # Remove path
    domain = url.split('/')[0]
    # Remove www prefix
    domain = re.sub(r'^www\.', '', domain)
    return domain


@dataclass
class ConflictResolution:
    """Result of a conflict resolution."""
    resolved_value: str
    resolution_reason: str
    all_values: list[tuple[str, Citation]]


def resolve_conflict(
    values: list[tuple[str, Citation]],
    strategy: str = "prefer_recent",
) -> ConflictResolution:
    """Resolve conflicting values from multiple sources.

    Tier-based resolution was removed. Supported strategies are now:
    ``prefer_recent``, ``prefer_specific``, and ``keep_all``. The
    default is ``prefer_recent``.
    """
    if not values:
        return ConflictResolution("", "No values provided", [])

    if len(values) == 1:
        return ConflictResolution(values[0][0], "Single source", values)

    if strategy == "prefer_recent":
        sorted_values = sorted(values, key=lambda x: x[1].retrieved_at, reverse=True)
        return ConflictResolution(
            sorted_values[0][0],
            "Most recent source",
            values,
        )

    if strategy == "prefer_specific":
        sorted_values = sorted(values, key=lambda x: len(x[0]), reverse=True)
        return ConflictResolution(
            sorted_values[0][0],
            "Most specific value",
            values,
        )

    if strategy == "keep_all":
        unique_values = list(dict.fromkeys(v[0] for v in values))
        combined = " | ".join(unique_values)
        return ConflictResolution(
            combined,
            f"Combined {len(unique_values)} values",
            values,
        )

    # Unknown strategy — fall back to most recent.
    return resolve_conflict(values, "prefer_recent")


def merge_attribute_values(
    existing: Optional[AttributeValue],
    new_value: str,
    new_citation: Optional[Citation] = None,
    strategy: str = "prefer_recent",
) -> AttributeValue:
    """Merge a new value into an existing attribute value.

    Each distinct value keeps its own sources; evidence is derived
    on read (see ``AttributeValue.evidence``).
    """
    if not existing:
        result = AttributeValue()
        result.add_value(new_value, new_citation)
        return result

    existing.add_value(new_value, new_citation)
    return existing


def merge_attribute_values_multi(
    existing: Optional[AttributeValue],
    new_value: str,
    citations: list[Citation],
    strategy: str = "prefer_recent",
) -> AttributeValue:
    """Merge a new value into an existing attribute value with multiple citations."""
    if not existing:
        result = AttributeValue()
        result.add_value_with_sources(new_value, citations)
        return result

    for citation in citations:
        existing.add_value(new_value, citation)
    if not citations:
        existing.add_value(new_value, None)
    return existing
