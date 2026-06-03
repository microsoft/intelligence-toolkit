"""
Source quality handling for Schemify.

Implements domain classification, source weighting, and conflict resolution.
"""

from dataclasses import dataclass
from typing import Optional
import re

from .models import Citation, SourceTier, AttributeValue


# Default domain classifications
AUTHORITATIVE_DOMAINS = {
    ".gov", ".edu", ".mil", ".int",
}

REPUTABLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "forbes.com", "bloomberg.com", "wsj.com",
    "nature.com", "science.org", "sciencedirect.com",
    "ieee.org", "acm.org", "springer.com",
    "who.int", "un.org", "worldbank.org", "imf.org",
    "nih.gov", "cdc.gov", "fbi.gov", "state.gov",
}

USER_GENERATED_DOMAINS = {
    "reddit.com", "quora.com", "medium.com",
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "linkedin.com", "youtube.com", "tiktok.com",
    "tumblr.com", "pinterest.com",
}

BLACKLISTED_DOMAINS = {
    # Add known unreliable sources here
}


def extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    # Remove protocol
    url = re.sub(r'^https?://', '', url.lower())
    # Remove path
    domain = url.split('/')[0]
    # Remove www prefix
    domain = re.sub(r'^www\.', '', domain)
    return domain


def classify_source(
    url: str,
    custom_whitelist: list[str] | None = None,
    custom_blacklist: list[str] | None = None
) -> SourceTier:
    """
    Classify a URL into a source quality tier.
    
    Args:
        url: The URL to classify
        custom_whitelist: Additional domains to treat as authoritative
        custom_blacklist: Additional domains to blacklist
        
    Returns:
        SourceTier classification
    """
    domain = extract_domain(url)
    whitelist_set = set(custom_whitelist or [])
    blacklist_set = set(custom_blacklist or [])
    
    # Check blacklist first
    if domain in BLACKLISTED_DOMAINS or domain in blacklist_set:
        return SourceTier.BLACKLISTED
    
    # Check custom whitelist
    if domain in whitelist_set:
        return SourceTier.AUTHORITATIVE
    
    # Check authoritative TLDs
    for tld in AUTHORITATIVE_DOMAINS:
        if domain.endswith(tld):
            return SourceTier.AUTHORITATIVE
    
    # Check reputable domains
    if domain in REPUTABLE_DOMAINS:
        return SourceTier.REPUTABLE
    
    # Check user-generated
    if domain in USER_GENERATED_DOMAINS:
        return SourceTier.USER_GENERATED
    
    return SourceTier.GENERAL


@dataclass
class ConflictResolution:
    """Result of a conflict resolution."""
    resolved_value: str
    resolution_reason: str
    all_values: list[tuple[str, Citation]]


def resolve_conflict(
    values: list[tuple[str, Citation]],
    strategy: str = "prefer_authoritative"
) -> ConflictResolution:
    """
    Resolve conflicting values from multiple sources.
    
    Args:
        values: List of (value, citation) tuples
        strategy: Resolution strategy:
            - "prefer_authoritative": Prefer higher-tier sources
            - "prefer_recent": Prefer more recently retrieved sources
            - "prefer_specific": Prefer longer/more specific values
            - "keep_all": Combine all values with separator
            
    Returns:
        ConflictResolution with the resolved value
    """
    if not values:
        return ConflictResolution("", "No values provided", [])
    
    if len(values) == 1:
        return ConflictResolution(values[0][0], "Single source", values)
    
    if strategy == "prefer_authoritative":
        # Sort by source tier (higher = better)
        sorted_values = sorted(values, key=lambda x: x[1].tier.value, reverse=True)
        return ConflictResolution(
            sorted_values[0][0],
            f"Preferred {sorted_values[0][1].tier.name} source",
            values
        )
    
    elif strategy == "prefer_recent":
        # Sort by retrieval date (newer = better)
        sorted_values = sorted(values, key=lambda x: x[1].retrieved_at, reverse=True)
        return ConflictResolution(
            sorted_values[0][0],
            "Most recent source",
            values
        )
    
    elif strategy == "prefer_specific":
        # Prefer longer, more specific values
        sorted_values = sorted(values, key=lambda x: len(x[0]), reverse=True)
        return ConflictResolution(
            sorted_values[0][0],
            "Most specific value",
            values
        )
    
    elif strategy == "keep_all":
        # Combine unique values
        unique_values = list(dict.fromkeys(v[0] for v in values))
        combined = " | ".join(unique_values)
        return ConflictResolution(
            combined,
            f"Combined {len(unique_values)} values",
            values
        )
    
    else:
        # Default: prefer authoritative
        return resolve_conflict(values, "prefer_authoritative")


def merge_attribute_values(
    existing: Optional[AttributeValue],
    new_value: str,
    new_citation: Optional[Citation] = None,
    strategy: str = "prefer_authoritative"
) -> AttributeValue:
    """
    Merge a new value into an existing attribute value.
    
    Supports multiple values per attribute - each distinct value
    gets its own list of sources.
    
    Args:
        existing: Existing attribute value (may be None)
        new_value: New value to merge
        new_citation: Citation for the new value (may be None)
        strategy: Conflict resolution strategy (for future use)
        
    Returns:
        Merged AttributeValue with all discovered values
    """
    if not existing:
        # No existing value, create new with this value
        result = AttributeValue()
        result.add_value(new_value, new_citation)
        result.compute_confidence()
        return result
    
    # Add to existing - will merge if same value or add as alternative
    existing.add_value(new_value, new_citation)
    existing.compute_confidence()
    return existing


def merge_attribute_values_multi(
    existing: Optional[AttributeValue],
    new_value: str,
    citations: list[Citation],
    strategy: str = "prefer_authoritative"
) -> AttributeValue:
    """
    Merge a new value into an existing attribute value with multiple citations.
    
    Args:
        existing: Existing attribute value (may be None)
        new_value: New value to merge
        citations: List of citations supporting this value
        strategy: Conflict resolution strategy (for future use)
        
    Returns:
        Merged AttributeValue with all discovered values
    """
    if not existing:
        result = AttributeValue()
        result.add_value_with_sources(new_value, citations)
        result.compute_confidence()
        return result
    
    # Add to existing with all sources
    for citation in citations:
        existing.add_value(new_value, citation)
    if not citations:
        existing.add_value(new_value, None)
    existing.compute_confidence()
    return existing


def filter_sources_by_quality(
    citations: list[Citation],
    min_tier: SourceTier = SourceTier.USER_GENERATED
) -> list[Citation]:
    """
    Filter citations by minimum quality tier.
    
    Args:
        citations: List of citations to filter
        min_tier: Minimum tier to include (by value)
        
    Returns:
        Filtered list of citations
    """
    return [c for c in citations if c.tier.value >= min_tier.value]
