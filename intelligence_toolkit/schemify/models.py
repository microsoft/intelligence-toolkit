"""
Data models for Schemify.

Implements attribute-level citation tracking, lightweight evidence
summaries, and schema evolution support.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
import json


class BudgetExceededError(Exception):
    """Raised when the estimated token budget is exceeded."""
    pass


_UNUSUAL_TERMINATORS = str.maketrans({
    "\u2028": " ",  # LINE SEPARATOR
    "\u2029": " ",  # PARAGRAPH SEPARATOR
    "\u0085": " ",  # NEXT LINE
})


def _clean_text(s):
    """Strip Unicode line terminators that trip editors/parsers.

    Why: scraped web content occasionally contains U+2028 / U+2029 /
    U+0085 inside text, which VS Code flags as "unusual line terminators"
    and which some JS consumers treat as real line breaks.
    """
    if isinstance(s, str):
        return s.translate(_UNUSUAL_TERMINATORS)
    return s


@dataclass
class Citation:
    """A citation from a web search result with source text evidence."""
    url: str
    title: str
    retrieved_at: datetime = field(default_factory=datetime.now)
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    snippet: Optional[str] = None  # The actual text from the response that this citation supports

    def __post_init__(self):
        self.url = _clean_text(self.url)
        self.title = _clean_text(self.title)
        self.snippet = _clean_text(self.snippet)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "retrieved_at": self.retrieved_at.isoformat(),
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        return cls(
            url=data["url"],
            title=data["title"],
            retrieved_at=datetime.fromisoformat(data.get("retrieved_at", datetime.now().isoformat())),
            snippet=data.get("snippet"),
        )


def _source_key(url: str) -> str:
    """Normalize a URL for distinct-source counting.

    Strips protocol, ``www.``, query, fragment, and trailing slashes so
    that ``https://www.example.com/path/`` and ``http://example.com/path``
    collapse to a single source.
    """
    if not url:
        return ""
    s = url.lower().split("?", 1)[0].split("#", 1)[0]
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    return s.rstrip("/")


_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize_value(s: str) -> str:
    """Casefold, drop punctuation, collapse whitespace.

    ``"Web-Based!"`` and ``"web based"`` both collapse to ``"web based"``.
    """
    if not s:
        return ""
    s = _PUNCT_RE.sub(" ", s.casefold())
    return _WS_RE.sub(" ", s).strip()


def _trigrams(s: str) -> set[str]:
    """Character trigrams over a normalized string, padded with spaces.

    Padding ensures short strings still yield discriminative shingles
    (``"eu"`` → ``{" eu", "eu "}``).
    """
    if not s:
        return set()
    padded = f" {s} "
    if len(padded) < 3:
        return {padded}
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _mean_pairwise_jaccard(values: list[str]) -> float:
    """Mean pairwise Jaccard similarity of trigram sets.

    Returns ``1.0`` for 0 or 1 distinct normalized values. For two or
    more, averages Jaccard over all unordered pairs.
    """
    distinct = sorted({_normalize_value(v) for v in values if v and v.strip()})
    if len(distinct) < 2:
        return 1.0
    grams = [_trigrams(v) for v in distinct]
    total = 0.0
    pairs = 0
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            a, b = grams[i], grams[j]
            union = a | b
            total += (len(a & b) / len(union)) if union else 1.0
            pairs += 1
    return total / pairs if pairs else 1.0


@dataclass
class Evidence:
    """Three orthogonal, auditable signals about an attribute value.

    Replaces the prior opaque ``confidence`` float. Each field is
    something a downstream reader can verify against the citation list
    themselves:

    * ``source_count`` — number of *distinct* citations supporting any
      value of this attribute (deduped by host+path).
    * ``agreement`` — mean pairwise Jaccard similarity over character
      trigrams of the normalized values. ``1.0`` when sources agree (or
      only one value exists); lower as values diverge.
    * ``last_seen_at`` — most recent ``retrieved_at`` across citations,
      or ``None`` if no citations exist. NOTE: this is *when we fetched*
      the page, not when the page was published. Publication date is not
      currently captured.
    """
    source_count: int
    agreement: float
    last_seen_at: Optional[datetime]

    def to_dict(self) -> dict:
        return {
            "source_count": self.source_count,
            "agreement": round(self.agreement, 3),
            "last_seen_at": (
                self.last_seen_at.isoformat() if self.last_seen_at else None
            ),
        }


@dataclass
class SourcedValue:
    """A single value with its supporting sources."""
    value: str
    sources: list[Citation] = field(default_factory=list)

    def __post_init__(self):
        self.value = _clean_text(self.value)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "sources": [s.to_dict() for s in self.sources],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SourcedValue":
        return cls(
            value=data["value"],
            sources=[Citation.from_dict(s) for s in data.get("sources", [])],
        )


@dataclass
class AttributeValue:
    """
    An attribute with support for multiple discovered values,
    each with its own sources.
    
    This enables:
    - Tracking conflicting values from different sources
    - Seeing which sources support which values
    - Choosing the best value based on source count/quality
    """
    values: list[SourcedValue] = field(default_factory=list)

    @property
    def value(self) -> str:
        """Get display value - the primary (best-sourced) value."""
        if not self.values:
            return ""
        # Return the value with the most sources (best supported)
        return max(self.values, key=lambda v: len(v.sources)).value
    
    @property
    def all_values(self) -> list[str]:
        """Get all distinct values as a list."""
        return list(set(v.value for v in self.values))
    
    @property
    def all_values_display(self) -> str:
        """Get all distinct values as a comma-separated string."""
        if not self.values:
            return ""
        if len(self.values) == 1:
            return self.values[0].value
        return ", ".join(sorted(set(v.value for v in self.values)))
    
    @property
    def primary_value(self) -> str:
        """Get the primary (best) value - the one with most/best sources."""
        if not self.values:
            return ""
        return max(self.values, key=lambda v: len(v.sources)).value
    
    @property
    def sources(self) -> list[Citation]:
        """Get all sources across all values."""
        all_sources = []
        for v in self.values:
            all_sources.extend(v.sources)
        return all_sources
    
    @property
    def primary_sources(self) -> list[Citation]:
        """Get sources for the primary value only."""
        if not self.values:
            return []
        primary = max(self.values, key=lambda v: len(v.sources))
        return primary.sources
    
    def add_value(self, value: str, source: Citation | None = None):
        """Add a value with its source, merging if value already exists."""
        value_lower = value.strip().lower()
        for existing in self.values:
            if existing.value.strip().lower() == value_lower:
                if source:
                    # Deduplicate by URL
                    if source.url not in {s.url for s in existing.sources}:
                        existing.sources.append(source)
                return
        sources = [source] if source else []
        self.values.append(SourcedValue(value=value.strip(), sources=sources))
    
    def add_value_with_sources(self, value: str, sources: list[Citation]):
        """Add a value with multiple sources at once."""
        if not sources:
            self.add_value(value, None)
            return
        for source in sources:
            self.add_value(value, source)
    
    def has_conflicts(self) -> bool:
        """Back-compat: true when normalized values disagree at all.

        Prefer ``evidence.agreement`` for a continuous measure.
        """
        norm = {_normalize_value(v.value) for v in self.values if v.value.strip()}
        norm.discard("")
        return len(norm) > 1

    @property
    def evidence(self) -> Evidence:
        """Derive the three evidence signals from current sources/values.

        Cheap to recompute; not cached. Sources are deduped by
        ``(host, path)`` so three URLs pointing at the same article
        count once.
        """
        seen: set[str] = set()
        distinct = 0
        last_seen: Optional[datetime] = None
        for s in self.sources:
            key = _source_key(s.url)
            if not key or key in seen:
                continue
            seen.add(key)
            distinct += 1
            if last_seen is None or s.retrieved_at > last_seen:
                last_seen = s.retrieved_at
        agreement = _mean_pairwise_jaccard([v.value for v in self.values])
        return Evidence(
            source_count=distinct,
            agreement=agreement,
            last_seen_at=last_seen,
        )

    def to_dict(self) -> dict:
        return {
            "value": self.value,  # Primary value for backward compat
            "values": [v.to_dict() for v in self.values],
            "evidence": self.evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttributeValue":
        # Support both old format (single value) and new format (values array)
        if "values" in data:
            values = [SourcedValue.from_dict(v) for v in data["values"]]
        else:
            # Legacy format: single value with sources
            values = [SourcedValue(
                value=data.get("value", ""),
                sources=[Citation.from_dict(s) for s in data.get("sources", [])],
            )]
        # Legacy ``confidence``/``verified``/``tier`` keys in older
        # snapshots are ignored — evidence is now derived from sources.
        return cls(values=values)


# Tokens that suggest a label is naming a *category* of things rather than
# a single specific entity. Used by ``_canonical_score`` so that when a
# record has been (often incorrectly) fuzzy-merged together with specific
# entity names, the specific name is preferred as the canonical label.
_CATEGORY_STOP_TOKENS: frozenset[str] = frozenset({
    "TOOLS", "ARCHIVES", "PLATFORMS", "DATABASES", "INITIATIVES",
    "FRAMEWORKS", "RESOURCES", "SERVICES", "SYSTEMS", "PROJECTS",
    "PROGRAMS", "ORGANIZATIONS", "AGENCIES", "GROUPS", "PARTNERS",
    "PARTNERSHIPS", "NETWORKS", "COALITIONS",
})


def _canonical_score(name, frequency):
    upper = name.upper()
    tokens = upper.replace(":", " ").split()
    word_count = len(tokens)
    stop_hits = sum(1 for t in tokens if t in _CATEGORY_STOP_TOKENS)
    is_category_shaped = (
        word_count >= 4
        or stop_hits >= 1
        or ":" in name
        or " AND " in f" {upper} "
        or " OR " in f" {upper} "
    )
    return (
        not is_category_shaped,
        -stop_hits,
        frequency,
        -word_count,
        -len(name),
        name,
    )


@dataclass
class Record:
    """
    An entity record with attribute-level citation tracking.
    
    Supports alias tracking: when fuzzy duplicate detection merges records,
    all name variants are kept as aliases and the most frequent is used as
    the canonical label.
    """
    label: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    additional_attributes: dict[str, AttributeValue] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)  # Alternate names for this entity
    alias_counts: dict[str, int] = field(default_factory=dict)  # Count of how often each name variant appears
    manual_canonical: bool = False  # If True, _update_canonical_label will not change the label
    completion_attempts: dict[str, int] = field(default_factory=dict)  # attr_name → targeted search attempt count
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_attribute(self, name: str) -> Optional[AttributeValue]:
        """Get an attribute value by name."""
        return self.attributes.get(name) or self.additional_attributes.get(name)
    
    def set_attribute(self, name: str, value: AttributeValue, is_schema_attr: bool = True):
        """Set an attribute value."""
        self.updated_at = datetime.now()
        if is_schema_attr:
            self.attributes[name] = value
        else:
            self.additional_attributes[name] = value
    
    def get_all_sources(self) -> list[Citation]:
        """Get all citations for this record."""
        sources = []
        for attr in self.attributes.values():
            sources.extend(attr.sources)
        for attr in self.additional_attributes.values():
            sources.extend(attr.sources)
        return sources
    
    def evidence_summary(self) -> dict:
        """Aggregate evidence signals across this record's attributes.

        Returns ``{n_attrs, n_sourced, mean_agreement, last_seen_at}``.
        ``mean_agreement`` averages per-attribute agreement scores over
        attributes that have any value; ``1.0`` means perfect agreement
        (or single-value attrs only).
        """
        all_attrs = list(self.attributes.values()) + list(self.additional_attributes.values())
        n_sourced = 0
        agreements: list[float] = []
        last_seen: Optional[datetime] = None
        for av in all_attrs:
            ev = av.evidence
            if ev.source_count > 0:
                n_sourced += 1
            if av.values:
                agreements.append(ev.agreement)
            if ev.last_seen_at and (
                last_seen is None or ev.last_seen_at > last_seen
            ):
                last_seen = ev.last_seen_at
        mean_agreement = (
            sum(agreements) / len(agreements) if agreements else 1.0
        )
        return {
            "n_attrs": len(all_attrs),
            "n_sourced": n_sourced,
            "mean_agreement": round(mean_agreement, 3),
            "last_seen_at": last_seen.isoformat() if last_seen else None,
        }
    
    def attribute_coverage(self, schema_attrs: Optional[list["SchemaAttribute"]] = None) -> float:
        """
        Compute attribute coverage ratio.
        
        Args:
            schema_attrs: If provided, compute coverage against these schema attributes.
                         Otherwise, count filled vs total attributes.
                         
        Returns:
            Float between 0.0 and 1.0 indicating coverage ratio.
        """
        if schema_attrs:
            # Coverage against schema attributes
            if not schema_attrs:
                return 0.0
            filled = sum(
                1 for attr in schema_attrs 
                if attr.name in self.attributes and self.attributes[attr.name].value
            )
            return filled / len(schema_attrs)
        else:
            # Coverage of any filled attributes
            all_attrs = list(self.attributes.values()) + list(self.additional_attributes.values())
            if not all_attrs:
                return 0.0
            filled = sum(1 for a in all_attrs if a.value)
            return filled / len(all_attrs)
    
    def has_good_coverage(self, schema_attrs: Optional[list["SchemaAttribute"]] = None, threshold: float = 0.6) -> bool:
        """
        Check if this record has good attribute coverage.
        
        Args:
            schema_attrs: Schema attributes to check against
            threshold: Coverage ratio threshold (default: 0.6 = 60%)
            
        Returns:
            True if coverage >= threshold
        """
        return self.attribute_coverage(schema_attrs) >= threshold
    
    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "aliases": self.aliases,
            "alias_counts": self.alias_counts,
            "attributes": {k: v.to_dict() for k, v in self.attributes.items()},
            "additional_attributes": {k: v.to_dict() for k, v in self.additional_attributes.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.completion_attempts:
            d["completion_attempts"] = self.completion_attempts
        return d
    
    def add_alias(self, name: str, count: int = 1):
        """
        Add a name variant as an alias and update counts.
        
        The canonical label is automatically updated to the most frequent variant.
        """
        name_normalized = name.strip()
        if not name_normalized:
            return
        
        # Update count for this name variant
        name_lower = name_normalized.lower()
        
        # Find existing key (case-insensitive)
        existing_key = None
        for key in self.alias_counts:
            if key.lower() == name_lower:
                existing_key = key
                break
        
        if existing_key:
            self.alias_counts[existing_key] += count
        else:
            self.alias_counts[name_normalized] = count
            # Add to aliases if not already present
            if name_normalized not in self.aliases and name_normalized.upper() != self.label.upper():
                self.aliases.append(name_normalized)
        
        # Update canonical label to most frequent variant
        self._update_canonical_label()
    
    def _update_canonical_label(self):
        """Update the canonical label to the best variant (category-aware).

        Picks the variant that is least "category-shaped" first, then most
        frequent, then shortest. Skipped entirely when ``manual_canonical`` is
        True so user pins are preserved across subsequent merges.
        """
        if not self.alias_counts:
            return
        if getattr(self, "manual_canonical", False):
            # Still rebuild aliases list, but don't change canonical label.
            self.aliases = [
                name for name in self.alias_counts.keys()
                if name.upper() != self.label.upper()
            ]
            return

        best_name = max(
            self.alias_counts.items(),
            key=lambda kv: _canonical_score(kv[0], kv[1]),
        )[0]
        new_label = best_name

        if new_label.upper() != self.label.upper():
            if self.label not in self.aliases:
                self.aliases.append(self.label)
            self.label = new_label

        self.aliases = [
            name for name in self.alias_counts.keys()
            if name.upper() != self.label.upper()
        ]
    
    def merge_from(self, other: "Record"):
        """
        Merge another record into this one.
        
        - Adds other's label as an alias
        - Merges all attribute values (combining sources)
        - Preserves alias counts for canonical label selection
        """
        from .sources import merge_attribute_values_multi
        
        self.updated_at = datetime.now()
        
        # Merge label as alias
        self.add_alias(other.label, other.alias_counts.get(other.label, 1))
        
        # Merge other's aliases
        for alias in other.aliases:
            count = other.alias_counts.get(alias, 1)
            self.add_alias(alias, count)
        
        # Merge attributes
        for attr_name, other_attr in other.attributes.items():
            if attr_name in self.attributes:
                # Merge values from other into existing
                for sourced_val in other_attr.values:
                    self.attributes[attr_name] = merge_attribute_values_multi(
                        self.attributes[attr_name],
                        sourced_val.value,
                        sourced_val.sources
                    )
            else:
                # Add new attribute
                self.attributes[attr_name] = other_attr
        
        # Merge additional attributes
        for attr_name, other_attr in other.additional_attributes.items():
            if attr_name in self.additional_attributes:
                for sourced_val in other_attr.values:
                    self.additional_attributes[attr_name] = merge_attribute_values_multi(
                        self.additional_attributes[attr_name],
                        sourced_val.value,
                        sourced_val.sources
                    )
            else:
                self.additional_attributes[attr_name] = other_attr
    
    def to_flat_dict(self, include_sources: bool = False, include_evidence: bool = False, collapse_additional: bool = True) -> dict:
        """Convert to flat dictionary for DataFrame export.
        
        Args:
            include_sources: If True, include <attr>_evidence column with source URLs
                           and alternative values as JSON (deprecated, use include_evidence)
            include_evidence: If True, include the grounding text in the evidence column
            collapse_additional: If True, combine all additional (non-schema) 
                               attributes into a single "Additional Attributes" column
        """
        import json
        
        # Treat include_sources as alias for include_evidence for backwards compat
        should_include_evidence = include_evidence or include_sources
        
        result: dict = {"label": self.label}
        
        # Include aliases if present
        if self.aliases:
            result["aliases"] = ", ".join(self.aliases)
        
        def build_evidence_json(attr: AttributeValue) -> str:
            """Build JSON evidence column keyed by attribute values."""
            if not attr.values:
                return ""
            
            # Build dict keyed by value -> list of sources with grounding text
            evidence_data: dict = {}
            
            for sv in attr.values:
                val = sv.value
                if val not in evidence_data:
                    evidence_data[val] = []
                
                for src in sv.sources:
                    src_entry = {"url": src.url}
                    # Always include grounding text if available
                    if src.snippet:
                        snippet = ' '.join(src.snippet.split())[:300]
                        if len(snippet) == 300:
                            snippet += "..."
                        src_entry["text"] = snippet
                    evidence_data[val].append(src_entry)
            
            return json.dumps(evidence_data, ensure_ascii=False) if evidence_data else ""
        
        # Schema attributes as individual columns
        for name, attr in self.attributes.items():
            result[name] = attr.value  # Clean primary value only
            if should_include_evidence and attr.values:
                evidence_json = build_evidence_json(attr)
                if evidence_json:
                    result[f"{name}_evidence"] = evidence_json
        
        # Additional attributes - collapse or expand
        if self.additional_attributes:
            if collapse_additional:
                # Collapse into single column as "key: value" pairs
                additional_parts = []
                for name, attr in self.additional_attributes.items():
                    if attr.value:
                        additional_parts.append(f"{name}: {attr.value}")
                result["Additional Attributes"] = "; ".join(additional_parts) if additional_parts else ""
            else:
                # Expand as separate columns with underscore prefix
                for name, attr in self.additional_attributes.items():
                    result[f"_{name}"] = attr.value
                    if should_include_evidence and attr.values:
                        evidence_json = build_evidence_json(attr)
                        if evidence_json:
                            result[f"_{name}_evidence"] = evidence_json
        
        # Flat record-level evidence summary for CSV/dataframe export.
        ev = self.evidence_summary()
        result["_sourced_attrs"] = ev["n_sourced"]
        result["_mean_agreement"] = ev["mean_agreement"]
        if ev["last_seen_at"]:
            result["_last_seen_at"] = ev["last_seen_at"]

        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        return cls(
            label=data["label"],
            aliases=data.get("aliases", []),
            alias_counts=data.get("alias_counts", {}),
            attributes={k: AttributeValue.from_dict(v) for k, v in data.get("attributes", {}).items()},
            additional_attributes={k: AttributeValue.from_dict(v) for k, v in data.get("additional_attributes", {}).items()},
            completion_attempts=data.get("completion_attempts", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
        )


@dataclass
class SchemaAttribute:
    """A schema attribute definition with exploration support."""
    name: str
    description: Optional[str] = None
    required: bool = False
    frequency: float = 0.0  # Fraction of records containing this attribute
    
    # Exploration support
    provisional_values: list[str] = field(default_factory=list)  # Values for combinatorial exploration (may evolve)
    canonical_values: list[str] = field(default_factory=list)  # Fixed values for normalization (immutable)
    is_closed_set: bool = False  # True if finite/bounded value set (e.g., continent, country)
    cardinality_threshold: int = 50  # Values below this = closed set
    values_explored: set = field(default_factory=set)  # Track which values have been explored
    
    # Value cardinality per entity
    is_multi_valued: bool = True  # Start as multi-valued; set to False if all entities have single value
    
    @property
    def normalization_values(self) -> list[str]:
        """Get values to use for normalization: canonical if set, else provisional."""
        return self.canonical_values if self.canonical_values else self.provisional_values
    
    @property
    def exploration_values(self) -> list[str]:
        """Get values to use for exploration: provisional if set, else canonical."""
        return self.provisional_values if self.provisional_values else self.canonical_values
    
    def classify_cardinality(self) -> bool:
        """
        Classify attribute as closed or open based on exploration values count.
        
        Closed set: Finite, bounded (continent, country, technology type)
        Open set: Unbounded (name, address, description)
        """
        self.is_closed_set = len(self.exploration_values) <= self.cardinality_threshold
        return self.is_closed_set
    
    def get_unexplored_values(self) -> list[str]:
        """Get exploration values that haven't been explored yet."""
        return [v for v in self.exploration_values if v not in self.values_explored]
    
    def mark_value_explored(self, value: str):
        """Mark a value as explored."""
        self.values_explored.add(value)
    
    def classify_value_cardinality(self, records: list["Record"]) -> bool:
        """
        Classify attribute as single or multi-valued based on entity data.
        
        Logic:
        - Start with multi-valued assumption (is_multi_valued = True)
        - If ALL entities have at most one value for this attribute, set to single-valued
        
        Args:
            records: List of records to analyze
            
        Returns:
            True if multi-valued, False if single-valued
        """
        for record in records:
            # Check in both attributes and additional_attributes
            attr_val = record.attributes.get(self.name) or record.additional_attributes.get(self.name)
            if attr_val and len(attr_val.values) > 1:
                # Found an entity with multiple values - this is multi-valued
                self.is_multi_valued = True
                return True
        
        # All entities have at most one value - this is single-valued
        self.is_multi_valued = False
        return False


@dataclass
class RecordSet:
    """
    A collection of records with schema management.
    
    Supports fuzzy duplicate detection: when adding records, similar labels
    are detected and merged, with all name variants kept as aliases.
    """
    category: str
    guidance: str
    records: list[Record] = field(default_factory=list)
    schema_attributes: list[SchemaAttribute] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    fuzzy_threshold: int = 85  # Similarity threshold for fuzzy matching (0-100)
    # User-supplied exclusion rules: each item is {"label": str, "reason": str}.
    # Surfaced into discovery/expansion prompts via build_exclusion_text() so
    # the agent stops re-proposing rejected entities.
    user_exclusions: list[dict] = field(default_factory=list)
    # Optional LLM-backed arbiter. When set, ``get_record_fuzzy`` will only
    # return matches that have been previously LLM-approved (cache hit on
    # the arbiter). Async paths consult the arbiter directly via
    # ``check_duplicates`` / ``deduplicate_fuzzy`` and warm the cache;
    # the sync ``add_record`` path falls through this gate, so unverified
    # fuzzy hits at add-time will NOT auto-merge.
    merge_arbiter: object | None = None
    # Pairs (frozenset of 2 labels) that the LLM arbiter has rejected.
    # Populated by ``deduplicate_fuzzy`` so future clustering passes skip
    # them.
    do_not_merge: set = field(default_factory=set)
    
    def get_record(self, label: str) -> Optional[Record]:
        """Get a record by label (case-insensitive)."""
        label_upper = label.upper()
        for record in self.records:
            if record.label.upper() == label_upper:
                return record
        return None
    
    def get_record_fuzzy(self, label: str, threshold: int | None = None) -> tuple[Optional[Record], int]:
        """
        Get a record by fuzzy label matching.
        
        Args:
            label: The label to search for
            threshold: Similarity threshold (0-100), uses self.fuzzy_threshold if None
            
        Returns:
            Tuple of (record, score) or (None, 0) if no match found
        """
        from .resolution import find_fuzzy_match
        
        threshold = threshold if threshold is not None else self.fuzzy_threshold
        
        # Build alias map for expanded matching
        alias_map = {
            r.label: r.aliases for r in self.records if r.aliases
        }
        
        matched_label, score = find_fuzzy_match(
            label,
            self.get_labels(),
            threshold=threshold,
            include_aliases=alias_map,
        )

        if matched_label:
            # If an arbiter is wired, refuse the merge unless it has been
            # previously LLM-approved (cache hit). This makes the sync
            # add-time fuzzy path safe by default: unverified fuzzy hits
            # become two records that the next async dedup pass will
            # re-evaluate.
            arbiter = getattr(self, "merge_arbiter", None)
            if arbiter is not None:
                cached = arbiter.get_cached(label, matched_label)
                if cached is None or not arbiter.approves(cached):
                    return None, 0
            return self.get_record(matched_label), score
        return None, 0
    
    def add_record(self, record: Record, use_fuzzy: bool = False) -> tuple[bool, Optional[Record]]:
        """
        Add a record if not duplicate. 
        
        Args:
            record: The record to add
            use_fuzzy: If True, use fuzzy matching to detect duplicates
            
        Returns:
            Tuple of (was_added, existing_record)
            - (True, None) if record was added as new
            - (False, existing) if duplicate found (existing may have been merged)
        """
        # Initialize alias counts if this is a new record
        if not record.alias_counts:
            record.alias_counts[record.label] = 1
        
        # Check for exact match first
        existing = self.get_record(record.label)
        if existing:
            existing.merge_from(record)
            return False, existing
        
        # Check for fuzzy match if enabled
        if use_fuzzy:
            existing, score = self.get_record_fuzzy(record.label)
            if existing:
                existing.merge_from(record)
                return False, existing
        
        # No duplicate - add as new record
        self.records.append(record)
        return True, None
    
    def get_labels(self) -> list[str]:
        """Get all record labels."""
        return [r.label for r in self.records]
    
    def get_well_covered_labels(self, threshold: float = 0.6, max_labels: int = 30) -> list[str]:
        """
        Get labels of records with good attribute coverage.
        
        Args:
            threshold: Coverage ratio threshold (default: 0.6 = 60%)
            max_labels: Maximum number of labels to return (to keep prompts manageable)
            
        Returns:
            List of labels for well-covered records
        """
        covered = [
            r.label for r in self.records 
            if r.has_good_coverage(self.schema_attributes, threshold)
        ]
        return covered[:max_labels]
    
    def build_exclusion_text(self, threshold: float = 0.6, max_labels: int = 30) -> str:
        """
        Build exclusion text for search queries.
        
        Args:
            threshold: Coverage ratio threshold
            max_labels: Maximum labels to include
            
        Returns:
            Formatted exclusion text, or empty string if no exclusions
        """
        sections: list[str] = []

        # User-supplied exclusion rules take priority — they tell the agent
        # WHY each entity (or class of entity) is out of scope, which helps
        # it skip lookalikes too. Rules can be label-based or predicates
        # over attributes (e.g. "Currency is empty", "Continent = Africa").
        if self.user_exclusions:
            from .resolution import format_exclusion_rule  # local import to avoid cycle

            rules = [format_exclusion_rule(r) for r in self.user_exclusions]
            rules = [r for r in rules if r]
            if rules:
                sections.append(
                    "User-defined exclusions — do NOT return entities matching "
                    "these criteria, and apply the stated reasons to similar "
                    "cases:\n" + "\n".join(f"- {r}" for r in rules)
                )

        labels = self.get_well_covered_labels(threshold, max_labels)
        if labels:
            labels_text = ", ".join(labels)
            sections.append(
                f"Do NOT include these already-documented examples: {labels_text}. "
                f"Focus on finding NEW and DIFFERENT examples not in this list."
            )

        return "\n\n".join(sections)
    
    async def deduplicate_fuzzy(
        self,
        threshold: int | None = None,
        arbiter=None,
    ) -> list[tuple[str, str, int]]:
        """
        Find and merge fuzzy duplicate records.

        Uses clustering to find groups of similar labels, then verifies each
        proposed merge with the LLM arbiter (if provided) before applying.
        Rejected pairs are added to ``do_not_merge`` so future passes skip
        them. Without an arbiter, falls back to naive accept-all (legacy
        behavior — should only be used in offline/no-key mode).

        Args:
            threshold: Similarity threshold (0-100), uses self.fuzzy_threshold if None.
            arbiter: Optional ``MergeArbiter`` to confirm each cluster merge.

        Returns:
            List of (canonical_label, merged_label, score) for each merge performed.
        """
        from .resolution import cluster_fuzzy_matches, find_all_fuzzy_matches

        threshold = threshold if threshold is not None else self.fuzzy_threshold

        labels = self.get_labels()
        clusters = cluster_fuzzy_matches(
            labels,
            threshold,
            do_not_merge=getattr(self, "do_not_merge", None),
        )

        if arbiter is not None:
            # Bind resolver so the arbiter sees full record context.
            arbiter.record_resolver = lambda lbl: self.get_record(lbl)

        # Ensure do_not_merge container exists; we may add to it on rejection.
        if not hasattr(self, "do_not_merge") or self.do_not_merge is None:
            self.do_not_merge = set()

        merges: list[tuple[str, str, int]] = []

        for cluster in clusters:
            cluster_records = [r for r in self.records if r.label in cluster]
            if len(cluster_records) < 2:
                continue

            base_record = max(
                cluster_records,
                key=lambda r: (len(r.attributes), sum(r.alias_counts.values(), 0)),
            )

            others = [r for r in cluster_records if r is not base_record]

            # Verify each candidate against the base via LLM.
            approved: list[Record] = []
            if arbiter is not None:
                pairs = [(base_record.label, o.label) for o in others]
                verdicts = await arbiter.verify_pairs(pairs)
                for o in others:
                    v = verdicts.get(arbiter._key(base_record.label, o.label))
                    if v is not None and arbiter.approves(v):
                        approved.append(o)
                    else:
                        # Record rejection so future runs don't re-propose it.
                        self.do_not_merge.add(frozenset({base_record.label, o.label}))
            else:
                approved = others

            for other in approved:
                matches = find_all_fuzzy_matches(
                    [base_record.label, other.label], threshold
                )
                score = matches[0][2] if matches else threshold
                merges.append((base_record.label, other.label, score))
                base_record.merge_from(other)
                self.records.remove(other)

        return merges
    
    def get_all_labels_with_aliases(self) -> list[str]:
        """Get all record labels including aliases."""
        labels = []
        for record in self.records:
            labels.append(record.label)
            labels.extend(record.aliases)
        return labels
        """Get all record labels."""
        return [r.label for r in self.records]
    
    def update_schema_frequencies(self):
        """Update attribute frequencies and value cardinality based on current records."""
        if not self.records:
            return
        
        attr_counts: dict[str, int] = {}
        for record in self.records:
            for name in record.attributes:
                attr_counts[name] = attr_counts.get(name, 0) + 1
            for name in record.additional_attributes:
                attr_counts[name] = attr_counts.get(name, 0) + 1
        
        # Update existing schema attributes
        for attr in self.schema_attributes:
            attr.frequency = attr_counts.get(attr.name, 0) / len(self.records)
            # Update single/multi-valued classification
            attr.classify_value_cardinality(self.records)
        
        # NOTE: Do NOT auto-add new schema attributes here. Promotions
        # should only happen via evolve_schema() or explicit agent
        # schema_changes, which apply proper frequency thresholds.
        # Previously this block added every attribute (even 0%-fill
        # additional_attributes) to schema_attributes, causing massive
        # bloat (500+ attrs when only ~10 have data).
    
    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "guidance": self.guidance,
            "records": [r.to_dict() for r in self.records],
            "schema_attributes": [
                {
                    "name": a.name, 
                    "description": a.description, 
                    "required": a.required, 
                    "frequency": a.frequency,
                    "is_multi_valued": a.is_multi_valued,
                    "is_closed_set": a.is_closed_set,
                    "provisional_values": a.provisional_values,
                    "canonical_values": a.canonical_values,
                }
                for a in self.schema_attributes
            ],
            "user_exclusions": list(self.user_exclusions),
            "do_not_merge": [sorted(list(p)) for p in (self.do_not_merge or set())],
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RecordSet":
        return cls(
            category=data["category"],
            guidance=data["guidance"],
            records=[Record.from_dict(r) for r in data.get("records", [])],
            schema_attributes=[
                SchemaAttribute(
                    name=a["name"],
                    description=a.get("description"),
                    required=a.get("required", False),
                    frequency=a.get("frequency", 0.0),
                    is_multi_valued=a.get("is_multi_valued", True),
                    is_closed_set=a.get("is_closed_set", False),
                    provisional_values=a.get("provisional_values", []),
                    canonical_values=a.get("canonical_values", []),
                )
                for a in data.get("schema_attributes", [])
            ],
            user_exclusions=list(data.get("user_exclusions", []) or []),
            do_not_merge={
                frozenset(p) for p in (data.get("do_not_merge") or [])
                if isinstance(p, (list, tuple)) and len(p) == 2
            },
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "RecordSet":
        return cls.from_dict(json.loads(json_str))


@dataclass
class SchemifyConfig:
    """Configuration for Schemify."""
    api_key: str
    
    # Model settings
    search_model: str = "gpt-5.2"  # For web search
    completion_model: str = "gpt-5.2"  # For structured extraction
    reasoning_effort: str = "none"  # Reasoning level for completion model
    temperature: float = 0.0  # Temperature for model outputs
    
    # Schema thresholds
    schema_inclusion_threshold: float = 0.5
    value_completion_threshold: float = 0.2
    parameter_limit: int = 20
    
    # Deduplication
    dedup_similarity_threshold: float = 0.95  # High threshold to avoid false merges
    dedup_auto_merge: bool = False
    
    # Multi-source verification
    verification_queries: int = 1
    require_corroboration: int = 1
    
    # Parallel exploration
    parallel_subcategories: int = 3
    
    # Reflection settings
    enable_reflection: bool = True  # Use LLM reflection for strategic queries
    reflection_threshold: int = 10  # Min entities before reflection triggers
    max_reflections: int = 3  # Max reflection rounds per run
    reflection_low_yield: float = 0.1  # Trigger reflection when yield rate < 10%
    
    # Completion settings (runs AFTER main query loop completes)
    completion_enabled: bool = True  # Enable post-run completion pass
    max_completion_calls_per_entity: int = 3  # Max completion LLM calls per entity
    
    # Cost controls
    max_budget: Optional[float] = None
    cost_alerts: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    
    # Source filtering
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    
    # Deterministic seeding for reproducible query ordering
    seed: Optional[int] = None
    
    # Lazy query generation (generate pairs/triples from productive results only)
    lazy_generation: bool = True

    # Early stop window (stop if last N queries yield 0 new entities; 0 = disabled)
    early_stop_window: int = 10

    # Multilingual search: if set, every agent-issued search query is
    # passed through this translator to fan out into additional source
    # languages. Each entry returned is ``(translated_query, lang_code)``.
    # Default (None) preserves single-language behaviour.
    query_translator: Optional[Callable[[str], Awaitable[list[tuple[str, str]]]]] = None

    # Target language for extracted attribute values. When set to
    # anything other than the empty string, an instruction is added to
    # the discovery prompt asking the LLM to normalize attribute values
    # into that language regardless of source-document language.
    target_language: str = "English"
