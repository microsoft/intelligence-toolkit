"""
Extraction engine for Schemify.

Handles entity discovery, record extraction, and expansion.
"""

import asyncio
import hashlib
import re
from typing import Any, Optional
import logging

from .models import (
    AttributeValue, Citation, Record, RecordSet, 
    SchemaAttribute, SchemifyConfig
)
from .llm import LLMClient
from .cache import Cache, NoOpCache
from .search import SearchProvider, OpenAISearchProvider
from .sources import merge_attribute_values, merge_attribute_values_multi, classify_source
from . import prompts
from . import schemas
from datetime import date as _date


def _date_suffix() -> str:
    """Suffix appended to web-search queries so grounding favours current sources."""
    return f"\n\n(Today's date is {_date.today().isoformat()}. Prefer current/up-to-date sources.)"

logger = logging.getLogger("schemify.extraction")


class ExtractionEngine:
    """
    Core extraction engine for discovering and extracting entities.
    """
    
    def __init__(
        self,
        config: SchemifyConfig,
        llm: LLMClient,
        cache: Cache | NoOpCache,
        search_provider: SearchProvider | None = None,
    ):
        self.config = config
        self.llm = llm
        self.cache = cache
        self.search = search_provider or OpenAISearchProvider(llm)
    
    def _log_with_context(self, message: str):
        """Log a message with the LLM's current progress context."""
        if self.llm._progress_context:
            logger.info(f"[{self.llm._progress_context}] {message}")
        else:
            logger.info(message)
    
    async def discover_entities(
        self,
        record_set: RecordSet,
        subcategory_focus: str = "",
        expand_immediately: bool = True,
    ) -> list[Record]:
        """
        Discover new entities via web search with two-phase approach:
        1. First search to discover entity labels
        2. Then targeted search for each entity to fill attributes
        
        Args:
            record_set: Existing record set
            subcategory_focus: Optional subcategory to focus on
            expand_immediately: Whether to do targeted search per entity
            
        Returns:
            List of newly discovered records
        """
        # Build the search query
        guidance = record_set.guidance or "Focus on notable and well-documented examples."
        
        if subcategory_focus:
            focus_text = f"Focus specifically on: {subcategory_focus}"
        else:
            focus_text = ""
        
        # Build exclusion list from well-covered entities
        exclusion_text = record_set.build_exclusion_text(threshold=0.6, max_labels=30)
        
        query = prompts.ENTITY_DISCOVERY_QUERY.format(
            category=record_set.category,
            guidance=guidance,
            subcategory_focus=focus_text,
            exclusion_list=exclusion_text,
        ) + _date_suffix()
        
        # Check cache - use content hash of exclusion labels for stable keys.
        # Include user-defined exclusions (label OR attribute predicates) so
        # editing them busts the cache.
        exclusion_labels = sorted(record_set.get_well_covered_labels())
        user_excl_sig_parts: list[str] = []
        for e in (record_set.user_exclusions or []):
            if e.get("attribute"):
                user_excl_sig_parts.append(
                    "attr::"
                    + (e.get("attribute") or "").strip().lower()
                    + "::" + (e.get("operator") or "equals").lower()
                    + "::" + "|".join(
                        str(v).strip().lower()
                        for v in (e.get("values") or [])
                    )
                    + "::" + (e.get("reason") or "").strip().lower()
                )
            else:
                user_excl_sig_parts.append(
                    "label::"
                    + (e.get("label") or "").strip().lower()
                    + "::" + (e.get("reason") or "").strip().lower()
                )
        user_excl_sig = "##".join(user_excl_sig_parts)
        exclusion_hash = hashlib.sha256(
            ("|".join(exclusion_labels) + "@@" + user_excl_sig).encode()
        ).hexdigest()[:12]
        cache_key = {
            "category": record_set.category,
            "guidance": guidance,
            "subcategory": subcategory_focus,
            "exclusion_hash": exclusion_hash,
        }
        cached = self.cache.get("entity_discovery", **cache_key)
        
        if cached:
            self._log_with_context("Using cached discovery results")
            text, citations_data = cached["text"], cached["citations"]
            citations = [Citation.from_dict(c) for c in citations_data]
        else:
            # Perform web search via search provider
            text, citations = await self.search.search(query)
            
            # Cache the results
            self.cache.set(
                {"text": text, "citations": [c.to_dict() for c in citations]},
                "entity_discovery",
                **cache_key
            )
        
        self._log_with_context(f"Discovery found {len(citations)} sources")
        
        # Phase 1: Extract records with whatever info is in the broad search
        records = await self._extract_records_from_text(
            text=text,
            citations=citations,
            record_set=record_set,
        )
        
        # Phase 2: Expand each new record with a targeted search
        if expand_immediately and records:
            self._log_with_context(f"Phase 2: Expanding {len(records)} new entities with targeted searches")
            
            # Get core attributes to search for (limit to top 5 for focused queries)
            core_attrs = [a.name for a in record_set.schema_attributes[:5]]
            
            # Expand in parallel with limited concurrency
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent expansions
            
            async def expand_one(record: Record) -> Record:
                async with semaphore:
                    return await self._targeted_entity_search(
                        record=record,
                        record_set=record_set,
                        target_attributes=core_attrs,
                    )
            
            tasks = [expand_one(r) for r in records]
            await asyncio.gather(*tasks)
        
        return records
    
    async def _targeted_entity_search(
        self,
        record: Record,
        record_set: RecordSet,
        target_attributes: list[str],
    ) -> Record:
        """
        Perform a targeted web search for a specific entity.
        
        This is more effective than extracting from broad category searches
        because it searches specifically for this entity's details.
        """
        # Build a focused query for this specific entity
        attr_list = ", ".join(target_attributes) if target_attributes else "key characteristics"
        
        query = f"""Search for detailed information about "{record.label}" as a {record_set.category}.

Find specific facts about: {attr_list}

{record_set.guidance or ""}

Provide concrete, verifiable details with specific names, dates, organizations, and statistics where available."""
        
        # Check cache
        cache_key = {
            "label": record.label,
            "category": record_set.category,
        }
        cached = self.cache.get("entity_targeted_search", **cache_key)
        
        if cached:
            text, citations_data = cached["text"], cached["citations"]
            citations = [Citation.from_dict(c) for c in citations_data]
        else:
            text, citations = await self.search.search(query)
            self.cache.set(
                {"text": text, "citations": [c.to_dict() for c in citations]},
                "entity_targeted_search",
                **cache_key
            )
        
        # Extract attributes from the targeted search
        await self._extract_attributes_into_record(
            record=record,
            text=text,
            citations=citations,
            target_attributes=target_attributes,
            record_set=record_set,
        )
        
        logger.debug(f"Expanded {record.label}: {len(record.attributes)} attributes")
        return record
    
    async def expand_record(
        self,
        record: Record,
        record_set: RecordSet,
        target_attributes: list[str] | None = None,
    ) -> Record:
        """
        Expand a record with additional information via web search.
        
        Args:
            record: The record to expand
            record_set: The parent record set
            target_attributes: Specific attributes to search for
            
        Returns:
            The expanded record
        """
        # Determine which attributes to search for
        if target_attributes:
            attrs_to_find = target_attributes
        else:
            # Use schema attributes + any missing additional attributes
            schema_attrs = [a.name for a in record_set.schema_attributes]
            missing = [a for a in schema_attrs if a not in record.attributes or not record.attributes[a].value]
            attrs_to_find = missing if missing else schema_attrs[:5]
        
        if not attrs_to_find:
            self._log_with_context(f"No attributes to expand for {record.label}")
            return record
        
        # Build search query
        query = prompts.ENTITY_EXPANSION_QUERY.format(
            label=record.label,
            category=record_set.category,
            attributes=", ".join(attrs_to_find),
            guidance=record_set.guidance or "",
        ) + _date_suffix()
        
        # Check cache. Note: the cache key intentionally excludes the
        # attribute list — the web_search result for an entity rarely
        # changes when callers narrow the schema, and reusing the cached
        # page text avoids a costly second browsing round trip during
        # re-verification after schema tweaks.
        cache_key = {
            "label": record.label,
            "category": record_set.category,
        }
        cached = self.cache.get("entity_expansion", **cache_key)
        
        if cached:
            self._log_with_context(f"Using cached expansion for {record.label}")
            text, citations_data = cached["text"], cached["citations"]
            citations = [Citation.from_dict(c) for c in citations_data]
        else:
            # Perform web search via search provider
            text, citations = await self.search.search(query)
            
            # Cache the results
            self.cache.set(
                {"text": text, "citations": [c.to_dict() for c in citations]},
                "entity_expansion",
                **cache_key
            )
        
        # Extract attributes from the text
        await self._extract_attributes_into_record(
            record=record,
            text=text,
            citations=citations,
            target_attributes=attrs_to_find,
            record_set=record_set,
        )
        
        return record
    
    async def complete_missing_values(
        self,
        record_set: RecordSet,
    ) -> list[Record]:
        """
        Complete missing attribute values across all records.
        
        Args:
            record_set: The record set to complete
            
        Returns:
            List of records that were updated
        """
        updated_records = []
        schema_attrs = {a.name for a in record_set.schema_attributes}
        
        # Find records with missing values
        records_to_complete = []
        for record in record_set.records:
            missing = []
            for attr_name in schema_attrs:
                attr_val = record.attributes.get(attr_name)
                if not attr_val or not attr_val.value:
                    missing.append(attr_name)
            
            if missing and len(missing) <= len(schema_attrs) * 0.8:  # Skip if mostly empty
                records_to_complete.append((record, missing))
        
        self._log_with_context(f"Completing {len(records_to_complete)} records with missing values")
        
        # Expand each record in parallel (limited concurrency)
        async def expand_one(record: Record, missing: list[str]) -> Optional[Record]:
            try:
                await self.expand_record(record, record_set, target_attributes=missing)
                return record
            except Exception as e:
                logger.error(f"Error expanding {record.label}: {e}")
                return None
        
        tasks = [expand_one(r, m) for r, m in records_to_complete]
        results = await asyncio.gather(*tasks)
        
        updated_records = [r for r in results if r is not None]
        return updated_records
    
    async def _extract_records_from_text(
        self,
        text: str,
        citations: list[Citation],
        record_set: RecordSet,
    ) -> list[Record]:
        """
        Extract structured records from search result text.
        Uses per-attribute citation tracking for accurate source attribution.
        """
        existing_labels = record_set.get_labels()
        
        # Get schema attributes for structured extraction
        # Pass full SchemaAttribute objects to enable enum constraints for canonical_values
        schema_attrs = record_set.schema_attributes
        
        # Use citation-aware schema when we have citations
        use_citations = len(citations) > 0
        extraction_schema = schemas.get_record_extraction_schema(
            schema_attrs if schema_attrs else [], 
            with_citations=use_citations
        )
        
        # Build sources list for the prompt
        sources_list = ""
        if use_citations:
            sources_list = "\n".join(
                f"[{i}] {c.title} ({c.url})" 
                for i, c in enumerate(citations)
            )
        
        # Choose prompt based on citation tracking
        prompt = prompts.RECORD_EXTRACTION_WITH_CITATIONS if use_citations else prompts.RECORD_EXTRACTION
        
        # Extract records
        variables = {
            "category": record_set.category,
            "guidance": record_set.guidance or "",
            "existing_labels": "\n".join(existing_labels) if existing_labels else "(none)",
            "text": text,
        }
        if use_citations:
            variables["sources_list"] = sources_list
        
        result = await self.llm.structured_completion(
            prompt=prompt,
            response_format=extraction_schema,
            variables=variables,
        )
        
        def _split_evidence_by_source(evidence: str, indices: list[int]) -> dict[int, str]:
            """Split evidence like 'Source [0] says X. Source [1] says Y.' into {0: 'Says X.', 1: 'Says Y.'}."""
            parts = re.split(r'Sources?\s*\[(\d+)\]\s*', evidence)
            # parts = ['prefix', '0', 'text0', '1', 'text1', ...]
            result = {}
            if len(parts) >= 3:
                for i in range(1, len(parts) - 1, 2):
                    idx = int(parts[i])
                    text = parts[i + 1].strip().rstrip('.')
                    if text:
                        # Capitalize first letter
                        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
                        result[idx] = text + '.'
            return result

        def _clean_snippet(text: str) -> str:
            """Remove any remaining citation references and clean up."""
            text = re.sub(r'Sources?\s*\[\d+(?:,\s*\d+)*\]\s*', '', text)
            text = re.sub(r'\(\[\d+(?:,\s*\d+)*\]\)', '', text)
            text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
            text = re.sub(r'\s{2,}', ' ', text).strip()
            text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
            return text

        # Helper to resolve citation indices to actual Citation objects, optionally with evidence
        def resolve_citations(indices: list[int], evidence: str = "") -> list[Citation]:
            # Split multi-source evidence into per-index portions
            per_source = _split_evidence_by_source(evidence, indices) if evidence else {}

            resolved = []
            for idx in indices:
                if 0 <= idx < len(citations):
                    source = citations[idx]
                    # Use index-specific evidence, fall back to full evidence, then source snippet
                    snippet = per_source.get(idx) or evidence or source.snippet
                    if snippet:
                        snippet = _clean_snippet(snippet)
                    citation_with_evidence = Citation(
                        url=source.url,
                        title=source.title,
                        retrieved_at=source.retrieved_at,
                        start_index=source.start_index,
                        end_index=source.end_index,
                        snippet=snippet or None,
                        tier=source.tier,
                    )
                    resolved.append(citation_with_evidence)
            return resolved
        
        # Convert to Record objects
        new_records = []
        for raw_record in result.get("records", []):
            label = raw_record.get("label", "").strip().upper()
            
            # Skip if duplicate
            if label in [l.upper() for l in existing_labels]:
                logger.debug(f"Skipping duplicate: {label}")
                continue
            
            record = Record(label=label)
            
            # Add schema attributes with proper citation tracking
            for schema_attr in schema_attrs:
                attr_name: str = schema_attr.name if hasattr(schema_attr, 'name') else str(schema_attr)
                attr_data = raw_record.get(attr_name, "")
                
                if use_citations and isinstance(attr_data, dict):
                    # New format: {value: str, citation_indices: [int], evidence: str}
                    value = attr_data.get("value", "")
                    citation_indices = attr_data.get("citation_indices", [])
                    evidence = attr_data.get("evidence", "")
                    if value:
                        attr_value = AttributeValue()
                        cited_sources = resolve_citations(citation_indices, evidence)
                        attr_value.add_value_with_sources(value, cited_sources)
                        attr_value.compute_confidence()
                        record.attributes[attr_name] = attr_value
                elif attr_data:
                    # Old format: plain string (fallback)
                    value = attr_data if isinstance(attr_data, str) else str(attr_data)
                    attr_value = AttributeValue()
                    attr_value.add_value_with_sources(value, citations)
                    attr_value.compute_confidence()
                    record.attributes[attr_name] = attr_value
            
            # Add additional attributes
            for extra in raw_record.get("additional_attributes", []):
                name = extra.get("name", "")
                value = extra.get("value", "")
                if name and value:
                    attr_value = AttributeValue()
                    if use_citations:
                        citation_indices = extra.get("citation_indices", [])
                        evidence = extra.get("evidence", "")
                        cited_sources = resolve_citations(citation_indices, evidence)
                        attr_value.add_value_with_sources(value, cited_sources)
                    else:
                        attr_value.add_value_with_sources(value, citations)
                    attr_value.compute_confidence()
                    record.additional_attributes[name] = attr_value
            
            new_records.append(record)
            existing_labels.append(label)  # Prevent duplicates within batch
        
        self._log_with_context(f"Extracted {len(new_records)} new records")
        
        # Check for cross-entity contamination in this batch
        if len(new_records) > 1:
            self.check_cross_entity_contamination(new_records)
        
        return new_records
    
    async def _extract_attributes_into_record(
        self,
        record: Record,
        text: str,
        citations: list[Citation],
        target_attributes: list[str],
        record_set: RecordSet,
    ):
        """
        Extract specific attributes from text into an existing record.
        Uses per-attribute citation tracking for accurate source attribution.

        Uses a dedicated single-entity prompt that pins the label and
        alias-aware fuzzy matching so rebranded / renamed entities still
        get matched.
        """
        # Look up full SchemaAttribute objects to get canonical_values for enum constraints
        schema_attr_map = {a.name: a for a in record_set.schema_attributes}
        target_schema_attrs = [
            schema_attr_map[name] if name in schema_attr_map else name
            for name in target_attributes
        ]
        
        # Use citation-aware schema when we have citations
        use_citations = len(citations) > 0
        extraction_schema = schemas.get_record_extraction_schema(
            target_schema_attrs, 
            with_citations=use_citations
        )
        
        # Build sources list for the prompt
        sources_list = ""
        if use_citations:
            sources_list = "\n".join(
                f"[{i}] {c.title} ({c.url})" 
                for i, c in enumerate(citations)
            )
        
        # Use the single-entity verification prompt when we have citations
        # (verification always goes through web search → citations).
        # Fall back to the general prompt for non-citation paths.
        if use_citations:
            prompt = prompts.SINGLE_ENTITY_VERIFICATION
            aliases_display = ", ".join(record.aliases) if record.aliases else "(none)"
            variables = {
                "entity_label": record.label,
                "entity_aliases": aliases_display,
                "category": record_set.category,
                "guidance": record_set.guidance or "",
                "sources_list": sources_list,
                "text": f"Information about {record.label}:\n\n{text}",
            }
        else:
            prompt = prompts.RECORD_EXTRACTION
            variables = {
                "category": record_set.category,
                "guidance": record_set.guidance or "",
                "existing_labels": record.label,
                "text": f"Information about {record.label}:\n\n{text}",
            }
        
        result = await self.llm.structured_completion(
            prompt=prompt,
            response_format=extraction_schema,
            variables=variables,
        )
        
        def _split_evidence_by_source(evidence: str, indices: list[int]) -> dict[int, str]:
            """Split evidence like 'Source [0] says X. Source [1] says Y.' into {0: 'Says X.', 1: 'Says Y.'}."""
            parts = re.split(r'Sources?\s*\[(\d+)\]\s*', evidence)
            result_map: dict[int, str] = {}
            if len(parts) >= 3:
                for i in range(1, len(parts) - 1, 2):
                    idx = int(parts[i])
                    text = parts[i + 1].strip().rstrip('.')
                    if text:
                        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
                        result_map[idx] = text + '.'
            return result_map

        def _clean_snippet(text: str) -> str:
            """Remove any remaining citation references and clean up."""
            text = re.sub(r'Sources?\s*\[\d+(?:,\s*\d+)*\]\s*', '', text)
            text = re.sub(r'\(\[\d+(?:,\s*\d+)*\]\)', '', text)
            text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
            text = re.sub(r'\s{2,}', ' ', text).strip()
            text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
            return text

        # Helper to resolve citation indices to actual Citation objects, with evidence
        def resolve_citations(indices: list[int], evidence: str = "") -> list[Citation]:
            # Split multi-source evidence into per-index portions
            per_source = _split_evidence_by_source(evidence, indices) if evidence else {}

            resolved = []
            for idx in indices:
                if 0 <= idx < len(citations):
                    source = citations[idx]
                    snippet = per_source.get(idx) or evidence or source.snippet
                    if snippet:
                        snippet = _clean_snippet(snippet)
                    citation_with_evidence = Citation(
                        url=source.url,
                        title=source.title,
                        retrieved_at=source.retrieved_at,
                        start_index=source.start_index,
                        end_index=source.end_index,
                        snippet=snippet or None,
                        tier=source.tier,
                    )
                    resolved.append(citation_with_evidence)
            return resolved
        
        def _merge_raw_record(raw_record: dict):
            """Merge attribute data from a raw LLM record into self.record."""
            for attr_name in target_attributes:
                attr_data = raw_record.get(attr_name, "")
                
                if use_citations and isinstance(attr_data, dict):
                    new_value = attr_data.get("value", "")
                    citation_indices = attr_data.get("citation_indices", [])
                    evidence = attr_data.get("evidence", "")
                    if new_value:
                        existing = record.attributes.get(attr_name)
                        cited_sources = resolve_citations(citation_indices, evidence)
                        record.attributes[attr_name] = merge_attribute_values_multi(
                            existing, new_value, cited_sources
                        )
                elif attr_data:
                    new_value = attr_data if isinstance(attr_data, str) else str(attr_data)
                    existing = record.attributes.get(attr_name)
                    citation = citations[0] if citations else None
                    record.attributes[attr_name] = merge_attribute_values(
                        existing, new_value, citation
                    )
            
            for extra in raw_record.get("additional_attributes", []):
                name = extra.get("name", "")
                value = extra.get("value", "")
                if name and value:
                    existing = record.additional_attributes.get(name)
                    if use_citations:
                        citation_indices = extra.get("citation_indices", [])
                        evidence = extra.get("evidence", "")
                        cited_sources = resolve_citations(citation_indices, evidence)
                        record.additional_attributes[name] = merge_attribute_values_multi(
                            existing, value, cited_sources
                        )
                    else:
                        citation = citations[0] if citations else None
                        record.additional_attributes[name] = merge_attribute_values(
                            existing, value, citation
                        )
        
        # -----------------------------------------------------------
        # Match returned records to *this* entity.
        # 1. Exact label match (fast path)
        # 2. Fuzzy / alias-aware match (handles rebranding etc.)
        # -----------------------------------------------------------
        records_out = result.get("records", [])
        matched_raw = None
        
        # Pass 1: exact match (case-insensitive)
        for raw_record in records_out:
            raw_label = raw_record.get("label", "").strip().upper()
            if raw_label == record.label.upper():
                matched_raw = raw_record
                break
        
        # Pass 2: fuzzy / alias match
        if matched_raw is None and records_out:
            from .resolution import find_fuzzy_match
            returned_labels = [
                raw_record.get("label", "").strip()
                for raw_record in records_out
            ]
            # Build alias map for the target entity
            alias_map = {record.label: list(record.aliases)}
            matched_label, score = find_fuzzy_match(
                record.label,
                returned_labels,
                threshold=70,  # lower threshold — we're matching *our* entity to LLM output
                include_aliases=alias_map,
            )
            if matched_label:
                for raw_record in records_out:
                    if raw_record.get("label", "").strip() == matched_label:
                        matched_raw = raw_record
                        # Record the alternate name as an alias
                        # (captures rebranding: Thorn Spotlight → Spotlight by Canary)
                        record.add_alias(matched_label)
                        logger.info(
                            "Verification fuzzy-matched '%s' → '%s' (score %d); "
                            "added as alias",
                            record.label, matched_label, score,
                        )
                        break
            else:
                logger.warning(
                    "Verification for '%s' returned %d record(s) but none matched "
                    "(labels: %s)",
                    record.label,
                    len(records_out),
                    ", ".join(returned_labels[:5]),
                )
        
        if matched_raw is not None:
            _merge_raw_record(matched_raw)
    
    @staticmethod
    def check_cross_entity_contamination(records: list[Record]) -> list[dict]:
        """
        Check for cross-entity attribute contamination in a batch of extracted records.
        
        Detects when the same evidence snippet is cited for the same attribute across
        multiple entities — a common LLM failure mode where attributes get mixed between
        entities in a single extraction batch.
        
        Args:
            records: List of records from a single extraction batch
            
        Returns:
            List of contamination warnings, each with attribute, evidence, and affected entities
        """
        if len(records) < 2:
            return []
        
        # Index: (attr_name, evidence_snippet) -> list of entity labels
        evidence_map: dict[tuple[str, str], list[str]] = {}
        
        for record in records:
            for attr_name, attr_val in record.attributes.items():
                for sv in attr_val.values:
                    for source in sv.sources:
                        if source.snippet:
                            # Normalize snippet for comparison (lowercase, strip whitespace)
                            key = (attr_name, source.snippet.strip().lower()[:200])
                            evidence_map.setdefault(key, []).append(record.label)
        
        warnings = []
        for (attr_name, snippet), labels in evidence_map.items():
            if len(set(labels)) > 1:
                warnings.append({
                    "attribute": attr_name,
                    "evidence_prefix": snippet[:80],
                    "entities": list(set(labels)),
                })
        
        if warnings:
            logger.warning(
                f"Cross-entity contamination detected: {len(warnings)} shared evidence snippets "
                f"across {sum(len(w['entities']) for w in warnings)} entity-attribute pairs"
            )
        
        return warnings
