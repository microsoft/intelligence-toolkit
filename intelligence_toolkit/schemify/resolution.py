"""
Resolution logic for Schemify.

Handles attribute name resolution, entity deduplication, and schema evolution.
Exploration logic has been moved to QueryQueue.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Optional
import logging

from rapidfuzz import fuzz, process

from .models import (
    AttributeValue, Record, RecordSet, SourcedValue,
    SchemaAttribute, SchemifyConfig
)
from .llm import LLMClient
from . import prompts
from . import schemas

logger = logging.getLogger("schemify.resolution")


class ResolutionEngine:
    """
    Handles attribute resolution, deduplication, and schema management.
    
    Note: Exploration logic (query generation, combinatorial search) has been
    moved to QueryQueue for cleaner separation of concerns.
    """
    
    def __init__(self, config: SchemifyConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
    
    async def resolve_attribute_names(self, record_set: RecordSet) -> dict[str, str]:
        """
        Resolve semantically equivalent attribute names to canonical forms.
        
        Args:
            record_set: The record set to process
            
        Returns:
            Mapping of original -> canonical attribute names
        """
        # Collect all attribute names
        all_attrs = set()
        for record in record_set.records:
            all_attrs.update(record.attributes.keys())
            all_attrs.update(record.additional_attributes.keys())
        
        if len(all_attrs) < 2:
            return {}
        
        logger.info(f"Resolving {len(all_attrs)} attribute names")
        
        # Ask LLM to identify equivalent names
        result = await self.llm.structured_completion(
            prompt=prompts.ATTRIBUTE_RESOLUTION,
            response_format=schemas.get_attribute_resolution_schema(),
            variables={"attribute_names": "\n".join(sorted(all_attrs))}
        )
        
        # Build mapping
        mapping = {}
        for item in result.get("mappings", []):
            original = item.get("original", "")
            canonical = item.get("canonical", "")
            if original and canonical and original != canonical:
                mapping[original] = canonical
        
        if mapping:
            logger.info(f"Resolved {len(mapping)} attribute names")
            self._apply_attribute_mapping(record_set, mapping)
        
        return mapping
    
    async def normalize_attribute_values(
        self, 
        record_set: RecordSet,
        attributes: list[str] | None = None,
    ) -> dict[str, dict[str, str]]:
        """
        Normalize attribute values to canonical forms.
        
        For closed-set attributes with provisional values, maps observed values
        to the closest canonical value. Handles case differences, compound values,
        and near-matches.
        
        Args:
            record_set: The record set to process
            attributes: Specific attributes to normalize (default: all closed-set)
            
        Returns:
            Dict mapping attribute_name -> {original_value: normalized_value}
        """
        if not record_set.records:
            return {}
        
        # Determine which attributes to normalize
        if attributes:
            target_attrs = [a for a in record_set.schema_attributes if a.name in attributes]
        else:
            # Default: closed-set attributes with canonical or provisional values
            target_attrs = [
                a for a in record_set.schema_attributes 
                if a.is_closed_set and a.normalization_values
            ]
        
        if not target_attrs:
            return {}
        
        all_mappings = {}
        
        for attr in target_attrs:
            # Skip attributes with canonical_values - they were already extracted with enum constraints
            if attr.canonical_values:
                logger.info(f"Skipping normalization for '{attr.name}' - already enum-constrained during extraction")
                continue
            
            # Use canonical_values if set, else provisional_values
            normalization_values = attr.normalization_values
            if not normalization_values:
                continue
            
            # Collect all observed values for this attribute
            observed_values: set[str] = set()
            for record in record_set.records:
                attr_val = record.attributes.get(attr.name)
                if attr_val and attr_val.value:
                    observed_values.add(attr_val.value)
            
            if not observed_values:
                continue
            
            # Check if normalization is needed (case-insensitive dedup check)
            canonical_lower = {v.lower(): v for v in normalization_values}
            needs_normalization = False
            for obs in observed_values:
                if obs not in normalization_values:
                    needs_normalization = True
                    break
            
            if not needs_normalization:
                continue
            
            source_type = "canonical" if attr.canonical_values else "provisional"
            logger.info(f"Normalizing values for '{attr.name}': {len(observed_values)} observed → {len(normalization_values)} {source_type} values")
            
            # Ask LLM to normalize
            result = await self.llm.structured_completion(
                prompt=prompts.VALUE_NORMALIZATION,
                response_format=schemas.get_value_normalization_schema(),
                variables={
                    "attribute_name": attr.name,
                    "canonical_values": "\n".join(f"- {v}" for v in normalization_values),
                    "observed_values": "\n".join(f"- {v}" for v in sorted(observed_values)),
                }
            )
            
            # Build mapping
            mapping = {}
            for item in result.get("mappings", []):
                original = item.get("original", "")
                normalized = item.get("normalized", "")
                if original and normalized and original != normalized:
                    mapping[original] = normalized
            
            if mapping:
                logger.info(f"  Normalized {len(mapping)} values for '{attr.name}'")
                self._apply_value_mapping(record_set, attr.name, mapping, retain_raw=True)
                all_mappings[attr.name] = mapping
        
        return all_mappings
    
    async def normalize_open_set_values(
        self,
        record_set: RecordSet,
        attributes: list[str] | None = None,
        fuzzy_threshold: int = 80,
        min_cluster_size: int = 2,
        use_llm: bool = True,
    ) -> dict[str, dict[str, str]]:
        """
        Normalize open-set attribute values by fuzzy-matching and clustering.
        
        For open-set attributes (no predefined canonical values), this method:
        1. Collects all observed values for each attribute
        2. Clusters similar values using fuzzy string matching
        3. Optionally uses LLM to pick the best canonical form for each cluster
        4. Maps all cluster members to the standardized value
        
        This handles cases like:
        - "Case Management System" vs "Case Management Platform" → "Case Management System"
        - "mobile app" vs "Mobile Application" → "Mobile App"
        - "AI/ML" vs "Machine Learning" vs "Artificial Intelligence" → cluster together
        
        Args:
            record_set: The record set to process
            attributes: Specific attributes to normalize (default: all open-set)
            fuzzy_threshold: Similarity threshold for clustering (0-100, default: 80)
            min_cluster_size: Minimum cluster size to process (default: 2)
            use_llm: If True, use LLM to pick best canonical form; else use most frequent
            
        Returns:
            Dict mapping attribute_name -> {original_value: normalized_value}
        """
        if not record_set.records:
            return {}
        
        # Determine which attributes to normalize
        if attributes:
            target_attrs = [
                a for a in record_set.schema_attributes 
                if a.name in attributes and not a.is_closed_set
            ]
        else:
            # Default: open-set attributes (no canonical/provisional values)
            target_attrs = [
                a for a in record_set.schema_attributes 
                if not a.is_closed_set and not a.normalization_values
            ]
        
        if not target_attrs:
            return {}
        
        all_mappings = {}
        
        for attr in target_attrs:
            # Collect all observed values with frequency counts
            value_counts: dict[str, int] = {}
            for record in record_set.records:
                attr_val = record.attributes.get(attr.name)
                if attr_val:
                    for sv in attr_val.values:
                        if sv.value:
                            value_counts[sv.value] = value_counts.get(sv.value, 0) + 1
            
            if len(value_counts) < 2:
                continue
            
            observed_values = list(value_counts.keys())
            
            # Cluster similar values
            clusters = cluster_fuzzy_matches(observed_values, threshold=fuzzy_threshold)
            
            # Filter to clusters meeting minimum size
            clusters = [c for c in clusters if len(c) >= min_cluster_size]
            
            if not clusters:
                continue
            
            logger.info(f"Clustering open-set values for '{attr.name}': {len(clusters)} clusters from {len(observed_values)} values")
            
            mapping = {}
            
            if use_llm and clusters:
                # Format clusters for LLM
                cluster_text = []
                for i, cluster in enumerate(clusters, 1):
                    # Sort by frequency (most common first)
                    sorted_cluster = sorted(cluster, key=lambda v: value_counts.get(v, 0), reverse=True)
                    values_with_counts = [f'"{v}" ({value_counts.get(v, 0)}x)' for v in sorted_cluster]
                    cluster_text.append(f"Cluster {i}: {', '.join(values_with_counts)}")
                
                # Ask LLM to pick canonical values
                result = await self.llm.structured_completion(
                    prompt=prompts.OPEN_SET_VALUE_CLUSTERING,
                    response_format=schemas.get_open_set_clustering_schema(),
                    variables={
                        "attribute_name": attr.name,
                        "clusters": "\n".join(cluster_text),
                    }
                )
                
                # Build mapping from LLM response
                cluster_list = list(clusters)
                for item in result.get("standardizations", []):
                    cluster_id = item.get("cluster_id", 0)
                    canonical = item.get("canonical_value", "")
                    
                    if 1 <= cluster_id <= len(cluster_list) and canonical:
                        cluster = cluster_list[cluster_id - 1]
                        for value in cluster:
                            if value != canonical:
                                mapping[value] = canonical
                        
                        # If canonical isn't in cluster, map the most common to it
                        if canonical not in cluster:
                            most_common = max(cluster, key=lambda v: value_counts.get(v, 0))
                            mapping[most_common] = canonical
                        
                        reasoning = item.get("reasoning", "")
                        logger.debug(f"  Cluster {cluster_id}: {len(cluster)} values → '{canonical}' ({reasoning})")
            else:
                # Without LLM, use the most frequent value as canonical
                for cluster in clusters:
                    most_common = max(cluster, key=lambda v: value_counts.get(v, 0))
                    for value in cluster:
                        if value != most_common:
                            mapping[value] = most_common
            
            if mapping:
                logger.info(f"  Standardized {len(mapping)} values for '{attr.name}'")
                self._apply_value_mapping(record_set, attr.name, mapping, retain_raw=True)
                all_mappings[attr.name] = mapping
        
        return all_mappings
    
    async def auto_normalize(
        self,
        record_set: RecordSet,
        attributes: list[str] | None = None,
        fuzzy_threshold: int = 75,
        cardinality_threshold: int = 50,
    ) -> dict[str, dict]:
        """
        Unified normalization pass that analyzes value distributions to
        automatically classify attributes as open/closed and normalize values.
        
        For each attribute:
        1. Collect all observed values with frequency counts
        2. Cluster near-duplicates via fuzzy string matching
        3. Ask the LLM to classify open/closed based on cluster distribution
           and select canonical values
        4. Apply the value mapping across all records
        5. Update SchemaAttribute metadata (is_closed_set, canonical_values)
        
        Args:
            record_set: The record set to process
            attributes: Specific attribute names to normalize (default: all)
            fuzzy_threshold: Similarity threshold for clustering (0-100)
            cardinality_threshold: Max canonical values for closed-set classification
            
        Returns:
            Dict mapping attribute_name -> {
                "classification": "closed" | "open",
                "unique_raw": int,
                "unique_clustered": int,
                "canonical_values": list[str],
                "mappings": dict[str, str],
            }
        """
        if not record_set.records:
            return {}
        
        # Determine target attributes
        if attributes:
            target_attrs = [
                a for a in record_set.schema_attributes
                if a.name in attributes
            ]
        else:
            target_attrs = list(record_set.schema_attributes)
        
        if not target_attrs:
            return {}
        
        total_records = len(record_set.records)
        results = {}
        
        for attr in target_attrs:
            # ---- 1. Collect observed values with frequency counts ----
            value_counts: dict[str, int] = {}
            for record in record_set.records:
                attr_val = record.attributes.get(attr.name)
                if attr_val:
                    for sv in attr_val.values:
                        if sv.value:
                            value_counts[sv.value] = value_counts.get(sv.value, 0) + 1
            
            if not value_counts:
                continue
            
            unique_raw = len(value_counts)
            
            # If already well-constrained (few unique values relative to records)
            # and canonical_values match observations, skip
            if attr.canonical_values and unique_raw <= len(attr.canonical_values) + 1:
                all_canonical = set(attr.canonical_values) | {"Other"}
                if all(v in all_canonical for v in value_counts):
                    logger.info(
                        f"Skipping '{attr.name}': already constrained "
                        f"({unique_raw} values within {len(attr.canonical_values)} canonical)"
                    )
                    continue
            
            # ---- 2. Cluster near-duplicates ----
            observed_values = list(value_counts.keys())
            clusters = cluster_fuzzy_matches(observed_values, threshold=fuzzy_threshold)
            
            # Build cluster membership set for fast lookup
            clustered_values: set[str] = set()
            for cluster in clusters:
                clustered_values.update(cluster)
            
            # Singletons = values not in any cluster
            singletons = [v for v in observed_values if v not in clustered_values]
            unique_clustered = len(clusters) + len(singletons)
            
            logger.info(
                f"Auto-normalizing '{attr.name}': {unique_raw} raw values → "
                f"{len(clusters)} clusters + {len(singletons)} singletons "
                f"= {unique_clustered} effective groups ({total_records} records)"
            )
            
            # ---- 3. Format clusters for LLM ----
            cluster_text_parts = []
            for i, cluster in enumerate(clusters, 1):
                sorted_cluster = sorted(
                    cluster,
                    key=lambda v: value_counts.get(v, 0),
                    reverse=True,
                )
                values_display = [
                    f'"{v}" ({value_counts.get(v, 0)}x)' for v in sorted_cluster
                ]
                cluster_text_parts.append(
                    f"Cluster {i}: {', '.join(values_display)}"
                )
            
            cluster_text = "\n".join(cluster_text_parts) if cluster_text_parts else "(none)"
            
            # Limit singletons sent to LLM to avoid massive prompts
            max_singletons = 80
            sorted_singletons = sorted(singletons, key=lambda v: value_counts.get(v, 0), reverse=True)
            singleton_parts = []
            for v in sorted_singletons[:max_singletons]:
                singleton_parts.append(f'- "{v}" ({value_counts.get(v, 0)}x)')
            if len(singletons) > max_singletons:
                omitted = len(singletons) - max_singletons
                singleton_parts.append(f"... and {omitted} more low-frequency singletons omitted")
            singleton_text = "\n".join(singleton_parts) if singleton_parts else "(none)"
            
            # ---- 4. Ask LLM to classify and normalize ----
            result = await self.llm.structured_completion(
                prompt=prompts.CARDINALITY_CLASSIFICATION,
                response_format=schemas.get_cardinality_classification_schema(),
                variables={
                    "attribute_name": attr.name,
                    "category": record_set.category,
                    "total_records": str(total_records),
                    "num_clusters": str(len(clusters)),
                    "clusters": cluster_text,
                    "singletons": singleton_text,
                },
            )
            
            classification = result.get("classification", "open")
            canonical_values = result.get("canonical_values", [])
            reasoning = result.get("reasoning", "")
            
            # Build value mapping
            mapping: dict[str, str] = {}
            for item in result.get("mappings", []):
                original = item.get("original", "")
                canonical = item.get("canonical", "")
                if original and canonical and original != canonical:
                    mapping[original] = canonical
            
            # ---- 4b. Map omitted singletons to nearest canonical (closed-set only) ----
            if classification == "closed" and canonical_values and len(singletons) > max_singletons:
                omitted_singletons = sorted_singletons[max_singletons:]
                for val in omitted_singletons:
                    if val in mapping:
                        continue
                    # Find the best fuzzy match among canonical values
                    best_match = process.extractOne(
                        val, canonical_values, scorer=fuzz.token_sort_ratio
                    )
                    if best_match and best_match[1] >= 50:
                        mapping[val] = best_match[0]
                    else:
                        mapping[val] = "REMOVE"
                
                logger.info(
                    f"  Mapped {len(omitted_singletons)} omitted singletons to canonical values"
                )
            
            logger.info(
                f"  '{attr.name}' classified as {classification} "
                f"({len(canonical_values)} canonical values, "
                f"{len(mapping)} remappings): {reasoning}"
            )
            
            # ---- 5. Apply mapping ----
            if mapping:
                self._apply_value_mapping(record_set, attr.name, mapping, retain_raw=True)
            
            # ---- 6. Update schema attribute metadata ----
            attr.is_closed_set = classification == "closed"
            if classification == "closed" and canonical_values:
                attr.canonical_values = canonical_values
                # Also set provisional to the same so normalization_values is consistent
                attr.provisional_values = list(canonical_values)
            
            results[attr.name] = {
                "classification": classification,
                "unique_raw": unique_raw,
                "unique_clustered": unique_clustered,
                "canonical_values": canonical_values,
                "mappings": mapping,
                "reasoning": reasoning,
            }
        
        return results

    async def cleanup_value_formats(
        self,
        record_set: RecordSet,
        attribute_formats: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        """
        Clean up poorly formatted values using LLM.
        
        For attributes like "Year Founded" where values might be "April 2014", 
        "Apr-14", "2017 (update in 2020)", etc., this uses an LLM to standardize
        to the expected format.
        
        Args:
            record_set: The record set to process
            attribute_formats: Dict mapping attribute name to expected format description.
                             e.g., {"Year Founded": "4-digit year like 2014"}
                             
        Returns:
            Dict mapping attribute_name -> {original_value: cleaned_value}
        """
        if not record_set.records:
            return {}
        
        all_mappings = {}
        
        for attr_name, expected_format in attribute_formats.items():
            # Collect all observed values for this attribute
            observed_values: set[str] = set()
            for record in record_set.records:
                attr_val = record.attributes.get(attr_name)
                if attr_val:
                    for sv in attr_val.values:
                        if sv.value:
                            observed_values.add(sv.value)
            
            if not observed_values:
                continue
            
            # Filter to values that might need cleanup (not already in expected format)
            # For years, check if value is not a clean 4-digit year
            needs_cleanup = set()
            for val in observed_values:
                val_stripped = val.strip()
                # Simple heuristic: if it's not a clean 4-digit year
                if not (val_stripped.isdigit() and len(val_stripped) == 4):
                    needs_cleanup.add(val)
            
            if not needs_cleanup:
                continue
            
            logger.info(f"Cleaning up {len(needs_cleanup)} values for '{attr_name}'")
            
            # Ask LLM to clean up
            result = await self.llm.structured_completion(
                prompt=prompts.VALUE_FORMAT_CLEANUP,
                response_format=schemas.get_value_normalization_schema(),
                variables={
                    "attribute_name": attr_name,
                    "expected_format": expected_format,
                    "observed_values": "\n".join(f"- {v}" for v in sorted(needs_cleanup)),
                }
            )
            
            # Build mapping
            mapping = {}
            for item in result.get("mappings", []):
                original = item.get("original", "")
                normalized = item.get("normalized", "")
                if original and normalized and original != normalized:
                    mapping[original] = normalized
            
            if mapping:
                logger.info(f"  Cleaned {len(mapping)} values for '{attr_name}'")
                self._apply_value_mapping(record_set, attr_name, mapping, retain_raw=True)
                all_mappings[attr_name] = mapping
        
        return all_mappings
    
    def fix_value_capitalization(
        self,
        record_set: RecordSet,
        attributes: list[str] | None = None,
    ) -> dict[str, dict[str, str]]:
        """
        Fix inconsistent Title Case produced by LLM extraction.
        
        Converts values like "Supply Chain Mapping And Traceability" to
        "Supply chain mapping and traceability" (sentence case), while
        preserving:
        - Acronyms (2+ consecutive uppercase chars: AI, OSINT, CSAM, UN)
        - The first character of the string (always uppercase)
        - Closed-set attribute values (already normalized to canonical forms)
        - Values that are already sentence case or all-lowercase
        
        Args:
            record_set: The record set to process
            attributes: Specific attribute names to fix (default: all open-set)
            
        Returns:
            Dict mapping attribute_name -> {original_value: fixed_value}
        """
        if not record_set.records:
            return {}
        
        # Determine which attributes to fix
        if attributes:
            target_attrs = [
                a for a in record_set.schema_attributes
                if a.name in attributes
            ]
        else:
            # Default: open-set attributes (closed-set values are canonical)
            target_attrs = [
                a for a in record_set.schema_attributes
                if not a.is_closed_set
            ]
        
        if not target_attrs:
            return {}
        
        all_mappings = {}
        
        for attr in target_attrs:
            mapping = {}
            
            for record in record_set.records:
                for attr_dict in (record.attributes, record.additional_attributes):
                    attr_val = attr_dict.get(attr.name)
                    if not attr_val:
                        continue
                    for sv in attr_val.values:
                        if not sv.value or sv.value in mapping:
                            continue
                        fixed = self._to_sentence_case(sv.value)
                        if fixed != sv.value:
                            mapping[sv.value] = fixed
            
            if mapping:
                logger.info(f"  Fixed capitalization for {len(mapping)} values in '{attr.name}'")
                self._apply_value_mapping(record_set, attr.name, mapping, retain_raw=False)
                all_mappings[attr.name] = mapping
        
        return all_mappings
    
    @staticmethod
    def _to_sentence_case(text: str) -> str:
        """
        Convert a Title Case string to sentence case, preserving acronyms.
        
        "Supply Chain Mapping And Traceability" -> "Supply chain mapping and traceability"
        "AI-powered OSINT platform" -> "AI-powered OSINT platform" (unchanged)
        "Blockchain analytics" -> "Blockchain analytics" (unchanged)
        """
        if not text:
            return text
        
        # Quick check: does it look like Title Case?
        # Count words that start with uppercase followed by lowercase
        words = text.split()
        if len(words) < 2:
            return text
        
        title_cased = sum(
            1 for w in words[1:]  # skip first word
            if len(w) > 1 and w[0].isupper() and w[1:].islower()
            and w.lower() in (
                'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
                'in', 'on', 'at', 'to', 'of', 'by', 'with', 'from', 'as', 'into',
                'through', 'during', 'before', 'after', 'between', 'under', 'over',
            )
        )
        
        if title_cased == 0:
            return text
        
        # Rebuild: lowercase each word unless it's an acronym or the first word
        result = []
        for i, word in enumerate(words):
            if i == 0:
                # Keep first word as-is
                result.append(word)
            elif re.match(r'^[A-Z]{2,}', word):
                # Acronym (2+ leading uppercase): keep as-is (AI, OSINT, CSAM, UN-backed)
                result.append(word)
            else:
                result.append(word.lower())
        
        return ' '.join(result)

    async def expand_enum_values(
        self,
        record_set: RecordSet,
        min_other_count: int = 3,
    ) -> dict[str, list[str]]:
        """
        Dynamically expand closed-set enum values when too many entities get "Other".
        
        When >= min_other_count entities are assigned "Other" for a closed-set attribute,
        triggers an LLM call to propose new canonical values and reclassify those entities.
        
        Args:
            record_set: The record set to process
            min_other_count: Minimum "Other" assignments to trigger expansion
            
        Returns:
            Dict mapping attribute_name -> list of newly added canonical values
        """
        if not record_set.records:
            return {}
        
        expansions = {}
        
        for attr in record_set.schema_attributes:
            if not attr.is_closed_set or not attr.canonical_values:
                continue
            
            # Find entities with "Other" for this attribute
            other_entities = []
            for record in record_set.records:
                attr_val = record.attributes.get(attr.name)
                if attr_val and attr_val.value == "Other":
                    other_entities.append(record.label)
            
            if len(other_entities) < min_other_count:
                continue
            
            logger.info(
                f"Expanding enum for '{attr.name}': {len(other_entities)} entities assigned 'Other'"
            )
            
            result = await self.llm.structured_completion(
                prompt=prompts.ENUM_EXPANSION,
                response_format=schemas.get_enum_expansion_schema(),
                variables={
                    "attribute_name": attr.name,
                    "category": record_set.category,
                    "current_values": "\n".join(f"- {v}" for v in attr.canonical_values),
                    "other_entities": "\n".join(f"- {e}" for e in other_entities),
                }
            )
            
            # Add new canonical values
            new_values = []
            for item in result.get("new_values", []):
                value = item.get("value", "").strip()
                if value and value not in attr.canonical_values:
                    attr.canonical_values.append(value)
                    new_values.append(value)
            
            # Reclassify entities
            reclassified = 0
            for item in result.get("reclassifications", []):
                entity_label = item.get("entity", "").strip().upper()
                new_value = item.get("new_value", "").strip()
                
                if not new_value or new_value == "Other":
                    continue
                
                # Find and update the record
                record = record_set.get_record(entity_label)
                if record:
                    attr_val = record.attributes.get(attr.name)
                    if attr_val:
                        # Replace "Other" with the new value
                        for sv in attr_val.values:
                            if sv.value == "Other":
                                sv.value = new_value
                                reclassified += 1
                                break
            
            if new_values:
                expansions[attr.name] = new_values
                logger.info(
                    f"  Added {len(new_values)} values to '{attr.name}', "
                    f"reclassified {reclassified} entities"
                )
        
        return expansions
    
    def consolidate_attribute_keys(self, record_set: RecordSet) -> dict[str, str]:
        """
        Deterministic, LLM-free cleanup of attribute-key drift.

        Two passes, applied in order:
          1. **Promote across buckets**: any key in ``additional_attributes`` that
             case-insensitively matches a ``schema_attributes`` name is merged
             into ``record.attributes`` under the canonical (schema) name.
          2. **Collapse within additional_attributes**: remaining case-insensitive
             duplicates are merged together; the most frequent casing across the
             corpus wins.

        Provenance keys with a ``(Raw)`` suffix are preserved verbatim.

        Returns:
            Mapping of ``old_key -> canonical_key`` that was applied (informational).
        """
        if not record_set.records:
            return {}

        schema_names = {a.name for a in record_set.schema_attributes}
        schema_lower = {n.lower(): n for n in schema_names}

        # Pass 1: build cross-bucket promotion map (additional -> core schema name).
        promote_map: dict[str, str] = {}
        for record in record_set.records:
            for key in list(record.additional_attributes.keys()):
                if key.lower().endswith("(raw)"):
                    continue
                canonical = schema_lower.get(key.lower())
                if canonical and key != canonical:
                    promote_map.setdefault(key, canonical)
                elif canonical and key == canonical:
                    # Same name in both buckets — fold additional into core.
                    promote_map.setdefault(key, canonical)

        promoted = 0
        for record in record_set.records:
            for old_key, canonical in promote_map.items():
                addl_val = record.additional_attributes.pop(old_key, None)
                if addl_val is None:
                    continue
                target = record.attributes.get(canonical)
                if target is None:
                    record.attributes[canonical] = addl_val
                else:
                    for sv in addl_val.values:
                        if not sv.sources:
                            target.add_value(sv.value, None)
                        else:
                            for src in sv.sources:
                                target.add_value(sv.value, src)
                promoted += 1

        # Pass 2: collapse case-insensitive duplicates remaining in additional_attributes.
        # First, pick the winning casing for each lowercase key (max frequency across records).
        casing_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for record in record_set.records:
            for key in record.additional_attributes:
                if key.lower().endswith("(raw)"):
                    continue
                casing_counts[key.lower()][key] += 1

        case_collapse: dict[str, str] = {}
        for lower, variants in casing_counts.items():
            if len(variants) < 2:
                continue
            winner = max(variants.items(), key=lambda kv: (kv[1], kv[0]))[0]
            for variant in variants:
                if variant != winner:
                    case_collapse[variant] = winner

        collapsed = 0
        for record in record_set.records:
            for old_key, canonical in case_collapse.items():
                addl_val = record.additional_attributes.pop(old_key, None)
                if addl_val is None:
                    continue
                target = record.additional_attributes.get(canonical)
                if target is None:
                    record.additional_attributes[canonical] = addl_val
                else:
                    for sv in addl_val.values:
                        if not sv.sources:
                            target.add_value(sv.value, None)
                        else:
                            for src in sv.sources:
                                target.add_value(sv.value, src)
                collapsed += 1

        applied = {**promote_map, **case_collapse}
        if applied:
            logger.info(
                f"consolidate_attribute_keys: promoted {promoted} additional→core, "
                f"collapsed {collapsed} case-variants ({len(applied)} distinct mappings)"
            )
        return applied

    async def merge_similar_attributes(
        self,
        record_set: RecordSet,
        fuzzy_threshold: int = 75,
    ) -> list[tuple[str, str]]:
        """
        Identify and merge near-duplicate schema attributes.
        
        Uses fuzzy matching to find candidate pairs, then LLM to confirm merges.
        Handles cases like "Technology" vs "Technology Type" or "Region" vs "Geographic Focus".
        
        Args:
            record_set: The record set to process
            fuzzy_threshold: Similarity threshold for candidate detection
            
        Returns:
            List of (merged_from, merged_into) pairs
        """
        attr_names = [a.name for a in record_set.schema_attributes]
        if len(attr_names) < 2:
            return []
        
        # Find similar attribute name pairs
        candidate_pairs = []
        for i, name1 in enumerate(attr_names):
            for name2 in attr_names[i + 1:]:
                score = fuzz.token_sort_ratio(name1.lower(), name2.lower())
                if score >= fuzzy_threshold:
                    candidate_pairs.append((name1, name2, score))
        
        if not candidate_pairs:
            return []
        
        logger.info(f"Found {len(candidate_pairs)} candidate attribute pairs for merge")
        
        # Ask LLM to confirm merges
        pairs_text = "\n".join(
            f"- \"{n1}\" vs \"{n2}\" (similarity: {s}%)"
            for n1, n2, s in candidate_pairs
        )
        
        result = await self.llm.structured_completion(
            prompt=prompts.ATTRIBUTE_MERGE,
            response_format=schemas.get_attribute_merge_schema(),
            variables={"attribute_pairs": pairs_text}
        )
        
        # Apply confirmed merges
        merged_pairs = []
        for item in result.get("merges", []):
            if not item.get("should_merge", False):
                continue
            
            attr1 = item.get("attr1", "")
            attr2 = item.get("attr2", "")
            canonical = item.get("canonical_name", "")
            
            if not canonical or canonical not in (attr1, attr2):
                continue
            
            old_name = attr2 if canonical == attr1 else attr1
            
            # Rename in mapping
            mapping = {old_name: canonical}
            self._apply_attribute_mapping(record_set, mapping)
            
            # Remove old schema attribute
            record_set.schema_attributes = [
                a for a in record_set.schema_attributes
                if a.name != old_name
            ]
            
            merged_pairs.append((old_name, canonical))
            logger.info(f"Merged attribute '{old_name}' into '{canonical}'")
        
        return merged_pairs

    def _apply_value_mapping(
        self, 
        record_set: RecordSet, 
        attr_name: str, 
        mapping: dict[str, str],
        retain_raw: bool = True,
    ):
        """
        Apply value normalization mapping to all records for a specific attribute.
        
        Handles multi-valued normalizations: if a normalized value contains "|",
        splits it into separate SourcedValue entries (inheriting sources from original).
        
        Args:
            record_set: The record set to process
            attr_name: Name of the attribute to normalize
            mapping: Dict of original value -> normalized value(s)
            retain_raw: If True, store original values in "{attr_name} (Raw)" attribute
        """
        raw_attr_name = f"{attr_name} (Raw)"
        
        for record in record_set.records:
            attr_val = record.attributes.get(attr_name)
            if not attr_val:
                continue
            
            # Store raw values before normalization (if any will be changed)
            if retain_raw:
                raw_values_to_store = []
                for sv in attr_val.values:
                    if sv.value in mapping:
                        raw_values_to_store.append(sv.value)
                
                if raw_values_to_store:
                    # Get or create raw attribute
                    if raw_attr_name not in record.additional_attributes:
                        record.additional_attributes[raw_attr_name] = AttributeValue()
                    raw_attr = record.additional_attributes[raw_attr_name]
                    
                    # Add raw values (deduplicated)
                    for raw_val in raw_values_to_store:
                        raw_attr.add_value(raw_val, None)
            
            # Process each SourcedValue in the attribute
            new_sourced_values = []
            needs_rebuild = False
            
            for sourced_val in attr_val.values:
                original_value = sourced_val.value
                
                if original_value not in mapping:
                    # No mapping needed, keep as-is
                    new_sourced_values.append(sourced_val)
                    continue
                
                new_value = mapping[original_value]
                
                if new_value == "REMOVE":
                    # Skip this value (don't add to new list)
                    needs_rebuild = True
                    continue
                
                # Check if this is a multi-value normalization (contains "|")
                if "|" in new_value:
                    # Split into separate SourcedValue entries, each inheriting the sources
                    needs_rebuild = True
                    for part in new_value.split("|"):
                        part = part.strip()
                        if part:
                            # Create new SourcedValue with same sources
                            new_sourced_values.append(SourcedValue(
                                value=part,
                                sources=list(sourced_val.sources)  # Copy sources
                            ))
                else:
                    # Simple single-value normalization
                    sourced_val.value = new_value
                    new_sourced_values.append(sourced_val)
                    needs_rebuild = True
            
            # Rebuild the values list if needed
            if needs_rebuild:
                if not new_sourced_values:
                    # All values were removed
                    del record.attributes[attr_name]
                    continue
                
                # Deduplicate by value (merge sources for same values)
                merged: dict[str, dict] = {}
                for sv in new_sourced_values:
                    val_lower = sv.value.lower()
                    if val_lower not in merged:
                        merged[val_lower] = {"value": sv.value, "sources": list(sv.sources)}
                    else:
                        # Merge sources, deduplicate by URL
                        existing_urls = {s.url for s in merged[val_lower]["sources"]}
                        for s in sv.sources:
                            if s.url not in existing_urls:
                                merged[val_lower]["sources"].append(s)
                                existing_urls.add(s.url)
                
                # Rebuild values list
                attr_val.values = [
                    SourcedValue(value=m["value"], sources=m["sources"])
                    for m in merged.values()
                ]

    def _apply_attribute_mapping(self, record_set: RecordSet, mapping: dict[str, str]):
        """Apply attribute name mapping to all records."""
        for record in record_set.records:
            # Process schema attributes
            attrs_to_rename = [(k, v) for k, v in record.attributes.items() if k in mapping]
            for old_name, attr_value in attrs_to_rename:
                new_name = mapping[old_name]
                # Merge if target exists
                if new_name in record.attributes:
                    existing = record.attributes[new_name]
                    # Merge all values from the old attribute into the existing one
                    for sourced_val in attr_value.values:
                        for source in sourced_val.sources:
                            existing.add_value(sourced_val.value, source)
                else:
                    record.attributes[new_name] = attr_value
                del record.attributes[old_name]
            
            # Process additional attributes
            addl_to_rename = [(k, v) for k, v in record.additional_attributes.items() if k in mapping]
            for old_name, attr_value in addl_to_rename:
                new_name = mapping[old_name]
                if new_name in record.additional_attributes:
                    existing = record.additional_attributes[new_name]
                    # Merge all values from the old attribute into the existing one
                    for sourced_val in attr_value.values:
                        for source in sourced_val.sources:
                            existing.add_value(sourced_val.value, source)
                else:
                    record.additional_attributes[new_name] = attr_value
                del record.additional_attributes[old_name]

    # ── Deterministic finalization (labels / aliases / units) ──────────

    # Attribute names that should be folded into ``record.aliases`` rather
    # than kept as ordinary attributes.
    _ALIAS_ATTR_NAMES = {
        "also known as",
        "aka",
        "alias",
        "aliases",
        "alternate name",
        "alternate names",
        "alternative name",
        "alternative names",
        "other name",
        "other names",
        "formerly known as",
        "previously known as",
    }

    # Maps a unit (lowercased) to its canonical attribute-suffix form.
    _UNIT_SUFFIXES = {
        "km²": "km2",
        "km2": "km2",
        "sq km": "km2",
        "sq. km": "km2",
        "square kilometres": "km2",
        "square kilometers": "km2",
        "mi²": "mi2",
        "mi2": "mi2",
        "sq mi": "mi2",
        "square miles": "mi2",
        "ha": "ha",
        "hectares": "ha",
        "m": "m",
        "metres": "m",
        "meters": "m",
        "km": "km",
        "kilometres": "km",
        "kilometers": "km",
        "kg": "kg",
        "kilograms": "kg",
        "t": "t",
        "tonnes": "t",
        "usd": "usd",
        "$": "usd",
        "eur": "eur",
        "€": "eur",
        "gbp": "gbp",
        "£": "gbp",
        "%": "pct",
        "percent": "pct",
        "years": "years",
        "yrs": "years",
    }

    _NUMERIC_VALUE_RE = re.compile(
        r"""
        ^\s*
        (?P<num>-?[\d,]*\.?\d+(?:e-?\d+)?)
        \s*
        (?P<unit>[^\d\s].*?)?
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def finalize_normalization(self, record_set: RecordSet) -> dict[str, int]:
        """
        Deterministic post-processing applied at the end of a run.

        Three independent passes:

        - **Labels** are upper-cased so the canonical entity name is stable.
        - Attribute values stored under any of ``_ALIAS_ATTR_NAMES`` (e.g.
          "Also known as") are folded into ``record.aliases`` and the
          attribute is dropped.
        - Numeric attributes whose values share a single unit (e.g. ``Area`` =
          "603 km²", "504 km²") get the unit lifted into the attribute name
          (``Area (km2)``) and the values are reduced to bare numerals.

        Returns a small dict of counts for logging.
        """
        if not record_set or not record_set.records:
            return {}

        stats = {"labels": 0, "aliases": 0, "units": 0, "pruned_attrs": 0}

        # ── 1. Uppercase labels ────────────────────────────────────────
        for record in record_set.records:
            upper = (record.label or "").strip().upper()
            if upper and upper != record.label:
                if record.label and record.label not in record.aliases:
                    record.aliases.append(record.label)
                record.label = upper
                stats["labels"] += 1
            # Dedup aliases (case-insensitive) and exclude the label itself
            seen: set[str] = set()
            deduped: list[str] = []
            label_upper = record.label.upper()
            for a in record.aliases:
                if not a:
                    continue
                key = a.upper()
                if key == label_upper or key in seen:
                    continue
                seen.add(key)
                deduped.append(a)
            record.aliases = deduped

        # ── 2. Fold alias-like attributes into record.aliases ─────────
        for record in record_set.records:
            for bucket in (record.attributes, record.additional_attributes):
                for key in list(bucket.keys()):
                    if key.strip().lower() not in self._ALIAS_ATTR_NAMES:
                        continue
                    attr_val = bucket.pop(key)
                    for sv in attr_val.values:
                        for piece in re.split(r"[;,/|]| or | and ", sv.value or ""):
                            piece = piece.strip().strip("\"'()")
                            if not piece:
                                continue
                            if piece.upper() == record.label.upper():
                                continue
                            if any(
                                piece.upper() == existing.upper()
                                for existing in record.aliases
                            ):
                                continue
                            record.aliases.append(piece)
                            stats["aliases"] += 1

        # ── 3. Lift units into numeric attribute names ────────────────
        renames = self._infer_unit_renames(record_set)
        for old_name, new_name in renames.items():
            self._rename_attribute_with_unit_strip(record_set, old_name, new_name)
            stats["units"] += 1

        # ── 4. Strip placeholder values ("Unknown", "N/A", …) ─────────
        # These slip in when the LLM declines to leave an attribute
        # empty. They make every downstream surface (CSV, dashboard)
        # look like the attribute carries real information when it
        # doesn't, and they prevent step 5 from pruning the attribute.
        placeholder_set = {
            "unknown", "n/a", "na", "none", "null",
            "not available", "not specified", "not disclosed",
            "not publicly disclosed", "not publicly available",
            "no information", "no data", "no info",
            "tbd", "tba", "-", "—", "?",
        }
        stripped = 0
        for record in record_set.records:
            for bucket in (record.attributes, record.additional_attributes):
                for key in list(bucket.keys()):
                    av = bucket[key]
                    keep_vals = []
                    for sv in getattr(av, "values", []) or []:
                        raw = (sv.value or "").strip()
                        if raw and raw.casefold() not in placeholder_set:
                            keep_vals.append(sv)
                    if not keep_vals:
                        bucket.pop(key)
                        stripped += 1
                    elif len(keep_vals) != len(av.values):
                        av.values = keep_vals
        stats["placeholders"] = stripped

        # ── 5. Dedupe + prune orphan schema attributes ────────────────
        # Schema can accumulate duplicates (e.g. case-insensitive
        # collisions, a `promote` of a name already present, or a
        # `rename` whose old entry survived). Collapse on case-folded
        # name, keeping the first occurrence's casing.
        seen_names: dict[str, int] = {}
        deduped_attrs = []
        for sa in record_set.schema_attributes:
            key = (sa.name or "").strip().casefold()
            if not key or key in seen_names:
                continue
            seen_names[key] = len(deduped_attrs)
            deduped_attrs.append(sa)
        dup_removed = len(record_set.schema_attributes) - len(deduped_attrs)

        # Now drop schema attributes that no record populates after the
        # placeholder strip. Without this, the dashboard prints empty
        # columns even when every value was "Unknown".
        def _populated(rec, name: str) -> bool:
            for bucket_name in ("attributes", "additional_attributes"):
                av = getattr(rec, bucket_name, {}).get(name)
                if av is None:
                    continue
                for sv in getattr(av, "values", []) or []:
                    raw = (sv.value or "").strip()
                    if raw and raw.casefold() not in placeholder_set:
                        return True
            return False

        populated_attrs = [
            sa for sa in deduped_attrs
            if any(_populated(r, sa.name) for r in record_set.records)
        ]
        orphan_pruned = len(deduped_attrs) - len(populated_attrs)
        record_set.schema_attributes = populated_attrs
        stats["pruned_attrs"] = dup_removed + orphan_pruned

        # Refresh frequencies so downstream consumers see consistent counts.
        try:
            record_set.update_schema_frequencies()
        except Exception:  # noqa: BLE001
            pass

        if any(stats.values()):
            logger.info(
                "Finalization: %d labels uppercased, %d aliases merged, "
                "%d attributes unit-tagged, %d placeholder values stripped, "
                "%d schema attributes pruned (%d dup, %d orphan)",
                stats["labels"], stats["aliases"], stats["units"],
                stats["placeholders"], stats["pruned_attrs"],
                dup_removed, orphan_pruned,
            )
        return stats

    def _infer_unit_renames(self, record_set: RecordSet) -> dict[str, str]:
        """For each schema attribute, decide if it should be renamed ``Name (unit)``."""
        renames: dict[str, str] = {}
        for attr in record_set.schema_attributes:
            current_name = attr.name
            # Skip if attribute already includes a parenthesized unit.
            if "(" in current_name and current_name.rstrip().endswith(")"):
                continue
            unit_counts: dict[str, int] = defaultdict(int)
            numeric_count = 0
            total_count = 0
            for record in record_set.records:
                attr_val = record.attributes.get(current_name)
                if not attr_val:
                    continue
                for sv in attr_val.values:
                    raw = (sv.value or "").strip()
                    if not raw:
                        continue
                    total_count += 1
                    parsed = self._parse_numeric_with_unit(raw)
                    if parsed is None:
                        continue
                    numeric_count += 1
                    unit = parsed[1]
                    if unit:
                        unit_counts[unit] += 1
            # Only rename if a clear majority of values are numeric and one
            # canonical unit dominates.
            if total_count < 3 or numeric_count < max(3, int(total_count * 0.7)):
                continue
            if not unit_counts:
                continue
            top_unit, top_count = max(unit_counts.items(), key=lambda kv: kv[1])
            if top_count < max(2, int(numeric_count * 0.6)):
                continue
            new_name = f"{current_name} ({top_unit})"
            renames[current_name] = new_name
        return renames

    def _rename_attribute_with_unit_strip(
        self,
        record_set: RecordSet,
        old_name: str,
        new_name: str,
    ) -> None:
        """Rename ``old_name`` → ``new_name`` and strip units from values."""
        for attr in record_set.schema_attributes:
            if attr.name == old_name:
                attr.name = new_name
                break
        for record in record_set.records:
            for bucket in (record.attributes, record.additional_attributes):
                if old_name not in bucket:
                    continue
                attr_val = bucket.pop(old_name)
                for sv in attr_val.values:
                    parsed = self._parse_numeric_with_unit(sv.value or "")
                    if parsed is not None:
                        sv.value = parsed[0]
                bucket[new_name] = attr_val

    @classmethod
    def _parse_numeric_with_unit(cls, raw: str) -> Optional[tuple[str, str]]:
        """Return ``(numeric_string, canonical_unit)`` or None if not numeric."""
        if not raw:
            return None
        match = cls._NUMERIC_VALUE_RE.match(raw.replace("\u00a0", " "))
        if not match:
            return None
        num = match.group("num").replace(",", "")
        try:
            float(num)
        except ValueError:
            return None
        unit_raw = (match.group("unit") or "").strip().lower()
        # Strip trailing punctuation/parentheses.
        unit_raw = unit_raw.rstrip(").,;").lstrip("(")
        if not unit_raw:
            return (num, "")
        canonical = cls._UNIT_SUFFIXES.get(unit_raw)
        if canonical is None:
            # Allow simple compound like "km²" matched above; otherwise reject.
            return None
        return (num, canonical)

    def decompose_compound_values(
        self,
        record_set: RecordSet,
        decompositions: list[dict],
    ) -> int:
        """
        Decompose compound attribute values into atomic values across attributes.
        
        When a value like "Open-source mobile app" is really two dimensions
        (license + deployment model), this splits it: the source attribute gets
        one piece and the target attribute gets the other.
        
        Args:
            record_set: The record set to modify
            decompositions: List of dicts, each with:
                - source_attribute: attribute containing the compound value
                - compound_value: the multi-dimensional value to decompose
                - replacements: list of {attribute, value} dicts for the atomic pieces
                
        Returns:
            Count of records changed
        """
        changes = 0
        schema_attr_names = {a.name for a in record_set.schema_attributes}
        
        for decomp in decompositions:
            src_attr = decomp.get("source_attribute", "")
            compound_val = decomp.get("compound_value", "")
            replacements = decomp.get("replacements", [])
            
            if not src_attr or not compound_val or not replacements:
                continue
            
            for record in record_set.records:
                attr_val = record.attributes.get(src_attr)
                if not attr_val:
                    continue
                
                # Check if any SourcedValue matches the compound value
                matched_sv = None
                for sv in attr_val.values:
                    if sv.value.lower() == compound_val.lower():
                        matched_sv = sv
                        break
                
                if not matched_sv:
                    continue
                
                # Apply each replacement
                for repl in replacements:
                    target_attr = repl.get("attribute", "")
                    target_val = repl.get("value", "")
                    if not target_attr or not target_val:
                        continue
                    
                    if target_attr == src_attr:
                        # Replace the compound value in-place on the source attribute
                        matched_sv.value = target_val
                    else:
                        # Set value on a different attribute
                        is_schema = target_attr in schema_attr_names
                        target_av = record.attributes.get(target_attr) if is_schema else record.additional_attributes.get(target_attr)
                        if not target_av or not target_av.value:
                            # Only set if target is empty (don't overwrite existing data)
                            new_av = AttributeValue()
                            new_av.add_value(target_val)
                            new_av.values[0].sources = list(matched_sv.sources)
                            record.set_attribute(target_attr, new_av, is_schema_attr=is_schema)
                
                changes += 1
        
        if changes:
            logger.info(f"Decomposed compound values in {changes} records")
        return changes

    async def check_duplicates(
        self,
        new_labels: list[str],
        existing_labels: list[str],
        use_fuzzy: bool = True,
        fuzzy_threshold: int = 85,
        existing_aliases: dict[str, list[str]] | None = None,
    ) -> dict[str, str]:
        """
        Check for semantic duplicates between new and existing labels.
        
        Uses a two-stage approach:
        1. Fast fuzzy matching (rapidfuzz) for obvious matches
        2. LLM-based semantic matching for uncertain cases (optional)
        
        Args:
            new_labels: Labels to check
            existing_labels: Existing labels in the dataset
            use_fuzzy: If True, use fuzzy matching first (default: True)
            fuzzy_threshold: Threshold for fuzzy matching (0-100)
            existing_aliases: Optional dict mapping labels to their aliases
            
        Returns:
            Mapping of new_label -> existing_label for duplicates
        """
        if not new_labels or not existing_labels:
            return {}
        
        duplicates = {}
        uncertain_labels = []
        
        # Stage 1: Fast fuzzy matching
        if use_fuzzy:
            for new_label in new_labels:
                matched_label, score = find_fuzzy_match(
                    new_label,
                    existing_labels,
                    threshold=fuzzy_threshold,
                    include_aliases=existing_aliases,
                )
                if matched_label:
                    duplicates[new_label] = matched_label
                    logger.debug(f"Fuzzy match: {new_label} -> {matched_label} (score: {score})")
                elif score > fuzzy_threshold - 20:
                    # Close to threshold - defer to LLM
                    uncertain_labels.append(new_label)
                # else: no match, skip
        else:
            uncertain_labels = new_labels
        
        # Stage 2: LLM-based semantic matching for uncertain cases
        if uncertain_labels and existing_labels:
            result = await self.llm.structured_completion(
                prompt=prompts.ENTITY_DEDUPLICATION,
                response_format=schemas.get_entity_deduplication_schema(),
                variables={
                    "new_labels": "\n".join(uncertain_labels),
                    "existing_labels": "\n".join(existing_labels),
                }
            )
            
            for item in result.get("duplicates", []):
                new_label = item.get("new_label", "")
                existing_label = item.get("existing_label", "")
                confidence = item.get("confidence", 0)
                
                if confidence >= self.config.dedup_similarity_threshold:
                    duplicates[new_label] = existing_label
                    logger.info(f"LLM duplicate: {new_label} -> {existing_label} ({confidence:.2f})")
        
        return duplicates
    
    def evolve_schema(self, record_set: RecordSet) -> list[SchemaAttribute]:
        """
        Evolve the schema based on attribute frequencies.
        
        Promotes frequent additional attributes to schema attributes,
        and demotes infrequent schema attributes.
        
        Returns:
            List of newly promoted SchemaAttribute objects (for query generation)
        """
        if not record_set.records:
            return []
        
        # Count attribute occurrences
        attr_counts: dict[str, int] = defaultdict(int)
        
        for record in record_set.records:
            for name, attr in record.attributes.items():
                if attr.value:
                    attr_counts[name] += 1
            for name, attr in record.additional_attributes.items():
                if attr.value:
                    attr_counts[name] += 1
        
        total_records = len(record_set.records)
        threshold_count = total_records * self.config.schema_inclusion_threshold
        
        # Determine which attributes should be in schema
        qualified_attrs = [
            (name, count / total_records)
            for name, count in attr_counts.items()
            if count >= threshold_count
        ]
        
        # Sort by frequency and limit
        qualified_attrs.sort(key=lambda x: x[1], reverse=True)
        qualified_attrs = qualified_attrs[:self.config.parameter_limit]
        
        # Update schema
        new_schema_names = {name for name, _ in qualified_attrs}
        old_schema_names = {a.name for a in record_set.schema_attributes}
        
        # Track newly added attributes
        new_attributes: list[SchemaAttribute] = []
        
        # Add new attributes
        for name, freq in qualified_attrs:
            if name not in old_schema_names:
                new_attr = SchemaAttribute(
                    name=name,
                    frequency=freq,
                )
                record_set.schema_attributes.append(new_attr)
                new_attributes.append(new_attr)
                logger.info(f"Promoted to schema: {name} ({freq:.1%})")
        
        # Update frequencies
        for attr in record_set.schema_attributes:
            attr.frequency = attr_counts.get(attr.name, 0) / total_records
        
        # Align records with new schema
        self._align_records_with_schema(record_set, new_schema_names)
        
        return new_attributes
    
    def _align_records_with_schema(self, record_set: RecordSet, schema_names: set[str]):
        """Align all records with the current schema."""
        for record in record_set.records:
            # Promote additional attributes to schema if qualified
            to_promote = []
            for name, attr in record.additional_attributes.items():
                if name in schema_names:
                    to_promote.append((name, attr))
            
            for name, attr in to_promote:
                if name not in record.attributes:
                    record.attributes[name] = attr
                del record.additional_attributes[name]
            
            # Demote schema attributes if no longer qualified
            to_demote = []
            for name, attr in record.attributes.items():
                if name not in schema_names and name != "label":
                    to_demote.append((name, attr))
            
            for name, attr in to_demote:
                record.additional_attributes[name] = attr
                del record.attributes[name]

    async def suggest_schema(self, category: str, guidance: str, max_attributes: int = 5) -> list[SchemaAttribute]:
        """
        Proactively suggest schema attributes for a category.
        
        Limits to a small number of core attributes to ensure focused,
        high-quality extraction rather than sparse broad extraction.
        
        Args:
            category: The entity category
            guidance: User guidance
            max_attributes: Maximum attributes to suggest (default: 5)
            
        Returns:
            List of suggested schema attributes
        """
        result = await self.llm.structured_completion(
            prompt=prompts.SCHEMA_SUGGESTION,
            response_format=schemas.get_schema_suggestion_schema(),
            variables={
                "category": category,
                "guidance": guidance or "",
            }
        )
        
        suggestions = []
        high_importance = []
        medium_importance = []
        
        for item in result.get("attributes", []):
            attr = SchemaAttribute(
                name=item.get("name", ""),
                description=item.get("description", ""),
                required=item.get("importance") == "high",
            )
            if item.get("importance") == "high":
                high_importance.append(attr)
            else:
                medium_importance.append(attr)
        
        # Take high importance first, fill remainder with medium
        suggestions = high_importance[:max_attributes]
        remaining = max_attributes - len(suggestions)
        if remaining > 0:
            suggestions.extend(medium_importance[:remaining])
        
        logger.info(f"Suggested {len(suggestions)} core schema attributes")
        return suggestions

    async def generate_attribute_values(
        self, 
        category: str, 
        guidance: str,
        attributes: list[SchemaAttribute],
        cardinality_threshold: int = 50,
    ) -> list[SchemaAttribute]:
        """
        Generate provisional values for schema attributes.
        
        Classifies each attribute as closed (finite) or open (unbounded) set
        based on the cardinality of generated values.
        
        Args:
            category: The entity category
            guidance: User guidance
            attributes: Schema attributes to generate values for
            cardinality_threshold: Max values for closed set classification
            
        Returns:
            Updated attributes with provisional values
        """
        if not attributes:
            return []
        
        attr_descriptions = "\n".join(
            f"- {a.name}: {a.description or 'No description'}"
            for a in attributes
        )
        
        result = await self.llm.structured_completion(
            prompt=prompts.ATTRIBUTE_VALUES_GENERATION,
            response_format=schemas.get_attribute_values_schema(),
            variables={
                "category": category,
                "guidance": guidance or "",
                "attributes": attr_descriptions,
            }
        )
        
        # Update attributes with provisional values
        attr_map = {a.name.lower(): a for a in attributes}
        
        for item in result.get("attributes", []):
            name = item.get("name", "")
            attr = attr_map.get(name.lower())
            if attr:
                attr.provisional_values = item.get("values", [])
                attr.cardinality_threshold = cardinality_threshold
                
                # Only update is_closed_set if user didn't explicitly set canonical_values
                # (canonical_values implies closed-set, don't let LLM override that)
                if not attr.canonical_values:
                    attr.is_closed_set = item.get("is_closed_set", False)
                    
                    # Override LLM classification if values exceed threshold
                    if len(attr.provisional_values) > cardinality_threshold:
                        attr.is_closed_set = False
                
                logger.info(
                    f"Attribute '{attr.name}': {len(attr.provisional_values)} values, "
                    f"{'closed' if attr.is_closed_set else 'open'} set"
                )
        
        return attributes


# ============================================================================
# Fuzzy Label Matching
# ============================================================================

# ----- Exclusion rules -----

_MISSING_TOKENS = {"", "n/a", "na", "none", "null", "unknown", "-", "—"}


def _rule_kind(rule: dict) -> str:
    """Return 'attribute' if the rule targets an attribute, else 'label'."""
    if not isinstance(rule, dict):
        return "label"
    if rule.get("attribute"):
        return "attribute"
    return rule.get("kind") or "label"


def _normalize_values(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    try:
        return [str(v) for v in values]
    except Exception:  # noqa: BLE001
        return []


def format_exclusion_rule(rule: dict) -> str:
    """Render an exclusion rule as a single-line human-readable string."""
    if not isinstance(rule, dict):
        return ""
    reason = (rule.get("reason") or "").strip()
    if _rule_kind(rule) == "attribute":
        attr = (rule.get("attribute") or "").strip()
        if not attr:
            return ""
        op = (rule.get("operator") or "equals").lower()
        values = _normalize_values(rule.get("values"))
        if op == "missing":
            body = f"`{attr}` is missing or unknown"
        elif op == "equals" and values:
            body = f"`{attr}` equals \"{values[0]}\""
        elif op == "in" and values:
            body = f"`{attr}` is one of [{', '.join(values)}]"
        elif op == "contains" and values:
            body = f"`{attr}` contains \"{values[0]}\""
        elif op == "regex" and values:
            body = f"`{attr}` matches regex /{values[0]}/"
        else:
            body = f"`{attr}` {op} {values}"
        return f"{body}" + (f" — {reason}" if reason else "")
    label = (rule.get("label") or "").strip()
    if not label:
        return ""
    return f"{label}" + (f" — {reason}" if reason else "")


def _attribute_value_strings(record, attribute: str) -> list[str]:
    """Return all string values an attribute holds on a record (multi-valued aware)."""
    out: list[str] = []
    for bucket_name in ("attributes", "additional_attributes"):
        bucket = getattr(record, bucket_name, {}) or {}
        av = bucket.get(attribute)
        if av is None:
            continue
        val = getattr(av, "value", None)
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            out.extend(str(v) for v in val if v is not None)
        else:
            out.append(str(val))
    return out


def record_matches_rule(record, rule: dict) -> bool:
    """Check whether a record satisfies a single exclusion rule."""
    if not isinstance(rule, dict):
        return False
    if _rule_kind(rule) == "label":
        target = (rule.get("label") or "").strip().casefold()
        if not target:
            return False
        labels = [getattr(record, "label", "")] + list(
            getattr(record, "aliases", []) or []
        )
        return any((s or "").strip().casefold() == target for s in labels)

    attr = (rule.get("attribute") or "").strip()
    if not attr:
        return False
    op = (rule.get("operator") or "equals").lower()
    values = [str(v).strip() for v in _normalize_values(rule.get("values"))]
    folded = [v.casefold() for v in values]

    raw_vals = _attribute_value_strings(record, attr)
    stripped = [v.strip() for v in raw_vals]

    if op == "missing":
        if not stripped:
            return True
        return all(v.strip().casefold() in _MISSING_TOKENS for v in stripped)
    if not stripped:
        return False
    if op == "equals":
        return any(v.casefold() == folded[0] for v in stripped) if folded else False
    if op == "in":
        if not folded:
            return False
        return any(v.casefold() in folded for v in stripped)
    if op == "contains":
        if not folded:
            return False
        return any(folded[0] in v.casefold() for v in stripped)
    if op == "regex":
        if not values:
            return False
        try:
            import re
            pat = re.compile(values[0], re.IGNORECASE)
        except re.error:
            return False
        return any(pat.search(v) for v in stripped)
    return False


def normalize_label(label: str) -> str:
    """
    Normalize a label for fuzzy matching.
    
    - Strips whitespace
    - Converts to uppercase
    - Removes common suffixes like "Inc.", "LLC", etc.
    """
    import re
    
    normalized = label.strip().upper()
    
    # Remove common corporate suffixes
    suffixes = [
        r'\bINC\.?$', r'\bLLC\.?$', r'\bLTD\.?$', r'\bCORP\.?$',
        r'\bCORPORATION$', r'\bCOMPANY$', r'\bCO\.?$',
        r'\bFOUNDATION$', r'\bORG\.?$', r'\bORGANIZATION$',
        r'\bPROJECT$', r'\bINITIATIVE$', r'\bPLATFORM$',
    ]
    for suffix in suffixes:
        normalized = re.sub(suffix, '', normalized).strip()
    
    # Remove trailing punctuation
    normalized = re.sub(r'[,.\-\s]+$', '', normalized)
    
    return normalized


def _length_ratio_ok(a: str, b: str, *, min_ratio: float = 0.6, min_len: int = 10) -> bool:
    """Return True iff both strings meet a minimum length AND the shorter is
    at least ``min_ratio`` of the longer's length.

    Used to gate substring-style fuzzy scorers (``partial_ratio``,
    ``token_set_ratio``) so a short specific name (e.g. "MEMEX") cannot be
    silently absorbed into a long category-like phrase that happens to
    contain it (e.g. "WEB SCRAPING ARCHIVES: MEMEX, TELLFINDER").
    """
    la, lb = len(a), len(b)
    if la < min_len or lb < min_len:
        return False
    shorter = min(la, lb)
    longer = max(la, lb)
    if longer == 0:
        return False
    return (shorter / longer) >= min_ratio


def find_fuzzy_match(
    label: str,
    existing_labels: list[str],
    threshold: int = 85,
    include_aliases: dict[str, list[str]] | None = None,
) -> tuple[str | None, int]:
    """
    Find a fuzzy match for a label among existing labels.
    
    Uses rapidfuzz for efficient fuzzy string matching with multiple algorithms:
    - Token sort ratio (handles word order differences)
    - Token set ratio (handles subset relationships)
    - Ratio (standard Levenshtein)
    
    Args:
        label: The label to find a match for
        existing_labels: List of existing labels to match against
        threshold: Minimum similarity score (0-100) to consider a match
        include_aliases: Optional dict mapping labels to their aliases for expanded matching
        
    Returns:
        Tuple of (matched_label, score) or (None, 0) if no match found
    """
    if not existing_labels:
        return None, 0
    
    normalized_label = normalize_label(label)
    
    # Build search space: existing labels + their aliases
    search_space: list[tuple[str, str]] = []  # (normalized, original_label)
    for existing in existing_labels:
        search_space.append((normalize_label(existing), existing))
        if include_aliases and existing in include_aliases:
            for alias in include_aliases[existing]:
                search_space.append((normalize_label(alias), existing))
    
    if not search_space:
        return None, 0
    
    # Use multiple fuzzy matching strategies and take the best
    best_match = None
    best_score = 0
    best_original = None
    
    for normalized_existing, original_label in search_space:
        # Standard ratio
        score1 = fuzz.ratio(normalized_label, normalized_existing)
        
        # Token sort ratio - handles word order differences
        # e.g., "Polaris Project" vs "Project Polaris"
        score2 = fuzz.token_sort_ratio(normalized_label, normalized_existing)
        
        # Token set ratio - handles partial matches and extra words
        # e.g., "Polaris" vs "Polaris Project"
        # Only use if both strings are reasonably long to avoid false positives
        score3 = 0
        if _length_ratio_ok(normalized_label, normalized_existing):
            score3 = fuzz.token_set_ratio(normalized_label, normalized_existing)
        
        # Partial ratio - only use for longer strings AND when lengths are
        # comparable (length-ratio guard) so short specific names cannot be
        # absorbed into long category-like phrases that contain them.
        score4 = 0
        if _length_ratio_ok(normalized_label, normalized_existing):
            score4 = fuzz.partial_ratio(normalized_label, normalized_existing)
        
        # Take the maximum score across strategies
        max_score = max(score1, score2, score3, score4)
        
        if max_score > best_score:
            best_score = max_score
            best_match = normalized_existing
            best_original = original_label
    
    if best_score >= threshold:
        return best_original, int(best_score)
    
    return None, 0


def find_all_fuzzy_matches(
    labels: list[str],
    threshold: int = 85,
) -> list[tuple[str, str, int]]:
    """
    Find all fuzzy matches among a list of labels.
    
    Returns groups of labels that are likely referring to the same entity.
    
    Args:
        labels: List of labels to check for duplicates
        threshold: Minimum similarity score (0-100) to consider a match
        
    Returns:
        List of (label1, label2, score) tuples for each match found
    """
    matches = []
    normalized = [(normalize_label(l), l) for l in labels]
    
    for i, (norm1, orig1) in enumerate(normalized):
        for norm2, orig2 in normalized[i + 1:]:
            # Calculate similarity with conservative approach
            score1 = fuzz.ratio(norm1, norm2)
            score2 = fuzz.token_sort_ratio(norm1, norm2)
            # Only use partial matching for longer strings
            score3 = 0
            if _length_ratio_ok(norm1, norm2):
                score3 = fuzz.token_set_ratio(norm1, norm2)
            score4 = 0
            if _length_ratio_ok(norm1, norm2):
                score4 = fuzz.partial_ratio(norm1, norm2)
            
            score = max(score1, score2, score3, score4)
            
            if score >= threshold:
                matches.append((orig1, orig2, score))
    
    return matches


def cluster_fuzzy_matches(
    labels: list[str],
    threshold: int = 85,
    do_not_merge: set[frozenset[str]] | None = None,
) -> list[set[str]]:
    """
    Cluster labels that are fuzzy matches into groups.
    
    Uses union-find to group transitively related labels.
    
    Args:
        labels: List of labels to cluster
        threshold: Minimum similarity score (0-100) to consider a match
        do_not_merge: Optional set of frozensets of label pairs that must not
            be unioned (user-curated constraints). Any candidate union that
            would place both members of a forbidden pair in the same cluster
            is skipped.
        
    Returns:
        List of sets, each containing labels that refer to the same entity
    """
    matches = find_all_fuzzy_matches(labels, threshold)
    
    if not matches:
        return []
    
    forbidden = do_not_merge or set()
    
    # Build clusters using union-find
    parent = {label: label for label in labels}
    members: dict[str, set[str]] = {label: {label} for label in labels}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def would_violate(set_a: set[str], set_b: set[str]) -> bool:
        for pair in forbidden:
            if len(pair) != 2:
                continue
            a, b = tuple(pair)
            if (a in set_a and b in set_b) or (b in set_a and a in set_b) \
               or (a in set_a and b in set_a) or (a in set_b and b in set_b):
                return True
        return False
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if would_violate(members[px], members[py]):
            return
        parent[px] = py
        members[py] |= members[px]
        members[px] = members[py]
    
    for label1, label2, _ in matches:
        union(label1, label2)
    
    # Group by root
    clusters = defaultdict(set)
    for label in labels:
        root = find(label)
        clusters[root].add(label)
    
    # Return only clusters with more than one member
    return [cluster for cluster in clusters.values() if len(cluster) > 1]
