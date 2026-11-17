# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""The Build Entity Dataset API as one object.

:class:`BuildEntityDataset` holds a :class:`~itk_entities.state.Session` and
forwards every call to the stage module that implements it. The stage
modules are plain functions over the session and can be called directly;
this class exists so an application can hold one object and poll it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from . import (
    audits,
    auto_runs,
    config,
    curation,
    exclusions,
    inputs,
    recategorize,
    refine,
    release,
    research,
    review,
    schema,
    state,
    storage,
    value_edits,
)
from .state import ResearchProgress, Session, UsageStats


class BuildEntityDataset:
    """Build, curate, audit and release an entity dataset researched from the web.

    Research runs in a background thread so a UI can poll ``progress``
    without blocking. Pass ``runs_dir`` to keep saved runs somewhere other
    than the default cache folder.
    """

    DEFAULT_SAFETY_PROMPT = audits.DEFAULT_SAFETY_PROMPT
    DEFAULT_ALIAS_SUGGEST_PROMPT = curation.DEFAULT_ALIAS_SUGGEST_PROMPT

    def __init__(self, runs_dir: Path | None = None) -> None:
        self.session = (
            Session() if runs_dir is None else Session(runs_dir=Path(runs_dir))
        )

    # -- live state ------------------------------------------------------

    @property
    def progress(self) -> ResearchProgress:
        return self.session.progress

    @progress.setter
    def progress(self, value: ResearchProgress) -> None:
        self.session.progress = value

    @property
    def usage(self) -> UsageStats:
        return self.session.usage

    @usage.setter
    def usage(self, value: UsageStats) -> None:
        self.session.usage = value

    @property
    def review_verification_progress(self) -> ResearchProgress:
        return self.session.review_verification_progress

    @property
    def recat_added(self) -> list[dict]:
        return self.session.recat_added

    @recat_added.setter
    def recat_added(self, value: list[dict]) -> None:
        self.session.recat_added = value

    @property
    def relevance_results(self) -> dict:
        return self.session.relevance_results

    @relevance_results.setter
    def relevance_results(self, value: dict) -> None:
        self.session.relevance_results = value

    # -- operations, in the order of the stage modules -------------------

    @property
    def is_running(self) -> bool:
        return state.is_running(self.session)

    @property
    def is_verifying_review(self) -> bool:
        return state.is_verifying_review(self.session)

    @property
    def review_verification_report(self) -> dict:
        return review.review_verification_report(self.session)

    @property
    def review_dataset_fingerprint(self) -> str:
        return review.review_dataset_fingerprint(self.session)

    def list_review_verifications(self) -> list[dict]:
        """List persisted captures for the loaded dataset, without changing it."""
        return review.list_review_verifications(self.session)

    def open_review_verification(self, directory: str) -> None:
        return review.open_review_verification(self.session, directory)

    def review_verification_bundle(self) -> bytes:
        return review.review_verification_bundle(self.session)

    def stop_review_verification(self) -> None:
        return review.stop_review_verification(self.session)

    def start_review_verification(
        self,
        proposals: list[dict],
        *,
        backend: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        concurrency: int = 4,
        max_sources: int = 12,
        resume_directory: str | None = None,
        retry_failed_sources: bool = False,
    ) -> None:
        """Verify an immutable snapshot, including loaded read-only datasets."""
        return review.start_review_verification(
            self.session,
            proposals,
            backend=backend,
            api_key=api_key,
            model=model,
            concurrency=concurrency,
            max_sources=max_sources,
            resume_directory=resume_directory,
            retry_failed_sources=retry_failed_sources,
        )

    def resume_review_verification(
        self,
        directory: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        concurrency: int = 4,
        retry_failed_sources: bool = False,
    ) -> None:
        return review.resume_review_verification(
            self.session,
            directory,
            api_key=api_key,
            model=model,
            concurrency=concurrency,
            retry_failed_sources=retry_failed_sources,
        )

    @property
    def dataset_json(self) -> dict | None:
        return self.session.dataset_json

    @property
    def dataframe(self) -> pd.DataFrame | None:
        return self.session.df

    @property
    def run_dir(self) -> Path | None:
        return self.session.run_dir

    def current_dataframe(self) -> pd.DataFrame | None:
        """Return a DataFrame snapshot of the live record set."""
        return state.current_dataframe(self.session)

    @property
    def schema_attributes(self) -> list[dict]:
        return state.schema_attributes(self.session)

    def refresh_progress(self) -> None:
        """Pull live counters from the running Schemify instance."""
        return state.refresh_progress(self.session)

    def reset(self) -> None:
        """Drop all in-memory state."""
        return state.reset(self.session)

    def start_research(
        self,
        api_key: str,
        category: str,
        guidance: str = "",
        schema_attributes: list[dict] | None = None,
        max_queries: int = 30,
        concurrency: int = 5,
        model: str = config.DEFAULT_MODEL,
        budget: float = 10.0,
        verify: bool = True,
        phase_split: tuple[float, float, float] = (0.6, 0.2, 0.2),
        seed_labels: list[str] | None = None,
        complete: bool = False,
    ) -> None:
        """Start research in a daemon background thread."""
        return research.start_research(
            self.session,
            api_key,
            category,
            guidance,
            schema_attributes,
            max_queries,
            concurrency,
            model,
            budget,
            verify,
            phase_split,
            seed_labels,
            complete,
        )

    def stop_research(self) -> None:
        """Snapshot current state and mark as stopped (thread runs to completion)."""
        return research.stop_research(self.session)

    @staticmethod
    def reference_labels_from_file(filename: str, raw: bytes) -> list[str]:
        """Parse an uploaded reference dataset and return the entity labels."""
        return inputs.reference_labels_from_file(filename, raw)

    def propose_search_languages(
        self,
        api_key: str,
        category: str,
        guidance: str = "",
        model: str = config.DEFAULT_MODEL,
    ) -> list[dict]:
        """Synchronous wrapper around :func:`multilingual.propose_search_languages`."""
        return research.propose_search_languages(
            self.session, api_key, category, guidance, model
        )

    def start_auto_mode(
        self,
        api_key: str,
        category: str,
        guidance: str = "",
        schema_attributes: list[dict] | None = None,
        *,
        reference_labels: list[str] | None = None,
        search_languages: list[str] | None = None,
        target_language: str = "English",
        max_iterations: int = 5,
        per_iter_query_budget: int = 30,
        normalize_every: int = 3,
        min_iterations: int = 2,
        concurrency: int = 5,
        model: str = config.DEFAULT_MODEL,
        budget: float = 10.0,
    ) -> None:
        """Run auto mode in a daemon background thread."""
        return auto_runs.start_auto_mode(
            self.session,
            api_key,
            category,
            guidance,
            schema_attributes,
            reference_labels=reference_labels,
            search_languages=search_languages,
            target_language=target_language,
            max_iterations=max_iterations,
            per_iter_query_budget=per_iter_query_budget,
            normalize_every=normalize_every,
            min_iterations=min_iterations,
            concurrency=concurrency,
            model=model,
            budget=budget,
        )

    def stop_auto_mode(self) -> None:
        """Ask the auto-mode worker to stop at the next iteration boundary."""
        return auto_runs.stop_auto_mode(self.session)

    def start_rediscovery_benchmark(
        self,
        api_key: str,
        category: str,
        guidance: str = "",
        schema_attributes: list[dict] | None = None,
        *,
        reference_labels: list[str],
        search_languages: list[str] | None = None,
        target_language: str = "English",
        max_iterations: int = 4,
        per_iter_query_budget: int = 25,
        gap_fill_query_budget: int = 200,
        gap_fill_cost_reserve_usd: float = 25.0,
        normalize_every: int = 3,
        min_iterations: int = 2,
        concurrency: int = 5,
        model: str = config.DEFAULT_MODEL,
        budget: float = 50.0,
    ) -> None:
        """Run a blind-rediscovery + gap-fill benchmark in a daemon thread."""
        return auto_runs.start_rediscovery_benchmark(
            self.session,
            api_key,
            category,
            guidance,
            schema_attributes,
            reference_labels=reference_labels,
            search_languages=search_languages,
            target_language=target_language,
            max_iterations=max_iterations,
            per_iter_query_budget=per_iter_query_budget,
            gap_fill_query_budget=gap_fill_query_budget,
            gap_fill_cost_reserve_usd=gap_fill_cost_reserve_usd,
            normalize_every=normalize_every,
            min_iterations=min_iterations,
            concurrency=concurrency,
            model=model,
            budget=budget,
        )

    def can_continue_research(self) -> bool:
        """True iff the in-memory Schemify state is alive enough to keep researching."""
        return state.can_continue_research(self.session)

    @property
    def is_read_only(self) -> bool:
        """True when the loaded dataset has no live Schemify instance."""
        return state.is_read_only(self.session)

    @property
    def read_only_reason(self) -> str | None:
        """Human-readable explanation of why the dataset is read-only, or None."""
        return state.read_only_reason(self.session)

    def continue_research(
        self,
        max_queries: int = 30,
        concurrency: int = 5,
        verify: bool = False,
        phase_split: tuple[float, float, float] = (0.2, 0.2, 0.6),
    ) -> bool:
        """Run additional research on top of the existing record set."""
        return research.continue_research(
            self.session, max_queries, concurrency, verify, phase_split
        )

    def count_unverified(self) -> tuple[int, int]:
        """Return ``(entities_with_unverified, total_unverified_values)``."""
        return state.count_unverified(self.session)

    def start_verification(self, concurrency: int = 12) -> None:
        """Run verification in a background daemon thread."""
        return refine.start_verification(self.session, concurrency)

    def append_guidance(self, note: str) -> bool:
        """Append a free-form user note to the record set's research guidance."""
        return research.append_guidance(self.session, note)

    @staticmethod
    def parse_candidate_file(filename: str, raw: bytes) -> list[str]:
        """Best-effort extraction of candidate names from an uploaded file."""
        return inputs.parse_candidate_file(filename, raw)

    def add_candidate_entities(self, names: list[str]) -> int:
        """Add user-supplied entity labels as blank seed records."""
        return research.add_candidate_entities(self.session, names)

    @property
    def exclusions(self) -> list[dict]:
        """Current user-supplied exclusion rules (label-based or attribute predicates)."""
        return exclusions.exclusions(self.session)

    def add_label_exclusion(
        self, label: str, reason: str = "", remove_existing: bool = True
    ) -> tuple[bool, int]:
        """Exclude a single named entity. See add_attribute_exclusion for predicate rules."""
        return exclusions.add_label_exclusion(
            self.session, label, reason, remove_existing
        )

    def add_exclusion(
        self, label: str, reason: str = "", remove_existing: bool = True
    ) -> tuple[bool, int]:
        return exclusions.add_exclusion(self.session, label, reason, remove_existing)

    def add_attribute_exclusion(
        self,
        attribute: str,
        operator: str,
        values: list[str] | None = None,
        reason: str = "",
        remove_existing: bool = True,
    ) -> tuple[bool, int]:
        """Exclude every entity that satisfies the given attribute predicate."""
        return exclusions.add_attribute_exclusion(
            self.session, attribute, operator, values, reason, remove_existing
        )

    def remove_exclusion(self, identifier: str | dict) -> bool:
        """Remove an exclusion rule. ``identifier`` may be a label string."""
        return exclusions.remove_exclusion(self.session, identifier)

    def rename_attribute(self, old_name: str, new_name: str) -> int:
        """Rename a schema attribute and update all records. Returns affected records."""
        return schema.rename_attribute(self.session, old_name, new_name)

    def remove_attribute(self, name: str) -> int:
        """Remove a schema attribute from the schema and all records."""
        return schema.remove_attribute(self.session, name)

    def start_normalize(self, attributes: list[str] | None = None) -> None:
        """Run schemify.normalize in a background thread for the given attributes."""
        return refine.start_normalize(self.session, attributes)

    def scan_harmful_content(
        self,
        prompt_template: str | None = None,
        *,
        concurrency: int = audits.SAFETY_SCAN_CONCURRENCY,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Use an LLM to flag potentially harmful values across records."""
        return audits.scan_harmful_content(
            self.session,
            prompt_template,
            concurrency=concurrency,
            progress_callback=progress_callback,
        )

    def remove_record_by_label(self, label: str) -> bool:
        """Drop a single record by exact label (case-insensitive). Returns True if removed."""
        return curation.remove_record_by_label(self.session, label)

    def list_alias_groups(self, only_with_aliases: bool = False) -> list[dict]:
        """Return every record's canonical label + its alias group."""
        return curation.list_alias_groups(self.session, only_with_aliases)

    @property
    def do_not_merge_pairs(self) -> list[list[str]]:
        """User-asserted pairs that must never be auto-merged again."""
        return curation.do_not_merge_pairs(self.session)

    def rename_record_canonical(self, current_label: str, new_canonical: str) -> bool:
        """Promote ``new_canonical`` to be the canonical label of the record."""
        return curation.rename_record_canonical(
            self.session, current_label, new_canonical
        )

    def remove_alias(self, label: str, alias: str) -> bool:
        """Drop ``alias`` from the record's alias list and counts. Does not."""
        return curation.remove_alias(self.session, label, alias)

    def remove_source(self, label: str, url_contains: str) -> int:
        """Drop citations whose URL contains ``url_contains`` from one record."""
        return curation.remove_source(self.session, label, url_contains)

    def split_alias(self, label: str, alias: str) -> bool:
        """Extract ``alias`` from the given record into a new standalone."""
        return curation.split_alias(self.session, label, alias)

    def merge_records(self, primary_label: str, other_label: str) -> bool:
        """Merge the record identified by ``other_label`` into the one."""
        return curation.merge_records(self.session, primary_label, other_label)

    def unpin_canonical(self, label: str) -> bool:
        """Allow automatic canonical re-selection on this record."""
        return curation.unpin_canonical(self.session, label)

    def suggest_alias_groups(
        self,
        max_groups: int = 25,
        prompt_template: str | None = None,
        *,
        chunk_size: int = curation.ALIAS_SUGGEST_CHUNK_SIZE,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Ask the LLM to propose record-level merges across the current."""
        return curation.suggest_alias_groups(
            self.session,
            max_groups,
            prompt_template,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
        )

    def apply_alias_suggestion(self, primary: str, members: list[str]) -> int:
        """Merge each ``members[i]`` into ``primary``. Returns merges done."""
        return curation.apply_alias_suggestion(self.session, primary, members)

    def audit_scope(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        budget: float = 4.0,
        confidence_threshold: float = 0.7,
        concurrency: int = 6,
        progress_cb=None,
    ) -> dict:
        """Flag records that aren't instances of the dataset's category."""
        return audits.audit_scope(
            self.session,
            api_key=api_key,
            model=model,
            budget=budget,
            confidence_threshold=confidence_threshold,
            concurrency=concurrency,
            progress_cb=progress_cb,
        )

    def add_schema_attribute(
        self,
        name: str,
        description: str = "",
        *,
        is_closed_set: bool = False,
        is_multi_valued: bool = False,
        canonical_values: list[str] | None = None,
        canonical_value_descriptions: dict | None = None,
        locked: bool = True,
    ) -> bool:
        """Add a schema attribute (empty until researched). Returns False if it already exists."""
        return schema.add_schema_attribute(
            self.session,
            name,
            description,
            is_closed_set=is_closed_set,
            is_multi_valued=is_multi_valued,
            canonical_values=canonical_values,
            canonical_value_descriptions=canonical_value_descriptions,
            locked=locked,
        )

    def audit_relevance(
        self,
        *,
        criterion: str,
        evidence_attribute: str,
        connections: list[str],
        context_attributes: list[str] | None = None,
        extra_evidence_attributes: list[str] | None = None,
        labels: list[str] | None = None,
        confidence_threshold: float = 0.7,
        concurrency: int = 6,
        progress_cb=None,
    ) -> dict:
        """Judge each record's cited evidence against an explicit-connection ``criterion``."""
        return audits.audit_relevance(
            self.session,
            criterion=criterion,
            evidence_attribute=evidence_attribute,
            connections=connections,
            context_attributes=context_attributes,
            extra_evidence_attributes=extra_evidence_attributes,
            labels=labels,
            confidence_threshold=confidence_threshold,
            concurrency=concurrency,
            progress_cb=progress_cb,
        )

    def apply_scope_exclusions(self, entries: list[dict]) -> int:
        """Exclude flagged records (also blocks their re-discovery). Returns count removed."""
        return audits.apply_scope_exclusions(self.session, entries)

    def audit_merge_quality(
        self,
        *,
        api_key: str | None = None,
        budget: float = 4.0,
        model: str | None = None,
        concurrency: int = 8,
        confidence_threshold: float = 0.7,
        progress_cb=None,
    ) -> dict:
        """Run the LLM-backed merge-quality audit."""
        return audits.audit_merge_quality(
            self.session,
            api_key=api_key,
            budget=budget,
            model=model,
            concurrency=concurrency,
            confidence_threshold=confidence_threshold,
            progress_cb=progress_cb,
        )

    def apply_audit_split(self, label: str, audit_entry: dict) -> int:
        """Apply a single split proposal (an entry from the audit's."""
        return audits.apply_audit_split(self.session, label, audit_entry)

    def audit_duplicate_clusters(
        self,
        *,
        api_key: str | None = None,
        budget: float = 4.0,
        model: str | None = None,
        concurrency: int = 8,
        confidence_threshold: float = 0.7,
        min_token_overlap: int = 2,
        progress_cb=None,
    ) -> dict:
        """Run the LLM-backed dedup audit (find under-merged clusters)."""
        return audits.audit_duplicate_clusters(
            self.session,
            api_key=api_key,
            budget=budget,
            model=model,
            concurrency=concurrency,
            confidence_threshold=confidence_threshold,
            min_token_overlap=min_token_overlap,
            progress_cb=progress_cb,
        )

    def apply_dedup_merge(
        self, canonical_label: str, member_labels: list[str], rationale: str = ""
    ) -> int:
        """Collapse every label in ``member_labels`` (other than."""
        return audits.apply_dedup_merge(
            self.session, canonical_label, member_labels, rationale
        )

    def apply_alias_correction(
        self, label: str, alias: str, *, belongs_to_label: str = "", rationale: str = ""
    ) -> bool:
        """Remove a misattached alias flagged by the duplicate audit (logged in history)."""
        return audits.apply_alias_correction(
            self.session,
            label,
            alias,
            belongs_to_label=belongs_to_label,
            rationale=rationale,
        )

    def propose_recategorization(
        self,
        constraints: str,
        *,
        attributes: list[str] | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Ask an LLM to propose tightened taxonomies for the current dataset."""
        return recategorize.propose_recategorization(
            self.session,
            constraints,
            attributes=attributes,
            api_key=api_key,
            model=model,
        )

    def apply_recategorization(
        self,
        proposal: dict,
        *,
        remove_out_of_scope: bool = False,
    ) -> dict:
        """Apply a reviewed proposal to the live dataset in place."""
        return recategorize.apply_recategorization(
            self.session, proposal, remove_out_of_scope=remove_out_of_scope
        )

    def recategorization_gaps(self, attributes: list[str]) -> dict[str, list[str]]:
        """Return, per attribute, the labels of records with no value."""
        return recategorize.recategorization_gaps(self.session, attributes)

    def recategorization_candidates(
        self,
        attributes: list[str],
        *,
        scope: str = "all",
        candidate_keywords: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Preview which records a web-search pass would touch (per attribute)."""
        return recategorize.recategorization_candidates(
            self.session,
            attributes,
            scope=scope,
            candidate_keywords=candidate_keywords,
            labels=labels,
        )

    def start_recategorization_search(
        self,
        attributes: list[str],
        *,
        mode: str = "augment",
        only_values: list[str] | None = None,
        candidate_keywords: list[str] | None = None,
        labels: list[str] | None = None,
        scope: str = "all",
        concurrency: int = 4,
    ) -> None:
        """Re-classify records against the new taxonomy using web search."""
        return recategorize.start_recategorization_search(
            self.session,
            attributes,
            mode=mode,
            only_values=only_values,
            candidate_keywords=candidate_keywords,
            labels=labels,
            scope=scope,
            concurrency=concurrency,
        )

    def import_dataset(
        self,
        data: dict,
        *,
        api_key: str | None = None,
        model: str = config.DEFAULT_MODEL,
        budget: float = 10.0,
    ) -> Path:
        """Import a dataset JSON as a new saved run so it can be edited, extended and re-published."""
        return storage.import_dataset(
            self.session, data, api_key=api_key, model=model, budget=budget
        )

    def update_schema_attribute(
        self,
        name: str,
        *,
        description: str | None = None,
        is_closed_set: bool | None = None,
        is_multi_valued: bool | None = None,
        canonical_values: list[str] | None = None,
        canonical_value_descriptions: dict | None = None,
        locked: bool | None = None,
    ) -> bool:
        """Edit an attribute's definition/taxonomy. With ``locked`` set, existing values."""
        return schema.update_schema_attribute(
            self.session,
            name,
            description=description,
            is_closed_set=is_closed_set,
            is_multi_valued=is_multi_valued,
            canonical_values=canonical_values,
            canonical_value_descriptions=canonical_value_descriptions,
            locked=locked,
        )

    def count_incomplete(self) -> tuple[int, int]:
        """(records with any empty schema attribute, empty cells) across the dataset."""
        return state.count_incomplete(self.session)

    def start_completion(self, concurrency: int = 8) -> None:
        """Fill missing schema attributes for every record (background)."""
        return refine.start_completion(self.session, concurrency)

    def start_relevance_filter(
        self,
        *,
        criterion: str,
        evidence_attribute: str,
        connections: list[str],
        evidence_description: str = "",
        research_evidence: bool = True,
        search_focus: str | None = None,
        labels: list[str] | None = None,
        context_attributes: list[str] | None = None,
        extra_evidence_attributes: list[str] | None = None,
        confidence_threshold: float = 0.7,
        concurrency: int = 8,
    ) -> None:
        """Background: optionally research a sourced evidence attribute for every record,."""
        return recategorize.start_relevance_filter(
            self.session,
            criterion=criterion,
            evidence_attribute=evidence_attribute,
            connections=connections,
            evidence_description=evidence_description,
            research_evidence=research_evidence,
            search_focus=search_focus,
            labels=labels,
            context_attributes=context_attributes,
            extra_evidence_attributes=extra_evidence_attributes,
            confidence_threshold=confidence_threshold,
            concurrency=concurrency,
        )

    @staticmethod
    def list_saved_runs(runs_dir: Path | None = None) -> list[dict]:
        """Return saved runs under ``runs_dir`` (the default runs folder when omitted), newest first."""
        return storage.list_saved_runs(runs_dir)

    def load_saved_run(
        self,
        data_path: str | Path,
        api_key: str | None = None,
        model: str = config.DEFAULT_MODEL,
        budget: float = 10.0,
    ) -> None:
        """Load a dataset previously written by ``_save_run``."""
        return storage.load_saved_run(self.session, data_path, api_key, model, budget)

    def load_dataset(self, data: dict) -> None:
        """Load a previously saved dataset JSON into this object."""
        return storage.load_dataset(self.session, data)

    def get_dataset_bytes_json(self) -> bytes:
        return storage.get_dataset_bytes_json(self.session)

    def get_dataset_bytes_csv(self) -> bytes:
        return storage.get_dataset_bytes_csv(self.session)

    def restore_records(self, records: list[dict], reason: str = "") -> int:
        """Re-add records (dicts from a previous dataset JSON) and lift their label."""
        return curation.restore_records(self.session, records, reason)

    def set_record_values(
        self,
        label: str,
        attribute: str,
        values: list[str],
        *,
        note: str = "",
        source_url: str = "",
    ) -> bool:
        """Replace one record's values for an attribute (manual curation)."""
        return value_edits.set_record_values(
            self.session, label, attribute, values, note=note, source_url=source_url
        )

    def select_single_values(
        self,
        attribute: str,
        *,
        context_attributes: list[str] | None = None,
        api_key: str | None = None,
        model: str | None = None,
        budget: float = 2.0,
        concurrency: int = 6,
        progress_cb=None,
    ) -> list[dict]:
        """LLM picks the one value per record that best fits the named entity, for."""
        return value_edits.select_single_values(
            self.session,
            attribute,
            context_attributes=context_attributes,
            api_key=api_key,
            model=model,
            budget=budget,
            concurrency=concurrency,
            progress_cb=progress_cb,
        )

    def apply_single_value_choices(self, attribute: str, choices: list[dict]) -> int:
        """Keep only the chosen value per record and mark ``attribute`` single-valued."""
        return value_edits.apply_single_value_choices(self.session, attribute, choices)

    def value_variant_changes(self, attribute: str) -> list[dict]:
        """Preview spelling variants of one name (``"Acme"`` / ``"Acme, Inc."`` /."""
        return value_edits.value_variant_changes(self.session, attribute)

    def merge_value_variants(self, attribute: str) -> int:
        """Give each name one spelling, within every record and across the dataset,."""
        return value_edits.merge_value_variants(self.session, attribute)

    def rename_attribute_value(
        self,
        attribute: str,
        old: str,
        new: str,
        *,
        description: str | None = None,
    ) -> int:
        """Rename one value of ``attribute`` in the schema and every record (sources kept)."""
        return value_edits.rename_attribute_value(
            self.session, attribute, old, new, description=description
        )

    def get_release_csv_bytes(self) -> bytes:
        """Flat CSV for publication: label, aliases, every schema attribute, sources."""
        return release.get_release_csv_bytes(self.session)

    def build_release_manifest(
        self, *, title: str, version: str, notes: str = ""
    ) -> dict:
        return release.build_release_manifest(
            self.session, title=title, version=version, notes=notes
        )

    def build_release_zip(
        self, *, version: str, notes: str = "", **dashboard_kwargs
    ) -> bytes:
        """One zip for publication: data.json, data.csv, manifest.json and dashboard/."""
        return release.build_release_zip(
            self.session, version=version, notes=notes, **dashboard_kwargs
        )

    def build_dashboard_zip(
        self,
        title: str,
        subtitle: str,
        dataset_label: str,
        primary_color: str,
        accent_color: str,
        logo_bytes: bytes | None = None,
        logo_filename: str | None = None,
        favicon_bytes: bytes | None = None,
        favicon_filename: str | None = None,
        views: list[str] | None = None,
        secondary_accent_color: str | None = None,
        colors: dict[str, str] | None = None,
        footer: str | None = None,
        labels: dict[str, str] | None = None,
        intro: str | None = None,
        intro_details: str | None = None,
    ) -> bytes:
        """Bundle dashboard.html + theme + data into a downloadable ZIP."""
        return release.build_dashboard_zip(
            self.session,
            title,
            subtitle,
            dataset_label,
            primary_color,
            accent_color,
            logo_bytes,
            logo_filename,
            favicon_bytes,
            favicon_filename,
            views,
            secondary_accent_color,
            colors,
            footer,
            labels,
            intro,
            intro_details,
        )
