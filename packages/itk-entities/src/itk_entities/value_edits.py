# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Editing attribute values: set, pick a single value, merge variants and rename."""

from __future__ import annotations

import asyncio
import copy
import logging
import time

from . import state, storage
from .state import Session

_logger = logging.getLogger(__name__)


def set_record_values(
    session: Session,
    label: str,
    attribute: str,
    values: list[str],
    *,
    note: str = "",
    source_url: str = "",
) -> bool:
    """Replace one record's values for an attribute (manual curation).

    Sources already cited for a kept value are preserved; new values get the
    optional ``source_url``/``note`` as their citation.
    """
    from itk_schemify.models import AttributeValue, Citation, SourcedValue

    rec = state.find_record(session, label)
    if rec is None:
        return False
    old = rec.attributes.get(attribute)
    old_sources = {sv.value: sv.sources for sv in (old.values if old else [])}
    all_old = [s for srcs in old_sources.values() for s in srcs]
    new_values = []
    for v in [v.strip() for v in values if v and v.strip()]:
        if v in old_sources:
            srcs = old_sources[v]
        elif source_url:
            srcs = [
                Citation(url=source_url, title="Manual curation", snippet=note or None)
            ]
        else:
            # Keep the record's existing evidence so the value stays sourced.
            srcs = list(all_old)
        new_values.append(SourcedValue(value=v, sources=list(srcs)))
    if new_values:
        rec.attributes[attribute] = AttributeValue(values=new_values)
    else:
        rec.attributes.pop(attribute, None)
    session.schemify.record_set.history.append({
        "op": "set_record_values",
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": rec.label,
        "attribute": attribute,
        "old": list(old_sources),
        "new": [sv.value for sv in new_values],
        "note": note,
    })
    storage.post_curation_refresh(session)
    return True


def select_single_values(
    session: Session,
    attribute: str,
    *,
    context_attributes: list[str] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    budget: float = 2.0,
    concurrency: int = 6,
    progress_cb=None,
) -> list[dict]:
    """LLM picks the one value per record that best fits the named entity, for
    records holding several values of ``attribute`` (e.g. accumulated descriptions).

    Returns reviewable choices; apply with :meth:`apply_single_value_choices`.
    """
    from itk_schemify import value_consolidation as _vc
    from itk_schemify.llm import LLMClient
    from itk_schemify.models import SchemifyConfig

    if not session.schemify or not session.schemify.record_set:
        return []
    llm = getattr(session.schemify, "llm", None)
    if llm is None:
        if not api_key:
            raise ValueError(
                "select_single_values requires api_key when no llm is attached"
            )
        cfg_kwargs = {"api_key": api_key, "max_budget": float(budget)}
        if model:
            cfg_kwargs["completion_model"] = model
        llm = LLMClient(SchemifyConfig(**cfg_kwargs))
    return asyncio.run(
        _vc.select_single_values(
            session.schemify.record_set,
            llm,
            attribute,
            context_attributes=context_attributes,
            concurrency=concurrency,
            progress_cb=progress_cb,
        )
    )


def apply_single_value_choices(
    session: Session, attribute: str, choices: list[dict]
) -> int:
    """Keep only the chosen value per record and mark ``attribute`` single-valued."""
    from itk_schemify import value_consolidation as _vc

    if not session.schemify or not session.schemify.record_set or not choices:
        return 0
    n = _vc.apply_single_value_choices(session.schemify.record_set, attribute, choices)
    session.schemify.record_set.history.append({
        "op": "select_single_values",
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attribute": attribute,
        "records": n,
    })
    storage.post_curation_refresh(session)
    return n


def value_variant_changes(session: Session, attribute: str) -> list[dict]:
    """Preview spelling variants of one name (``"Acme"`` / ``"Acme, Inc."`` /
    ``"Acme (ACM)"``) that :meth:`merge_value_variants` would collapse, both within a
    record and across records, as ``{"label", "before", "after"}`` per changed record."""

    from itk_schemify import value_consolidation as _vc

    if not session.schemify or not session.schemify.record_set:
        return []
    rs = session.schemify.record_set
    trial = copy.deepcopy(rs)
    _vc.apply_value_variant_changes(trial, attribute)
    _vc.apply_dataset_variant_changes(trial, attribute)
    changes = []
    for record, after_rec in zip(rs.records, trial.records):
        av, after_av = (
            record.attributes.get(attribute),
            after_rec.attributes.get(attribute),
        )
        before = [sv.value for sv in (av.values if av else [])]
        after = [sv.value for sv in (after_av.values if after_av else [])]
        if after != before:
            changes.append({"label": record.label, "before": before, "after": after})
    return changes


def merge_value_variants(session: Session, attribute: str) -> int:
    """Give each name one spelling, within every record and across the dataset,
    keeping the fullest form within a record and the most used form across records.
    All sources are kept. Returns the number of records changed."""
    from itk_schemify import value_consolidation as _vc

    if not session.schemify or not session.schemify.record_set:
        return 0
    rs = session.schemify.record_set
    within = _vc.apply_value_variant_changes(rs, attribute)
    across = _vc.apply_dataset_variant_changes(rs, attribute)
    rs.history.append({
        "op": "merge_value_variants",
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attribute": attribute,
        "records": within,
        "records_across_dataset": across,
    })
    storage.post_curation_refresh(session)
    return within + across


def rename_attribute_value(
    session: Session,
    attribute: str,
    old: str,
    new: str,
    *,
    description: str | None = None,
) -> int:
    """Rename one value of ``attribute`` in the schema and every record (sources kept).
    Returns the number of records changed."""
    if not session.schemify or not session.schemify.record_set:
        return 0
    rs = session.schemify.record_set
    schema = next((a for a in rs.schema_attributes if a.name == attribute), None)
    in_schema = schema is not None and (
        old in schema.canonical_values
        or old in schema.provisional_values
        or old in (schema.canonical_value_descriptions or {})
    )
    in_records = any(
        sv.value == old
        for r in rs.records
        for sv in (r.attributes[attribute].values if attribute in r.attributes else [])
    )
    if not (in_schema or in_records):
        return 0
    if in_schema:
        for name in ("canonical_values", "provisional_values"):
            setattr(
                schema, name, [new if v == old else v for v in getattr(schema, name)]
            )
        descs = dict(schema.canonical_value_descriptions or {})
        if old in descs or description is not None:
            text = descs.pop(old, None)
            descs[new] = description if description is not None else text
        schema.canonical_value_descriptions = descs
    changed = 0
    for record in rs.records:
        av = record.attributes.get(attribute)
        hit = False
        for sv in av.values if av else []:
            if sv.value == old:
                sv.value, hit = new, True
        changed += hit
    rs.history.append({
        "op": "rename_attribute_value",
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attribute": attribute,
        "old": old,
        "new": new,
        "records": changed,
    })
    storage.post_curation_refresh(session)
    return changed
