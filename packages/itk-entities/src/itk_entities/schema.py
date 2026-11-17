# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Editing the schema: add, rename, update and remove attributes."""

from __future__ import annotations

import logging

from . import storage
from .state import Session

_logger = logging.getLogger(__name__)


def rename_attribute(session: Session, old_name: str, new_name: str) -> int:
    """Rename a schema attribute and update all records. Returns affected records."""
    if not session.schemify or not session.schemify.record_set:
        return 0
    old = (old_name or "").strip()
    new = (new_name or "").strip()
    if not old or not new or old == new:
        return 0
    rs = session.schemify.record_set
    # Schema attributes
    for sa in rs.schema_attributes:
        if sa.name == old:
            sa.name = new
            break
    affected = 0
    for record in rs.records:
        for bucket in (record.attributes, record.additional_attributes):
            if old in bucket:
                bucket[new] = bucket.pop(old)
                affected += 1
    rs.update_schema_frequencies()
    session.df = session.schemify.to_dataframe()
    session.dataset_json = storage.build_dataset_json(session)
    try:
        storage.snapshot_partial(session, category=rs.category)
    except Exception as e:  # noqa: BLE001
        _logger.warning("snapshot_partial failed: %s", e)
    return affected


def remove_attribute(session: Session, name: str) -> int:
    """Remove a schema attribute from the schema and all records."""
    if not session.schemify or not session.schemify.record_set:
        return 0
    target = (name or "").strip()
    if not target:
        return 0
    rs = session.schemify.record_set
    rs.schema_attributes = [sa for sa in rs.schema_attributes if sa.name != target]
    affected = 0
    for record in rs.records:
        for bucket in (record.attributes, record.additional_attributes):
            if target in bucket:
                bucket.pop(target, None)
                affected += 1
    rs.update_schema_frequencies()
    session.df = session.schemify.to_dataframe()
    session.dataset_json = storage.build_dataset_json(session)
    try:
        storage.snapshot_partial(session, category=rs.category)
    except Exception as e:  # noqa: BLE001
        _logger.warning("snapshot_partial failed: %s", e)
    return affected


def add_schema_attribute(
    session: Session,
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
    from itk_schemify.models import SchemaAttribute

    if not session.schemify or not session.schemify.record_set:
        return False
    rs = session.schemify.record_set
    if any(a.name.casefold() == name.strip().casefold() for a in rs.schema_attributes):
        return False
    rs.schema_attributes.append(
        SchemaAttribute(
            name=name.strip(),
            description=description,
            is_closed_set=is_closed_set,
            is_multi_valued=is_multi_valued,
            canonical_values=list(canonical_values or []),
            canonical_value_descriptions=dict(canonical_value_descriptions or {}),
            locked=locked,
        )
    )
    session.dataset_json = storage.build_dataset_json(session)
    return True


def update_schema_attribute(
    session: Session,
    name: str,
    *,
    description: str | None = None,
    is_closed_set: bool | None = None,
    is_multi_valued: bool | None = None,
    canonical_values: list[str] | None = None,
    canonical_value_descriptions: dict | None = None,
    locked: bool | None = None,
) -> bool:
    """Edit an attribute's definition/taxonomy. With ``locked`` set, existing values
    outside ``canonical_values`` are dropped at the next refresh."""
    if not session.schemify or not session.schemify.record_set:
        return False
    attr = next(
        (a for a in session.schemify.record_set.schema_attributes if a.name == name),
        None,
    )
    if attr is None:
        return False
    if description is not None:
        attr.description = description
    if is_closed_set is not None:
        attr.is_closed_set = is_closed_set
    if is_multi_valued is not None:
        attr.is_multi_valued = is_multi_valued
    if canonical_values is not None:
        attr.canonical_values = [v.strip() for v in canonical_values if v and v.strip()]
        attr.provisional_values = list(attr.canonical_values)
        if attr.canonical_values:
            attr.is_closed_set = True
    if canonical_value_descriptions is not None:
        attr.canonical_value_descriptions = {
            k.strip(): v.strip()
            for k, v in canonical_value_descriptions.items()
            if k and v
        }
    if locked is not None:
        attr.locked = locked
    storage.post_curation_refresh(session)
    return True
