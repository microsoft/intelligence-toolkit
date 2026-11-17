# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""User exclusion rules: by label or by attribute value, applied to current and future records."""

from __future__ import annotations

import logging

from . import storage
from .state import Session

_logger = logging.getLogger(__name__)


def exclusions(session: Session) -> list[dict]:
    """Current user-supplied exclusion rules (label-based or attribute predicates)."""
    if not session.schemify or not session.schemify.record_set:
        return []
    return list(session.schemify.record_set.user_exclusions or [])


def add_label_exclusion(
    session: Session, label: str, reason: str = "", remove_existing: bool = True
) -> tuple[bool, int]:
    """Exclude a single named entity. See add_attribute_exclusion for predicate rules."""
    if not session.schemify or not session.schemify.record_set:
        return (False, 0)
    lbl = (label or "").strip()
    if not lbl:
        return (False, 0)
    rule = {"label": lbl, "reason": (reason or "").strip()}
    return _upsert_exclusion(session, rule, remove_existing=remove_existing)


def add_exclusion(
    session: Session, label: str, reason: str = "", remove_existing: bool = True
) -> tuple[bool, int]:
    return add_label_exclusion(session, label, reason, remove_existing)


def add_attribute_exclusion(
    session: Session,
    attribute: str,
    operator: str,
    values: list[str] | None = None,
    reason: str = "",
    remove_existing: bool = True,
) -> tuple[bool, int]:
    """Exclude every entity that satisfies the given attribute predicate.

    Supported operators: ``missing``, ``equals``, ``in``, ``contains``, ``regex``.
    """
    if not session.schemify or not session.schemify.record_set:
        return (False, 0)
    attr = (attribute or "").strip()
    op = (operator or "equals").strip().lower()
    if not attr or op not in {"missing", "equals", "in", "contains", "regex"}:
        return (False, 0)
    vals = [str(v).strip() for v in (values or []) if str(v).strip()]
    if op != "missing" and not vals:
        return (False, 0)
    rule = {
        "attribute": attr,
        "operator": op,
        "values": vals,
        "reason": (reason or "").strip(),
    }
    return _upsert_exclusion(session, rule, remove_existing=remove_existing)


def remove_exclusion(session: Session, identifier: str | dict) -> bool:
    """Remove an exclusion rule. ``identifier`` may be a label string
    (matches label rules) or a rule dict (compared by attribute+operator+values).
    """
    if not session.schemify or not session.schemify.record_set:
        return False
    rs = session.schemify.record_set
    before = len(rs.user_exclusions or [])
    rs.user_exclusions = [
        e
        for e in (rs.user_exclusions or [])
        if not _rule_matches_identifier(e, identifier)
    ]
    if len(rs.user_exclusions) == before:
        return False
    session.dataset_json = storage.build_dataset_json(session)
    try:
        storage.snapshot_partial(session, category=rs.category)
    except Exception as e:  # noqa: BLE001
        _logger.warning("snapshot_partial failed: %s", e)
    return True


def _upsert_exclusion(
    session: Session, rule: dict, remove_existing: bool = True
) -> tuple[bool, int]:
    rs = session.schemify.record_set
    rs.user_exclusions = list(rs.user_exclusions or [])
    # Dedupe — same kind+target overwrites the previous reason.
    for i, existing in enumerate(rs.user_exclusions):
        if _rules_equivalent(existing, rule):
            rs.user_exclusions[i] = rule
            break
    else:
        rs.user_exclusions.append(rule)

    removed = _remove_records_matching_rule(session, rule) if remove_existing else 0

    session.df = session.schemify.to_dataframe()
    session.dataset_json = storage.build_dataset_json(session)
    try:
        storage.snapshot_partial(session, category=rs.category)
    except Exception as e:  # noqa: BLE001
        _logger.warning("snapshot_partial failed: %s", e)
    return (True, removed)


def _rules_equivalent(a: dict, b: dict) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if bool(a.get("attribute")) != bool(b.get("attribute")):
        return False
    if a.get("attribute"):
        return (
            (a.get("attribute") or "").strip().casefold()
            == (b.get("attribute") or "").strip().casefold()
            and (a.get("operator") or "equals").lower()
            == (b.get("operator") or "equals").lower()
            and [str(v).strip().casefold() for v in (a.get("values") or [])]
            == [str(v).strip().casefold() for v in (b.get("values") or [])]
        )
    return (a.get("label") or "").strip().casefold() == (
        b.get("label") or ""
    ).strip().casefold()


def _rule_matches_identifier(rule: dict, identifier) -> bool:
    if isinstance(identifier, dict):
        return _rules_equivalent(rule, identifier)
    ident = (identifier or "").strip().casefold()
    if not ident:
        return False
    if rule.get("attribute"):
        return False
    return (rule.get("label") or "").strip().casefold() == ident


def _remove_records_matching_rule(session: Session, rule: dict) -> int:
    if not session.schemify or not session.schemify.record_set:
        return 0
    from itk_schemify.resolution import record_matches_rule

    rs = session.schemify.record_set
    keep = []
    removed = 0
    for r in rs.records:
        if record_matches_rule(r, rule):
            removed += 1
        else:
            keep.append(r)
    if removed:
        rs.records = keep
        rs.update_schema_frequencies()
    return removed


def _remove_records_matching(session: Session, label: str) -> int:
    """Legacy helper kept for backward compatibility."""
    return _remove_records_matching_rule(session, {"label": label})
