# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Record curation: aliases, canonical names, merges, splits, removals and restores."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable

from . import state, storage
from .state import Session

_logger = logging.getLogger(__name__)


# Max labels per LLM call when asking for alias-merge suggestions.
# Larger datasets are chunked across multiple calls instead of being
# silently truncated.
ALIAS_SUGGEST_CHUNK_SIZE = 200


DEFAULT_ALIAS_SUGGEST_PROMPT = (
    "You are curating an entity dataset about: {category}.\n\n"
    "Below are entity labels currently treated as DISTINCT records. "
    "Identify groups of two or more labels in this list that refer to "
    "the SAME real-world entity (e.g. acronym + full name, vendor "
    "renaming, common typo). Be conservative: do NOT group entities "
    "that merely belong to the same category, are made by the same "
    "organisation, or share a generic word.\n\n"
    "Reply with a JSON array. Each item must be an object with:\n"
    '  - "primary": the label that should be the canonical name '
    "(prefer the most specific / official name; avoid category-like "
    "phrases such as 'TOOLS', 'PLATFORMS', 'ARCHIVES').\n"
    '  - "members": list of OTHER labels that should be merged into '
    "the primary as aliases. Must be a subset of the input labels.\n"
    '  - "reason": one short sentence justifying the merge.\n'
    "If no groupings are warranted, reply with an empty array: [].\n\n"
    "Labels:\n{labels}\n"
)


def _parse_alias_suggestions(raw: str) -> list[dict]:
    """Best-effort JSON extraction from an LLM reply.

    The model is asked to return a JSON array of merge suggestions but
    occasionally wraps it in prose or fenced code blocks; tolerate that.
    Returns a list of plain dicts (no validation against the dataset —
    callers do that).
    """
    text = (raw or "").strip()
    if not text:
        return []
    # Strip leading/trailing markdown code fences if present.
    if text.startswith("```"):
        # Drop the opening fence (with optional language tag) and the
        # trailing fence.
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        # Fall back to locating the first '[' ... ']' span.
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def remove_record_by_label(session: Session, label: str) -> bool:
    """Drop a single record by exact label (case-insensitive). Returns True if removed."""
    if not session.schemify or not session.schemify.record_set:
        return False
    rs = session.schemify.record_set
    target = (label or "").strip().casefold()
    if not target:
        return False
    keep = []
    removed = False
    for r in rs.records:
        if (r.label or "").strip().casefold() == target and not removed:
            removed = True
            continue
        keep.append(r)
    if removed:
        rs.records = keep
        rs.update_schema_frequencies()
        session.df = session.schemify.to_dataframe()
        session.dataset_json = storage.build_dataset_json(session)
        try:
            storage.snapshot_partial(session, category=rs.category)
        except Exception:  # noqa: BLE001
            pass
    return removed


def list_alias_groups(session: Session, only_with_aliases: bool = False) -> list[dict]:
    """Return every record's canonical label + its alias group.

    Each entry is ``{"label", "aliases", "alias_counts", "total_count"}``.
    Sorted by ``total_count`` desc so the biggest merged groups (the
    most likely culprits for over-aggressive merging) appear first.
    """
    if not session.schemify or not session.schemify.record_set:
        return []
    rs = session.schemify.record_set
    groups: list[dict] = []
    for r in rs.records:
        aliases = list(r.aliases or [])
        if only_with_aliases and not aliases:
            continue
        counts = dict(r.alias_counts or {})
        groups.append({
            "label": r.label,
            "aliases": aliases,
            "alias_counts": counts,
            "total_count": sum(counts.values()) if counts else 1,
        })
    groups.sort(key=lambda g: (-g["total_count"], g["label"]))
    return groups


def do_not_merge_pairs(session: Session) -> list[list[str]]:
    """User-asserted pairs that must never be auto-merged again."""
    if not session.schemify or not session.schemify.record_set:
        return []
    rs = session.schemify.record_set
    pairs = getattr(rs, "do_not_merge", None) or set()
    return [sorted(list(p)) for p in pairs]


def _record_do_not_merge(session: Session, a: str, b: str) -> None:
    rs = session.schemify.record_set
    if not hasattr(rs, "do_not_merge") or rs.do_not_merge is None:
        rs.do_not_merge = set()
    a_norm = (a or "").strip()
    b_norm = (b or "").strip()
    if a_norm and b_norm and a_norm.casefold() != b_norm.casefold():
        rs.do_not_merge.add(frozenset((a_norm, b_norm)))


def rename_record_canonical(
    session: Session, current_label: str, new_canonical: str
) -> bool:
    """Promote ``new_canonical`` to be the canonical label of the record
    currently identified by ``current_label`` (or one of its aliases).

    - If ``new_canonical`` already exists as an alias, it is promoted and
      the previous label is demoted to an alias.
    - If it is a brand-new string, it is added (with count 1) before
      promotion.
    - A ``manual_canonical = True`` marker is set on the record so
      subsequent alias additions/merges do not silently rename it.
    """
    rec = state.find_record(session, current_label)
    if rec is None:
        return False
    new_label = (new_canonical or "").strip()
    if not new_label:
        return False
    if new_label.upper() == (rec.label or "").upper():
        rec.manual_canonical = True
        storage.post_curation_refresh(session)
        return True
    # Ensure the new label is in alias_counts (with at least count 1).
    existing_key = None
    for key in rec.alias_counts:
        if key.lower() == new_label.lower():
            existing_key = key
            break
    if existing_key is None:
        rec.alias_counts[new_label] = 1
    else:
        new_label = existing_key  # preserve original casing of the alias
    # Demote current label to alias.
    if rec.label and rec.label not in rec.aliases:
        rec.aliases.append(rec.label)
    if rec.label and rec.label not in rec.alias_counts:
        rec.alias_counts[rec.label] = 1
    rec.label = new_label
    rec.aliases = [
        name for name in rec.alias_counts.keys() if name.upper() != rec.label.upper()
    ]
    rec.manual_canonical = True
    storage.post_curation_refresh(session)
    return True


def remove_alias(session: Session, label: str, alias: str) -> bool:
    """Drop ``alias`` from the record's alias list and counts. Does not
    re-introduce it as a separate record."""
    rec = state.find_record(session, label)
    if rec is None:
        return False
    target = (alias or "").strip()
    if not target:
        return False
    # Don't allow deleting the canonical via this path.
    if target.upper() == (rec.label or "").upper():
        return False
    existing_key = None
    for key in rec.alias_counts:
        if key.lower() == target.lower():
            existing_key = key
            break
    changed = False
    if existing_key is not None:
        rec.alias_counts.pop(existing_key, None)
        changed = True
    rec.aliases = [a for a in rec.aliases if (a or "").lower() != target.lower()]
    if changed:
        storage.post_curation_refresh(session)
    return changed


def remove_source(session: Session, label: str, url_contains: str) -> int:
    """Drop citations whose URL contains ``url_contains`` from one record.

    Values left without any citation and attributes left without values
    are removed too, so claims supported only by a bad source disappear.
    Returns the number of citations removed."""
    rec = state.find_record(session, label)
    needle = (url_contains or "").strip()
    if rec is None or not needle:
        return 0
    removed = 0
    for attr in list(rec.attributes):
        av = rec.attributes[attr]
        kept_values = []
        for sv in av.values:
            before = len(sv.sources)
            sv.sources = [c for c in sv.sources if needle not in (c.url or "")]
            removed += before - len(sv.sources)
            if sv.sources or before == 0:
                kept_values.append(sv)
        av.values = kept_values
        if not kept_values:
            del rec.attributes[attr]
    if removed:
        session.schemify.record_set.history.append({
            "op": "remove_source",
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": rec.label,
            "url_contains": needle,
            "citations": removed,
        })
        storage.post_curation_refresh(session)
    return removed


def split_alias(session: Session, label: str, alias: str) -> bool:
    """Extract ``alias`` from the given record into a new standalone
    record (preserving its alias count) and register a do-not-merge
    constraint between the two so subsequent fuzzy passes won't
    re-merge them."""
    if not session.schemify or not session.schemify.record_set:
        return False
    from itk_schemify.models import Record

    rec = state.find_record(session, label)
    if rec is None:
        return False
    target = (alias or "").strip()
    if not target:
        return False
    if target.upper() == (rec.label or "").upper():
        return False

    existing_key = None
    for key in rec.alias_counts:
        if key.lower() == target.lower():
            existing_key = key
            break
    if existing_key is None:
        return False

    count = int(rec.alias_counts.pop(existing_key, 1) or 1)
    rec.aliases = [a for a in rec.aliases if (a or "").lower() != target.lower()]

    # Create a fresh record from the extracted alias. Use upper-case to
    # match the rest of the dataset convention.
    new_rec = Record(label=existing_key.upper())
    new_rec.alias_counts[new_rec.label] = count
    # Mark as user-pinned so it won't be silently renamed if it picks up
    # a more frequent alias later.
    new_rec.manual_canonical = True
    session.schemify.record_set.records.append(new_rec)

    # Register the do-not-merge constraint (both directions of label).
    _record_do_not_merge(session, rec.label, new_rec.label)

    storage.post_curation_refresh(session)
    return True


def merge_records(session: Session, primary_label: str, other_label: str) -> bool:
    """Merge the record identified by ``other_label`` into the one
    identified by ``primary_label``. The primary's canonical label is
    preserved (and pinned) so the merge never flips canonicals on the
    user."""
    if not session.schemify or not session.schemify.record_set:
        return False
    primary = state.find_record(session, primary_label)
    other = state.find_record(session, other_label)
    if primary is None or other is None or primary is other:
        return False
    # Pin canonical on the primary so the absorbed counts can't promote
    # an alias from ``other`` to canonical.
    primary.manual_canonical = True
    primary.merge_from(other)
    try:
        session.schemify.record_set.records.remove(other)
    except ValueError:
        pass
    # Drop any stale do-not-merge constraint between these two — the
    # user has now explicitly opted in.
    rs = session.schemify.record_set
    if getattr(rs, "do_not_merge", None):
        rs.do_not_merge = {
            p for p in rs.do_not_merge if not (primary.label in p and other.label in p)
        }
    storage.post_curation_refresh(session)
    return True


def unpin_canonical(session: Session, label: str) -> bool:
    """Allow automatic canonical re-selection on this record."""
    rec = state.find_record(session, label)
    if rec is None:
        return False
    if getattr(rec, "manual_canonical", False):
        rec.manual_canonical = False
        # Recompute now so the UI reflects the change.
        rec._update_canonical_label()
        storage.post_curation_refresh(session)
        return True
    return False


def suggest_alias_groups(
    session: Session,
    max_groups: int = 25,
    prompt_template: str | None = None,
    *,
    chunk_size: int = ALIAS_SUGGEST_CHUNK_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Ask the LLM to propose record-level merges across the current
    dataset.

    Labels are partitioned into chunks of ``chunk_size`` and one LLM
    call is dispatched per chunk so the whole dataset is considered
    instead of being silently truncated. Suggestions are de-duplicated
    across chunks (by primary+members signature).

    Returns a list of ``{"primary": str, "members": list[str],
    "reason": str}`` dicts. Members and primary are validated against
    the current set of canonical labels so applying a suggestion is
    always well-defined.
    """
    if not session.schemify or not session.schemify.record_set:
        return []
    try:
        from itk_core.client import OpenAIClient
        from itk_core.defaults import DEFAULT_TEMPERATURE
    except ImportError as e:
        _logger.warning("alias-suggest: AI client import failed: %s", e)
        return []

    rs = session.schemify.record_set
    labels = [r.label for r in rs.records if (r.label or "").strip()]
    if len(labels) < 2:
        return []

    # Build deterministic chunks. For datasets larger than chunk_size
    # we shuffle the chunk boundaries with a stable hash so labels
    # that *should* group don't end up always in different chunks
    # (e.g. alphabetical-by-default would split "IBM" and "International
    # Business Machines").
    chunk = max(2, int(chunk_size))
    if len(labels) > chunk:
        # Stable but spread-out ordering: round-robin into ``num_chunks``
        # buckets by index. Two passes would be required to catch all
        # cross-chunk merges, but a single pass already finds within-chunk
        # ones reliably.
        num_chunks = (len(labels) + chunk - 1) // chunk
        buckets: list[list[str]] = [[] for _ in range(num_chunks)]
        for i, lbl in enumerate(labels):
            buckets[i % num_chunks].append(lbl)
        chunks = buckets
    else:
        chunks = [list(labels)]

    template = (prompt_template or DEFAULT_ALIAS_SUGGEST_PROMPT).strip()
    client = OpenAIClient()
    label_lookup = {lbl.casefold(): lbl for lbl in labels}
    cleaned: list[dict] = []
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()

    for idx, chunk_labels in enumerate(chunks):
        if progress_callback:
            try:
                progress_callback(idx, len(chunks))
            except Exception as e:  # noqa: BLE001
                _logger.warning("alias-suggest progress raised: %s", e)
        labels_block = "\n".join(f"- {lbl}" for lbl in chunk_labels)
        prompt = template.format(
            category=rs.category or "(unspecified)",
            labels=labels_block,
        )
        try:
            raw = client.generate_chat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=DEFAULT_TEMPERATURE,
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning(
                "alias-suggest: chunk %d/%d failed: %s", idx + 1, len(chunks), e
            )
            continue

        for s in _parse_alias_suggestions(raw or ""):
            primary_raw = (s.get("primary") or "").strip()
            primary = label_lookup.get(primary_raw.casefold())
            members_raw = s.get("members") or []
            members: list[str] = []
            seen_m = set()
            for m in members_raw:
                if not isinstance(m, str):
                    continue
                resolved = label_lookup.get(m.strip().casefold())
                if (
                    resolved
                    and resolved != primary
                    and resolved.casefold() not in seen_m
                ):
                    members.append(resolved)
                    seen_m.add(resolved.casefold())
            if not (primary and members):
                continue
            key = (
                primary.casefold(),
                tuple(sorted(m.casefold() for m in members)),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cleaned.append({
                "primary": primary,
                "members": members,
                "reason": (s.get("reason") or "").strip(),
            })
            if len(cleaned) >= max_groups:
                break
        if len(cleaned) >= max_groups:
            break

    if progress_callback:
        try:
            progress_callback(len(chunks), len(chunks))
        except Exception as e:  # noqa: BLE001
            _logger.warning("alias-suggest progress raised: %s", e)
    return cleaned


def apply_alias_suggestion(session: Session, primary: str, members: list[str]) -> int:
    """Merge each ``members[i]`` into ``primary``. Returns merges done."""
    applied = 0
    for m in members or []:
        if merge_records(session, primary, m):
            applied += 1
    return applied


def restore_records(session: Session, records: list[dict], reason: str = "") -> int:
    """Re-add records (dicts from a previous dataset JSON) and lift their label
    exclusions so research won't drop them again. Returns count restored."""
    from itk_schemify.models import Record

    if not session.schemify or not session.schemify.record_set:
        return 0
    rs = session.schemify.record_set
    present = {(r.label or "").casefold() for r in rs.records}
    restored = []
    for d in records or []:
        label = (d.get("label") or "").strip()
        if not label or label.casefold() in present:
            continue
        rs.records.append(Record.from_dict(d))
        present.add(label.casefold())
        restored.append(label)
    wanted = {l.casefold() for l in restored}
    rs.user_exclusions = [
        e
        for e in (rs.user_exclusions or [])
        if e.get("attribute") or (e.get("label") or "").casefold() not in wanted
    ]
    if restored:
        rs.history.append({
            "op": "restore_records",
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "labels": restored,
            "reason": reason,
        })
        storage.post_curation_refresh(session)
    return len(restored)
