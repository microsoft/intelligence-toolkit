# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Auto-mode orchestrator for Build Entity Dataset.

Drives the existing Schemify primitives (``run_agentic`` →
``verify_unverified`` → ``normalize``) in a loop until an LLM judge
declares the dataset complete (or a hard ceiling is reached).

The orchestrator is intentionally a pure async function so it can be
called both from notebook scripts and from the ITK background-thread
wrapper. State and progress are propagated via simple callbacks rather
than shared globals.

IMPORTANT: Normalization is a REQUIRED phase in any iterative workflow.
It must run:
  1. Periodically during discovery loops (every N iterations)
  2. At the end of any research pass (before export/finalize)

Normalization uses the LLM to:
  - Detect and consolidate value variants (case, spacing, synonyms)
  - Split multi-valued attributes into filterable arrays
  - Map synonyms to canonical forms
  - Classify attributes as closed-set or open-set

Without normalization, the dataset will have duplicate values that should
be merged (e.g., "Human Trafficking" vs "human trafficking"), making
filtering and analysis less effective. Always call schemify.normalize()
after discovery and before export.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

_logger = logging.getLogger(__name__)


@dataclass
class CompletenessVerdict:
    """LLM judge's read on whether to stop iterating."""

    complete: bool
    confidence: float
    reason: str
    missing_gaps: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "complete": bool(self.complete),
            "confidence": float(self.confidence),
            "reason": self.reason,
            "missing_gaps": list(self.missing_gaps),
            "suggested_queries": list(self.suggested_queries),
        }


@dataclass
class IterationSummary:
    """A single iteration's outcome — fed back into the judge prompt."""

    index: int
    new_entities: int
    total_entities: int
    queries_run: int
    cost_usd: float
    phase_split: tuple[float, float, float]
    judge: Optional[CompletenessVerdict] = None


@dataclass
class AutoRunResult:
    """Final result of a full auto-mode run."""

    iterations_completed: int
    final_entity_count: int
    final_cost: float
    stop_reason: str
    per_iteration_history: list[IterationSummary]


JUDGE_PROMPT = """\
You are a research-completeness judge. Decide whether the dataset under
construction is COMPLETE ENOUGH to stop iterating, or whether another
research pass is likely to keep adding meaningful entities or fill
substantive attribute gaps.

Category being researched: {category}

Original guidance from the user:
{guidance}

Schema attributes (the columns we are trying to fill):
{schema_summary}

Current dataset state:
- Total entities: {entity_count}
- Attribute coverage (per-attribute share of entities with a non-empty
  value): {coverage_summary}
- Most-frequent values seen per attribute (top 5):
{value_distribution}

Recent iteration history (most recent last):
{history_summary}

--Decision rules--

* Mark COMPLETE only when the iteration history shows clearly
  diminishing returns AND the attribute coverage looks broadly adequate
  for the user's stated goal. Discovery plateaus at small entity counts
  often signal stuck queries, not true completeness — be skeptical.
* Provide a confidence between 0 and 1. A confidence below 0.7 should
  almost never accompany ``complete = true``.
* If NOT complete, list the most important unaddressed gaps (e.g.
  geographies, sub-types, audiences) and 3–8 concrete follow-up search
  queries the next iteration should prioritise.
"""

JUDGE_SCHEMA: dict[str, Any] = {
    "json_schema": {
        "name": "completeness_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "complete": {"type": "boolean"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
                "missing_gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "suggested_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "complete",
                "confidence",
                "reason",
                "missing_gaps",
                "suggested_queries",
            ],
        },
    }
}


def _adaptive_phase_split(
    iteration: int, max_iterations: int
) -> tuple[float, float, float]:
    """Earlier iterations favour discovery, later iterations favour completion.

    Linear interpolation between (0.7, 0.2, 0.1) on iteration 1 and
    (0.2, 0.2, 0.6) on the final iteration. With ``max_iterations == 1``
    we run the balanced default (0.6, 0.2, 0.2).
    """
    if max_iterations <= 1:
        return (0.6, 0.2, 0.2)
    t = max(0.0, min(1.0, (iteration - 1) / (max_iterations - 1)))
    discovery = 0.7 + (0.2 - 0.7) * t
    targeted = 0.2
    completion = 0.1 + (0.6 - 0.1) * t
    # Normalise to guard against floating-point drift.
    total = discovery + targeted + completion
    return (discovery / total, targeted / total, completion / total)


def _format_schema_summary(record_set) -> str:
    attrs = getattr(record_set, "schema_attributes", None) or []
    if not attrs:
        return "(no schema attributes defined)"
    lines = []
    for a in attrs:
        desc = getattr(a, "description", "") or ""
        lines.append(f"- {a.name}: {desc.strip() or '(no description)'}")
    return "\n".join(lines)


def _format_coverage(record_set) -> str:
    attrs = getattr(record_set, "schema_attributes", None) or []
    records = getattr(record_set, "records", None) or []
    if not attrs or not records:
        return "(no data yet)"
    parts = []
    for a in attrs:
        filled = sum(
            1 for r in records
            if a.name in r.attributes and r.attributes[a.name].value
        )
        ratio = filled / len(records) if records else 0.0
        parts.append(f"{a.name}={ratio:.0%}")
    return ", ".join(parts)


def _format_value_distribution(record_set, top: int = 5) -> str:
    attrs = getattr(record_set, "schema_attributes", None) or []
    records = getattr(record_set, "records", None) or []
    if not attrs or not records:
        return "(no data yet)"
    lines: list[str] = []
    for a in attrs:
        counts: dict[str, int] = {}
        for r in records:
            av = r.attributes.get(a.name) or r.additional_attributes.get(a.name)
            if not av:
                continue
            for sv in av.values:
                val = (sv.value or "").strip()
                if not val:
                    continue
                counts[val] = counts.get(val, 0) + 1
        if not counts:
            lines.append(f"  - {a.name}: (no values)")
            continue
        top_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
        joined = ", ".join(f"{v}({n})" for v, n in top_items)
        lines.append(f"  - {a.name}: {joined}")
    return "\n".join(lines) or "(no values yet)"


def _format_history(history: list[IterationSummary]) -> str:
    if not history:
        return "(no prior iterations)"
    parts = []
    for h in history:
        verdict_bit = ""
        if h.judge is not None:
            verdict_bit = (
                f", judge={'COMPLETE' if h.judge.complete else 'continue'}"
                f"@{h.judge.confidence:.2f}"
            )
        parts.append(
            f"  - iter {h.index}: +{h.new_entities} new (total {h.total_entities}), "
            f"{h.queries_run} queries, ${h.cost_usd:.2f}, "
            f"phase_split={h.phase_split}{verdict_bit}"
        )
    return "\n".join(parts)


async def judge_completeness(
    llm,
    record_set,
    history: list[IterationSummary],
) -> CompletenessVerdict:
    """Single LLM call to decide whether to keep iterating."""
    try:
        result = await llm.structured_completion(
            prompt=JUDGE_PROMPT,
            response_format=JUDGE_SCHEMA,
            variables={
                "category": getattr(record_set, "category", "") or "(unknown)",
                "guidance": getattr(record_set, "guidance", "") or "(none)",
                "schema_summary": _format_schema_summary(record_set),
                "entity_count": len(getattr(record_set, "records", []) or []),
                "coverage_summary": _format_coverage(record_set),
                "value_distribution": _format_value_distribution(record_set),
                "history_summary": _format_history(history),
            },
        )
    except Exception as e:  # noqa: BLE001
        _logger.warning("judge_completeness LLM call failed: %s", e)
        return CompletenessVerdict(
            complete=False,
            confidence=0.0,
            reason=f"Judge call failed: {e}",
        )

    return CompletenessVerdict(
        complete=bool(result.get("complete", False)),
        confidence=float(result.get("confidence", 0.0) or 0.0),
        reason=str(result.get("reason", "") or ""),
        missing_gaps=[str(g) for g in (result.get("missing_gaps") or []) if str(g).strip()],
        suggested_queries=[
            str(q) for q in (result.get("suggested_queries") or []) if str(q).strip()
        ],
    )


def _build_guidance_overlay(
    base_guidance: str,
    verdict: CompletenessVerdict,
) -> str:
    """Append the judge's gap/query hints onto the base guidance.

    The verdict is volatile per iteration, so we always rebuild from the
    *base* guidance — never compound previous overlays on top of each
    other.
    """
    base = (base_guidance or "").rstrip()
    bits: list[str] = []
    if verdict.missing_gaps:
        bits.append(
            "Auto-mode focus for the next iteration — these gaps need "
            "attention: " + "; ".join(verdict.missing_gaps)
        )
    if verdict.suggested_queries:
        bits.append(
            "Concrete query ideas to consider: "
            + "; ".join(f"\"{q}\"" for q in verdict.suggested_queries)
        )
    if not bits:
        return base
    overlay = "\n\n".join(bits)
    return f"{base}\n\n{overlay}" if base else overlay


async def run_auto_loop(
    schemify,
    *,
    max_iterations: int = 5,
    per_iter_query_budget: int = 30,
    concurrency: int = 5,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str, dict], None]] = None,
    judge_callback: Optional[
        Callable[[Any, list[IterationSummary]], Awaitable[CompletenessVerdict]]
    ] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    normalize_every: int = 3,
    min_iterations: int = 2,
    confidence_threshold: float = 0.7,
) -> AutoRunResult:
    """Drive Schemify in a loop until the judge says stop (or cap is hit).

    Args:
        schemify: An initialised ``Schemify`` instance.
        max_iterations: Hard ceiling on iterations.
        per_iter_query_budget: ``max_queries`` per discovery pass.
        concurrency: Parallel web-search concurrency.
        output_dir: Run directory for snapshots / logs.
        progress_callback: ``(phase, info)`` callback — phase is one of
            ``"Discovery"``, ``"Verification"``, ``"Normalize"``,
            ``"Judging"``, ``"Done"``.
        judge_callback: Async judge override (mostly for testing).
            Defaults to :func:`judge_completeness` against the schemify
            LLM.
        should_stop: Cooperative cancellation predicate — checked at the
            top of every iteration.
        normalize_every: Run normalize every N iterations during the
            loop. A final normalize always runs after the loop exits.
        min_iterations: Never stop before this many iterations have
            completed, even if the judge says complete.
        confidence_threshold: Minimum judge confidence required to
            actually stop (paired with ``complete=True``).
    """
    if schemify is None or schemify.record_set is None:
        raise ValueError("Schemify must be initialised before run_auto_loop")

    record_set = schemify.record_set
    base_guidance = record_set.guidance or ""
    llm = schemify.llm

    history: list[IterationSummary] = []
    stop_reason = "max_iterations"
    iterations_completed = 0

    def _report(phase: str, **info):
        if progress_callback is None:
            return
        try:
            progress_callback(phase, info)
        except Exception as e:  # noqa: BLE001
            _logger.warning("auto progress_callback failed: %s", e)

    def _cost() -> float:
        try:
            return float(getattr(llm, "total_cost", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def _entities() -> int:
        return len(getattr(record_set, "records", []) or [])

    def _queries() -> int:
        return int(getattr(schemify, "_query_counter", 0) or 0)

    for i in range(1, max_iterations + 1):
        if should_stop is not None and should_stop():
            stop_reason = "user_stop"
            break

        entities_before = _entities()
        queries_before = _queries()
        phase_split = _adaptive_phase_split(i, max_iterations)

        _report(
            "Discovery",
            iteration=i,
            max_iterations=max_iterations,
            phase_split=phase_split,
            entity_count=entities_before,
        )

        try:
            await schemify.run_agentic(
                max_queries=per_iter_query_budget,
                concurrency=concurrency,
                output_dir=output_dir,
                phase_split=phase_split,
            )
        except Exception as e:  # noqa: BLE001
            _logger.exception("auto-mode discovery iteration %d failed", i)
            stop_reason = f"discovery_error: {e}"
            break

        _report(
            "Verification",
            iteration=i,
            max_iterations=max_iterations,
            entity_count=_entities(),
        )
        try:
            await schemify.verify_unverified(concurrency=max(2, concurrency))
        except Exception as e:  # noqa: BLE001
            _logger.warning("auto-mode verify iteration %d failed: %s", i, e)

        ran_normalize = False
        if normalize_every > 0 and (i % normalize_every == 0):
            _report(
                "Normalize",
                iteration=i,
                max_iterations=max_iterations,
                entity_count=_entities(),
            )
            try:
                await schemify.normalize()
                ran_normalize = True
            except Exception as e:  # noqa: BLE001
                _logger.warning("auto-mode normalize iteration %d failed: %s", i, e)

        summary = IterationSummary(
            index=i,
            new_entities=max(0, _entities() - entities_before),
            total_entities=_entities(),
            queries_run=max(0, _queries() - queries_before),
            cost_usd=_cost(),
            phase_split=phase_split,
        )

        verdict: Optional[CompletenessVerdict] = None
        if i >= min_iterations:
            _report(
                "Judging",
                iteration=i,
                max_iterations=max_iterations,
                entity_count=_entities(),
            )
            try:
                judge = judge_callback or (
                    lambda rs, hist: judge_completeness(llm, rs, hist)
                )
                verdict = await judge(record_set, history + [summary])
            except Exception as e:  # noqa: BLE001
                _logger.warning("auto-mode judge iteration %d failed: %s", i, e)
                verdict = CompletenessVerdict(
                    complete=False,
                    confidence=0.0,
                    reason=f"Judge error: {e}",
                )
            summary.judge = verdict

            # Re-bias next iteration's guidance based on what the judge
            # surfaced. Always rebuild from base_guidance so previous
            # overlays don't compound.
            if not verdict.complete or verdict.confidence < confidence_threshold:
                record_set.guidance = _build_guidance_overlay(base_guidance, verdict)
            else:
                record_set.guidance = base_guidance

        history.append(summary)
        iterations_completed = i

        if (
            verdict is not None
            and verdict.complete
            and verdict.confidence >= confidence_threshold
            and i >= min_iterations
        ):
            stop_reason = "judge_complete"
            break

    # Always restore base guidance for the final passes so we don't
    # carry transient overlays into the saved snapshot.
    record_set.guidance = base_guidance

    _report(
        "Normalize",
        iteration=iterations_completed,
        max_iterations=max_iterations,
        entity_count=_entities(),
        final=True,
    )
    try:
        await schemify.normalize()
    except Exception as e:  # noqa: BLE001
        _logger.warning("auto-mode final normalize failed: %s", e)

    try:
        schemify.finalize(output_dir=output_dir)
    except TypeError:
        try:
            schemify.finalize()
        except Exception as e:  # noqa: BLE001
            _logger.warning("auto-mode finalize failed: %s", e)
    except Exception as e:  # noqa: BLE001
        _logger.warning("auto-mode finalize failed: %s", e)

    _report(
        "Done",
        iteration=iterations_completed,
        max_iterations=max_iterations,
        entity_count=_entities(),
        stop_reason=stop_reason,
    )

    return AutoRunResult(
        iterations_completed=iterations_completed,
        final_entity_count=_entities(),
        final_cost=_cost(),
        stop_reason=stop_reason,
        per_iteration_history=history,
    )


# ---------------------------------------------------------------------------
# Rediscovery benchmark — Phase B (compare) + Phase C (gap-fill probe)
# ---------------------------------------------------------------------------

_PUNCT_RX = None  # lazily compiled
_TOKEN_RX = None  # lazily compiled

# Tokens that contribute no discriminating signal in this domain. Without
# this filter, e.g. reference "Microsoft PhotoDNA" gets matched to any
# random "Microsoft …" entity by token overlap.
_STOPWORDS: frozenset[str] = frozenset({
    # Generic English
    "the", "a", "an", "of", "and", "or", "for", "in", "on", "to", "at",
    "by", "from", "with", "is", "as", "it", "its", "be", "this", "that",
    # Domain-generic descriptors (anti-trafficking / tooling)
    "human", "trafficking", "anti", "antitrafficking", "exploit",
    "exploitation", "abuse", "abusive", "victim", "victims", "survivor",
    "survivors", "child", "children", "labor", "labour", "forced",
    "sexual", "sex", "material", "content", "csam",
    # Generic tool/platform descriptors
    "tool", "tools", "app", "apps", "application", "applications",
    "platform", "platforms", "system", "systems", "service", "services",
    "program", "programme", "programs", "programmes", "project", "projects",
    "initiative", "initiatives", "campaign", "campaigns",
    "hotline", "hotlines", "report", "reporting", "data", "dataset",
    "datasets", "portal", "portals", "website", "online", "api", "ai",
    "ml", "machine", "learning", "model", "models", "detection",
    "detector", "monitoring", "screening", "screen", "screener",
    "intelligence", "investigation", "investigations", "investigative",
    "investigator", "case", "management", "manager", "managed",
    "suspicious", "activity", "transaction", "transactions",
    "complaint", "complaints", "referral", "referrals", "support",
    "supporting", "network", "alliance", "centre", "center",
    "rating", "ratings", "risk", "compliance", "supply", "chain",
    "trace", "tracing", "trafficker", "traffickers",
    # Geo descriptors
    "national", "international", "global", "regional", "country", "world",
    # Misc generic
    "modern", "slavery", "slave", "task", "force", "module", "modules",
    "assessment", "software", "secure", "secured", "based", "web",
    "communication", "interactive", "pilot", "pilots", "mechanism",
    "mechanisms", "solution", "solutions",
})


def _normalize_label(s: str) -> str:
    """Casefold + strip non-alphanumerics for comparison.

    Aggressive enough that "STOP THE TRAFFIK APP" matches "Stop the
    Traffik app", "STOPtheTRAFFIK App!", etc., but not so loose that
    "MEMEX" matches "memex search".
    """
    global _PUNCT_RX  # noqa: PLW0603
    if _PUNCT_RX is None:
        import re as _re  # noqa: PLC0415
        _PUNCT_RX = _re.compile(r"[^a-z0-9]+")
    return _PUNCT_RX.sub("", (s or "").casefold())


def _token_set(s: str) -> frozenset[str]:
    """Return the set of discriminating tokens in a label.

    Lowercased, punctuation-split, stopwords removed, dropping tokens
    shorter than 3 chars unless purely numeric (years stay).
    """
    global _TOKEN_RX  # noqa: PLW0603
    if _TOKEN_RX is None:
        import re as _re  # noqa: PLC0415
        _TOKEN_RX = _re.compile(r"[a-z0-9]+")
    raw = _TOKEN_RX.findall((s or "").casefold())
    out: set[str] = set()
    for t in raw:
        if t in _STOPWORDS:
            continue
        if len(t) < 3 and not t.isdigit():
            continue
        out.add(t)
    return frozenset(out)


def _fuzzy_match(
    ref_key: str,
    ref_tokens: frozenset[str],
    discovered_keys: dict[str, str],
    discovered_token_index: list[tuple[frozenset[str], str]],
    *,
    jaccard_threshold: float = 0.6,
    containment_threshold: float = 0.85,
) -> Optional[str]:
    """Return a discovered canonical label that fuzzy-matches the reference.

    Strategy:
    1. Exact key match.
    2. Substring containment of the normalized key, either direction.
    3. Token-set match: at least one ``strong`` shared token (length ≥ 5,
       not a stopword), AND either Jaccard ≥ jaccard_threshold OR
       smaller-set containment ≥ containment_threshold with ≥ 2 shared
       tokens overall. The "strong token" requirement is what keeps
       common noise tokens like "case", "monitoring", "csam" from
       driving spurious matches.
    """
    if ref_key and ref_key in discovered_keys:
        return discovered_keys[ref_key]

    if ref_key and len(ref_key) >= 8:
        for k, canon in discovered_keys.items():
            if not k or len(k) < 8:
                continue
            if ref_key in k or k in ref_key:
                # Guard against tiny shared substrings driving false positives.
                if min(len(ref_key), len(k)) >= max(8, int(0.6 * max(len(ref_key), len(k)))):
                    return canon

    if not ref_tokens:
        return None

    strong_ref = {t for t in ref_tokens if len(t) >= 5}

    best: Optional[str] = None
    best_score = 0.0
    for tokens, canon in discovered_token_index:
        if not tokens:
            continue
        common = ref_tokens & tokens
        if len(common) < 2:
            continue
        strong_common = strong_ref & tokens
        if not strong_common:
            # No discriminating overlap — only generic descriptors match.
            continue
        union = ref_tokens | tokens
        jaccard = len(common) / len(union) if union else 0.0
        containment = len(common) / min(len(ref_tokens), len(tokens))
        if jaccard < jaccard_threshold and containment < containment_threshold:
            continue
        score = max(jaccard, containment)
        if score > best_score:
            best_score = score
            best = canon
    return best


def _collect_discovered_keys(record_set) -> dict[str, str]:
    """Return {normalized_key: canonical_label} for every record + alias."""
    out: dict[str, str] = {}
    for r in getattr(record_set, "records", None) or []:
        canon = getattr(r, "label", "") or ""
        for name in [canon] + list(getattr(r, "aliases", None) or []):
            k = _normalize_label(name)
            if k:
                out.setdefault(k, canon)
    return out


def _collect_discovered_token_index(
    record_set,
) -> list[tuple[frozenset[str], str]]:
    """Return [(token_set, canonical_label)] for every record + alias.

    Stable list (not a dict) because two distinct entities can share
    identical token sets and we want to consider both when matching.
    """
    out: list[tuple[frozenset[str], str]] = []
    for r in getattr(record_set, "records", None) or []:
        canon = getattr(r, "label", "") or ""
        for name in [canon] + list(getattr(r, "aliases", None) or []):
            ts = _token_set(name)
            if ts:
                out.append((ts, canon))
    return out


def classify_reference_matches(
    record_set,
    reference_labels: list[str],
) -> dict[str, Any]:
    """Diff reference labels against discovered entities, exact + fuzzy.

    Returns a dict with:
      - ``reference_dedup``: unique reference labels (in input order)
      - ``exact_matches``: list of (ref, discovered_canonical)
      - ``fuzzy_matches``: list of (ref, discovered_canonical)
      - ``missed``: list of reference labels with no exact or fuzzy hit
    """
    discovered_keys = _collect_discovered_keys(record_set)
    token_index = _collect_discovered_token_index(record_set)

    seen: set[str] = set()
    ref_dedup: list[str] = []
    for raw in reference_labels or []:
        s = (raw or "").strip()
        k = _normalize_label(s)
        if not k or k in seen:
            continue
        seen.add(k)
        ref_dedup.append(s)

    exact: list[tuple[str, str]] = []
    fuzzy: list[tuple[str, str]] = []
    missed: list[str] = []
    for ref in ref_dedup:
        rk = _normalize_label(ref)
        if rk in discovered_keys:
            exact.append((ref, discovered_keys[rk]))
            continue
        hit = _fuzzy_match(rk, _token_set(ref), discovered_keys, token_index)
        if hit:
            fuzzy.append((ref, hit))
        else:
            missed.append(ref)

    return {
        "reference_dedup": ref_dedup,
        "exact_matches": exact,
        "fuzzy_matches": fuzzy,
        "missed": missed,
    }


def _record_has_sources(rec) -> bool:
    """True iff at least one attribute value on the record cites a URL."""
    for av in list((getattr(rec, "attributes", None) or {}).values()) + list(
        (getattr(rec, "additional_attributes", None) or {}).values()
    ):
        for sv in getattr(av, "values", None) or []:
            srcs = getattr(sv, "sources", None) or []
            for src in srcs:
                url = getattr(src, "url", None) or (
                    src.get("url") if isinstance(src, dict) else None
                )
                if url:
                    return True
    return False


@dataclass
class GapFillResult:
    """Outcome of the rediscovery comparison + gap-fill probe."""

    reference_count: int
    rediscovered_before: int          # exact-key blind-discovery overlap
    fuzzy_matches_before: int         # additional fuzzy hits (token-set)
    missed_before: list[str]          # reference names not in blind set
    recovered_via_gap_fill: int       # of missed_before, now grounded
    still_missing: list[str]          # never sourced even after gap-fill
    gap_fill_queries_run: int
    gap_fill_cost: float

    @property
    def blind_recall(self) -> float:
        if not self.reference_count:
            return 0.0
        return self.rediscovered_before / self.reference_count

    @property
    def blind_recall_fuzzy(self) -> float:
        if not self.reference_count:
            return 0.0
        return (
            self.rediscovered_before + self.fuzzy_matches_before
        ) / self.reference_count

    @property
    def total_recall(self) -> float:
        if not self.reference_count:
            return 0.0
        return (
            self.rediscovered_before
            + self.fuzzy_matches_before
            + self.recovered_via_gap_fill
        ) / self.reference_count

    def to_dict(self) -> dict:
        return {
            "reference_count": self.reference_count,
            "rediscovered_before": self.rediscovered_before,
            "fuzzy_matches_before": self.fuzzy_matches_before,
            "missed_before_count": len(self.missed_before),
            "recovered_via_gap_fill": self.recovered_via_gap_fill,
            "still_missing_count": len(self.still_missing),
            "blind_recall": round(self.blind_recall, 4),
            "blind_recall_fuzzy": round(self.blind_recall_fuzzy, 4),
            "total_recall": round(self.total_recall, 4),
            "gap_fill_queries_run": self.gap_fill_queries_run,
            "gap_fill_cost": round(self.gap_fill_cost, 4),
            "still_missing_sample": list(self.still_missing[:50]),
        }


async def gap_fill_against_reference(
    schemify,
    reference_labels: list[str],
    *,
    gap_fill_query_budget: int = 200,
    concurrency: int = 5,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str, dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> GapFillResult:
    """Compare discovered set to reference, then probe missed names.

    Phase B: diff `reference_labels` against the schemify record set
    (canonical labels + aliases, normalized to alphanumerics only).

    Phase C: seed each missed reference label as a blank candidate record
    and run a SINGLE targeted-only ``run_agentic`` pass (phase split
    ``(0.0, 1.0, 0.0)``) so the agent issues per-entity grounded
    searches. After the pass, any seeded record that gained at least one
    URL-cited attribute value counts as "recovered"; the rest stay in
    ``still_missing``.

    Records added during gap-fill that remain unsourced are pruned from
    the record set so the final dataset only contains entities backed by
    web evidence.
    """
    if schemify is None or schemify.record_set is None:
        raise ValueError("Schemify must be initialised before gap_fill_against_reference")

    record_set = schemify.record_set
    llm = schemify.llm

    def _report(phase: str, **info):
        if progress_callback is None:
            return
        try:
            progress_callback(phase, info)
        except Exception as e:  # noqa: BLE001
            _logger.warning("gap_fill progress_callback failed: %s", e)

    def _cost() -> float:
        try:
            return float(getattr(llm, "total_cost", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def _queries() -> int:
        return int(getattr(schemify, "_query_counter", 0) or 0)

    # ---- Phase B: compare (exact + fuzzy) -------------------------------
    classify = classify_reference_matches(record_set, reference_labels)
    reference_dedup = classify["reference_dedup"]
    exact_matches: list[tuple[str, str]] = classify["exact_matches"]
    fuzzy_matches: list[tuple[str, str]] = classify["fuzzy_matches"]
    missed: list[str] = classify["missed"]

    _report(
        "GapFillCompare",
        reference_count=len(reference_dedup),
        rediscovered_before=len(exact_matches),
        fuzzy_matches_before=len(fuzzy_matches),
        missed_before=len(missed),
    )

    if not missed:
        return GapFillResult(
            reference_count=len(reference_dedup),
            rediscovered_before=len(exact_matches),
            fuzzy_matches_before=len(fuzzy_matches),
            missed_before=[],
            recovered_via_gap_fill=0,
            still_missing=[],
            gap_fill_queries_run=0,
            gap_fill_cost=0.0,
        )

    # ---- Phase C: seed missed labels + targeted-only research ------------
    from intelligence_toolkit.schemify.models import Record  # noqa: PLC0415

    seeded_keys: set[str] = set()
    for name in missed:
        if should_stop is not None and should_stop():
            break
        k = _normalize_label(name)
        if not k:
            continue
        rec = Record(label=name.upper())
        ok, _ = record_set.add_record(rec, use_fuzzy=False)
        if ok:
            seeded_keys.add(k)

    _report(
        "GapFillSeed",
        seeded=len(seeded_keys),
        missed_total=len(missed),
    )

    # Make sure the agent biases toward filling these specific stubs,
    # not chasing entirely new entities.
    base_guidance = record_set.guidance or ""
    overlay = (
        "Gap-fill probe: the following entity names were seeded as blank "
        "candidate records from a reference dataset, but the broad-discovery "
        "phase did NOT surface them. For each one, search the open web with "
        "the entity name as the primary anchor and gather grounded citations "
        "for as many schema attributes as possible. Do NOT discover new "
        "entities in this pass — focus exclusively on filling these stubs. "
        f"Seeded names ({len(seeded_keys)}): "
        + "; ".join(missed[:60])
        + (" …" if len(missed) > 60 else "")
    )
    record_set.guidance = f"{base_guidance}\n\n{overlay}" if base_guidance else overlay

    queries_before = _queries()
    cost_before = _cost()
    _report(
        "GapFillResearch",
        entity_count=len(record_set.records),
        budget=gap_fill_query_budget,
    )
    try:
        await schemify.run_agentic(
            max_queries=gap_fill_query_budget,
            concurrency=concurrency,
            output_dir=output_dir,
            phase_split=(0.0, 1.0, 0.0),
        )
    except Exception as e:  # noqa: BLE001
        _logger.exception("gap-fill run_agentic failed: %s", e)

    record_set.guidance = base_guidance

    # ---- Verify what got sourced ----------------------------------------
    recovered_keys: set[str] = set()
    surviving: list = []
    pruned_unsourced = 0
    for r in list(record_set.records):
        rk = _normalize_label(getattr(r, "label", ""))
        if rk in seeded_keys:
            if _record_has_sources(r):
                recovered_keys.add(rk)
                surviving.append(r)
            else:
                pruned_unsourced += 1
                continue
        else:
            surviving.append(r)
    record_set.records = surviving

    still_missing = [
        name for name in missed
        if _normalize_label(name) not in recovered_keys
    ]

    _report(
        "GapFillDone",
        recovered=len(recovered_keys),
        still_missing=len(still_missing),
        pruned_unsourced=pruned_unsourced,
        queries_run=max(0, _queries() - queries_before),
        cost=round(_cost() - cost_before, 4),
    )

    return GapFillResult(
        reference_count=len(reference_dedup),
        rediscovered_before=len(exact_matches),
        fuzzy_matches_before=len(fuzzy_matches),
        missed_before=missed,
        recovered_via_gap_fill=len(recovered_keys),
        still_missing=still_missing,
        gap_fill_queries_run=max(0, _queries() - queries_before),
        gap_fill_cost=max(0.0, _cost() - cost_before),
    )
