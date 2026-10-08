"""LLM-backed verifier for fuzzy-matched entity merges.

Fuzzy string similarity alone cannot tell that
"INTELLIGENCE TOOLKIT" (a Microsoft Research data-synthesis toolkit) and
"INTELLIGRADE" (IWF's CSAM grading system) are different entities. The
arbiter asks an LLM to confirm — with full attribute and source context —
before any automated merge is applied.

Public API:
    MergeArbiter(llm).verify_pairs(pairs) -> dict[(a,b)->Verdict]
    MergeArbiter(llm).verify_pair(a, b) -> Verdict

Pairs are tuples of (canonical_label, candidate_label). Records are
passed via a resolver callback so this module stays decoupled from
RecordSet shape.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeVerdict:
    same_entity: bool
    confidence: float
    reason: str


_PROMPT = """\
You are auditing automated entity-deduplication decisions. A fuzzy
string-similarity scorer flagged these label pairs as POSSIBLE duplicates,
but fuzzy scores are unreliable — many false positives share prefixes
or tokens without referring to the same real-world entity.

For each pair decide whether the two labels refer to the SAME real-world
entity (the same tool, organization, platform, product, hotline, etc.),
or to DIFFERENT entities that happen to have similar names.

Be CONSERVATIVE. When in doubt, return same_entity=false. It is far
worse to incorrectly merge two distinct entities (which corrupts
attributes with values from the wrong source) than to leave a duplicate.

Treat as SAME entity:
  - Pure name variants ("MICROSOFT CORP" vs "MICROSOFT CORPORATION")
  - Acronym vs full name ("UNODC" vs "UNITED NATIONS OFFICE ON DRUGS AND CRIME")
  - Product with/without org prefix ("POLARIS HOTLINE" vs "NATIONAL HUMAN TRAFFICKING HOTLINE BY POLARIS"), if context confirms
  - Typos or punctuation differences

Treat as DIFFERENT entities (DO NOT MERGE):
  - Different products from the same organization
  - Different organizations sharing a common word
  - A platform vs a project that uses it
  - Names that merely share a prefix or token ("INTELLIGRADE" vs "INTELLIGENCE TOOLKIT")
  - Different geographic instances (e.g., a hotline per country)
  - Sibling products on the same vendor catalog page — a shared prefix
    like "TRANSFORMATIVE TECH: X" / "TRANSFORMATIVE TECH: Y" or a shared
    URL like `vendor.com/products` is evidence they were listed on the
    same page, NOT that they are the same product. Compare the
    `description` fields: if the two entities clearly describe different
    functionality (e.g. a mobile app vs. a remediation system, a dataset
    vs. an analysis tool), they are DIFFERENT even when organization,
    source domain, and naming template all match.

For each pair, return:
  - pair_id (echo back)
  - same_entity (bool)
  - confidence (0.0-1.0; how confident in the decision)
  - reason (one short sentence)

Pairs to evaluate:
{pairs_block}
"""


_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "merge_arbitration",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_id": {"type": "integer"},
                            "same_entity": {"type": "boolean"},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["pair_id", "same_entity", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        },
    },
}


def _record_brief(record) -> dict:
    """Compact, LLM-friendly summary of a Record's identifying attributes."""
    if record is None:
        return {}

    def _attr(name: str):
        a = getattr(record, "attributes", {}).get(name) \
            or getattr(record, "additional_attributes", {}).get(name)
        if a is None:
            return None
        v = getattr(a, "value", None)
        if not v:
            vs = getattr(a, "values", None) or []
            v = [getattr(sv, "value", None) for sv in vs if getattr(sv, "value", None)]
            v = v or None
        return v

    src_domains: list[str] = []
    seen: set[str] = set()
    for bucket in ("attributes", "additional_attributes"):
        for av in (getattr(record, bucket, {}) or {}).values():
            for sv in (getattr(av, "values", None) or []):
                for s in (getattr(sv, "sources", None) or []):
                    url = getattr(s, "url", None) or (s.get("url") if isinstance(s, dict) else None)
                    if not url:
                        continue
                    try:
                        host = url.split("//", 1)[-1].split("/", 1)[0].lower()
                    except Exception:
                        continue
                    if host and host not in seen:
                        seen.add(host)
                        src_domains.append(host)
                        if len(src_domains) >= 6:
                            break
                if len(src_domains) >= 6:
                    break
            if len(src_domains) >= 6:
                break
        if len(src_domains) >= 6:
            break

    return {
        "label": getattr(record, "label", ""),
        "aliases": list(getattr(record, "aliases", []) or [])[:8],
        "description": _attr("Tool Description") or _attr("Description") or "",
        "organization": _attr("Organization Name") or _attr("Organization") or "",
        "organization_type": _attr("Organization Type") or "",
        "target_users": _attr("Target Users") or "",
        "trafficking_type": _attr("Trafficking Type") or "",
        "source_domains": src_domains,
    }


class MergeArbiter:
    """Verifies fuzzy-match merge candidates with an LLM."""

    def __init__(
        self,
        llm,
        *,
        record_resolver: Optional[Callable[[str], object]] = None,
        confidence_threshold: float = 0.8,
        batch_size: int = 12,
        cache: Optional[dict] = None,
    ) -> None:
        self.llm = llm
        self.record_resolver = record_resolver
        self.confidence_threshold = confidence_threshold
        self.batch_size = max(1, int(batch_size))
        self._cache: dict[tuple[str, str], MergeVerdict] = cache if cache is not None else {}

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted([(a or "").strip().upper(), (b or "").strip().upper()]))

    def get_cached(self, a: str, b: str) -> Optional[MergeVerdict]:
        return self._cache.get(self._key(a, b))

    def remember(self, a: str, b: str, verdict: MergeVerdict) -> None:
        self._cache[self._key(a, b)] = verdict

    async def verify_pair(self, a: str, b: str) -> MergeVerdict:
        result = await self.verify_pairs([(a, b)])
        return result[self._key(a, b)]

    async def verify_pairs(
        self, pairs: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], MergeVerdict]:
        """Return a verdict for every unique pair. Cached pairs reused."""
        unique: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        out: dict[tuple[str, str], MergeVerdict] = {}
        for a, b in pairs:
            k = self._key(a, b)
            if k in seen:
                continue
            seen.add(k)
            if k in self._cache:
                out[k] = self._cache[k]
                continue
            unique.append((a, b))

        if not unique:
            return out

        if self.llm is None:
            for a, b in unique:
                v = MergeVerdict(False, 0.0, "no LLM configured")
                self.remember(a, b, v)
                out[self._key(a, b)] = v
            return out

        for i in range(0, len(unique), self.batch_size):
            batch = unique[i : i + self.batch_size]
            batch_out = await self._call_llm(batch)
            for k, v in batch_out.items():
                self._cache[k] = v
                out[k] = v
        return out

    async def _call_llm(
        self, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], MergeVerdict]:
        blocks: list[str] = []
        for pid, (a, b) in enumerate(pairs, start=1):
            rec_a = self.record_resolver(a) if self.record_resolver else None
            rec_b = self.record_resolver(b) if self.record_resolver else None
            blocks.append(
                f"Pair {pid}:\n"
                f"  A: {json.dumps(_record_brief(rec_a) or {'label': a}, ensure_ascii=False)}\n"
                f"  B: {json.dumps(_record_brief(rec_b) or {'label': b}, ensure_ascii=False)}"
            )
        try:
            result = await self.llm.structured_completion(
                prompt=_PROMPT,
                response_format=_SCHEMA,
                variables={"pairs_block": "\n\n".join(blocks)},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("merge arbiter LLM call failed: %s — refusing all", e)
            out: dict[tuple[str, str], MergeVerdict] = {}
            for a, b in pairs:
                out[self._key(a, b)] = MergeVerdict(
                    False, 0.0, f"arbiter error: {e}"
                )
            return out

        decisions = {
            int(d.get("pair_id", -1)): d
            for d in (result.get("decisions") or [])
            if isinstance(d, dict)
        }
        out: dict[tuple[str, str], MergeVerdict] = {}
        for pid, (a, b) in enumerate(pairs, start=1):
            d = decisions.get(pid)
            if d is None:
                v = MergeVerdict(False, 0.0, "missing arbiter decision")
            else:
                v = MergeVerdict(
                    same_entity=bool(d.get("same_entity")),
                    confidence=float(d.get("confidence") or 0.0),
                    reason=str(d.get("reason") or ""),
                )
            out[self._key(a, b)] = v
        return out

    def approves(self, verdict: MergeVerdict) -> bool:
        return bool(
            verdict.same_entity and verdict.confidence >= self.confidence_threshold
        )
