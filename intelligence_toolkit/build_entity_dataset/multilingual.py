# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Multilingual helpers for the auto mode of Build Entity Dataset.

Two responsibilities:

* :func:`propose_search_languages` — ask an LLM to recommend source
  languages worth searching in for the user's category/guidance, so the
  user can confirm before launching auto mode.
* :func:`make_query_translator` — produce a per-query translator that
  turns one English query into ``[(translated, lang)]`` pairs across the
  selected source languages. Results from foreign sources are extracted
  into English by the existing LLM extraction pipeline.

These helpers are intentionally cheap: one small structured-completion
call per query, cached by ``(query, languages)`` so repeated queries do
not redo translation work.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

_logger = logging.getLogger(__name__)


PROPOSE_LANGUAGES_PROMPT = """\
You recommend SOURCE LANGUAGES for web research about the following topic.

The user wants to build a comprehensive dataset by searching the web in
multiple languages and translating the results into English. Recommend
languages whose documents would meaningfully expand coverage beyond
English-only sources — typically the working languages of regions where
the relevant entities operate, the languages of major stakeholders, or
languages with strong native publishing on the subject.

Category: {category}

Additional guidance / constraints:
{guidance}

Return at most 8 distinct languages, including English. For each
language give the ISO 639-1 code, the English name, and a brief
rationale grounded in the topic (1 short sentence).
"""

PROPOSE_LANGUAGES_SCHEMA: dict[str, Any] = {
    "json_schema": {
        "name": "search_languages",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "code": {"type": "string"},
                            "name": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["code", "name", "rationale"],
                    },
                },
            },
            "required": ["languages"],
        },
    }
}


TRANSLATE_QUERY_PROMPT = """\
Translate the following web-search query into each requested language so
that a native speaker would type it into a search engine. Keep proper
nouns (organization names, product names, place names) in their most
commonly searched form — usually the original Latin spelling unless the
language uses a different script natively for that name.

Target languages (ISO 639-1 codes): {codes}

Search query:
\"\"\"{query}\"\"\"

For each target language, return the translated query verbatim. Do not
add explanations or quotation marks. If a language is not meaningfully
different from the original query (e.g. the query is already in that
language, or is mostly a product name), still return it — it is fine to
repeat the source query.
"""


TRANSLATE_QUERY_SCHEMA: dict[str, Any] = {
    "json_schema": {
        "name": "query_translations",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "code": {"type": "string"},
                            "query": {"type": "string"},
                        },
                        "required": ["code", "query"],
                    },
                },
            },
            "required": ["translations"],
        },
    }
}


def _normalize_lang_codes(languages: list[str]) -> list[str]:
    """Lowercase + dedupe ISO codes while preserving input order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in languages or []:
        code = (raw or "").strip().lower()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


async def propose_search_languages(
    llm, category: str, guidance: str = ""
) -> list[dict]:
    """Ask the LLM to suggest source languages for this topic.

    Always returns English first when present. The caller is expected to
    present this list to the user for editing before launching auto
    mode.

    Returns a list of ``{"code", "name", "rationale"}`` dicts.
    """
    cat = (category or "").strip()
    if not cat:
        return [{"code": "en", "name": "English", "rationale": "Default."}]
    try:
        result = await llm.structured_completion(
            prompt=PROPOSE_LANGUAGES_PROMPT,
            response_format=PROPOSE_LANGUAGES_SCHEMA,
            variables={"category": cat, "guidance": (guidance or "").strip() or "(none)"},
        )
    except Exception as e:  # noqa: BLE001
        _logger.warning("propose_search_languages failed: %s", e)
        return [{"code": "en", "name": "English", "rationale": "Fallback."}]

    languages = result.get("languages") if isinstance(result, dict) else None
    if not isinstance(languages, list) or not languages:
        return [{"code": "en", "name": "English", "rationale": "Fallback."}]

    seen: set[str] = set()
    cleaned: list[dict] = []
    for item in languages:
        if not isinstance(item, dict):
            continue
        code = (item.get("code") or "").strip().lower()
        name = (item.get("name") or "").strip()
        rationale = (item.get("rationale") or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        cleaned.append({"code": code, "name": name or code.upper(), "rationale": rationale})

    # Surface English first if present, for UX consistency.
    cleaned.sort(key=lambda d: 0 if d["code"] == "en" else 1)
    return cleaned or [{"code": "en", "name": "English", "rationale": "Fallback."}]


def make_query_translator(
    llm,
    languages: list[str],
    cache=None,
) -> Callable[[str], Awaitable[list[tuple[str, str]]]]:
    """Build an async translator: ``query -> [(translated, lang), ...]``.

    The translator always includes the original English query (as the
    ``en`` entry) and asks the LLM for one translation per non-English
    target. Short-circuits with no LLM call when ``languages`` is empty
    or English-only.

    ``cache``, if supplied, must follow the ``schemify.cache.Cache``
    contract (``get(op, **kw)`` / ``set(value, op, **kw)``). Translations
    are keyed by the query plus the sorted language set so repeated
    queries inside a run skip the LLM round-trip.
    """
    codes = _normalize_lang_codes(languages)
    # Make sure English is present so the original query always shows up
    # in the fan-out — otherwise users would lose English coverage as
    # soon as they configured any non-English language.
    if not codes:
        codes = ["en"]
    elif "en" not in codes:
        codes = ["en", *codes]

    non_en = [c for c in codes if c != "en"]
    cache_key_langs = ",".join(sorted(codes))

    async def translate(query: str) -> list[tuple[str, str]]:
        q = (query or "").strip()
        if not q:
            return []
        if not non_en:
            return [(q, "en")]

        if cache is not None:
            try:
                cached = cache.get("auto_translate_query", query=q, langs=cache_key_langs)
            except Exception as e:  # noqa: BLE001
                _logger.warning("translate cache.get failed: %s", e)
                cached = None
            if isinstance(cached, list) and cached:
                return [(str(t), str(l)) for t, l in cached if t and l]

        try:
            result = await llm.structured_completion(
                prompt=TRANSLATE_QUERY_PROMPT,
                response_format=TRANSLATE_QUERY_SCHEMA,
                variables={
                    "codes": ", ".join(non_en),
                    "query": q.replace('"""', '"'),
                },
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("translate_query LLM call failed: %s", e)
            return [(q, "en")]

        translations: list[tuple[str, str]] = [(q, "en")]
        items = result.get("translations") if isinstance(result, dict) else None
        if isinstance(items, list):
            seen_codes: set[str] = {"en"}
            for item in items:
                if not isinstance(item, dict):
                    continue
                code = (item.get("code") or "").strip().lower()
                text = (item.get("query") or "").strip()
                if not code or not text or code in seen_codes:
                    continue
                seen_codes.add(code)
                translations.append((text, code))

        if cache is not None:
            try:
                cache.set(translations, "auto_translate_query", query=q, langs=cache_key_langs)
            except Exception as e:  # noqa: BLE001
                _logger.warning("translate cache.set failed: %s", e)

        return translations

    return translate


def estimate_query_multiplier(languages: list[str]) -> int:
    """How many search calls each agent-issued query will fan out into."""
    codes = _normalize_lang_codes(languages)
    if not codes:
        return 1
    return len(codes) if "en" in codes else len(codes) + 1
