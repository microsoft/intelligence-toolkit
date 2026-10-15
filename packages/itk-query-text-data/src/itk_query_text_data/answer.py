# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Turn search results into a themed, source-linked report."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from itk_query_text_data import prompts, schemas
from itk_query_text_data.analysis import source_details, system_message
from itk_query_text_data.models import Answer

if TYPE_CHECKING:
    from itk_query_text_data.llm import LLM
    from itk_query_text_data.models import Corpus, SearchResult, TokenHandler

log = logging.getLogger(__name__)

_SOURCE_SPAN = re.compile(r"(\s*)\[source: ([^\]]+)\]")
_EVIDENCE_PREFIX = "**Source evidence**:"
_COMMENTARY_PREFIX = "**AI commentary**:"
_REPORT_FIELDS = ("answer", "report_title", "report_overview", "report_implications")


def link_sources(text: str, corpus: Corpus) -> tuple[str, list[int]]:
    """Turn ``[source: 1, 2]`` citations into anchor links.

    Ids that are not chunks in the corpus are dropped from the links and the
    returned references; a citation with no valid id is removed.
    """
    references: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        parts = [part.strip() for part in re.split(r"[,;]", match.group(2))]
        ids = [int(part) for part in parts if part.isdigit() and int(part) in corpus]
        if not ids:
            return ""
        references.update(ids)
        links = ", ".join(f"[{i}](#source-{i})" for i in ids)
        return f"{match.group(1)}[source: {links}]"

    return _SOURCE_SPAN.sub(replace, text), sorted(references)


def _parse_theme(response: str) -> dict[str, Any] | None:
    try:
        theme = json.loads(response)
    except json.JSONDecodeError:
        log.warning("Skipping theme summary that is not valid JSON")
        return None
    if not isinstance(theme, dict) or not theme.get("theme_points"):
        log.warning("Skipping theme summary without points")
        return None
    for point in theme["theme_points"]:
        for key, prefix in (
            ("point_evidence", _EVIDENCE_PREFIX),
            ("point_commentary", _COMMENTARY_PREFIX),
        ):
            if key in point:
                value = point[key].strip()
                point[key] = value if value.startswith(prefix) else f"{prefix} {value}"
    theme.setdefault("theme_title", "")
    return theme


def _report_markdown(
    result: SearchResult, themes: list[dict[str, Any]], report: dict[str, str]
) -> str:
    output = (
        f"## Query\n\n*{result.query}*\n\n"
        f"## Expanded Query\n\n*{result.expanded_query}*\n\n"
        f"## Answer\n\n{report['answer']}\n\n"
        f"## Analysis\n\n### {report['report_title']}\n\n{report['report_overview']}\n\n"
    )
    for theme in themes:
        output += f"#### Theme: {theme['theme_title']}\n\n"
        for point in theme["theme_points"]:
            output += (
                f"##### {point.get('point_title', '')}\n\n"
                f"{point.get('point_evidence', '')}\n\n"
                f"{point.get('point_commentary', '')}\n\n"
            )
    return output + f"#### Implications\n\n{report['report_implications']}\n\n"


async def build_answer(result: SearchResult, corpus: Corpus, llm: LLM) -> Answer:
    """Summarize each theme of relevant chunks, then integrate them into one report."""
    titles = list(result.themes)
    summaries = await asyncio.gather(
        *(
            llm.complete(
                system_message(
                    prompts.theme_summarization_prompt,
                    chunks=[
                        f"{chunk_id}: {corpus[chunk_id].to_prompt()}"
                        for chunk_id in chunk_ids
                        if chunk_id in corpus
                    ],
                    theme=title,
                    previous_themes=titles[:position],
                    query=result.expanded_query,
                ),
                response_format=schemas.theme_summarization_format,
            )
            for position, (title, chunk_ids) in enumerate(result.themes.items())
        )
    )
    themes = [theme for s in summaries if (theme := _parse_theme(s)) is not None]

    integration = await llm.complete(
        system_message(
            prompts.theme_integration_prompt,
            content=[json.dumps(theme) for theme in themes],
            query=result.query,
        ),
        response_format=schemas.theme_integration_format,
    )
    parsed = json.loads(integration)
    report = {field: str(parsed.get(field, "")) for field in _REPORT_FIELDS}

    markdown, references = link_sources(
        _report_markdown(result, themes, report), corpus
    )
    markdown += "## Sources\n\n" + source_details(references, corpus)
    return Answer(
        markdown=markdown,
        references=tuple(references),
        report=report,
        themes=tuple(themes),
    )


async def condense_answer(
    query: str,
    answer: Answer,
    llm: LLM,
    *,
    instructions: str = prompts.user_prompt,
    on_token: TokenHandler | None = None,
) -> str:
    """Rewrite the full report as a shorter answer following ``instructions``."""
    template = (
        f"{prompts.report_prompt} {instructions} "
        f"{prompts.list_prompts['safety_prompt']}"
    )
    return await llm.complete(
        system_message(template, query=query, answer=answer.markdown),
        on_token=on_token,
    )
