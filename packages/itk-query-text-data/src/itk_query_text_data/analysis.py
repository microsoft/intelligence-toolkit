# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Live thematic analysis of relevant chunks while a search runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from itk_query_text_data import prompts, schemas
from itk_query_text_data.models import AnalysisUpdated, ThemeStructure

if TYPE_CHECKING:
    from collections.abc import Iterable

    from itk_query_text_data.llm import LLM, Message
    from itk_query_text_data.models import Corpus, EventHandler, TokenHandler

ALL_RELEVANT = "All relevant chunks"
_COMMENTARY_CHUNKS_PER_THEME = 3


def system_message(template: str, **variables: Any) -> list[Message]:
    return [{"role": "system", "content": template.format(**variables)}]


def _as_chunk_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def source_details(chunk_ids: Iterable[int], corpus: Corpus) -> str:
    """Collapsible markdown blocks with the full text of each chunk."""
    output = ""
    for chunk_id in chunk_ids:
        if chunk_id not in corpus:
            continue
        chunk = corpus[chunk_id]
        output += (
            f"#### Source {chunk_id}\n\n<details>\n\n"
            f"##### Text chunk: {chunk.source} ({chunk.index})\n\n{chunk.text}\n\n"
            "</details>\n\n[Back to top](#final-report)\n\n"
        )
    return output


class ThematicAnalysis:
    """Groups relevant chunks into themed points, updating every ``interval`` new chunks.

    With ``interval == 0`` no LLM analysis runs and every relevant chunk
    falls into a single ``"All relevant chunks"`` theme.
    """

    def __init__(
        self,
        query: str,
        corpus: Corpus,
        llm: LLM,
        interval: int,
        on_event: EventHandler | None = None,
    ) -> None:
        self._query = query
        self._corpus = corpus
        self._llm = llm
        self._interval = interval
        self._on_event = on_event
        self._pending: list[int] = []
        self._discovered: list[int] = []
        self._points: dict[Any, str] = {}
        self._point_sources: dict[Any, list[int]] = {}
        self._themes: dict[str, list[Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._interval > 0

    @property
    def structure(self) -> ThemeStructure:
        return ThemeStructure(
            points=dict(self._points),
            point_sources={k: tuple(v) for k, v in self._point_sources.items()},
            themes={k: tuple(v) for k, v in self._themes.items()},
        )

    async def add(self, chunk_ids: Iterable[int]) -> None:
        for chunk_id in chunk_ids:
            self._discovered.append(chunk_id)
            self._pending.append(chunk_id)
        if self.enabled and len(self._pending) >= self._interval:
            await self._update()

    async def complete(self) -> None:
        if self.enabled and self._pending:
            await self._update()

    async def _update(self) -> None:
        sources = "\n\n".join(
            f"{chunk_id}:\n\n{self._corpus[chunk_id].to_prompt()}"
            for chunk_id in self._pending
        )
        self._pending = []
        current = {
            "points": self._points,
            "point_sources": self._point_sources,
            "themes": self._themes,
        }
        response = await self._llm.complete(
            system_message(
                prompts.thematic_update_prompt,
                sources=sources,
                query=self._query,
                structure=json.dumps(current, indent=2),
            ),
            response_format=schemas.thematic_update_format,
        )
        update = json.loads(response)
        self._themes = {}
        for point in update.get("updates", []):
            point_id = _as_chunk_id(point["point_id"]) or point["point_id"]
            if point["point_title"] or point_id not in self._points:
                self._points[point_id] = point["point_title"]
            sources_for_point = self._point_sources.setdefault(point_id, [])
            for raw_id in point["source_ids"]:
                chunk_id = _as_chunk_id(raw_id)
                if chunk_id in self._corpus and chunk_id not in sources_for_point:
                    sources_for_point.append(chunk_id)
        for theme in update.get("themes", []):
            self._themes[theme["theme_title"]] = [
                _as_chunk_id(point_id) or point_id for point_id in theme["point_ids"]
            ]
        if self._on_event is not None:
            self._on_event(AnalysisUpdated(self.structure, self.markdown()))

    def markdown(self) -> str:
        """Themes and points with linked sources, followed by the source texts."""
        output = ""
        cited: set[int] = set()
        for title, point_ids in self._themes.items():
            output += f"- **{title}**\n"
            for point_id in point_ids:
                if point_id not in self._point_sources:
                    continue
                sources = self._point_sources[point_id]
                links = ", ".join(f"[{x}](#source-{x})" for x in sources)
                output += f"  - {self._points.get(point_id, '')} [source: {links}]\n"
                cited.update(sources)
        if cited:
            output += "\n## Sources\n\n" + source_details(sorted(cited), self._corpus)
        return output

    def clustered_ids(self) -> dict[str, tuple[int, ...]]:
        """Relevant chunk ids per theme, in order of first citation."""
        if not self.enabled:
            return {ALL_RELEVANT: tuple(self._discovered)} if self._discovered else {}
        clusters = {}
        for title, point_ids in self._themes.items():
            ids: dict[int, None] = {}
            for point_id in point_ids:
                ids.update(dict.fromkeys(self._point_sources.get(point_id, ())))
            clusters[title] = tuple(ids)
        return clusters


async def generate_commentary(
    query: str,
    themes: dict[str, tuple[int, ...]],
    analysis_markdown: str,
    corpus: Corpus,
    llm: LLM,
    *,
    on_token: TokenHandler | None = None,
) -> str:
    """Short AI commentary on the themes found so far, using up to three chunks per theme."""
    selected: dict[int, None] = {}
    for chunk_ids in themes.values():
        selected.update(dict.fromkeys(chunk_ids[:_COMMENTARY_CHUNKS_PER_THEME]))
    chunks = "\n\n".join(
        f"{chunk_id}:\n\n{corpus[chunk_id].to_prompt()}"
        for chunk_id in selected
        if chunk_id in corpus
    )
    return await llm.complete(
        system_message(
            prompts.commentary_prompt,
            query=query,
            structure=analysis_markdown,
            chunks=chunks,
        ),
        on_token=on_token,
    )
