import json

import pytest

from intelligence_toolkit.query_text_data.answer_builder import answer_query
from intelligence_toolkit.query_text_data.classes import AnswerObject


class _DummyCommentary:
    def __init__(self):
        self.structure = {
            "themes": {"Theme Alpha": ["point-1"]},
            "points": {"point-1": "Point One"},
            "point_sources": {"point-1": [1]},
        }

    def get_clustered_cids(self):
        return {"Theme Alpha": [1]}


class _DummyProcessedChunks:
    cid_to_text = {
        1: json.dumps(
            {"title": "Doc", "chunk_id": 1, "text_chunk": "Chunk body"}
        )
    }


@pytest.mark.asyncio
async def test_answer_query_returns_report_with_sources(mocker):
    mocker.patch(
        "intelligence_toolkit.query_text_data.answer_builder.utils.prepare_messages",
        side_effect=lambda *args, **kwargs: {"args": args, "kwargs": kwargs},
    )
    mocker.patch(
        "intelligence_toolkit.query_text_data.answer_builder.utils.map_generate_text",
        new=mocker.AsyncMock(
            return_value=[
                json.dumps(
                    {
                        "theme_title": "Theme Alpha",
                        "theme_points": [
                            {
                                "point_title": "Point One",
                                "point_evidence": "**Source evidence**: Point One [source: 1]",
                                "point_commentary": "**AI commentary**: Point One",
                            }
                        ],
                    }
                )
            ]
        ),
    )
    mocker.patch(
        "intelligence_toolkit.query_text_data.answer_builder.utils.generate_text",
        return_value=json.dumps(
            {
                "answer": "Some answer",
                "report_title": "Aligned Themes",
                "report_overview": "Overview text",
                "report_implications": "Implication text",
            }
        ),
    )

    result = await answer_query(
        ai_configuration={"model": "test"},
        query="What happened?",
        expanded_query="Detailed question",
        processed_chunks=_DummyProcessedChunks(),
        commentary=_DummyCommentary(),
    )

    assert isinstance(result, AnswerObject)
    assert result.references == [1]
    assert "[source: [1](#source-1)]" in result.extended_answer
    assert "#### Source 1" in result.extended_answer
    assert "Doc (1)" in result.referenced_chunks
