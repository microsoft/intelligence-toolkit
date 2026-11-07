# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""The itk_core adapters, period handling, and resilience to malformed model replies."""

import asyncio
import json

import itk_query_text_data as qtd
import pytest
from itk_core.client import STREAM_CURSOR
from itk_core.utils import hash_text
from itk_query_text_data.analysis import ALL_RELEVANT, OTHER_RELEVANT, ThematicAnalysis
from itk_query_text_data.llm import _TokenRelay
from qtd2_support import QUERY, FakeLLM, fake_vector

# --- ItkEmbedder --------------------------------------------------------------


class _StoreEmbedder:
    """Mimics BaseEmbedder.embed_store_many: items may come back in any order."""

    def __init__(self, *, reverse: bool = True, drop: str | None = None) -> None:
        self.reverse = reverse
        self.drop = drop
        self.received: list[list[dict]] = []

    async def embed_store_many(self, data, callbacks=None, cache_data=True):
        self.received.append(data)
        results = [
            {
                "hash": item["hash"],
                "text": item["text"],
                "vector": fake_vector(item["text"]),
            }
            for item in data
            if item["text"] != self.drop
        ]
        return list(reversed(results)) if self.reverse else results


def test_itk_embedder_aligns_vectors_by_hash_not_position():
    store = _StoreEmbedder(reverse=True)
    texts = ["sea level rise", "interest rates", "heat waves"]
    vectors = asyncio.run(qtd.ItkEmbedder(store).embed(texts))
    assert vectors == [fake_vector(text) for text in texts]
    assert [item["hash"] for item in store.received[0]] == [hash_text(t) for t in texts]


def test_itk_embedder_embeds_duplicate_texts_once():
    store = _StoreEmbedder(reverse=False)
    vectors = asyncio.run(qtd.ItkEmbedder(store).embed(["flood", "rain", "flood"]))
    assert len(store.received[0]) == 2
    assert vectors[0] == vectors[2] == fake_vector("flood")


def test_itk_embedder_reports_missing_vectors():
    store = _StoreEmbedder(drop="rain")
    with pytest.raises(ValueError, match="no vector for 1 of 2"):
        asyncio.run(qtd.ItkEmbedder(store).embed(["flood", "rain"]))


# --- OpenAILLM ----------------------------------------------------------------


_CLIENT_CALLS: list[dict] = []


class _FakeClient:
    def __init__(self, configuration) -> None:
        self.configuration = configuration

    async def generate_chat_async(
        self, messages, stream=False, callbacks=None, **kwargs
    ):
        _CLIENT_CALLS.append({"messages": messages, "stream": stream, **kwargs})
        text = ""
        for token in ("Hel", "lo"):
            text += token
            for callback in callbacks or ():
                callback.on_llm_new_token(text + STREAM_CURSOR)
        for callback in callbacks or ():
            callback.on_llm_new_token(text)
        return text


@pytest.fixture()
def openai_llm(monkeypatch):
    import itk_core.client

    monkeypatch.setattr(itk_core.client, "OpenAIClient", _FakeClient)
    _CLIENT_CALLS.clear()
    return qtd.OpenAILLM(configuration=object())


def test_openai_llm_forwards_only_given_options(openai_llm):
    messages = [{"role": "system", "content": "hi"}]
    result = asyncio.run(openai_llm.complete(messages, max_tokens=1, logit_bias={1: 5}))
    assert result == "Hello"
    [call] = _CLIENT_CALLS
    assert call == {
        "messages": messages,
        "stream": False,
        "max_tokens": 1,
        "logit_bias": {1: 5},
    }


def test_openai_llm_streams_text_without_cursor(openai_llm):
    tokens: list[str] = []
    result = asyncio.run(
        openai_llm.complete(
            [{"role": "system", "content": "hi"}], on_token=tokens.append
        )
    )
    assert result == "Hello"
    assert tokens == ["Hel", "Hello", "Hello"]
    assert _CLIENT_CALLS[0]["stream"] is True


def test_token_relay_keeps_text_that_has_no_cursor():
    seen: list[str] = []
    _TokenRelay(seen.append).on_llm_new_token("done")
    assert seen == ["done"]


# --- periods ------------------------------------------------------------------


def _period_corpus() -> qtd.Corpus:
    return qtd.chunk_documents([
        qtd.Document("a.txt", "Sea level rise flooded coastal cities.", period="2023"),
        qtd.Document("b.txt", "Coastal cities built flood defenses.", period="2024"),
        qtd.Document("c.txt", "Interest rates rose."),
    ])


def test_periods_flow_from_documents_to_chunks_and_prompts():
    corpus = _period_corpus()
    assert corpus.periods == {"2023": (1,), "2024": (2,)}
    assert json.loads(corpus[1].to_prompt())["period"] == "2023"
    assert "period" not in json.loads(corpus[3].to_prompt())
    restored = qtd.Corpus.from_records(corpus.to_records())
    assert restored.periods == corpus.periods


def test_concept_graph_builds_one_graph_per_period():
    graph = qtd.build_concept_graph(
        _period_corpus(), max_cluster_size=4, min_edge_weight=1, min_node_degree=1
    )
    assert set(graph.period_graphs) == {"2023", "2024"}
    for period_graph in graph.period_graphs.values():
        assert set(period_graph) <= set(graph.graph)


# --- thematic analysis resilience ---------------------------------------------


class _ScriptedLLM(FakeLLM):
    """Returns the queued thematic-analysis replies in order."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self.replies = list(replies)

    def _thematic_analysis(self, content):
        return None, self.replies.pop(0)


def _corpus():
    """Four chunks with ids 1..4, one per document."""
    return qtd.chunk_documents([
        qtd.Document(f"{name}.txt", text)
        for name, text in zip(
            "abcd", ["Floods.", "Heat waves.", "Sea level.", "Carbon."], strict=True
        )
    ])


def _analysis(replies: list[str], interval: int = 2) -> ThematicAnalysis:
    return ThematicAnalysis(QUERY, _corpus(), _ScriptedLLM(replies), interval)


def test_unthemed_relevant_chunks_fall_back_to_a_single_group():
    analysis = _analysis([json.dumps({"updates": [], "themes": []})])
    asyncio.run(analysis.add([1, 2]))
    assert analysis.clustered_ids() == {ALL_RELEVANT: (1, 2)}


def test_relevant_chunks_outside_themes_are_kept_in_other_group():
    reply = json.dumps({
        "updates": [{"point_id": "P1", "point_title": "Flooding", "source_ids": [1]}],
        "themes": [{"theme_title": "Water", "point_ids": ["P1"]}],
    })
    analysis = _analysis([reply])
    asyncio.run(analysis.add([1, 2]))
    assert analysis.clustered_ids() == {"Water": (1,), OTHER_RELEVANT: (2,)}


def test_invalid_analysis_update_keeps_previous_structure_and_retries():
    good = json.dumps({
        "updates": [
            {"point_id": "P1", "point_title": "Flooding", "source_ids": [1, 2]}
        ],
        "themes": [{"theme_title": "Water", "point_ids": ["P1"]}],
    })
    later = json.dumps({
        "updates": [{"point_id": "P2", "point_title": "Heat", "source_ids": [3, 4]}],
        "themes": [
            {"theme_title": "Water", "point_ids": ["P1"]},
            {"theme_title": "Heat", "point_ids": ["P2"]},
        ],
    })
    analysis = _analysis([good, "not json", later])

    async def run():
        await analysis.add([1, 2])
        await analysis.add([3, 4])  # broken reply: structure must survive
        assert dict(analysis.structure.themes) == {"Water": ("P1",)}
        assert analysis.clustered_ids() == {"Water": (1, 2), OTHER_RELEVANT: (3, 4)}
        await analysis.complete()  # pending chunks 3, 4 are retried

    asyncio.run(run())
    assert analysis.clustered_ids() == {"Water": (1, 2), "Heat": (3, 4)}


class _BrokenReportLLM(FakeLLM):
    def _final_report(self, content):
        return None, "not json"


def test_invalid_report_integration_keeps_theme_summaries():
    corpus = _corpus()
    result = qtd.SearchResult(
        query=QUERY,
        expanded_query=QUERY,
        tests=(),
        relevant_chunk_ids=(1,),
        themes={ALL_RELEVANT: (1,)},
    )
    answer = asyncio.run(qtd.build_answer(result, corpus, _BrokenReportLLM()))
    assert len(answer.themes) == 1
    assert dict(answer.report) == {
        "answer": "",
        "report_title": "",
        "report_overview": "",
        "report_implications": "",
    }
    assert answer.references == (1,)
