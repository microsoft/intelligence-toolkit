# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Characterization tests for the v1 query_text_data workflow.

They drive the public ``QueryTextData`` API end to end over a fixed corpus with a
deterministic fake LLM and embedder, and compare outputs to golden snapshots. The
v2 rewrite must reproduce these results, or deliberately diverge (see the
``test_v1_quirk_*`` tests, which pin known v1 bugs that v2 is expected to fix).
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import (
    CORPUS_DIR,
    RecordingCallback,
    assert_matches_golden,
)

from intelligence_toolkit.query_text_data.api import QueryTextData, QueryTextDataStage
from intelligence_toolkit.query_text_data.classes import ChunkSearchConfig

if os.environ.get("PYTHONHASHSEED") != "0":
    pytest.skip(
        "v1 output depends on str hash order; run with PYTHONHASHSEED=0 "
        "(uv run poe test_characterization)",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[4]
QUERY = "What are the impacts of climate change?"
CORPUS_FILES = ["climate.txt", "health.txt", "economy.txt", "regions.csv"]
CHUNK_SIZE = 60
PROCESS_ARGS = {"max_cluster_size": 4, "min_edge_weight": 1, "min_node_degree": 1}


def search_config(analysis_update_interval: int) -> ChunkSearchConfig:
    return ChunkSearchConfig(
        adjacent_test_steps=1,
        community_relevance_tests=2,
        community_ranking_chunks=3,
        relevance_test_batch_size=2,
        relevance_test_budget=8,
        irrelevant_community_restart=2,
        analysis_update_interval=analysis_update_interval,
    )


def corpus_paths() -> list[str]:
    # POSIX paths: v1 derives file labels by splitting on "/" only.
    return [(CORPUS_DIR / name).as_posix() for name in CORPUS_FILES]


def build_workflow(fake_embedder) -> tuple[QueryTextData, dict]:
    qtd = QueryTextData()
    qtd.set_ai_config(ai_configuration=None, embedding_cache=False)
    qtd.set_embedder(fake_embedder)
    file_cb, chunk_cb = RecordingCallback(), RecordingCallback()
    stages = [qtd.stage.name]
    qtd.process_data_from_files(corpus_paths(), chunk_size=CHUNK_SIZE, callbacks=[file_cb])
    stages.append(qtd.stage.name)
    qtd.process_text_chunks(**PROCESS_ARGS, callbacks=[chunk_cb])
    stages.append(qtd.stage.name)
    asyncio.run(qtd.embed_text_chunks())
    stages.append(qtd.stage.name)
    return qtd, {"stages": stages, "file_batches": file_cb.batches, "chunk_batches": chunk_cb.batches}


def run_query(qtd: QueryTextData, interval: int) -> dict:
    histories: list[list] = []
    chunk_updates: list[int] = []
    analysis_cb, commentary_cb, condense_cb = (
        RecordingCallback(),
        RecordingCallback(),
        RecordingCallback(),
    )

    async def flow():
        expanded = await qtd.anchor_query_to_concepts(QUERY, top_concepts=10)
        relevant, summary = await qtd.detect_relevant_text_chunks(
            QUERY,
            expanded,
            search_config(interval),
            chunk_progress_callback=lambda h: histories.append(list(h)),
            chunk_callback=lambda chunks: chunk_updates.append(len(chunks)),
            analysis_callback=analysis_cb,
            commentary_callback=commentary_cb,
        )
        stage_after_search = qtd.stage.name
        answer = await qtd.answer_query_with_relevant_chunks()
        commentary = await qtd.generate_analysis_commentary()
        return expanded, relevant, summary, stage_after_search, answer, commentary

    expanded, relevant, summary, stage_after_search, answer, commentary = asyncio.run(flow())
    condensed = qtd.condense_answer(callbacks=[condense_cb])
    return {
        "expanded_query": expanded,
        "relevant_cids": relevant,
        "search_summary": summary,
        "test_history": histories[-1],
        "progress_history_lengths": [len(h) for h in histories],
        "chunk_callback_counts": chunk_updates,
        "stage_after_search": stage_after_search,
        "stage_after_answer": qtd.stage.name,
        "commentary_structure": qtd.commentary.structure,
        "clustered_cids": list(qtd.commentary.get_clustered_cids().items()),
        "analysis_callback_tokens": analysis_cb.tokens,
        "answer": {
            "extended_answer": answer.extended_answer,
            "references": answer.references,
            "referenced_chunk_keys": sorted(answer.referenced_chunks),
            "net_new_sources": answer.net_new_sources,
        },
        "commentary": commentary,
        "commentary_callback_tokens": commentary_cb.tokens,
        "condensed_answer": condensed,
        "condense_callback_tokens": condense_cb.tokens,
    }


def test_chunking(fake_embedder):
    qtd, progress = build_workflow(fake_embedder)
    assert_matches_golden(
        "chunking",
        {
            "label_to_chunks": {k: [json.loads(c) for c in v] for k, v in qtd.label_to_chunks.items()},
            "label_order": list(qtd.label_to_chunks),
            "file_batches": progress["file_batches"],
        },
    )


def test_processing(fake_embedder):
    qtd, progress = build_workflow(fake_embedder)
    pc = qtd.processed_chunks
    graphs = {
        period: {
            "nodes": sorted((n, d.get("count"), d.get("community")) for n, d in g.nodes(data=True)),
            "edges": sorted((*sorted((u, v)), d.get("weight")) for u, v, d in g.edges(data=True)),
        }
        for period, g in pc.period_concept_graphs.items()
    }
    communities = sorted(
        (
            hc.node,
            hc.level,
            pc.community_to_label[hc.cluster],
            pc.community_to_label.get(hc.parent_cluster) if hc.parent_cluster is not None else None,
            hc.is_final_cluster,
        )
        for hc in pc.hierarchical_communities
    )
    assert_matches_golden(
        "processing",
        {
            "stages": progress["stages"],
            "chunk_batches": progress["chunk_batches"],
            "cid_to_title": {cid: json.loads(t)["title"] for cid, t in pc.cid_to_text.items()},
            "text_to_cid_matches": all(pc.text_to_cid[t] == cid for cid, t in pc.cid_to_text.items()),
            "previous_cid": pc.previous_cid,
            "next_cid": pc.next_cid,
            "period_to_cids": pc.period_to_cids,
            "cid_to_concepts": pc.cid_to_concepts,
            "concept_to_cids": pc.concept_to_cids,
            "graphs": graphs,
            "communities": communities,
            "community_labels": sorted(pc.community_to_label.values()),
        },
    )


def test_embedding(fake_embedder):
    qtd, _ = build_workflow(fake_embedder)
    assert sorted(qtd.cid_to_vector) == sorted(qtd.processed_chunks.cid_to_text)
    assert_matches_golden(
        "embedding",
        {cid: vector[:4] for cid, vector in qtd.cid_to_vector.items()},
    )


def test_concept_community_graph(fake_embedder):
    qtd, _ = build_workflow(fake_embedder)
    networks = qtd.build_concept_community_graph()
    assert_matches_golden(
        "community_graph",
        {
            level: {
                label: {
                    "nodes": sorted(g.nodes()),
                    "edges": sorted(tuple(sorted(e)) for e in g.edges()),
                }
                for label, g in label_to_network.items()
            }
            for level, label_to_network in networks.items()
        },
    )


@pytest.mark.parametrize("interval", [0, 3], ids=["no_live_analysis", "live_analysis"])
def test_query_flow(fake_llm, fake_embedder, interval):
    qtd, _ = build_workflow(fake_embedder)
    result = run_query(qtd, interval)
    result["llm_calls"] = fake_llm.calls
    assert_matches_golden(f"query_flow_interval_{interval}", result)


def test_prepare_for_new_query_and_answer_reset_state(fake_llm, fake_embedder):
    qtd, _ = build_workflow(fake_embedder)
    run_query(qtd, 0)

    qtd.prepare_for_new_answer()
    assert qtd.stage == QueryTextDataStage.CHUNKS_MINED
    assert qtd.answer_object is None
    assert qtd.relevant_cids is not None

    qtd.prepare_for_new_query()
    assert qtd.stage == QueryTextDataStage.CHUNKS_EMBEDDED
    for attr in ("query", "expanded_query", "chunk_search_config", "relevant_cids", "search_summary", "answer_object"):
        assert getattr(qtd, attr) is None
    # Processed data survives so another query can run without re-embedding.
    assert qtd.cid_to_vector

    second = run_query(qtd, 0)
    assert second["stage_after_answer"] == QueryTextDataStage.QUESTION_ANSWERED.name


def test_chunks_csv_round_trip(fake_embedder, tmp_path):
    qtd, _ = build_workflow(fake_embedder)
    path = tmp_path / "chunks.csv"
    qtd.get_chunks_as_df().to_csv(path)

    restored = QueryTextData()
    restored.import_chunks_from_str(str(path))
    assert dict(restored.label_to_chunks) == dict(qtd.label_to_chunks)


def test_reset_workflow_clears_everything(fake_embedder):
    qtd, _ = build_workflow(fake_embedder)
    qtd.reset_workflow()
    assert qtd.stage == QueryTextDataStage.INITIAL
    assert qtd.label_to_chunks is None
    assert qtd.processed_chunks is None
    assert qtd.cid_to_vector is None


# --- Known v1 quirks. v2 should fix these; update or invert the tests when it does. ---


def test_v1_quirk_next_cid_uses_last_file_chunk_count(fake_embedder):
    """process_chunks computes next_cid with the chunk count of the *last* file."""
    qtd, _ = build_workflow(fake_embedder)
    pc = qtd.processed_chunks
    titles = {cid: json.loads(t)["title"] for cid, t in pc.cid_to_text.items()}
    file_last_cids = {}
    for cid in sorted(titles):
        file_last_cids[titles[cid]] = cid
    crosses_file_boundary = [
        cid for cid, nxt in pc.next_cid.items() if titles.get(nxt) != titles[cid]
    ]
    missing_links = [
        cid
        for cid in sorted(titles)
        if cid != file_last_cids[titles[cid]] and cid not in pc.next_cid
    ]
    assert crosses_file_boundary or missing_links


def test_v1_quirk_windows_paths_keep_full_path_as_label(fake_embedder, tmp_path):
    """convert_files_to_chunks only splits on '/', so backslash paths keep directories."""
    qtd = QueryTextData()
    if os.name == "nt":
        path = str(CORPUS_DIR / "climate.txt")
    else:
        # On POSIX '\\' is a valid filename character, so a literal backslash name
        # reproduces the same quirk.
        target = tmp_path / "corpus\\climate.txt"
        shutil.copyfile(CORPUS_DIR / "climate.txt", target)
        path = str(target)
    labels = list(qtd.process_data_from_files([path], chunk_size=CHUNK_SIZE))
    expected = path.split("/")[-1].replace(" ", "_").replace("(", "").replace(")", "")
    assert "\\" in expected
    assert labels == [expected]


_COMMUNITY_SNAPSHOT_SCRIPT = """
import json, sys
from pathlib import Path
from intelligence_toolkit.query_text_data.api import QueryTextData
corpus = Path(sys.argv[1])
files = [(corpus / n).as_posix() for n in sys.argv[2].split(",")]
qtd = QueryTextData()
qtd.process_data_from_files(files, chunk_size=int(sys.argv[3]))
qtd.process_text_chunks(**json.loads(sys.argv[4]))
pc = qtd.processed_chunks
print(json.dumps(sorted((hc.node, pc.community_to_label[hc.cluster]) for hc in pc.hierarchical_communities)))
"""


def test_v1_quirk_communities_depend_on_hash_seed():
    """Tie-breaks iterate over sets of strings, so labels vary with PYTHONHASHSEED."""
    args = [
        sys.executable,
        "-c",
        _COMMUNITY_SNAPSHOT_SCRIPT,
        str(CORPUS_DIR),
        ",".join(CORPUS_FILES),
        str(CHUNK_SIZE),
        json.dumps(PROCESS_ARGS),
    ]
    seeds = range(6)
    procs = [
        subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        for seed in seeds
    ]
    outputs = set()
    for proc in procs:
        stdout, _ = proc.communicate(timeout=300)
        assert proc.returncode == 0
        outputs.add(stdout.strip().splitlines()[-1])
    assert len(outputs) > 1, f"labels identical across {len(seeds)} hash seeds"


def test_v1_quirk_answer_without_analysis_uses_only_last_unprocessed_chunks(fake_llm, fake_embedder):
    """With interval=0 the answer is built from Commentary.unprocessed_chunks."""
    qtd, _ = build_workflow(fake_embedder)
    result = run_query(qtd, 0)
    clustered = dict(result["clustered_cids"])
    assert list(clustered) == ["All relevant chunks"]
    assert sorted(clustered["All relevant chunks"]) == result["relevant_cids"]
