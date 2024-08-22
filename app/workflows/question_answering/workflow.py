# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

import pandas as pd
import streamlit as st
from seaborn import color_palette
from streamlit_agraph import Config, Edge, Node, agraph

import python.question_answering.input_processor as input_processor
import python.question_answering.prompts as prompts
import python.question_answering.question_answerer as question_answerer
from app.util import ui_components
from app.util.download_pdf import add_download_pdf
from app.util.openai_wrapper import UIOpenAIConfiguration
from app.util.session_variables import SessionVariables
from app.workflows.question_answering import config
from python.AI.base_embedder import BaseEmbedder
from python.AI.defaults import CHUNK_SIZE
from python.AI.local_embedder import LocalEmbedder
from python.AI.openai_embedder import OpenAIEmbedder

sv_home = SessionVariables("home")
ai_configuration = UIOpenAIConfiguration().get_configuration()


def embedder() -> BaseEmbedder:
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        if sv_home.local_embeddings.value:
            return LocalEmbedder(
                db_name=config.cache_name,
                max_tokens=ai_configuration.max_tokens,
            )

        return OpenAIEmbedder(
            configuration=ai_configuration,
            db_name=config.cache_name,
        )
    except Exception as e:
        st.error(f"Error creating connection: {e}")
        st.stop()


def get_concept_graph(
    placeholder, G, concept_to_community, community_to_concepts, width, height, key
):
    """
    Implements the concept graph visualization
    """
    node_names = set()
    nodes = []
    edges = []
    max_degree = max([G.degree(node) for node in G.nodes()])
    num_communities = len(community_to_concepts.keys())
    community_colors = color_palette("husl", num_communities)
    sorted_communities = sorted(
        community_to_concepts.keys(),
        key=lambda x: len(community_to_concepts[x]),
        reverse=True,
    )
    community_to_color = dict(zip(sorted_communities, community_colors))
    for node in G.nodes():
        degree = G.degree(node)
        size = 5 + 20 * degree / max_degree
        vadjust = -size * 2 - 3
        color = community_to_color[concept_to_community[node]]
        color = "#%02x%02x%02x" % tuple([int(255 * x) for x in color])
        nodes.append(
            Node(
                title=node,
                id=node,
                label=node,
                size=size,
                color=color,
                shape="dot",
                timestep=0.001,
                font={"vadjust": vadjust, "size": size},
            )
        )

    for u, v, d in G.edges(data=True):
        edges.append(Edge(source=u, target=v, color="lightgray"))

    config = Config(
        width=width,
        height=height,
        directed=False,
        physics=True,
        hierarchical=False,
        key=key,
        linkLength=100,
    )
    with placeholder:
        return_value = agraph(nodes=nodes, edges=edges, config=config)
    return return_value


class ProgressBatchCallback:
    """Class for progress callbacks."""

    def __init__(self):
        self.current_batch = 0
        self.total_batches = 0

    def on_batch_change(self, current: int, total: int):
        """Handle when a new token is generated."""
        self.current_batch = current
        self.total_batches = total


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


def create(sv: SessionVariables, workflow=None):
    sv_home = SessionVariables("home")
    intro_tab, uploader_tab, graph_tab, search_tab, report_tab = st.tabs(
        [
            "Question answering workflow:",
            "Upload data",
            "Explore concept graph",
            "Search for answers",
            "Generate AI answer reports",
        ]
    )

    with intro_tab:
        st.markdown(get_intro())
    with uploader_tab:
        st.markdown("##### Upload data for processing")
        files = st.file_uploader(
            "Upload PDF text files",
            type=["pdf", "txt", "json"],
            accept_multiple_files=True,
            key=sv.upload_key.value,
        )
        if files is not None and st.button("Process files"):
            # functions.chunk_files(sv, files)
            file_pb = st.progress(0, "Processing files...")

            def on_file_batch_change(current, total):
                file_pb.progress(
                    int(current * 100 / total), text=f"Processed {current} files..."
                )

            file_callback = ProgressBatchCallback()
            file_callback.on_batch_change = on_file_batch_change
            sv.text_to_chunks.value = input_processor.process_file_bytes(
                input_file_bytes={file.name: file.getvalue() for file in files},
                callbacks=[file_callback],
            )

            chunk_pb = st.progress(0, "Processing text chunks...")

            def on_chunk_batch_change(current, total):
                chunk_pb.progress(
                    int(current * 100 / total),
                    text=f"Processed {current} text chunks...",
                )

            chunk_callback = ProgressBatchCallback()
            chunk_callback.on_batch_change = on_chunk_batch_change

            (
                sv.text_to_vectors.value,
                sv.concept_graph.value,
                sv.community_to_concepts.value,
                sv.concept_to_community.value,
                sv.concept_to_chunks.value,
                sv.chunk_to_concepts.value,
                sv.previous_chunk.value,
                sv.next_chunk.value,
            ) = input_processor.process_chunks(
                text_to_chunks=sv.text_to_chunks.value,
                embedder=embedder(),
                embedding_cache=sv_home.save_cache.value,
                max_cluster_size=50,
                callbacks=[chunk_callback],
            )
            chunk_pb.empty()
            file_pb.empty()
        num_files = len(sv.text_to_chunks.value.keys())
        num_chunks = sum([len(cs) for f, cs in sv.text_to_chunks.value.items()])
        G = sv.concept_graph.value
        if num_files > 0 and G is not None:
            st.success(
                f"Chunked **{num_files}** file{'s' if num_files > 1 else ''} into **{num_chunks}** chunks of up to **{CHUNK_SIZE}** tokens. Extracted concept graph with **{len(G.nodes())}** concepts and **{len(G.edges())}** cooccurrences."
            )
    with graph_tab:
        c1, c2 = st.columns([5, 2])
        selection = None
        with c1:
            gp = st.empty()
            if sv.concept_graph.value is not None:
                selection = get_concept_graph(
                    gp,
                    G,
                    sv.concept_to_community.value,
                    sv.community_to_concepts.value,
                    800,
                    700,
                    "graph",
                )
        with c2:
            if sv.concept_graph.value is not None and selection is not None:
                selected_chunks = sv.concept_to_chunks.value[selection]
                selected_chunks_df = pd.DataFrame(
                    [
                        {"Matching text chunks (double click to expand)": chunk}
                        for chunk in selected_chunks
                    ]
                )
                st.markdown(f"**Selected concept: {selection}**")
                st.dataframe(selected_chunks_df, hide_index=True, height=650)
    with search_tab:
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])
        with c1:
            st.number_input(
                "Depth (semantic search tests)",
                value=sv.semantic_search_depth.value,
                key=sv.semantic_search_depth.key,
                min_value=0,
            )
        with c2:
            st.number_input(
                "Breadth (community sample tests)",
                value=sv.relational_search_depth.value,
                key=sv.relational_search_depth.key,
                min_value=0,
            )
        with c3:
            st.number_input(
                "Detail (test relevant adjacent)",
                value=sv.structural_search_steps.value,
                key=sv.structural_search_steps.key,
                min_value=0,
            )
        with c4:
            st.number_input(
                "Relevance test batch size",
                value=sv.relevance_test_batch_size.value,
                key=sv.relevance_test_batch_size.key,
                min_value=0,
            )
        with c5:
            st.number_input(
                "Maximum relevance tests",
                value=sv.relevance_test_limit.value,
                key=sv.relevance_test_limit.key,
                min_value=0,
            )
        with c6:
            st.number_input(
                "Answer update batch size",
                value=sv.answer_update_batch_size.value,
                key=sv.answer_update_batch_size.key,
                min_value=0,
            )
        c1, c2 = st.columns([6, 1])
        with c1:
            st.text_input(
                "Question", value=sv.last_question.value, key=sv.last_question.key
            )
        with c2:
            regenerate = st.button(
                "Ask",
                key="search_answers",
                use_container_width=True,
            )

        c1, c2 = st.columns([1, 1])

        with c1:
            chunk_placeholder = st.empty()
            chunk_progress_placeholder = st.empty()
            answer_progress_placeholder = st.empty()
        with c2:
            answer_placeholder = st.empty()

        def on_chunk_progress(message):
            chunk_progress_placeholder.markdown(message)

        def on_answer_progress(message):
            answer_progress_placeholder.markdown(message)

        def on_chunk_relevant(message):
            chunk_placeholder.dataframe(
                pd.DataFrame(
                    columns=["Relevant text chunks (double click to expand)"],
                    data=message,
                ),
                hide_index=True,
                height=400,
                use_container_width=True,
            )

        def on_answer(message):
            answer_placeholder.markdown(message[0])

        chunk_progress_placeholder.markdown(sv.chunk_progress.value)
        answer_progress_placeholder.markdown(sv.answer_progress.value)
        chunk_placeholder.dataframe(
            pd.DataFrame(
                columns=["Relevant text chunks (double click to expand)"],
                data=sv.relevant_chunks.value,
            ),
            hide_index=True,
            height=400,
            use_container_width=True,
        )
        answer_text = (
            sv.partial_answers.value[0] if len(sv.partial_answers.value) > 0 else ""
        )
        answer_placeholder.markdown(answer_text)

        if sv.last_question.value != "" and regenerate:
            sv.relevant_chunks.value = []
            sv.partial_answers.value = []
            sv.chunk_progress.value = ""
            sv.answer_progress.value = ""
            chunk_progress_placeholder.markdown(sv.chunk_progress.value)
            answer_progress_placeholder.markdown(sv.answer_progress.value)
            chunk_placeholder.dataframe(
                pd.DataFrame(
                    columns=["Relevant text chunks (double click to expand)"],
                    data=sv.relevant_chunks.value,
                ),
                hide_index=True,
                height=400,
                use_container_width=True,
            )
            answer_text = (
                sv.partial_answers.value[0] if len(sv.partial_answers.value) > 0 else ""
            )
            answer_placeholder.markdown(answer_text)
            (
                sv.relevant_chunks.value,
                sv.partial_answers.value,
                sv.chunk_progress.value,
                sv.answer_progress.value,
            ) = question_answerer.answer_question(
                ai_configuration,
                sv.last_question.value,
                sv.text_to_chunks.value,
                sv.chunk_to_concepts.value,
                sv.concept_to_chunks.value,
                sv.text_to_vectors.value,
                sv.concept_graph.value,
                sv.community_to_concepts.value,
                sv.concept_to_community.value,
                sv.previous_chunk.value,
                sv.next_chunk.value,
                embedder=embedder(),
                embedding_cache=sv_home.save_cache.value,
                select_logit_bias=5,
                semantic_search_depth=sv.semantic_search_depth.value,
                structural_search_steps=sv.structural_search_steps.value,
                community_search_breadth=sv.relational_search_depth.value,
                relevance_test_limit=sv.relevance_test_limit.value,
                relevance_test_batch_size=sv.relevance_test_batch_size.value,
                answer_batch_size=5,
                augment_top_concepts=10,
                chunk_progress_callback=on_chunk_progress,
                answer_progress_callback=on_answer_progress,
                chunk_callback=on_chunk_relevant,
                answer_callback=on_answer,
            )

    with report_tab:
        if sv.partial_answers.value == []:
            st.warning("Search for answers to continue.")
        else:
            c1, c2 = st.columns([2, 3])

            with c1:
                variables = {
                    "question": sv.last_question.value,
                    "answers": sv.partial_answers.value,
                }
                generate, messages, reset = ui_components.generative_ai_component(
                    sv.system_prompt, variables
                )
                if reset:
                    sv.system_prompt.value["user_prompt"] = prompts.user_prompt
                    st.rerun()
            with c2:
                report_placeholder = st.empty()
                gen_placeholder = st.empty()
                if generate:
                    on_callback = ui_components.create_markdown_callback(
                        report_placeholder
                    )
                    result = ui_components.generate_text(
                        messages, callbacks=[on_callback]
                    )
                    sv.final_report.value = result

                    validation, messages_to_llm = ui_components.validate_ai_report(
                        messages, result
                    )
                    sv.report_validation.value = validation
                    sv.report_validation_messages.value = messages_to_llm
                    st.rerun()
                else:
                    if sv.final_report.value == "":
                        gen_placeholder.warning(
                            "Press the Generate button to create an AI report for the current question."
                        )
                report_placeholder.markdown(sv.final_report.value)

                if len(sv.final_report.value) > 0:
                    is_download_disabled = sv.final_report.value == ""
                    name = (
                        sv.final_report.value.split("\n")[0]
                        .replace("#", "")
                        .strip()
                        .replace(" ", "_")
                    )

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.download_button(
                            "Download AI answer report as MD",
                            data=sv.final_report.value,
                            file_name=f"{name}.md",
                            mime="text/markdown",
                            disabled=sv.final_report.value == "",
                            key="qa_download_button",
                        )
                    with c2:
                        add_download_pdf(
                            f"{name}.pdf",
                            sv.final_report.value,
                            "Download AI answer report as PDF",
                            disabled=is_download_disabled,
                        )

                    ui_components.build_validation_ui(
                        sv.report_validation.value,
                        sv.report_validation_messages.value,
                        sv.final_report.value,
                        workflow,
                    )
