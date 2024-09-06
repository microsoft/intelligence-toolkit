# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

import pandas as pd
import streamlit as st
from seaborn import color_palette
from streamlit_agraph import Config, Edge, Node, agraph

import toolkit.query_text_data.helper_functions as helper_functions
import toolkit.query_text_data.input_processor as input_processor
import toolkit.query_text_data.prompts as prompts
import toolkit.query_text_data.question_answerer as question_answerer
from app.util import ui_components
from app.util.download_pdf import add_download_pdf
from app.util.openai_wrapper import UIOpenAIConfiguration
from app.util.session_variables import SessionVariables
from app.workflows.query_text_data import config
from toolkit.AI.base_embedder import BaseEmbedder
from toolkit.AI.defaults import CHUNK_SIZE
from toolkit.AI.local_embedder import LocalEmbedder
from toolkit.AI.openai_embedder import OpenAIEmbedder
from toolkit.graph.graph_fusion_encoder_embedding import (
    generate_graph_fusion_encoder_embedding,
)
from toolkit.query_text_data.pattern_detector import (
    combine_chunk_text_and_explantion,
    detect_converging_pairs,
    explain_chunk_significance,
)

sv_home = SessionVariables("home")
ai_configuration = UIOpenAIConfiguration().get_configuration()


def create_progress_callback(template: str):
    pb = st.progress(0, "Preparing...")

    def on_change(current, total):
        pb.progress(
            (current) / total if (current) / total < 100 else 100,
            text=template.format(current, total),
        )

    callback = ProgressBatchCallback()
    callback.on_batch_change = on_change
    return pb, callback


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


text_embedder = embedder()


def get_concept_graph(
    placeholder, G, concept_to_community, community_to_concepts, width, height, key
):
    """
    Implements the concept graph visualization
    """
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
        if node == "dummynode":
            continue
        degree = G.degree(node)
        size = 5 + 20 * degree / max_degree
        vadjust = -size * 2 - 3
        community = concept_to_community[node] if node in concept_to_community else -1
        color = (
            community_to_color[community]
            if community in community_to_color
            else (0.75, 0.75, 0.75)
        )
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
        if u == "dummynode" or v == "dummynode":
            continue
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


async def create(sv: SessionVariables, workflow=None):
    sv_home = SessionVariables("home")
    intro_tab, uploader_tab, graph_tab, search_tab, report_tab = st.tabs(
        [
            "Query text data workflow:",
            "Upload data",
            "Explore concept graph",
            "Generate incremental answers",
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
        window_size = st.selectbox(
            "Analysis time window",
            key=sv.analysis_window_size.key,
            options=[str(x) for x in input_processor.PeriodOption._member_names_],
        )
        window_period = input_processor.PeriodOption[window_size]
        if files is not None and st.button("Process files"):
            file_pb, file_callback = create_progress_callback(
                "Loaded {} of {} files..."
            )
            sv.file_to_chunks.value = input_processor.process_file_bytes(
                input_file_bytes={file.name: file.getvalue() for file in files},
                analysis_window_size=window_period,
                callbacks=[file_callback],
            )

            chunk_pb, chunk_callback = create_progress_callback(
                "Processed {} of {} chunks..."
            )
            (
                sv.cid_to_text.value,
                sv.text_to_cid.value,
                sv.period_concept_graphs.value,
                sv.community_to_concepts.value,
                sv.concept_to_community.value,
                sv.concept_to_cids.value,
                sv.cid_to_concepts.value,
                sv.previous_cid.value,
                sv.next_cid.value,
                sv.period_to_cids.value,
                sv.node_period_counts.value,
                sv.edge_period_counts.value,
            ) = input_processor.process_chunks(
                file_to_chunks=sv.file_to_chunks.value,
                max_cluster_size=50,
                callbacks=[chunk_callback],
            )
            if window_period != input_processor.PeriodOption.NONE:
                gfee_pb, gfee_callback = create_progress_callback(
                    "Embedded {} of {} concept nodes..."
                )
                sv.node_to_period_to_pos.value, sv.node_to_period_to_shift.value = (
                    generate_graph_fusion_encoder_embedding(
                        period_to_graph=sv.period_concept_graphs.value,
                        node_to_label=sv.concept_to_community.value,
                        correlation=True,
                        diaga=True,
                        laplacian=True,
                        callbacks=[gfee_callback],
                    )
                )
                period_pb, period_callback = create_progress_callback(
                    "Analyzed {} of {} periods..."
                )
                sv.cid_to_converging_pairs.value = detect_converging_pairs(
                    sv.period_to_cids.value,
                    sv.cid_to_concepts.value,
                    sv.node_to_period_to_pos.value,
                    callbacks=[period_callback],
                )
                explain_pb, explain_callback = create_progress_callback(
                    "Explained patterns in {} of {} periods..."
                )
                sv.cid_to_summary.value = explain_chunk_significance(
                    sv.period_to_cids.value,
                    sv.cid_to_converging_pairs.value,
                    sv.node_period_counts.value,
                    sv.edge_period_counts.value,
                    callbacks=[explain_callback],
                )
                sv.cid_to_explained_text.value = combine_chunk_text_and_explantion(
                    sv.cid_to_text.value, sv.cid_to_summary.value
                )
                gfee_pb.empty()
                period_pb.empty()
                explain_pb.empty()
            else:
                sv.cid_to_explained_text.value = sv.cid_to_text.value
            embed_pb, embed_callback = create_progress_callback(
                "Embedded {} of {} text chunks..."
            )
            sv.cid_to_vector.value = await helper_functions.embed_texts(
                sv.cid_to_explained_text.value,
                text_embedder,
                config.cache_name,
                callbacks=[embed_callback],
            )
            chunk_pb.empty()
            file_pb.empty()
            embed_pb.empty()
        num_files = len(sv.file_to_chunks.value.keys())
        num_chunks = sum([len(cs) for f, cs in sv.file_to_chunks.value.items()])
        num_periods = len(sv.period_to_cids.value) - 1

        G = (
            sv.period_concept_graphs.value["ALL"]
            if sv.period_concept_graphs.value is not None
            else None
        )
        if num_files > 0 and G is not None:
            message = f"Chunked **{num_files}** file{'s' if num_files > 1 else ''} into **{num_chunks}** chunks of up to **{CHUNK_SIZE}** tokens. Extracted concept graph with **{len(G.nodes())}** concepts and **{len(G.edges())}** cooccurrences, spanning **{num_periods}** periods."
            message = message.replace("**1** periods", "**1** period")
            st.success(message)
    with graph_tab:
        if sv.period_concept_graphs.value is not None:
            G = sv.period_concept_graphs.value["ALL"]
            c1, c2 = st.columns([5, 2])
            selection = None
            with c1:
                gp = st.empty()
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
                if selection is not None:
                    selected_cids = sv.concept_to_cids.value[selection]
                    selected_cids_df = pd.DataFrame(
                        [
                            {
                                "Matching text chunks (double click to expand)": sv.cid_to_text.value[
                                    cid
                                ]
                            }
                            for cid in selected_cids
                        ]
                    )
                    st.markdown(f"**Selected concept: {selection}**")
                    st.dataframe(selected_cids_df, hide_index=True, height=650)
    with search_tab:
        with st.expander("Search options", expanded=False):
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
            with c1:
                st.number_input(
                    "Relevance test budget",
                    value=sv.relevance_test_budget.value,
                    key=sv.relevance_test_budget.key,
                    min_value=0,
                )
            with c2:
                st.number_input(
                    "Tests/community/round",
                    value=sv.relevance_test_batch_size.value,
                    key=sv.relevance_test_batch_size.key,
                    min_value=0,
                )
            with c3:
                st.number_input(
                    "Restart on irrelevant communities",
                    value=sv.irrelevant_community_restart.value,
                    key=sv.irrelevant_community_restart.key,
                    min_value=0,
                )
            with c4:
                st.number_input(
                    "Test relevant neighbours",
                    value=sv.adjacent_chunk_steps.value,
                    key=sv.adjacent_chunk_steps.key,
                    min_value=0,
                )
            with c5:
                st.number_input(
                    "Relevant chunks/answer update",
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
                data=[sv.cid_to_text.value[x] for x in sv.relevant_cids.value],
            ),
            hide_index=True,
            height=300,
            use_container_width=True,
        )
        answer_text = (
            sv.partial_answers.value[0] if len(sv.partial_answers.value) > 0 else ""
        )
        answer_placeholder.markdown(answer_text)

        if sv.last_question.value != "" and regenerate:
            sv.relevant_cids.value = []
            sv.partial_answers.value = []
            sv.chunk_progress.value = ""
            sv.answer_progress.value = ""
            chunk_progress_placeholder.markdown(sv.chunk_progress.value)
            answer_progress_placeholder.markdown(sv.answer_progress.value)
            chunk_placeholder.dataframe(
                pd.DataFrame(
                    columns=["Relevant text chunks (double click to expand)"],
                    data=[sv.cid_to_text[x] for x in sv.relevant_cids.value],
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
                sv.relevant_cids.value,
                sv.partial_answers.value,
                sv.chunk_progress.value,
                sv.answer_progress.value,
            ) = await question_answerer.answer_question(
                ai_configuration=ai_configuration,
                question=sv.last_question.value,
                cid_to_text=sv.cid_to_explained_text.value,
                cid_to_concepts=sv.cid_to_concepts.value,
                concept_to_cids=sv.concept_to_cids.value,
                cid_to_vector=sv.cid_to_vector.value,
                concept_graph=sv.period_concept_graphs.value["ALL"],
                community_to_concepts=sv.community_to_concepts.value,
                concept_to_community=sv.concept_to_community.value,
                previous_cid=sv.previous_cid.value,
                next_cid=sv.next_cid.value,
                embedder=embedder(),
                embedding_cache=sv_home.save_cache.value,
                select_logit_bias=5,
                adjacent_search_steps=sv.adjacent_chunk_steps.value,
                relevance_test_budget=sv.relevance_test_budget.value,
                community_relevance_tests=sv.relevance_test_batch_size.value,
                relevance_test_batch_size=sv.relevance_test_batch_size.value,
                irrelevant_community_restart=sv.irrelevant_community_restart.value,
                answer_batch_size=sv.answer_update_batch_size.value,
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
