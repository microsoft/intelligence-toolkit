# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

import pandas as pd
import streamlit as st
from seaborn import color_palette
from collections import defaultdict
from streamlit_agraph import Config, Edge, Node, agraph

import app.util.example_outputs_ui as example_outputs_ui
import app.workflows.query_text_data.functions as functions
import toolkit.query_text_data.answer_builder as answer_builder
import toolkit.query_text_data.graph_builder as graph_builder
import toolkit.query_text_data.helper_functions as helper_functions
import toolkit.query_text_data.input_processor as input_processor
import toolkit.query_text_data.prompts as prompts
import toolkit.query_text_data.relevance_assessor as relevance_assessor
from app.util import ui_components
from app.util.download_pdf import add_download_pdf
from app.util.openai_wrapper import UIOpenAIConfiguration
from app.util.session_variables import SessionVariables
from toolkit.AI.defaults import CHUNK_SIZE
from toolkit.graph.graph_fusion_encoder_embedding import (
    create_concept_to_community_hierarchy,
    generate_graph_fusion_encoder_embedding,
)
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback
from toolkit.query_text_data import config
from toolkit.query_text_data.pattern_detector import (
    combine_chunk_text_and_explantion,
    detect_converging_pairs,
    explain_chunk_significance,
)


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


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


def get_concept_graph(
    placeholder, G, hierarchical_communities, width, height, key
):
    """
    Implements the concept graph visualization
    """
    nodes = []
    edges = []
    max_degree = max([G.degree(node) for node in G.nodes()])
    concept_to_community = hierarchical_communities.final_level_hierarchical_clustering()
    community_to_concepts = defaultdict(set)
    for concept, community in concept_to_community.items():
        community_to_concepts[community].add(concept)

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

async def create(sv: SessionVariables, workflow=None):
    sv_home = SessionVariables("home")
    intro_tab, uploader_tab, graph_tab, search_tab, report_tab, examples_tab = st.tabs(
        [
            "Query Text Data workflow:",
            "Prepare data",
            "Explore concept graph",
            "Generate AI extended answers",
            "Generate AI answer reports",
            "View example outputs"
        ]
    )

    with intro_tab:
        st.markdown(get_intro(), unsafe_allow_html=True)
    with uploader_tab:
        st.markdown("##### Upload data for processing")
        files = st.file_uploader(
            "Upload PDF text files",
            type=["pdf", "txt", "json", "csv"],
            accept_multiple_files=True,
            key=sv.upload_key.value,
        )
        # window_size = st.selectbox(
        #     "Analysis time window",
        #     key=sv.analysis_window_size.key,
        #     options=[str(x) for x in input_processor.PeriodOption._member_names_],
        # )
        # window_period = input_processor.PeriodOption[window_size]
        window_period = input_processor.PeriodOption.NONE
        local_embedding = st.toggle(
            "Use local embeddings",
            sv.answer_local_embedding_enabled.value,
            help="Use local embeddings to index nodes. If disabled, the model will use OpenAI embeddings.",
        )
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
                sv.hierarchical_communities.value,
                sv.community_to_label.value,
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
                concept_to_community_hierarchy, max_cluster_per_level, max_level = create_concept_to_community_hierarchy(
                    sv.hierarchical_communities.value)
                sv.node_to_period_to_pos.value, sv.node_to_period_to_shift.value = (
                    generate_graph_fusion_encoder_embedding(
                        period_to_graph=sv.period_concept_graphs.value,
                        node_to_label=concept_to_community_hierarchy,
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
            text_embedder = functions.embedder(local_embedding)

            sv.cid_to_vector.value = await helper_functions.embed_texts(
                sv.cid_to_explained_text.value,
                text_embedder,
                sv_home.save_cache.value,
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
            message = f"Chunked **{num_files}** file{'s' if num_files > 1 else ''} into **{num_chunks}** chunks of up to **{CHUNK_SIZE}** tokens. Extracted concept graph with **{len(G.nodes())}** concepts and **{len(G.edges())}** cooccurrences"
            if num_periods > 1:
                message += ", spanning **{num_periods}** periods."
            else:
                message += "."
            message = message.replace("**1** periods", "**1** period")
            st.success(message)
    with graph_tab:
        if sv.period_concept_graphs.value is not None:
            G = sv.period_concept_graphs.value["ALL"]
            c1, c2 = st.columns([5, 2])
            selection = None
            with c1:

                level_to_label_to_network = graph_builder.build_meta_graph(G, sv.hierarchical_communities.value)
                selected_level_labels = [''] + [str(v) for v in level_to_label_to_network[0].keys()]
                selected_label = st.selectbox("Select topic area", options=selected_level_labels, key=f"{workflow}_community_label")
                gp = st.empty()
                if selected_label not in ['', None]:
                    selection = get_concept_graph(
                        gp,
                        level_to_label_to_network[0][selected_label],
                        sv.hierarchical_communities.value,
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
                    st.dataframe(selected_cids_df, hide_index=True, height=650, use_container_width=True)
    with search_tab:
        with st.expander("Options", expanded=False):
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
            with c1:
                st.number_input(
                    "Relevance test budget",
                    value=sv.relevance_test_budget.value,
                    key=sv.relevance_test_budget.key,
                    min_value=0,
                    help="The query method works by asking an LLM to evaluate the relevance of potentially-relevant text chunks, returning a single token, yes/no judgement. This parameter allows the user to cap the number of relvance tests that may be performed prior to generating an answer using all relevant chunks. Larger budgets will generally give better answers for a greater cost."
                )
            with c2:
                st.number_input(
                    "Tests/topic/round",
                    value=sv.relevance_test_batch_size.value,
                    key=sv.relevance_test_batch_size.key,
                    min_value=0,
                    help="How many relevant tests to perform for each topic in each round. Larger values reduce the likelihood of prematurely discarding topics whose relevant chunks may not be at the top of the similarity-based ranking, but may result in smaller values of `Relevance test budget` being spread across fewer topics and thus not capturing the full breadth of the data."
                )
            with c3:
                st.number_input(
                    "Restart on irrelevant topics",
                    value=sv.irrelevant_community_restart.value,
                    key=sv.irrelevant_community_restart.key,
                    min_value=0,
                    help="When this number of topics in a row fail to return any relevant chunks in their `Tests/topic/round`, return to the start of the topic ranking and continue testing `Tests/topic/round` text chunks from each topic with (a) relevance in the previous round and (b) previously untested text chunks. Higher values can avoid prematurely discarding topics that are relevant but whose relevant chunks are not at the top of the similarity-based ranking, but may result in a larger number of irrelevant topics being tested multiple times."
                )
            with c4:
                st.number_input(
                    "Test relevant neighbours",
                    value=sv.adjacent_chunk_steps.value,
                    key=sv.adjacent_chunk_steps.key,
                    min_value=0,
                    help="If a text chunk is relevant to the query, then adjacent text chunks in the original document may be able to add additional context to the relevant points. The value of this parameter determines how many chunks before and after each relevant text chunk will be evaluated at the end of the process (or `Relevance test budget`) if they are yet to be tested."
                )
            with c5:
                st.number_input(
                    "Relevant chunks/answer update",
                    value=sv.answer_update_batch_size.value,
                    key=sv.answer_update_batch_size.key,
                    min_value=0,
                    help="Determines how many relevant chunks at a time are incorporated into the extended answer in progress. Higher values may require fewer updates, but may miss more details from the chunks provided."
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
            if len(sv.partial_answers.value) > 0 and len(sv.partial_answers.value[-1]) > 0:
                st.download_button(
                    "Download extended answer as MD",
                    data=sv.partial_answers.value[-1],
                    file_name="extended_answer.md",
                    mime="text/markdown",
                    key="qa_extended_download_button",
                )

        def on_chunk_progress(message):
            chunk_progress_placeholder.markdown(message, unsafe_allow_html=True)

        def on_answer_progress(message):
            answer_progress_placeholder.markdown(message, unsafe_allow_html=True)

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
            answer_placeholder.markdown(message[-1], unsafe_allow_html=True)

        chunk_progress_placeholder.markdown(sv.chunk_progress.value, unsafe_allow_html=True)
        answer_progress_placeholder.markdown(sv.answer_progress.value, unsafe_allow_html=True)
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
            sv.partial_answers.value[-1] if len(sv.partial_answers.value) > 0 else ""
        )
        answer_placeholder.markdown(answer_text, unsafe_allow_html=True)

        if sv.last_question.value != "" and regenerate:
            sv.relevant_cids.value = []
            sv.partial_answers.value = []
            sv.chunk_progress.value = ""
            sv.answer_progress.value = ""
            chunk_progress_placeholder.markdown(sv.chunk_progress.value, unsafe_allow_html=True)
            answer_progress_placeholder.markdown(sv.answer_progress.value, unsafe_allow_html=True)
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
                sv.partial_answers.value[-1] if len(sv.partial_answers.value) > 0 else ""
            )
            answer_placeholder.markdown(answer_text, unsafe_allow_html=True)

            (
                sv.relevant_cids.value,
                sv.chunk_progress.value,
            ) = await relevance_assessor.detect_relevant_chunks(
                ai_configuration=ai_configuration,
                question=sv.last_question.value,
                cid_to_text=sv.cid_to_explained_text.value,
                cid_to_concepts=sv.cid_to_concepts.value,
                cid_to_vector=sv.cid_to_vector.value,
                hierarchical_communities=sv.hierarchical_communities.value,
                community_to_label=sv.community_to_label.value,
                previous_cid=sv.previous_cid.value,
                next_cid=sv.next_cid.value,
                embedder=functions.embedder(local_embedding),
                embedding_cache=sv_home.save_cache.value,
                select_logit_bias=5,
                adjacent_search_steps=sv.adjacent_chunk_steps.value,
                community_ranking_chunks=sv.relevance_test_batch_size.value,
                relevance_test_budget=sv.relevance_test_budget.value,
                community_relevance_tests=sv.relevance_test_batch_size.value,
                relevance_test_batch_size=sv.relevance_test_batch_size.value,
                irrelevant_community_restart=sv.irrelevant_community_restart.value,
                chunk_progress_callback=on_chunk_progress,
                chunk_callback=on_chunk_relevant,
            )

            # (
            #     sv.partial_answers.value,
            #     sv.answer_progress.value,
            # ) = answer_builder.answer_question(
            #     ai_configuration=ai_configuration,
            #     question=sv.last_question.value,
            #     relevant_cids=sv.relevant_cids.value,
            #     cid_to_text=sv.cid_to_explained_text.value,
            #     answer_batch_size=sv.answer_update_batch_size.value,
            #     answer_progress_callback=on_answer_progress,
            #     answer_callback=on_answer,
            # )
        if st.button("Generate intermediate answers"):
            intermediate_answers = await answer_builder.answer_question(
                ai_configuration=ai_configuration,
                question=sv.last_question.value,
                relevant_cids=sv.relevant_cids.value,
                cid_to_text=sv.cid_to_explained_text.value,
                answer_batch_size=sv.answer_update_batch_size.value,
            )
            st.write(intermediate_answers)
            

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

                    # validation, messages_to_llm = ui_components.validate_ai_report(
                    #     messages, result
                    # )
                    # sv.report_validation.value = validation
                    # sv.report_validation_messages.value = messages_to_llm
                    st.rerun()
                else:
                    if sv.final_report.value == "":
                        gen_placeholder.warning(
                            "Press the Generate button to create an AI report for the current question."
                        )
                report_placeholder.markdown(sv.final_report.value, unsafe_allow_html=True)

                if len(sv.final_report.value) > 0:
                    is_download_disabled = sv.final_report.value == ""
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.download_button(
                            "Download AI answer report as MD",
                            data=sv.final_report.value,
                            file_name=f"condensed_answer.md",
                            mime="text/markdown",
                            disabled=sv.final_report.value == "",
                            key="qa_download_button",
                        )
                    with c2:
                        add_download_pdf(
                            f"condensed_answer.pdf",
                            sv.final_report.value,
                            "Download AI answer report as PDF",
                            disabled=is_download_disabled,
                        )

                    # ui_components.build_validation_ui(
                    #     sv.report_validation.value,
                    #     sv.report_validation_messages.value,
                    #     sv.final_report.value,
                    #     workflow,
                    # )
    with examples_tab:
        example_outputs_ui.create_example_outputs_ui(examples_tab, workflow)
