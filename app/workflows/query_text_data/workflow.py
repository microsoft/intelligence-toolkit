# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import os
import random
import re

import pandas as pd
import streamlit as st

import app.util.embedder as embedder
import app.util.example_outputs_ui as example_outputs_ui
import app.workflows.query_text_data.functions as functions
import intelligence_toolkit.query_text_data.prompts as prompts
from app.util import ui_components
from app.util.download_pdf import add_download_pdf
from app.util.openai_wrapper import UIOpenAIConfiguration
from app.util.session_variables import SessionVariables
from intelligence_toolkit.AI.defaults import CHUNK_SIZE
from intelligence_toolkit.query_text_data.api import QueryTextDataStage
from intelligence_toolkit.query_text_data.classes import (
    ChunkSearchConfig,
)
from intelligence_toolkit.AI.classes import LLMCallback

sv_home = SessionVariables("home")
ai_configuration = UIOpenAIConfiguration().get_configuration()

def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()

async def create(sv: SessionVariables, workflow=None):
    if "search_answers" not in st.session_state.keys():
        st.session_state["search_answers"] = False
    gen_answer = False
    sv_home = SessionVariables("home")
    ui_components.check_ai_configuration()

    qtd = sv.workflow_object.value
    qtd.set_ai_config(ai_configuration, sv_home.save_cache.value)
    intro_tab, uploader_tab, graph_tab, search_tab, report_tab, examples_tab = st.tabs(
        [
            "Query Text Data workflow:",
            "Prepare data",
            "Explore concept graph",
            "Generate AI research report",
            "Generate AI answer reports",
            "View example outputs"
        ]
    )

    if f"{workflow}_uploader_index" not in st.session_state:
        st.session_state[f"{workflow}_uploader_index"] = str(random.randint(0, 100))
    with intro_tab:
        file_content = get_intro()
        st.markdown(file_content, unsafe_allow_html=True)
        add_download_pdf(
            f"{workflow}_introduction_tutorial.pdf",
            file_content,
            ":floppy_disk: Download as PDF",
        )
    with uploader_tab:
        st.markdown("##### Upload data for processing")

        upload_type = "Raw files"
        files = None
        file_chunks = None
        if upload_type == "Raw files":
            files = st.file_uploader(
                "Upload PDF text files",
                type=["pdf", "txt", "json", "csv"],
                accept_multiple_files=True,
                key="qtd_uploader_" + st.session_state[f"{workflow}_uploader_index"],
            )
        else:
            file_chunks = st.file_uploader(
                "Upload processed chunks",
                type=["csv"],
                key="chunk_uploader_" + st.session_state[f"{workflow}_uploader_index"],
            )

        local_embedding = st.toggle(
            "Use local embeddings",
            key=sv.answer_local_embedding_enabled.key,
            value=sv.answer_local_embedding_enabled.value,
            help="Use local embeddings to index nodes. If disabled, the model will use OpenAI embeddings.",
        )
        qtd.set_embedder(embedder.create_embedder(local_embedding, 20))

        if st.button("Process files") and (
            files is not None or file_chunks is not None
        ):
            qtd.reset_workflow()
            if upload_type == "Raw files":
                file_pb, file_callback = functions.create_progress_callback(
                    "Loaded {} of {} files..."
                )
                qtd.process_data_from_files(
                    input_file_bytes={file.name: file.getvalue() for file in files},
                    callbacks=[file_callback],
                )
                file_pb.empty()
            else:
                qtd.import_chunks_from_str(file_chunks)

            chunk_pb, chunk_callback = functions.create_progress_callback(
                "Processed {} of {} chunks..."
            )
            qtd.process_text_chunks(callbacks=[chunk_callback])

            embed_pb, embed_callback = functions.create_progress_callback(
                "Embedded {} of {} text chunks..."
            )
            await qtd.embed_text_chunks(callbacks=[embed_callback])
            chunk_pb.empty()
            embed_pb.empty()
            st.rerun()

        if qtd.stage.value >= QueryTextDataStage.CHUNKS_PROCESSED.value:
            num_files = len(qtd.label_to_chunks.keys())
            num_chunks = sum([len(cs) for f, cs in qtd.label_to_chunks.items()])
            num_periods = len(qtd.processed_chunks.period_to_cids) - 1
            G = qtd.processed_chunks.period_concept_graphs["ALL"]
            message = f"Chunked **{num_files}** file{'s' if num_files > 1 else ''} into **{num_chunks}** chunks of up to **{CHUNK_SIZE}** tokens. Extracted concept graph with **{len(G.nodes())}** concepts and **{len(G.edges())}** cooccurrences"
            if num_periods > 1:
                message += ", spanning **{num_periods}** periods."
            else:
                message += "."
            message = message.replace("**1** periods", "**1** period")
            st.success(message)

        # if qtd.label_to_chunks and upload_type == "Raw files":
        #     st.download_button(
        #         label="Download chunk data",
        #         help="Export chunk data as CSV",
        #         data=qtd.get_chunks_as_df().to_csv(),
        #         file_name=f"processed_data_{len(qtd.label_to_chunks)}_query_text.csv",
        #         mime="text/csv",
        #     )

    with graph_tab:
        if qtd.stage.value < QueryTextDataStage.CHUNKS_PROCESSED.value:
            st.warning("Process files to continue.")
        else:
            G = qtd.processed_chunks.period_concept_graphs["ALL"]
            c1, c2 = st.columns([5, 2])
            selection = None
            with c1:

                level_to_label_to_network = qtd.build_concept_community_graph()
                selected_level_labels = [''] + [str(v) for v in level_to_label_to_network[0].keys()]
                selected_label = st.selectbox("Select topic area", options=selected_level_labels, key=f"{workflow}_community_label")
                gp = st.empty()
                if selected_label not in ['', None]:
                    selection = functions.get_concept_graph(
                        gp,
                        level_to_label_to_network[0][selected_label],
                        qtd.processed_chunks.hierarchical_communities,
                        800,
                        700,
                        "graph",
                    )
            with c2:
                if selection is not None:
                    selected_cids = qtd.processed_chunks.concept_to_cids[selection]
                    selected_cids_df = pd.DataFrame(
                        [
                            {
                                "Matching text chunks (double click to expand)": qtd.processed_chunks.cid_to_text[
                                    cid
                                ]
                            }
                            for cid in selected_cids
                        ]
                    )
                    st.markdown(f"**Selected concept: {selection}**")
                    st.dataframe(selected_cids_df, hide_index=True, height=650, use_container_width=True)
    with search_tab:
        if qtd.stage.value < QueryTextDataStage.CHUNKS_EMBEDDED.value:
            st.warning(f"Process files to continue.")
        else:
            with st.expander("Advanced Options", expanded=False):
                c1, c2, c3, c4, c5 = st.columns(5)
                
                with c1:
                    st.number_input(
                        "Tests/topic/round",
                        value=sv.relevance_test_batch_size.value,
                        key=sv.relevance_test_batch_size.key,
                        min_value=0,
                        help="How many relevant tests to perform for each topic in each round. Larger values reduce the likelihood of prematurely discarding topics whose relevant chunks may not be at the top of the similarity-based ranking, but may result in smaller values of `Relevance test budget` being spread across fewer topics and thus not capturing the full breadth of the data."
                    )
                with c2:
                    st.number_input(
                        "Restart on irrelevant topics",
                        value=sv.irrelevant_community_restart.value,
                        key=sv.irrelevant_community_restart.key,
                        min_value=0,
                        help="When this number of topics in a row fail to return any relevant chunks in their `Tests/topic/round`, return to the start of the topic ranking and continue testing `Tests/topic/round` text chunks from each topic with (a) relevance in the previous round and (b) previously untested text chunks. Higher values can avoid prematurely discarding topics that are relevant but whose relevant chunks are not at the top of the similarity-based ranking, but may result in a larger number of irrelevant topics being tested multiple times."
                    )
                with c3:
                    st.number_input(
                        "Test relevant neighbours",
                        value=sv.adjacent_test_steps.value,
                        key=sv.adjacent_test_steps.key,
                        min_value=0,
                        help="If a text chunk is relevant to the query, then adjacent text chunks in the original document may be able to add additional context to the relevant points. The value of this parameter determines how many chunks before and after each relevant text chunk will be evaluated at the end of the process (or `Relevance test budget`) if they are yet to be tested."
                    )
                with c4:
                    st.number_input(
                        "Target chunks per cluster",
                        value=sv.target_chunks_per_cluster.value,
                        key=sv.target_chunks_per_cluster.key,
                        min_value=0,
                        help="The average number of text chunks to target per cluster, which determines the text chunks that will be evaluated together and in parallel to other clusters. Larger values will generally result in more related text chunks being evaluated in parallel, but may also result in information loss from unprocessed content."
                    )
                with c5:
                    st.checkbox(
                        "Show search process",
                        key=sv.show_search_process.key,
                        value=sv.show_search_process.value,
                        help="Show the search process in the UI, including the progress of chunk relevance tests and the search for relevant chunks."
                    )
            query_panel = st.container()
            main_panel = st.container()

            with query_panel:
                c1, c2 = st.columns([5, 1])
                with c1:
                    query_placeholder = st.empty()
                with c2:
                    budget_placeholder = st.empty()
                anchored_query_placeholder = st.empty()
            with main_panel:
                if sv.show_search_process.value:
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        chunk_placeholder = st.empty()
                        chunk_placeholder.dataframe(
                            pd.DataFrame(
                                columns=["Relevant text chunks (double click to expand)"],
                                data=[qtd.processed_chunks.cid_to_text[x] for x in qtd.relevant_cids] if qtd.relevant_cids != None else [],
                            ),
                            hide_index=True,
                            height=400,
                            use_container_width=True,
                        )
                        chunk_progress_placeholder = st.empty()
                        if qtd.search_summary is not None:
                            chunk_progress_placeholder.markdown(qtd.search_summary, unsafe_allow_html=True)
                else:
                    c2 = st.container()
                if sv.anchored_query.value != "":    
                        anchored_query_placeholder.markdown(f"**Expanded query:** {sv.anchored_query.value}")
                with c2:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown("#### Live analysis")
                        analysis_placeholder = st.empty()
                        commentary_placeholder = st.empty()
                        if qtd.stage.value >= QueryTextDataStage.CHUNKS_MINED.value:
                            st.download_button(
                                "Download live analysis as MD",
                                data=sv.thematic_analysis.value + "\n" + sv.thematic_commentary.value,
                                file_name=re.sub(r'[^\w\s]','',sv.query.value).replace(' ', '_')+"_analysis.md",
                                mime="text/markdown",
                                key="live_analysis_download_button",
                                disabled=qtd.stage.value < QueryTextDataStage.QUESTION_ANSWERED.value,
                            )
                    with c2:
                        

                        st.markdown("#### Final report")
                        answer_spinner = st.empty()
                        answer_placeholder = st.empty()
                        if qtd.stage.value >= QueryTextDataStage.CHUNKS_MINED.value:
                            st.download_button(
                                "Download research report as MD",
                                data=qtd.answer_object.extended_answer if qtd.answer_object is not None else "",
                                file_name=re.sub(r'[^\w\s]','',sv.query.value).replace(' ', '_')+".md",
                                mime="text/markdown",
                                key="research_report_download_button",
                                disabled=qtd.stage.value < QueryTextDataStage.QUESTION_ANSWERED.value,
                            )
                            analysis_placeholder.markdown(sv.thematic_analysis.value, unsafe_allow_html=True)
                            commentary_placeholder.markdown(sv.thematic_commentary.value, unsafe_allow_html=True)
                        if qtd.stage.value == QueryTextDataStage.QUESTION_ANSWERED.value:
                            answer_placeholder.markdown(qtd.answer_object.extended_answer, unsafe_allow_html=True)

                    def do_search():
                        st.session_state["search_answers"] = True
                        sv.query.value = st.session_state[sv.query.key]
                        qtd.prepare_for_new_query()
                        sv.chunk_progress.value = ""
                        sv.answer_progress.value = ""
                        answer_placeholder.markdown("")
                        main_panel.empty()
                    query_placeholder.text_input(
                        "Query",
                        key=sv.query.key,
                        on_change=lambda: do_search()
                    )
                    budget_placeholder.number_input(
                        "Relevance test budget",
                        value=sv.relevance_test_budget.value,
                        key=sv.relevance_test_budget.key,
                        min_value=0,
                        help="The query method works by asking an LLM to evaluate the relevance of potentially-relevant text chunks, returning a single token, yes/no judgement. This parameter allows the user to cap the number of relvance tests that may be performed prior to generating an answer using all relevant chunks. Larger budgets will generally give better answers for a greater cost."
                    )
                    if sv.query.value != "" and st.session_state["search_answers"]:
                        st.session_state["search_answers"] = False
                        sv.anchored_query.value = await qtd.anchor_query_to_concepts(
                            query=sv.query.value,
                            top_concepts=500,
                        )
                        anchored_query_placeholder.markdown(f"**Expanded query:** {sv.anchored_query.value}")
                        def on_chunk_progress(message):
                            if sv.show_search_process.value:
                                chunk_progress_placeholder.markdown(message, unsafe_allow_html=True)

                        def on_chunk_relevant(message):
                            if sv.show_search_process.value:
                                chunk_placeholder.dataframe(
                                    pd.DataFrame(
                                        columns=["Relevant text chunks (double click to expand)"],
                                        data=message,
                                    ),
                                    hide_index=True,
                                    height=400,
                                    use_container_width=True,
                                )

                        analysis_callback = LLMCallback()
                        def on_llm_new_token_analysis(message):
                            analysis_placeholder.markdown(message, unsafe_allow_html=True)
                            sv.thematic_analysis.value = message
                        analysis_callback.on_llm_new_token = on_llm_new_token_analysis

                        commentary_callback = LLMCallback()
                        def on_llm_new_token_commentary(message):
                            commentary_placeholder.markdown(message, unsafe_allow_html=True)
                            sv.thematic_commentary.value = message
                        commentary_callback.on_llm_new_token = on_llm_new_token_commentary

                        await qtd.detect_relevant_text_chunks(
                            query=sv.query.value,
                            expanded_query=sv.anchored_query.value,
                            chunk_search_config=ChunkSearchConfig(
                                relevance_test_budget=sv.relevance_test_budget.value,
                                relevance_test_batch_size=sv.relevance_test_batch_size.value,
                                community_ranking_chunks=sv.relevance_test_batch_size.value,
                                irrelevant_community_restart=sv.irrelevant_community_restart.value,
                                adjacent_test_steps=sv.adjacent_test_steps.value,
                                community_relevance_tests=sv.relevance_test_batch_size.value,
                            ),
                            chunk_progress_callback=on_chunk_progress,
                            chunk_callback=on_chunk_relevant,
                            analysis_callback=analysis_callback,
                            commentary_callback=commentary_callback,
                        )
                        st.rerun()
                    if gen_answer or qtd.stage.value == QueryTextDataStage.CHUNKS_MINED.value:
                        with answer_spinner:
                            with st.spinner("Generating research report..."):
                                await asyncio.gather(
                                    qtd.answer_query_with_relevant_chunks(),
                                    qtd.generate_analysis_commentary()                      
                                )
                                st.rerun()
    with report_tab:
        if qtd.stage.value < QueryTextDataStage.QUESTION_ANSWERED.value:
            st.warning("Generate a research report to continue.")
        else:
            c1, c2 = st.columns([2, 3])

            with c1:
                variables = {
                    "query": qtd.query,
                    "answer": qtd.answer_object.extended_answer,
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
                            "Press the Generate button to create an AI report for the current query."
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
