# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
import re
import pandas as pd

import streamlit as st
from util import ui_components
from util.download_pdf import add_download_pdf
from util.session_variables import SessionVariables
from util.openai_wrapper import UIOpenAIConfiguration
from python.AI.embedder import Embedder
from streamlit_agraph import Config, Edge, Node, agraph
from python.AI import utils, text_splitter
from python.AI.defaults import CHUNK_SIZE
import python.question_answering.prompts as prompts
import python.question_answering.process_texts as process_texts
import python.question_answering.search_answers as search_answers
from seaborn import color_palette
from workflows.question_answering import config

sv_home = SessionVariables("home")

def embedder():
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        return Embedder(
            ai_configuration, config.cache_dir, sv_home.local_embeddings.value
        )
    except Exception as e:
        st.error(f"Error creating connection: {e}")
        st.stop()

def get_concept_graph(placeholder, G, concept_to_community, community_to_concepts, width, height, key):
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
        community_to_concepts.keys(), key=lambda x: len(community_to_concepts[x]), reverse=True
    )
    community_to_color = dict(zip(sorted_communities, community_colors))
    for node in G.nodes():
        degree = G.degree(node)
        size = 5 + 20 * degree / max_degree
        vadjust = -size*2 - 3
        color = community_to_color[concept_to_community[node]]
        color = '#%02x%02x%02x' % tuple([int(255 * x) for x in color])
        nodes.append(
            Node(
                title=node,
                id=node,
                label=node,
                size=size,
                color=color,
                shape='dot',
                timestep=0.1,
                font={"vadjust": vadjust, "size": size},
            )
        )

    for u, v, d in G.edges(data=True):
        edges.append(Edge(source=u, target=v, color='lightgray'))

    config = Config(
        width=width,
        height=height,
        directed=False,
        physics=True,
        hierarchical=False,
        key=key,
        linkLength=100
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
    intro_tab, uploader_tab, graph_tab, search_tab, report_tab = st.tabs([
        "Question answering workflow:",
        "Upload data",
        "Explore concept graph",
        "Search for answers",
        "Generate AI answer reports",
    ])

    with intro_tab:
        st.markdown(get_intro())
    with uploader_tab:
        st.markdown("##### Upload data for processing")
        files = st.file_uploader(
            "Upload PDF text files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=sv.upload_key.value,
        )
        if files is not None and st.button("Process files"):
            # functions.chunk_files(sv, files)
            file_pb = st.progress(0, "Processing files...")
            def on_file_batch_change(current, total):
                file_pb.progress(int(current * 100 / total), text="Processing files...")

            file_callback = ProgressBatchCallback()
            file_callback.on_batch_change = on_file_batch_change
            text_to_pages = process_texts.process_files(
                input_file_bytes={file.name: file.getvalue() for file in files},
                text_splitter=text_splitter.TextSplitter(), # type: ignore
                callbacks=[file_callback]
            )
            
            chunk_pb = st.progress(0, "Processing text chunks...")
            def on_chunk_batch_change(current, total):
                chunk_pb.progress(int(current * 100 / total), text="Processing text chunks...")

            chunk_callback = ProgressBatchCallback()
            chunk_callback.on_batch_change = on_chunk_batch_change

            (
                sv.text_to_chunks.value,
                sv.text_to_vectors.value,
                sv.concept_graph.value,
                sv.community_to_concepts.value,
                sv.concept_to_community.value,
                sv.concept_to_chunks.value,
                sv.chunk_to_concepts.value,
                sv.previous_chunk.value,
                sv.next_chunk.value
            ) = process_texts.process_chunks(
                text_to_chunks=text_to_pages,
                embedder=embedder(),
                embedding_cache=sv_home.save_cache.value,
                max_cluster_size=50,
                callbacks=[chunk_callback]
            )
            chunk_pb.empty()
            file_pb.empty()
        num_files = len(sv.text_to_chunks.value.keys())
        num_chunks = sum([len(cs) for f, cs in sv.text_to_chunks.value.items()])
        G = sv.concept_graph.value
        if num_files > 0:
            st.success(
                f"Chunked **{num_files}** file{'s' if num_files > 1 else ''} into **{num_chunks}** chunks of up to **{CHUNK_SIZE}** tokens. Extracted concept graph with **{len(G.nodes())}** concepts and **{len(G.edges())}** cooccurrences."
            )
    with graph_tab:
        c1, c2 = st.columns([5, 2])
        selection = None
        with c1:
            gp = st.empty()
            if sv.concept_graph.value is not None:
                selection = get_concept_graph(gp, G, sv.concept_to_community.value, sv.community_to_concepts.value, 800, 700, "graph")
        with c2:
            if sv.concept_graph.value is not None and selection is not None:
                selected_chunks = sv.concept_to_chunks.value[selection]
                selected_chunks_df = pd.DataFrame(
                    [
                        {"Matching text chunks (double click to expand)": chunk} for chunk in selected_chunks
                    ]
                )
                st.markdown(f"**Selected concept: {selection}**")
                st.dataframe(selected_chunks_df, hide_index=True, height=650)
    with search_tab:
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            st.number_input(
                "Terminate on chunks tested",
                value=sv.terminate_on_chunks_tested.value,
                key=sv.terminate_on_chunks_tested.key,
                min_value=0,
            )
        with c2:
            st.number_input(
                "Terminate on relevant chunks",
                value=sv.terminate_on_relevant_chunks.value,
                key=sv.terminate_on_relevant_chunks.key,
                min_value=0,
            )
        with c3:
            st.number_input(
                "Terminate on successive irrelevant",
                value=sv.terminate_on_successive_irrelevant.value,
                key=sv.terminate_on_successive_irrelevant.key,
                min_value=0,
            )
        with c4:
            st.number_input(
                "Switch on successive irrelevant",
                value=sv.switch_on_successive_irrelevant.value,
                key=sv.switch_on_successive_irrelevant.key,
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
        progress_placeholder = st.markdown('Answer search progress')
        with c1:
            chunk_placeholder = st.empty()
        with c2:
            answer_placeholder = st.empty()

        chunk_placeholder.dataframe(pd.DataFrame(columns=["Relevant text chunks (double click to expand)"], data=sv.relevant_chunks.value), hide_index=True, height=500, use_container_width=True)
        answer_placeholder.dataframe(pd.DataFrame(columns=["Partial answers (double click to expand)"], data=sv.partial_answers.value), hide_index=True, height=500, use_container_width=True)
        if sv.last_question.value != "" and regenerate:
            def on_progress(message):
                progress_text = f"Chunks tested: {message['chunks_tested']}, Relevant chunks: {message['relevant_chunks']}, Successive irrelevant: {message['successive_irrelevant']}"
                sv.progress_message.value = progress_text
                progress_placeholder.markdown(progress_text)
            def on_chunk_relevant(message):
                chunk_placeholder.dataframe(pd.DataFrame(columns=["Relevant text chunks (double click to expand)"], data=message), hide_index=True, height=500, use_container_width=True)
            def on_answer(message):
                answer_placeholder.dataframe(pd.DataFrame(columns=["Partial answers (double click to expand)"], data=message), hide_index=True, height=500, use_container_width=True)



            sv.relevant_chunks.value, sv.partial_answers.value = search_answers.search_answers(
                sv.last_question.value,
                sv.text_to_chunks.value,
                sv.chunk_to_concepts.value,
                sv.concept_to_chunks.value,
                sv.text_to_vectors.value,
                sv.community_to_concepts.value,
                sv.concept_to_community.value,
                sv.previous_chunk.value,
                sv.next_chunk.value,
                embedder=embedder(),
                embedding_cache=sv_home.save_cache.value,
                select_logit_bias=5,
                terminate_on_chunks_tested=sv.terminate_on_chunks_tested.value,
                terminate_on_relevant_chunks=sv.terminate_on_relevant_chunks.value,
                terminate_on_successive_irrelevant=sv.terminate_on_successive_irrelevant.value,
                switch_on_successive_irrelevant=sv.switch_on_successive_irrelevant.value,
                relevance_batch_size=5,
                answer_batch_size=5,
                augment_top_concepts=20,
                progress_callback=on_progress,
                chunk_callback=on_chunk_relevant,
                answer_callback=on_answer
            )
        

   
    with report_tab:
        if sv.partial_answers.value == []:
            st.warning("Search for answers to continue.")
        else:
            c1, c2 = st.columns([2, 3])

            with c1:
                variables = {
                    "question": sv.last_question.value,
                    "answers": sv.partial_answers.value
                }
                generate, messages, reset = ui_components.generative_ai_component(
                    sv.system_prompt, variables
                )
                if reset:
                    sv.answering_system_prompt.value["user_prompt"] = (
                        prompts.user_prompt
                    )
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
