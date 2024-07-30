# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import json
import os
import re
from collections import Counter

import numpy as np
import scipy.spatial.distance
import streamlit as st
import workflows.question_answering.classes as classes
import workflows.question_answering.functions as functions
import workflows.question_answering.prompts as prompts
from util import ui_components
from util.df_functions import get_current_time
from util.download_pdf import add_download_pdf
from util.session_variables import SessionVariables

from python.AI import utils
from python.AI.defaults import CHUNK_SIZE


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


def create(sv: SessionVariables, workflow=None):
    sv_home = SessionVariables("home")
    intro_tab, uploader_tab, mining_tab, report_tab = st.tabs([
        "Question answering workflow:",
        "Upload data",
        "Mine & match questions",
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
            key=sv.answering_upload_key.value,
        )
        if files is not None and st.button("Chunk and embed files"):
            functions.chunk_files(sv, files)

        num_files = len(sv.answering_files.value)
        num_chunks = sum(len(f.chunk_texts) for f in sv.answering_files.value.values())
        if num_files > 0:
            st.success(
                f"Chunked **{num_files}** file{'s' if num_files > 1 else ''} into **{num_chunks}** chunks of up to **{CHUNK_SIZE}** tokens."
            )
    with mining_tab:
        c1, c2, c3, c4, c5 = st.columns([4, 1, 1, 1, 1])
        with c1:
            question = st.text_input(
                "Question", value=sv.answering_last_lazy_question.value
            )
        with c2:
            answering_target_matches = st.number_input(
                "Target QA matches",
                min_value=1,
                step=1,
                value=sv.answering_target_matches.value,
            )
        with c3:
            answering_source_diversity = st.number_input(
                "Target source diversity",
                min_value=1,
                step=1,
                value=sv.answering_source_diversity.value,
            )
        with c4:
            max_iterations = st.number_input(
                "Max iterations",
                min_value=1,
                step=1,
                value=sv.answering_max_iterations.value,
            )
        with c5:
            regenerate = st.button(
                "Mine matching questions",
                key="lazy_regenerate",
                use_container_width=True,
            )
            if regenerate:
                sv.answering_matches.value = ""
                # sv.answering_status_history.value = ''
                # sv.answering_status_history.value = ''
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("#### Question mining")
            if len(sv.answering_matches.value) > 0:
                txt = ""
                seen_qs = set()
                for d in sv.answering_context_list.value:
                    if d.id not in seen_qs:
                        txt += d.generate_outline(level=4) + "\n\n"
                        seen_qs.update([d.id])
                add_download_pdf(
                    f"question_mining_{get_current_time()}.pdf",
                    txt,
                    "Download mining questions",
                )
            lazy_answering_placeholder = st.empty()
        with c2:
            st.markdown("#### Question matching")
            lazy_matches_placeholder = st.empty()
        lazy_answering_placeholder.markdown(
            sv.answering_status_history.value, unsafe_allow_html=True
        )
        lazy_matches_placeholder.markdown(
            sv.answering_matches.value, unsafe_allow_html=True
        )

        if question != "" and regenerate:
            sv.answering_context_list.value = []
            sv.answering_question_history.value = []
            sv.answering_next_q_id.value = 1
            sv.answering_surface_questions.value = {}
            sv.answering_deeper_questions.value = {}
            sv.answering_report_validation.value = {}
            sv.answering_target_matches.value = answering_target_matches
            sv.answering_source_diversity.value = answering_source_diversity
            sv.answering_max_iterations.value = max_iterations
            sv.answering_last_lazy_question.value = question
            sv.answering_status_history.value = ""
            sv.answering_matches.value = ""
            sv.answering_question_history.value.append(question)
            lazy_answering_placeholder.markdown(
                sv.answering_status_history.value, unsafe_allow_html=True
            )
            lazy_matches_placeholder.markdown(
                sv.answering_matches.value, unsafe_allow_html=True
            )
            all_units = []
            for file in sv.answering_files.value.values():
                all_units += [
                    ("chunk", (file, cx), v)
                    for cx, v in file.chunk_vectors.items()
                    if v is not None
                ]
            for qid, q in sv.answering_surface_questions.value.items():
                if q.vector is not None:
                    all_units.append(("question", q, q.vector))
                    q.answer_texts[0]
                    av = q.answer_vectors[0]
                    if av is not None:
                        all_units.append(("answer", q, av))
            matched_qs = []
            status_history = ""
            iteration = 0
            source_counts = Counter()
            used_chunks = set()
            functions_embedder = functions.embedder()

            while True:
                qe = np.array(
                    functions_embedder.embed_store_one(
                        question, sv_home.save_cache.value
                    )
                )

                iteration += 1
                cosine_distances = sorted(
                    [
                        (t, c, scipy.spatial.distance.cosine(qe, v))
                        for (t, c, v) in all_units
                    ],
                    key=lambda x: x[2],
                    reverse=False,
                )
                chunk_index = sv.answering_target_matches.value
                for ix, (t, c, d) in enumerate(
                    cosine_distances[: sv.answering_target_matches.value]
                ):
                    if t == "chunk":
                        chunk_index = ix
                        break

                matched_qs = [c for t, c, d in cosine_distances[:chunk_index]]
                if (
                    len(matched_qs) == sv.answering_target_matches.value
                    or iteration > sv.answering_max_iterations.value
                ):
                    iteration_string = (
                        f"Iteration **{iteration}**"
                        if iteration <= sv.answering_max_iterations.value
                        else f"Exceeded maximum iterations of **{sv.answering_max_iterations.value}**"
                    )
                    status_history += (
                        f"{iteration_string}...<br/><br/>Matched user question to **{len(matched_qs)}** mined questions:<br/>- "
                        + "<br/>- ".join([
                            f"Q{matched_q.id}: {matched_q.text}"
                            for matched_q in matched_qs
                        ])
                        + "<br/><br/>"
                    )
                    sv.answering_status_history.value = status_history
                    report_input = f"**User question**: {sv.answering_question_history.value[0]}\n\n**Augmented question**: {sv.answering_question_history.value[-1]}\n\n"
                    seen_qs = set()
                    for t, c, d in cosine_distances:
                        if t in ["question", "answer"]:
                            if c.id not in seen_qs:
                                delta = c.generate_outline(level=6)
                                qs = c.list_all_sub_questions()
                                candidate = report_input + delta
                                candidate_tokens = ui_components.return_token_count(
                                    candidate
                                )
                                if candidate_tokens <= sv.answering_outline_limit.value:
                                    report_input = candidate
                                    seen_qs.update([q.id for q in qs])
                                else:
                                    break
                    sv.answering_matches.value = report_input
                    # st.session_state['generate_lazy_answer'] = True
                    st.rerun()
                status_history += f"**Iteration {iteration}**...<br/><br/>Matched user question to **{len(matched_qs)}** of **{sv.answering_target_matches.value}** mined questions before reaching an unmined chunk"
                if len(matched_qs) > 0:
                    status_history += ":<br/>- " + "<br/>- ".join([
                        f"Q{matched_q.id}: {matched_q.text}" for matched_q in matched_qs
                    ])
                status_history += "<br/><br/>"

                new_questions = []
                target_diversity = min(
                    sv.answering_source_diversity.value, len(sv.answering_files.value)
                )
                for j in range(len(cosine_distances)):
                    t, c, d = cosine_distances[j]

                    if t == "chunk" and c not in used_chunks:
                        f = c[0]
                        cx = c[1]
                        source = c[0].id
                        max_source_counts = source_counts.most_common(target_diversity)
                        max_count = (
                            max_source_counts[0][1] if len(max_source_counts) > 0 else 0
                        )
                        num_max = len([
                            x for x in max_source_counts if x[1] == max_count
                        ])
                        source_count = source_counts[source]
                        # print(f'Got a chunk with source count {source_count}. Max count is {max_count} and num max is {num_max}.')
                        if (
                            source_count > 0
                            and source_count == max_count
                            and num_max < target_diversity
                        ):
                            pass
                            # print('Cannot use')
                        else:
                            # print('Can use')
                            used_chunks.add(c)
                            cosine_distances.pop(j)
                            source_counts.update([source])
                            break

                status_history += f"Mining the next most similar chunk for question-answer pairs: chunk **{cx}** from file **{f.name}**...<br/>"
                variables = {"text": f.chunk_texts[cx], "file_id": f.id}
                messages = utils.prepare_messages(
                    system_message=prompts.extraction_system_prompt,
                    user_message=prompts.extraction_user_prompt,
                    variables=variables,
                )
                on_callback = ui_components.create_markdown_callback(
                    lazy_answering_placeholder, prefix=status_history
                )
                qas_raw = ui_components.generate_text(messages, callbacks=[on_callback])
                status_history += qas_raw + "<br/><br/>"
                try:
                    functions_embedder = functions.embedder()
                    qas = json.loads(qas_raw)
                    for qa in qas:
                        q = qa["question"]
                        a = qa["answer"]
                        raw_refs = qa["source"]
                        file_page_refs = [
                            tuple([int(x[1:]) for x in r.split(";")]) for r in raw_refs
                        ]

                        q_vec = np.array(
                            functions_embedder.embed_store_one(
                                q, sv_home.save_cache.value
                            )
                        )
                        a_vec = np.array(
                            functions_embedder.embed_store_one(
                                a, sv_home.save_cache.value
                            )
                        )

                        qid = sv.answering_next_q_id.value
                        sv.answering_next_q_id.value += 1
                        q = classes.Question(f, q, q_vec, 0, qid)
                        sv.answering_context_list.value.append(q)
                        new_questions.append(q)
                        print(f"Created question {qid} from file {f.id}.")
                        f.add_question(q)
                        sv.answering_surface_questions.value[qid] = q
                        f.add_answer(qid, a, file_page_refs, a_vec)
                        all_units.append(("question", q, q.vector))
                    for t, c, v in list(all_units):
                        if t == "chunk" and c[0].id == f.id and c[1] == cx:
                            all_units.remove((t, c, v))

                    status_history += (
                        "Augmenting user question with partial answers:<br/>"
                    )
                    new_question = functions.update_question(
                        sv.answering_question_history.value,
                        new_questions,
                        lazy_answering_placeholder,
                        status_history,
                    )
                    status_history += new_question + "<br/><br/>"
                    sv.answering_question_history.value.append(new_question)
                except Exception as e:
                    print("Error processing JSON")
                    print(e)

    with report_tab:
        # if 'generate_lazy_answer' in st.session_state:
        #     del st.session_state['generate_lazy_answer']
        if sv.answering_matches.value == "":
            st.warning("Mine question matches to continue.")
        else:
            c1, c2 = st.columns([2, 3])

            with c1:
                variables = {
                    "question": sv.answering_question_history.value[-1],
                    "outline": sv.answering_matches.value,
                    "source_diversity": sv.answering_source_diversity.value,
                }
                generate, messages, reset = ui_components.generative_ai_component(
                    sv.answering_system_prompt, variables
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
                    sv.answering_lazy_answer_text.value = result

                    validation, messages_to_llm = ui_components.validate_ai_report(
                        messages, result
                    )
                    sv.answering_report_validation.value = validation
                    sv.answering_report_validation_messages.value = messages_to_llm
                    st.rerun()
                else:
                    if sv.answering_lazy_answer_text.value == "":
                        gen_placeholder.warning(
                            "Press the Generate button to create an AI report for the current question."
                        )
                report_placeholder.markdown(sv.answering_lazy_answer_text.value)
                report_data = sv.answering_lazy_answer_text.value
                is_download_disabled = report_data == ""
                name = (
                    sv.answering_lazy_answer_text.value.split("\n")[0]
                    .replace("#", "")
                    .strip()
                    .replace(" ", "_")
                )
                full_text = (
                    sv.answering_lazy_answer_text.value
                    + "\n\n## Supporting FAQ\n\n"
                    + re.sub(
                        r" Q[\d]+: ",
                        " ",
                        "\n\n".join(sv.answering_matches.value.split("\n\n")[2:]),
                        re.MULTILINE,
                    ).replace("###### ", "### ")
                )

                if len(sv.answering_lazy_answer_text.value) > 0:
                    report_data = sv.answering_lazy_answer_text.value
                    is_download_disabled = report_data == ""
                    name = (
                        sv.answering_lazy_answer_text.value.split("\n")[0]
                        .replace("#", "")
                        .strip()
                        .replace(" ", "_")
                    )
                    full_text = (
                        sv.answering_lazy_answer_text.value
                        + "\n\n## Supporting FAQ\n\n"
                        + re.sub(
                            r" Q[\d]+: ",
                            " ",
                            "\n\n".join(sv.answering_matches.value.split("\n\n")[2:]),
                            re.MULTILINE,
                        ).replace("###### ", "### ")
                    )

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.download_button(
                            "Download AI answer report as MD",
                            data=full_text,
                            file_name=f"{name}.md",
                            mime="text/markdown",
                            disabled=sv.answering_lazy_answer_text.value == "",
                            key="lazy_download_button",
                        )
                    with c2:
                        add_download_pdf(
                            f"{name}.pdf",
                            full_text,
                            "Download AI answer report as PDF",
                            disabled=is_download_disabled,
                        )

                    ui_components.build_validation_ui(
                        sv.answering_report_validation.value,
                        sv.answering_report_validation_messages.value,
                        sv.answering_lazy_answer_text.value,
                        workflow,
                    )
