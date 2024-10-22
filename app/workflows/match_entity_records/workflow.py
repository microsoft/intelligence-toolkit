# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import io
import os

import numpy as np
import polars as pl
import streamlit as st

import app.util.example_outputs_ui as example_outputs_ui
import app.util.session_variables as home_vars
import app.workflows.match_entity_records.functions as functions
import app.workflows.match_entity_records.variables as rm_variables
import toolkit.match_entity_records.prompts as prompts
from app.util import ui_components
from app.util.download_pdf import add_download_pdf
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback
from toolkit.match_entity_records import (
    AttributeToMatch,
    MatchEntityRecords,
    RecordsModel,
)


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()
    
async def create(sv: rm_variables.SessionVariable, workflow=None) -> None:
    sv_home = home_vars.SessionVariables("home")
    ui_components.check_ai_configuration()
    mer = MatchEntityRecords()

    intro_tab, uploader_tab, process_tab, evaluate_tab, examples_tab = st.tabs(
        [
            "Match Entity Records workflow:",
            "Upload record datasets",
            "Detect record groups",
            "Evaluate record groups",
            "View example outputs",
        ]
    )
    selected_df = None
    with intro_tab:
        file_content = get_intro()
        st.markdown(file_content)
        add_download_pdf(
            f"{workflow}_introduction_tutorial.pdf",
            file_content,
            ":floppy_disk: Download as PDF",
        )
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            selected_file, selected_df, changed = ui_components.multi_csv_uploader(
                "Upload multiple CSVs",
                sv.matching_uploaded_files,
                workflow + "uploader",
                sv.matching_max_rows_to_process,
            )
        with model_col:
            st.markdown("##### Map columns to data model")
            if selected_df is None:
                st.warning("Upload and select a file to continue")
            else:
                selected_df = pl.from_pandas(selected_df).lazy()
                cols = selected_df.columns
                entity_id_col = ""
                ready = False
                dataset = st.text_input(
                    "Dataset name",
                    key=f"{selected_file}_dataset_name",
                    value=selected_file.split(".")[0] if selected_file else "",
                    help="Used to track which dataset each record came from; not used in the matching process itself.",
                )
                name_col = st.selectbox(
                    "Entity name column",
                    cols,
                    help="The column containing the name of the entity to be matched. This column is required.",
                )
                entity_id_col = st.selectbox(
                    "Entity ID column (optional)",
                    cols,
                    help="The column containing the unique identifier of the entity to be matched. If left blank, a unique ID will be generated for each entity based on the row number.",
                )
                filtered_cols = [
                    c for c in cols if c not in [entity_id_col, name_col, ""]
                ]
                att_cols = st.multiselect(
                    "Entity attribute columns",
                    filtered_cols,
                    help="Columns containing attributes of the entity to be matched. These columns will be used to match entities based on their similarity.",
                )
                ready = (
                    dataset is not None
                    and len(dataset) > 0
                    and len(att_cols) > 0
                    and name_col != ""
                )
                b1, b2 = st.columns([1, 1])
                with b1:
                    if st.button(
                        "Add records to model",
                        disabled=not ready,
                        use_container_width=True,
                    ):
                        model = RecordsModel(
                            dataframe=selected_df.collect(),
                            name_column=name_col,
                            columns=att_cols,
                            dataframe_name=dataset,
                            id_column=entity_id_col,
                        )
                        dataset_added = mer.add_df_to_model(model)
                        sv.matching_dfs.value[dataset] = dataset_added
                with b2:
                    if st.button(
                        "Reset data model",
                        disabled=len(sv.matching_dfs.value) == 0,
                        use_container_width=True,
                    ):
                        mer.clear_model_dfs()
                        sv.matching_dfs.value = {}
                        sv.matching_merged_df.value = pl.DataFrame()
                        st.rerun()
                if len(sv.matching_dfs.value) > 0:
                    recs = sum(len(df) for df in sv.matching_dfs.value.values())
                    st.success(
                        f"Data model has **{len(sv.matching_dfs.value)}** datasets with **{recs}** total records."
                    )
                    if not mer.model_dfs:
                        mer.model_dfs = sv.matching_dfs.value

    with process_tab:
        if len(sv.matching_dfs.value) == 0:
            st.warning("Upload data files to continue")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("##### Configure text embedding model")
                attr_options = mer.attribute_options
                sv.matching_mapped_atts.value = []

                num_atts = 0
                while True:
                    if f"att{num_atts}_vals" not in st.session_state:
                        break
                    num_atts += 1

                def att_ui(i, any_empty, changed, attsaa):
                    st.markdown(f"**Attribute {i + 1}**")
                    is_assigned = False
                    b1, b2 = st.columns([3, 1])
                    if f"att{i}_vals" not in st.session_state:
                        st.session_state[f"att{i}_vals"] = []
                    if f"att{i}_name" not in st.session_state:
                        st.session_state[f"att{i}_name"] = ""

                    with b1:
                        att_vals = st.multiselect(
                            "Values",
                            key=f"{i}_value",
                            default=st.session_state[f"att{i}_vals"]
                            if st.session_state[f"att{i}_vals"] in attr_options
                            else [],
                            options=attr_options,
                            help="Select all columns that represent the same attribute across datasets.",
                        )
                        if len(att_vals) > 0:
                            is_assigned = True
                    with b2:
                        att_name = st.text_input(
                            "Label (optional)",
                            key=f"{i}_name",
                            value=st.session_state[f"att{i}_name"],
                            help="The name to assign to this attribute in the merged dataset. If left blank, the first value selected will be used.",
                        )
                        att_name_original = att_name
                        if att_name == "" and len(att_vals) > 0:
                            att_name = sorted(att_vals)[0].split("::")[0]

                        attsaa.append(
                            AttributeToMatch({"label": att_name, "columns": att_vals})
                        )

                        if st.session_state[f"att{i}_vals"] != att_vals:
                            st.session_state[f"att{i}_vals"] = att_vals
                            changed = True
                        if st.session_state[f"att{i}_name"] != att_name_original:
                            st.session_state[f"att{i}_name"] = att_name_original
                            changed = True
                        if not is_assigned:
                            any_empty = True
                    return any_empty, changed, attsaa

                any_empty = False
                changed = False
                attsa = []
                for i in range(num_atts):
                    any_empty, changed, attsa = att_ui(i, any_empty, changed, attsa)

                if not any_empty:
                    _, changed, attsa = att_ui(num_atts, any_empty, changed, attsa)
                if changed:
                    st.rerun()

                local_embedding = st.toggle(
                    "Use local embeddings",
                    sv.matching_local_embedding_enabled.value,
                    help="Use local embeddings to index nodes. If disabled, the model will use OpenAI embeddings.",
                )
                st.markdown("##### Configure similarity thresholds")
                b1, b2 = st.columns([1, 1])
                with b1:
                    record_distance = st.number_input(
                        "Matching record distance (max)",
                        min_value=0.001,
                        max_value=1.0,
                        step=0.001,
                        format="%f",
                        value=sv.matching_sentence_pair_embedding_threshold.value,
                        help="The maximum cosine distance between two records in the embedding space for them to be considered a match. Lower values will result in fewer closer matches overall.",
                    )
                with b2:
                    name_similarity = st.number_input(
                        "Matching name similarity (min)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        value=sv.matching_sentence_pair_jaccard_threshold.value,
                        help="The minimum Jaccard similarity between the character trigrams of the names of two records for them to be considered a match. Higher values will result in fewer closer name matches.",
                    )

                if st.button("Detect record groups", use_container_width=True):
                    sv.matching_evaluations.value = pl.DataFrame()
                    sv.matching_report_validation.value = {}
                    if (
                        record_distance
                        != sv.matching_sentence_pair_embedding_threshold.value
                    ):
                        sv.matching_sentence_pair_embedding_threshold.value = (
                            record_distance
                        )
                    if (
                        name_similarity
                        != sv.matching_sentence_pair_jaccard_threshold.value
                    ):
                        sv.matching_sentence_pair_jaccard_threshold.value = (
                            name_similarity
                        )

                    with st.spinner("Detecting groups..."):
                        # if (
                        #     len(sv.matching_merged_df.value) == 0
                        #     or sv.matching_sentence_pair_embedding_threshold.value
                        #     != sv.matching_last_sentence_pair_embedding_threshold.value
                        # ):
                        sv.matching_last_sentence_pair_embedding_threshold.value = (
                            sv.matching_sentence_pair_embedding_threshold.value
                        )
                        sv.matching_merged_df.value = mer.build_model_df(attsa)
                        all_sentences_data = mer.sentences_vector_data

                        pb = st.progress(0, "Embedding text batches...")

                        def on_embedding_batch_change(current, total):
                            pb.progress(
                                (current) / total,
                                f"Embedding text {current} of {total}...",
                            )

                        callback = ProgressBatchCallback()
                        callback.on_batch_change = on_embedding_batch_change

                        functions_embedder = functions.embedder(local_embedding)
                        data_embeddings = await functions_embedder.embed_store_many(
                            all_sentences_data, [callback], sv_home.save_cache.value
                        )

                        all_sentences = [x["text"] for x in all_sentences_data]
                        all_embeddings = [
                            np.array(
                                next(
                                    d["vector"]
                                    for d in data_embeddings
                                    if d["text"] == f
                                )
                            )
                            for f in all_sentences
                        ]
                        mer.embeddings = all_embeddings
                        mer.all_sentences = all_sentences

                        pb.empty()
                        sv.matching_matches_df.value = mer.detect_record_groups(
                            sv.matching_sentence_pair_embedding_threshold.value,
                            sv.matching_sentence_pair_jaccard_threshold.value,
                        )

                        st.rerun()
                if len(sv.matching_matches_df.value) > 0:
                    st.markdown(
                        f"Identified **{len(sv.matching_matches_df.value['Group ID'].unique())}** record groups."
                    )
            with c2:
                st.markdown("##### Record groups")
                if len(sv.matching_matches_df.value) > 0:
                    data = sv.matching_matches_df.value
                    st.dataframe(
                        data, height=700, use_container_width=True, hide_index=True
                    )
                    st.download_button(
                        "Download record groups",
                        data=data.write_csv(),
                        file_name="record_groups.csv",
                        mime="text/csv",
                    )

    with evaluate_tab:
        b1, b2 = st.columns([2, 3])
        with b1:
            batch_size = 100
            data = sv.matching_matches_df.value.drop(
                [
                    "Entity ID",
                    "Dataset",
                    "Name similarity",
                ]
            ).to_pandas()
            generate, batch_messages, reset = (
                ui_components.generative_batch_ai_component(
                    sv.matching_system_prompt, {}, "data", data, batch_size
                )
            )
            if reset:
                sv.matching_system_prompt.value["user_prompt"] = prompts.user_prompt
                st.rerun()
        with b2:
            st.markdown("##### AI evaluation of record groups")
            prefix = "```\nGroup ID,Relatedness,Explanation\n"
            placeholder = st.empty()
            gen_placeholder = st.empty()

            if generate:
                for messages in batch_messages:
                    callback = ui_components.create_markdown_callback(
                        placeholder, prefix
                    )
                    response = ui_components.generate_text(messages, [callback])

                    if len(response.strip()) > 0:
                        prefix = prefix + response + "\n"
                result = prefix.replace("```\n", "").strip()
                sv.matching_evaluations.value = result
                lines = result.split("\n")

                if len(lines) > 30:
                    lines = lines[:30]
                    result = "\n".join(lines)

                # validation, messages_to_llm = ui_components.validate_ai_report(
                #     batch_messages[0], result
                # )
                # sv.matching_report_validation.value = validation
                # sv.matching_report_validation_messages.value = messages_to_llm
                st.rerun()
            else:
                if len(sv.matching_evaluations.value) == 0:
                    gen_placeholder.warning(
                        "Press the Generate button to create an AI report for the current record matches."
                    )
            placeholder.empty()

            if len(sv.matching_evaluations.value) > 0:
                try:
                    csv = pl.read_csv(io.StringIO(sv.matching_evaluations.value))
                    value = csv.drop_nulls()
                    jdf = sv.matching_matches_df.value.join(
                        value, on="Group ID", how="inner"
                    )
                    st.dataframe(
                        value.to_pandas(),
                        height=700,
                        use_container_width=True,
                        hide_index=True,
                    )
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.download_button(
                            "Download AI match reports",
                            data=csv.write_csv(),
                            file_name="record_group_match_reports.csv",
                            mime="text/csv",
                        )
                    with c2:
                        st.download_button(
                            "Download integrated results",
                            data=jdf.write_csv(),
                            file_name="integrated_record_match_results.csv",
                            mime="text/csv",
                        )
                except:
                    st.markdown(sv.matching_evaluations.value)
                    add_download_pdf(
                        "record_groups_evaluated.pdf",
                        sv.matching_evaluations.value,
                        "Download AI match report",
                    )

    with examples_tab:
        example_outputs_ui.create_example_outputs_ui(examples_tab, workflow)