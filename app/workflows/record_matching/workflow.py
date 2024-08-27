# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import io
from collections import defaultdict

import pandas as pd
import polars as pl
import streamlit as st

import app.util.session_variables as home_vars
import app.workflows.record_matching.functions as functions
import app.workflows.record_matching.variables as rm_variables
import toolkit.record_matching.prompts as prompts
from app.util import ui_components
from app.util.download_pdf import add_download_pdf
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback
from toolkit.record_matching import get_readme as get_intro
from toolkit.record_matching.detect import (
    build_matches,
    build_matches_dataset,
    build_near_map,
    build_nearest_neighbors,
    build_sentence_pair_scores,
    convert_to_sentences,
)
from toolkit.record_matching.prepare_model import format_dataset


def create(sv: rm_variables.SessionVariable, workflow=None) -> None:
    sv_home = home_vars.SessionVariables("home")

    intro_tab, uploader_tab, process_tab, evaluate_tab = st.tabs(
        [
            "Record matching workflow:",
            "Upload data to match",
            "Detect record groups",
            "Evaluate record groups",
        ]
    )
    selected_df = None
    with intro_tab:
        st.markdown(get_intro())
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            selected_file, selected_df = ui_components.multi_csv_uploader(
                "Upload multiple CSVs",
                sv.matching_uploaded_files,
                sv.matching_upload_key.value,
                "matching_uploader",
                sv.matching_max_rows_to_process,
            )
        with model_col:
            st.markdown("##### Map columns to data model")
            if selected_df is None:
                st.warning("Upload and select a file to continue")
            else:
                selected_df = pl.from_pandas(selected_df).lazy()
                cols = ["", *selected_df.columns]
                entity_col = ""
                ready = False
                dataset = st.text_input(
                    "Dataset name",
                    key=f"{selected_file}_dataset_name",
                    help="Used to track which dataset each record came from; not used in the matching process itself.",
                )
                name_col = st.selectbox(
                    "Entity name column",
                    cols,
                    help="The column containing the name of the entity to be matched. This column is required.",
                )
                entity_col = st.selectbox(
                    "Entity ID column (optional)",
                    cols,
                    help="The column containing the unique identifier of the entity to be matched. If left blank, a unique ID will be generated for each entity based on the row number.",
                )
                filtered_cols = [c for c in cols if c not in [entity_col, name_col]]
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
                        sv.matching_dfs.value[dataset] = format_dataset(
                            selected_df.collect(),
                            att_cols,
                            name_col,
                            entity_col,
                            sv.matching_max_rows_to_process.value,
                        )
                with b2:
                    if st.button(
                        "Reset data model",
                        disabled=len(sv.matching_dfs.value) == 0,
                        use_container_width=True,
                    ):
                        sv.matching_dfs.value = {}
                        sv.matching_merged_df.value = pl.DataFrame()
                        st.rerun()
                if len(sv.matching_dfs.value) > 0:
                    recs = sum(len(df) for df in sv.matching_dfs.value.values())
                    st.success(
                        f"Data model has **{len(sv.matching_dfs.value)}** datasets with **{recs}** total records."
                    )

    with process_tab:
        if len(sv.matching_dfs.value) == 0:
            st.warning("Upload data files to continue")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("##### Configure text embedding model")
                all_atts = []
                for dataset, merged_df in sv.matching_dfs.value.items():
                    all_atts.extend(
                        [
                            f"{c}::{dataset}"
                            for c in merged_df.columns
                            if c not in ["Entity ID", "Entity name"]
                        ]
                    )
                    all_atts = sorted(all_atts)
                options = sorted(all_atts)
                renaming = defaultdict(dict)
                atts_to_datasets = defaultdict(list)
                sv.matching_mapped_atts.value = []
                # por ex, inicialmente tenho 3 datasets, cada um com 3 atributos
                # att0_vals = [dataset1::att1, dataset2::att1, dataset3::att1]
                # att1_vals = [dataset1::att2, dataset2::att2, dataset3::att2]
                # att2_vals = [dataset1::att3, dataset2::att3, dataset3::att3]

                num_atts = 0
                while True:
                    if f"att{num_atts}_vals" not in st.session_state:
                        break
                    num_atts += 1

                def att_ui(i):
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
                            if st.session_state[f"att{i}_vals"] in options
                            else [],
                            options=options,
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
                            att_name = att_vals[0].split("::")[0]

                    for val in att_vals:
                        col, dataset = val.split("::")
                        renaming[dataset][col] = att_name
                        atts_to_datasets[att_name].append(dataset)
                    return is_assigned, att_vals, att_name_original

                any_empty = False
                changed = False
                for i in range(num_atts):
                    is_assigned, att_vals, att_name_original = att_ui(i)
                    if st.session_state[f"att{i}_vals"] != att_vals:
                        st.session_state[f"att{i}_vals"] = att_vals
                        changed = True
                    if st.session_state[f"att{i}_name"] != att_name_original:
                        st.session_state[f"att{i}_name"] = att_name_original
                        changed = True
                    if not is_assigned:
                        any_empty = True
                if not any_empty:
                    att_ui(num_atts)
                if changed:
                    st.rerun()

                st.markdown("##### Configure similarity thresholds")
                b1, b2 = st.columns([1, 1])
                with b1:
                    record_distance = st.number_input(
                        "Matching record distance (max)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
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
                        if (
                            len(sv.matching_merged_df.value) == 0
                            or sv.matching_sentence_pair_embedding_threshold.value
                            != sv.matching_last_sentence_pair_embedding_threshold.value
                        ):
                            sv.matching_last_sentence_pair_embedding_threshold.value = (
                                sv.matching_sentence_pair_embedding_threshold.value
                            )
                            aligned_dfs = []
                            for dataset, merged_df in sv.matching_dfs.value.items():
                                rdf = merged_df.clone()
                                rdf = rdf.rename(renaming[dataset])
                                # drop columns that were not renamed
                                for col in merged_df.columns:
                                    if (
                                        col not in ["Entity ID", "Entity name"]
                                        and col not in renaming[dataset].values()
                                    ):
                                        rdf = rdf.drop(col)
                                for att, datasets in atts_to_datasets.items():
                                    if dataset not in datasets:
                                        rdf = rdf.with_columns(pl.lit("").alias(att))
                                rdf = rdf.with_columns(pl.lit(dataset).alias("Dataset"))
                                rdf = rdf.select(sorted(rdf.columns))
                                aligned_dfs.append(rdf)
                            string_dfs = []
                            for merged_df in aligned_dfs:
                                # convert all columns to strings
                                for col in merged_df.columns:
                                    merged_df = merged_df.with_columns(
                                        pl.col(col).cast(pl.Utf8)
                                    )
                                string_dfs.append(merged_df)
                            sv.matching_merged_df.value = pl.concat(string_dfs)
                            sv.matching_merged_df.value = (
                                sv.matching_merged_df.value.filter(
                                    pl.col("Entity name") != ""
                                )
                            )
                            sv.matching_merged_df.value = (
                                sv.matching_merged_df.value.with_columns(
                                    (pl.col("Entity ID").cast(pl.Utf8))
                                    + "::"
                                    + pl.col("Dataset").alias("Unique ID")
                                )
                            )
                            all_sentences = convert_to_sentences(
                                sv.matching_merged_df.value
                            )

                            pb = st.progress(0, "Embedding text batches...")

                            def on_embedding_batch_change(current, total):
                                pb.progress(
                                    (current) / total,
                                    f"Embedding text batch {current} of {total}...",
                                )

                            callback = ProgressBatchCallback()
                            callback.on_batch_change = on_embedding_batch_change

                            functions_embedder = functions.embedder()

                            embeddings = functions_embedder.embed_store_many(
                                all_sentences, [callback], sv_home.save_cache.value
                            )
                            pb.empty()

                            distances, indices = build_nearest_neighbors(embeddings)

                            near_map = build_near_map(
                                distances,
                                indices,
                                all_sentences,
                                sv.matching_sentence_pair_embedding_threshold.value,
                            )

                            sv.matching_sentence_pair_scores.value = (
                                build_sentence_pair_scores(
                                    near_map, all_sentences, sv.matching_merged_df.value
                                )
                            )

                        merged_df = sv.matching_merged_df.value
                        entity_to_group, matches, pair_to_match = build_matches(
                            sv.matching_sentence_pair_scores.value,
                            merged_df,
                            sv.matching_sentence_pair_jaccard_threshold.value,
                        )

                        sv.matching_matches_df.value = pl.DataFrame(
                            list(matches),
                            schema=["Group ID", *sv.matching_merged_df.value.columns],
                        ).sort(
                            by=["Group ID", "Entity name", "Dataset"], descending=False
                        )

                        sv.matching_matches_df.value = build_matches_dataset(
                            sv.matching_matches_df.value, pair_to_match, entity_to_group
                        )

                        st.rerun()
                if len(sv.matching_matches_df.value) > 0:
                    st.markdown(
                        f"Identified **{len(sv.matching_matches_df.value)}** record groups."
                    )
            with c2:
                data = sv.matching_matches_df.value
                st.markdown("##### Record groups")
                if len(sv.matching_matches_df.value) > 0:
                    if sv_home.protected_mode.value:
                        unique_names = sv.matching_matches_df.value[
                            "Entity name"
                        ].unique()
                        for i, name in enumerate(unique_names, start=1):
                            data = data.with_columns(
                                data["Entity name"].replace(name, f"Entity_{i}")
                            )

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
                unique_names = sv.matching_matches_df.value["Entity name"].unique()
                for messages in batch_messages:
                    callback = ui_components.create_markdown_callback(
                        placeholder, prefix
                    )
                    response = ui_components.generate_text(messages, [callback])

                    if len(response.strip()) > 0:
                        prefix = prefix + response + "\n"
                    if sv_home.protected_mode.value:
                        for i, name in enumerate(unique_names, start=1):
                            prefix = prefix.replace(name, f"Entity_{i}")

                result = prefix.replace("```\n", "").strip()
                sv.matching_evaluations.value = result
                lines = result.split("\n")

                if len(lines) > 30:
                    lines = lines[:30]
                    result = "\n".join(lines)

                validation, messages_to_llm = ui_components.validate_ai_report(
                    batch_messages[0], result
                )
                sv.matching_report_validation.value = validation
                sv.matching_report_validation_messages.value = messages_to_llm
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
                    st.download_button(
                        "Download AI match report",
                        data=jdf.write_csv(),
                        file_name="record_groups_evaluated.csv",
                        mime="text/csv",
                    )
                except:
                    st.markdown(sv.matching_evaluations.value)
                    add_download_pdf(
                        "record_groups_evaluated.pdf",
                        sv.matching_evaluations.value,
                        "Download AI match report",
                    )

                report = (
                    pd.DataFrame(sv.matching_evaluations.value).to_json()
                    if type(sv.matching_evaluations.value) == pl.DataFrame
                    else sv.matching_evaluations.value
                )
                ui_components.build_validation_ui(
                    sv.matching_report_validation.value,
                    sv.matching_report_validation_messages.value,
                    report,
                    workflow,
                )
