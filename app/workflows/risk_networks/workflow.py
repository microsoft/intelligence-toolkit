# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

# ruff: noqa
import json
import os
import re
from collections import defaultdict

import networkx as nx
import pandas as pd
import streamlit as st
import workflows.risk_networks.functions as functions
import workflows.risk_networks.variables as rn_variables
from st_aggrid import (
    AgGrid,
    ColumnsAutoSizeMode,
    DataReturnMode,
    GridOptionsBuilder,
    GridUpdateMode,
)
from streamlit_agraph import agraph
from util import ui_components
from util.session_variables import SessionVariables

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback
from toolkit.risk_networks import config
from toolkit.risk_networks import get_readme as get_intro
from toolkit.risk_networks import graph_functions, prompts
from toolkit.risk_networks.identify import project_entity_graph, trim_nodeset
from toolkit.risk_networks.model import prepare_entity_attribute
from toolkit.risk_networks.network import build_network_from_entities, generate_final_df
from toolkit.risk_networks.node_community import get_community_nodes
from toolkit.risk_networks.protected_mode import protect_data
from toolkit.risk_networks.text_format import format_data_columns


def create(sv: rn_variables.SessionVariables, workflow=None):
    sv_home = SessionVariables("home")

    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)

    intro_tab, uploader_tab, process_tab, view_tab, report_tab = st.tabs(
        [
            "Network analysis workflow:",
            "Create data model",
            "Process data model",
            "Explore networks",
            "Generate AI network reports",
        ]
    )
    df = None
    with intro_tab:
        st.markdown(get_intro())
    with uploader_tab:
        uploader_col, model_col = st.columns([3, 2])
        with uploader_col:
            _selected_file, df = ui_components.multi_csv_uploader(
                "Upload multiple CSVs",
                sv.network_uploaded_files,
                sv.network_upload_key.value,
                "network_uploader",
                sv.network_max_rows_to_process,
            )
        with model_col:
            st.markdown("##### Map columns to model")
            if df is None:
                st.warning("Upload and select a file to continue")
            else:
                options = ["", *df.columns.to_numpy()]
                link_type = st.radio(
                    "Link type",
                    [
                        "Entity-Attribute",
                        "Entity-Flag",
                        "Entity-Group",
                    ],
                    horizontal=True,
                    help="Select the type of link to create. Entity-Attribute links connect entities with shared attributes, Entity-Flag links connect entities to risk flags, and Entity-Group links enable filtering of detected networks by specified grouping attributes.",
                )
                entity_col = st.selectbox(
                    "Entity ID column",
                    options,
                    help="The column containing unique entity identifiers, shared across datasets.",
                )
                model_links = None
                attribute_label = ""
                if link_type == "Entity-Attribute":
                    value_cols = st.multiselect(
                        "Attribute value column(s) to link on",
                        options,
                        help="The column(s) containing attribute values that would only be shared by closely related entities.",
                    )
                    attribute_col = st.selectbox(
                        "Attribute type",
                        ["Use column name", "Use custom name"],
                        disabled=True,
                        index=0,
                        help="Where the name of the attribute comes from: the selected column or the user.",
                    )
                    model_links = sv.network_attribute_links.value
                elif link_type == "Entity-Flag":
                    value_cols = st.multiselect(
                        "Flag value column(s)",
                        options,
                        help="The column(s) containing risk flags associated with entities.",
                    )
                    attribute_col = st.selectbox(
                        "Flag type",
                        ["Use column name", "Use custom name"],
                        disabled=True,
                        index=0,
                        help="Where the name of the flag comes from: the selected column or the user.",
                    )
                    flag_agg = st.selectbox(
                        "Flag format",
                        ["Instance", "Count"],
                        help="How flags are represented: as individual instances or as aggregate counts in a flag value column.",
                    )
                    model_links = sv.network_flag_links.value
                elif link_type == "Entity-Group":
                    value_cols = st.multiselect(
                        "Group value column(s) to group on",
                        options,
                        help="The column(s) containing group values that are shared by groups of broadly related entities.",
                    )
                    attribute_col = st.selectbox(
                        "Group type",
                        ["Use column name", "Use custom name"],
                        help="Where the name of the group comes from: the selected column or the user.",
                    )
                    model_links = sv.network_group_links.value
                b1, b2 = st.columns([1, 1])
                with b1:
                    if st.button(
                        "Add links to model",
                        disabled=entity_col == ""
                        or attribute_col == ""
                        or len(value_cols) == 0
                        or link_type == "",
                    ):
                        with st.spinner("Adding links to model..."):
                            value_col = ""
                            df = format_data_columns(df, value_cols, entity_col)
                            if sv_home.protected_mode.value:
                                (
                                    df,
                                    entities_renamed,
                                    attributes_renamed,
                                ) = protect_data(
                                    df,
                                    value_cols,
                                    entity_col,
                                    sv.network_entities_renamed.value,
                                )
                                sv.network_entities_renamed.value = entities_renamed
                                sv.network_attributes_renamed.value = attributes_renamed

                            if link_type == "Entity-Attribute":
                                attribute_links, node_types = prepare_entity_attribute(
                                    df,
                                    entity_col,
                                    attribute_col,
                                    value_cols,
                                    attribute_label,
                                )
                                sv.network_node_types.value = node_types
                                sv.network_overall_graph.value = (
                                    graph_functions.build_undirected_graph(
                                        network_attribute_links=attribute_links
                                    )
                                )
                            elif link_type == "Entity-Flag":
                                # groupby entity and sum flag counts
                                gdf = df.groupby([entity_col]).sum().reset_index()
                                gdf["attribute_col"] = attribute_label
                                gdf["attribute_col2"] = attribute_label
                                if attribute_col in [
                                    "Use column name",
                                    "Use custom name",
                                ]:
                                    sv.network_flag_types.value.add(attribute_label)
                                    if flag_agg == "Instance":
                                        gdf["count_col"] = 1
                                        model_links.append(
                                            gdf[
                                                [
                                                    entity_col,
                                                    "attribute_col",
                                                    value_col,
                                                    "count_col",
                                                ]
                                            ].values.tolist()
                                        )
                                    elif flag_agg == "Count":
                                        gdf[value_col] = gdf[value_col].astype(int)
                                        model_links.append(
                                            gdf[
                                                [
                                                    entity_col,
                                                    "attribute_col",
                                                    "attribute_col2",
                                                    value_col,
                                                ]
                                            ].values.tolist()
                                        )
                                else:
                                    sv.network_flag_types.value.update(
                                        df[attribute_label].unique().tolist()
                                    )
                                    if flag_agg == "Instance":
                                        model_links.append(
                                            gdf[
                                                [
                                                    entity_col,
                                                    attribute_col,
                                                    value_col,
                                                ]
                                            ].values.tolist()
                                        )
                                    elif flag_agg == "Count":
                                        gdf[value_col] = gdf[value_col].astype(int)
                                        model_links.append(
                                            gdf[
                                                [
                                                    entity_col,
                                                    attribute_col,
                                                    "attribute_col2",
                                                    value_col,
                                                ]
                                            ].values.tolist()
                                        )
                                functions.build_integrated_flags(sv)
                            elif link_type == "Entity-Group":
                                if attribute_col in [
                                    "Use column name",
                                    "Use custom name",
                                ]:
                                    df["attribute_col"] = attribute_label
                                    sv.network_group_types.value.add(attribute_label)
                                    model_links.append(
                                        df[
                                            [entity_col, "attribute_col", value_col]
                                        ].values.tolist()
                                    )
                                else:
                                    sv.network_group_types.value.update(
                                        df[attribute_label].unique().tolist()
                                    )
                                    model_links.append(
                                        df[
                                            [entity_col, attribute_col, value_col]
                                        ].values.tolist()
                                    )
                with b2:
                    if st.button(
                        "Clear data model",
                        disabled=entity_col == ""
                        or attribute_col == ""
                        or len(value_cols) == 0
                        or link_type == "",
                    ):
                        sv.network_attribute_links.value = []
                        sv.network_flag_links.value = []
                        sv.network_group_links.value = []
                        sv.network_overall_graph.value = None
                        sv.network_entity_graph.value = None
                        sv.network_community_df.value = pd.DataFrame()
                        sv.network_integrated_flags.value = pd.DataFrame()

            num_entities = 0
            num_attributes = 0
            num_edges = 0
            num_flags = 0
            groups = set()
            for link_list in sv.network_group_links.value:
                for link in link_list:
                    groups.add(f"{link[1]}{ATTRIBUTE_VALUE_SEPARATOR}{link[2]}")
            if sv.network_overall_graph.value is not None:
                all_nodes = sv.network_overall_graph.value.nodes()
                entity_nodes = [
                    node for node in all_nodes if node.startswith(config.entity_label)
                ]
                sv.network_attributes_list.value = [
                    node
                    for node in all_nodes
                    if not node.startswith(config.entity_label)
                ]
                num_entities = len(entity_nodes)
                num_attributes = len(all_nodes) - num_entities
                num_edges = len(sv.network_overall_graph.value.edges())

                original_df = pd.DataFrame(
                    sv.network_attributes_list.value, columns=["Attribute"]
                )
                atributes_entities = []
                unique_names = original_df["Attribute"].unique()
                for i, name in enumerate(unique_names, start=1):
                    name_format = name.split("==")[0].strip()
                    atributes_entities.append(
                        (
                            name,
                            f"{name_format}=={name_format}_{i!s}",
                        )
                    )

                sv.network_attributes_renamed.value = atributes_entities

            if len(sv.network_integrated_flags.value) > 0:
                num_flags = sv.network_integrated_flags.value["count"].sum()
            if num_entities > 0:
                st.markdown("##### Data model summary")
                st.markdown(
                    f"*Number of entities*: {num_entities}<br/>*Number of attributes*: {num_attributes}<br/>*Number of links*: {num_edges}<br/>*Number of flags*: {num_flags}<br/>*Number of groups*: {len(groups)}",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Add links to the model to continue.")

    with process_tab:
        index_col, part_col = st.columns([1, 1])
        components = None
        with index_col:
            st.markdown("##### Index nodes (optional)")
            fuzzy_options = sorted(
                [
                    config.entity_label,
                    *list(sv.network_node_types.value),
                ]
            )
            network_indexed_node_types = st.multiselect(
                "Select node types to fuzzy match",
                default=sv.network_indexed_node_types.value,
                options=fuzzy_options,
                help="Select the node types to embed into a multi-dimensional semantic space for fuzzy matching.",
            )
            if st.button("Index nodes", disabled=len(network_indexed_node_types) == 0):
                pb = st.progress(0, "Embedding text batches...")

                def on_embedding_batch_change(current, total):
                    pb.progress(
                        (current) / total,
                        f"Embedding text batch {current} of {total}...",
                    )

                callback = ProgressBatchCallback()
                callback.on_batch_change = on_embedding_batch_change

                sv.network_indexed_node_types.value = network_indexed_node_types

                (
                    sv.network_embedded_texts.value,
                    sv.network_nearest_text_distances.value,
                    sv.network_nearest_text_indices.value,
                ) = graph_functions.index_nodes(
                    sv.network_indexed_node_types.value,
                    sv.network_overall_graph.value,
                    [callback],
                    sv_home.local_embeddings.value,
                    sv_home.save_cache.value,
                )
                st.rerun()
            nodes_indexed = len(sv.network_embedded_texts.value)
            if nodes_indexed > 0:
                st.markdown(f"*Number of nodes indexed*: {nodes_indexed}")
            st.markdown("##### Infer links (optional)")
            network_similarity_threshold = st.number_input(
                "Similarity threshold for fuzzy matching (max)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=sv.network_similarity_threshold.value,
                help="The maximum cosine similarity threshold for inferring links between nodes based on their embeddings. Higher values will infer more links, but may also infer more false positives.",
            )
            if st.button(
                "Infer links",
                disabled=len(sv.network_nearest_text_distances.value) == 0,
            ):
                pb = st.progress(0, text="Inferring links...")

                def on_embedding_batch_change(current, total):
                    pb.progress(int(current * 100 / total), text="Inferring links...")

                callback = ProgressBatchCallback()
                callback.on_batch_change = on_embedding_batch_change

                sv.network_similarity_threshold.value = network_similarity_threshold

                sv.network_inferred_links.value = graph_functions.infer_nodes(
                    network_similarity_threshold,
                    sv.network_embedded_texts.value,
                    sv.network_nearest_text_indices.value,
                    sv.network_nearest_text_distances.value,
                    [callback],
                )
                pb.empty()
                st.rerun()

            link_list = graph_functions.create_links(sv.network_inferred_links.value)
            ilc = len(link_list)

            if ilc > 0:
                st.markdown(f"*Number of links inferred*: {ilc}")
                idf = pd.DataFrame(link_list, columns=["text", "similar"])
                idf["text"] = idf["text"].str.replace(
                    config.entity_label + ATTRIBUTE_VALUE_SEPARATOR, ""
                )
                idf["similar"] = idf["similar"].str.replace(
                    config.entity_label + ATTRIBUTE_VALUE_SEPARATOR, ""
                )
                idf = idf.sort_values(by=["text", "similar"]).reset_index(drop=True)
                st.dataframe(idf, hide_index=True, use_container_width=True)

        with part_col:
            st.markdown("##### Identify networks")
            adf = pd.DataFrame(sv.network_attributes_list.value, columns=["Attribute"])

            search = st.text_input("Search for attributes to remove", "")
            if search != "":
                adf = adf[adf["Attribute"].str.contains(search, case=False)]

            selected_rows = ui_components.dataframe_with_selections(
                adf,
                sv.network_additional_trimmed_attributes.value,
                "Attribute",
                "Remove",
                key="remove_attribute_table",
            )
            sv.network_additional_trimmed_attributes.value = selected_rows[
                "Attribute"
            ].tolist()

            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                network_max_attribute_degree = st.number_input(
                    "Maximum attribute degree",
                    min_value=1,
                    value=sv.network_max_attribute_degree.value,
                    help="The maximum number of entities that can share an attribute before it is removed from the network.",
                )
            with c2:
                network_max_cluster_size = st.number_input(
                    "Max network size",
                    min_value=2,
                    value=sv.network_max_cluster_size.value,
                    help="Any network with edges >= max_cluster_size will be isolated into a subnetwork",
                )
            with c3:
                network_supporting_attribute_types = st.multiselect(
                    "Supporting attribute types",
                    default=sv.network_supporting_attribute_types.value,
                    options=sorted(sv.network_node_types.value),
                    help="Attribute types that should not be used to detect networks (e.g., because of potential noise/unreliability) but which should be added back into detected networks for context.",
                )
            comm_count = 0
            with c4:
                st.text("")
                st.text("")
                identify = st.button("Identify networks")
            if identify:
                sv.network_max_attribute_degree.value = network_max_attribute_degree
                sv.network_max_cluster_size.value = network_max_cluster_size
                sv.network_supporting_attribute_types.value = (
                    network_supporting_attribute_types
                )

                with st.spinner("Identifying networks..."):
                    sv.network_table_index.value += 1
                    # Create a new graph P in which entities are connected if they share an attribute
                    (trimmed_degrees, trimmed_nodes) = trim_nodeset(
                        sv.network_overall_graph.value,
                        sv.network_additional_trimmed_attributes.value,
                        sv.network_max_attribute_degree.value,
                    )

                    sv.network_trimmed_attributes.value = (
                        pd.DataFrame(
                            trimmed_degrees,
                            columns=["Attribute", "Linked Entities"],
                        )
                        .sort_values("Linked Entities", ascending=False)
                        .reset_index(drop=True)
                    )
                    P = project_entity_graph(
                        sv.network_overall_graph.value,
                        trimmed_nodes,
                        sv.network_inferred_links.value,
                        sv.network_supporting_attribute_types.value,
                    )

                    (
                        sv.network_community_nodes.value,
                        sv.network_entity_to_community_ix.value,
                    ) = get_community_nodes(
                        P,
                        sv.network_max_network_size.value,
                        sv.network_max_cluster_size.value,
                    )

                    N = build_network_from_entities(
                        sv.network_overall_graph.value,
                        sv.network_entity_to_community_ix.value,
                        sv.network_integrated_flags.value,
                        sv.network_trimmed_attributes.value,
                        sv.network_inferred_links.value,
                    )

                entity_records = generate_final_df(
                    sv.network_community_nodes.value,
                    sv.network_integrated_flags.value,
                )

                sv.network_entity_df.value = pd.DataFrame(
                    entity_records,
                    columns=[
                        "Entity ID",
                        "Entity Flags",
                        "Network ID",
                        "Network Entities",
                        "Network Flags",
                        "Flagged",
                        "Flags/Entity",
                        "Flagged/Unflagged",
                    ],
                )
                sv.network_entity_df.value = sv.network_entity_df.value.sort_values(
                    by=["Flagged/Unflagged"], ascending=False
                ).reset_index(drop=True)
                sv.network_table_index.value += 1
                st.rerun()
            comm_count = len(sv.network_community_nodes.value)

            if comm_count > 0:
                comm_sizes = [
                    len(comm)
                    for comm in sv.network_community_nodes.value
                    if len(comm) > 1
                ]
                max_comm_size = max(comm_sizes)
                trimmed_atts = len(sv.network_trimmed_attributes.value)
                st.markdown(
                    f"*Networks identified: {comm_count} ({len(comm_sizes)} with multiple entities, maximum {max_comm_size})*"
                )
                st.markdown(
                    f"*Attributes removed because of high degree*: {trimmed_atts}"
                )

                adf = pd.DataFrame(
                    sv.network_attributes_list.value, columns=["Attribute"]
                )
                if trimmed_atts > 0:
                    st.dataframe(
                        sv.network_trimmed_attributes.value,
                        hide_index=True,
                        use_container_width=True,
                    )

                entities = sv.network_entity_df.value.copy()
                entities_renamed = []
                for i, name in enumerate(entities["Entity ID"], start=1):
                    entities_renamed.append((name, f"Entity ID_{i!s}"))
                sv.network_entities_renamed.value = entities_renamed

    with view_tab:
        if len(sv.network_entity_df.value) == 0:
            st.warning("Detect networks to continue.")
        else:
            with st.expander("View entity networks", expanded=True):
                b1, b2, b3, _b4 = st.columns([1, 1, 1, 4])
                with b1:
                    show_entities = st.checkbox(
                        "Show entities", value=sv.network_last_show_entities.value
                    )
                with b2:
                    show_groups = st.checkbox(
                        "Show groups", value=sv.network_last_show_groups.value
                    )
                with b3:
                    dl_button = st.empty()

                show_df = sv.network_entity_df.value.copy()
                if show_groups != sv.network_last_show_groups.value:
                    sv.network_last_show_groups.value = show_groups
                    sv.network_table_index.value += 1
                    st.rerun()
                if show_entities != sv.network_last_show_entities.value:
                    sv.network_last_show_entities.value = show_entities
                    sv.network_table_index.value += 1
                    st.rerun()
                if show_groups:
                    for group_links in sv.network_group_links.value:
                        df = pd.DataFrame(
                            group_links, columns=["Entity ID", "Group", "Value"]
                        ).replace("nan", "")
                        df = df[df["Value"] != ""]
                        # Use group values as columns with values in them
                        df = df.pivot_table(
                            index="Entity ID",
                            columns="Group",
                            values="Value",
                            aggfunc="first",
                        ).reset_index()
                        show_df = show_df.merge(df, on="Entity ID", how="left")
                last_df = show_df.copy()
                if not show_entities:
                    last_df = (
                        last_df.drop(columns=["Entity ID", "Entity Flags"])
                        .drop_duplicates()
                        .reset_index(drop=True)
                    )
                dl_button.download_button(
                    "Download network data",
                    last_df.to_csv(index=False),
                    "network_data.csv",
                    "Download network data",
                )
                gb = GridOptionsBuilder.from_dataframe(last_df)
                gb.configure_default_column(
                    flex=1,
                    wrapText=True,
                    wrapHeaderText=True,
                    enablePivot=False,
                    enableValue=False,
                    enableRowGroup=False,
                )
                gb.configure_selection(selection_mode="single", use_checkbox=False)
                gb.configure_side_bar()
                gridoptions = gb.build()
                response = AgGrid(
                    last_df,
                    key=f"report_grid_{sv.network_table_index.value}",
                    height=400,
                    gridOptions=gridoptions,
                    enable_enterprise_modules=False,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=False,
                    header_checkbox_selection_filtered_only=False,
                    use_checkbox=False,
                    enable_quicksearch=True,
                    reload_data=True,
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                )

            selected_entity = (
                response["selected_rows"][0]["Entity ID"]
                if len(response["selected_rows"]) > 0
                and "Entity ID" in response["selected_rows"][0]
                else ""
            )
            selected_network = (
                response["selected_rows"][0]["Network ID"]
                if len(response["selected_rows"]) > 0
                else ""
            )

            if selected_network != "":
                if (
                    selected_network != sv.network_selected_community.value
                    or selected_entity != sv.network_selected_entity.value
                ):
                    sv.network_report.value = ""
                    sv.network_report_validation.value = {}

                sv.network_selected_entity.value = selected_entity
                sv.network_selected_community.value = selected_network
                c_nodes = sv.network_community_nodes.value[selected_network]
                N = functions.build_network_from_entities(
                    sv, sv.network_overall_graph.value, c_nodes
                )

                if selected_entity != "":
                    qualified_selected = f"{config.entity_label}{ATTRIBUTE_VALUE_SEPARATOR}{selected_entity}"
                    context = "Upload risk flags to see risk exposure report."
                    if len(sv.network_integrated_flags.value) > 0:
                        rdf = sv.network_integrated_flags.value.copy()
                        rdf = rdf[rdf["qualified_entity"].isin(c_nodes)]
                        rdf = (
                            rdf[["qualified_entity", "flag", "count"]]
                            .groupby(["qualified_entity", "flag"])
                            .sum()
                            .reset_index()
                        )
                        all_flagged = rdf[rdf["count"] > 0]["qualified_entity"].unique()
                        path_to_source = defaultdict(list)
                        target_flags = rdf[
                            rdf["qualified_entity"] == qualified_selected
                        ]["count"].sum()
                        net_flags = rdf["count"].sum() - target_flags
                        net_flagged = len(all_flagged)
                        if qualified_selected in all_flagged:
                            net_flagged -= 1
                        context = "##### Risk Exposure Report\n\n"
                        for flagged in all_flagged:
                            all_paths = [
                                list(x)
                                for x in nx.all_shortest_paths(
                                    N, flagged, qualified_selected
                                )
                            ]
                            for path in all_paths:
                                if len(path) > 1:
                                    chain = ""
                                    for j, step in enumerate(path):
                                        indent = "".join(["  "] * j)
                                        if "]\n" in step:
                                            step = "".join(step.split("]\n")[1:])
                                            step = "\n".join(step.split("; "))
                                        if config.entity_label in step:
                                            step_risks = rdf[
                                                rdf["qualified_entity"] == step
                                            ]["count"].sum()
                                            step = (
                                                step.split(ATTRIBUTE_VALUE_SEPARATOR)[1]
                                                + f" [linked to {step_risks} flags]"
                                            )
                                        else:
                                            step_entities = nx.degree(N, step)
                                            step = (
                                                f"\n{indent}".join(step.split("\n"))
                                                + f" [linked to {step_entities} entities]"
                                            )
                                        chain += indent + f"{step}\n"
                                        if j < len(path) - 1:
                                            chain += indent + "--->\n"
                                    source = chain.split("\n--->")[0]
                                    path = chain.split("\n--->")[1]
                                    path_to_source[path].append(source)
                        paths = len(path_to_source.keys())
                        context += f"The selected entity **{selected_entity}** has **{target_flags}** direct flags and is linked to **{net_flags}** indirect flags via **{paths}** paths from **{net_flagged}** related entities:\n\n"
                        if net_flagged == 0:
                            context = context[:-3] + "."
                        for ix, (path, sources) in enumerate(path_to_source.items()):
                            context += f"**Path {ix + 1}**\n\n```\n"
                            for source in sources:
                                context += f"{source}\n"
                            context += f"---> {path}\n```\n\n"
                        context = context.replace("**1** steps", "**1** step")
                        context = context.replace("**1** flags", "**1** flag")
                    sv.network_risk_exposure.value = context
                else:
                    sv.network_risk_exposure.value = ""

                full_links_df = pd.DataFrame(
                    list(N.edges()), columns=["source", "target"]
                )
                full_links_df["attribute"] = full_links_df["target"].apply(
                    lambda x: x.split(ATTRIBUTE_VALUE_SEPARATOR)[0]
                )
                N1 = functions.simplify_graph(N)
                merged_nodes_df = pd.DataFrame(
                    [(n, d["type"], d["flags"]) for n, d in N1.nodes(data=True)],
                    columns=["node", "type", "flags"],
                )
                merged_links_df = pd.DataFrame(
                    list(N1.edges()), columns=["source", "target"]
                )
                merged_links_df["attribute"] = merged_links_df["target"].apply(
                    lambda x: x.split(ATTRIBUTE_VALUE_SEPARATOR)[0]
                )
                c1, c2 = st.columns([2, 1])

                with c1:
                    gp = st.container()
                with c2:
                    graph_type = st.radio(
                        "Graph type", ["Full", "Simplified"], horizontal=True
                    )
                    st.markdown(sv.network_risk_exposure.value)
                with gp:
                    if graph_type == "Full":
                        if selected_entity != "":
                            gp.markdown(
                                f"##### Entity {selected_entity} in Network {selected_network} (full)"
                            )
                        else:
                            gp.markdown(f"##### Network {selected_network} (full)")

                        nodes, edges, g_config = functions.get_entity_graph(
                            N,
                            f"{config.entity_label}{ATTRIBUTE_VALUE_SEPARATOR}{selected_entity}",
                            full_links_df,
                            1000,
                            700,
                            [config.entity_label, *list(sv.network_node_types.value)],
                        )
                        agraph(nodes=nodes, edges=edges, config=g_config)  # type: ignore
                    elif graph_type == "Simplified":
                        if selected_entity != "":
                            gp.markdown(
                                f"##### Entity {selected_entity} in Network {selected_network} (simplified)"
                            )
                        else:
                            gp.markdown(
                                f"##### Network {selected_network} (simplified)"
                            )
                        nodes, edges, g_config = functions.get_entity_graph(
                            N1,
                            f"{config.entity_label}{ATTRIBUTE_VALUE_SEPARATOR}{selected_entity}",
                            merged_links_df,
                            1000,
                            700,
                            [config.entity_label, *list(sv.network_node_types.value)],
                        )
                        agraph(nodes=nodes, edges=edges, config=g_config)  # type: ignore
                sv.network_merged_links_df.value = merged_links_df
                sv.network_merged_nodes_df.value = merged_nodes_df
            else:
                st.warning(
                    "Select column headers to rank networks by that attribute. Use quickfilter or column filters to narrow down the list of networks. Select a network to continue."
                )

    with report_tab:
        if (
            sv.network_selected_entity.value == ""
            and sv.network_selected_community.value == ""
        ):
            st.warning("Select a network or entity to continue.")
        else:
            c1, c2 = st.columns([2, 3])
            with c1:
                selected_entity = sv.network_selected_entity.value

                variables = {
                    "entity_id": sv.network_selected_entity.value,
                    "network_id": sv.network_selected_community.value,
                    "max_flags": sv.network_max_entity_flags.value,
                    "mean_flags": sv.network_mean_flagged_flags.value,
                    "exposure": sv.network_risk_exposure.value,
                    "network_nodes": sv.network_merged_nodes_df.value.to_csv(
                        index=False
                    ),
                    "network_edges": sv.network_merged_links_df.value.to_csv(
                        index=False
                    ),
                }
                sv.network_system_prompt.value = prompts.list_prompts
                generate, messages, reset = ui_components.generative_ai_component(
                    sv.network_system_prompt, variables
                )
                if reset:
                    sv.network_system_prompt.value["user_prompt"] = prompts.user_prompt
                    st.rerun()
            with c2:
                if sv.network_selected_entity.value != "":
                    st.markdown(f"##### Selected entity: {selected_entity}")
                else:
                    st.markdown(
                        f"##### Selected network: {sv.network_selected_community.value}"
                    )
                report_placeholder = st.empty()
                gen_placeholder = st.empty()
                if selected_entity != selected_entity:
                    sv.network_report.value = ""
                    sv.network_report_validation.value = {}

                callback = ui_components.create_markdown_callback(report_placeholder)
                if generate:
                    result = ui_components.generate_text(messages, [callback])
                    sv.network_report.value = result

                    validation, messages_to_llm = ui_components.validate_ai_report(
                        messages, result
                    )
                    sv.network_report_validation.value = validation
                    sv.network_report_validation_messages.value = messages_to_llm
                    st.rerun()
                else:
                    if len(sv.network_report.value) == 0:
                        gen_placeholder.warning(
                            "Press the Generate button to create an AI report for the current network."
                        )

                report_data = sv.network_report.value
                report_placeholder.markdown(report_data)

                ui_components.report_download_ui(report_data, "network_report")

                ui_components.build_validation_ui(
                    sv.network_report_validation.value,
                    sv.network_report_validation_messages.value,
                    sv.network_report.value,
                    workflow,
                )
