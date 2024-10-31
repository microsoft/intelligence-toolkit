# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import os

# ruff: noqa
import pandas as pd
import polars as pl
import streamlit as st
from app.util.download_pdf import add_download_pdf
from toolkit.detect_entity_networks.classes import FlagAggregatorType
import workflows.detect_entity_networks.functions as functions
import workflows.detect_entity_networks.variables as rn_variables
import app.util.example_outputs_ui as example_outputs_ui
from st_aggrid import (AgGrid, ColumnsAutoSizeMode, DataReturnMode,
                       GridOptionsBuilder, GridUpdateMode)
from streamlit_agraph import Edge, Node, agraph, Config
from util import ui_components
from util.session_variables import SessionVariables

from toolkit.detect_entity_networks import prompts
from toolkit.detect_entity_networks.config import (ENTITY_LABEL,
                                                   SIMILARITY_THRESHOLD_MAX,
                                                   SIMILARITY_THRESHOLD_MIN)
from toolkit.detect_entity_networks.explore_networks import (
    build_network_from_entities, get_entity_graph, simplify_entities_graph)
from toolkit.detect_entity_networks.exposure_report import build_exposure_report
from toolkit.detect_entity_networks.index_and_infer import (
    index_nodes,
)
from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


async def create(sv: rn_variables.SessionVariables, workflow=None):
    sv_home = SessionVariables("home")
    ui_components.check_ai_configuration()
    den = sv.workflow_object.value

    intro_tab, uploader_tab, process_tab, view_tab, report_tab, examples_tab = st.tabs(
        [
            "Detect Entity Networks workflow:",
            "Create data model",
            "Process data model",
            "Explore networks",
            "Generate AI network reports",
            "View example outputs"
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
        uploader_col, model_col = st.columns([3, 2])
        with uploader_col:
            _, selected_df, changed = ui_components.multi_csv_uploader(
                "Upload multiple CSVs",
                sv.network_uploaded_files,
                workflow + "_uploader",
                sv.network_max_rows_to_process,
            )
        with model_col:
            st.markdown("##### Map columns to model")
            if selected_df is None:
                st.warning("Upload and select a file to continue")
            else:
                options = ["", *selected_df.columns.to_numpy()]
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
                if link_type == "Entity-Attribute":
                    value_cols = st.multiselect(
                        "Attribute value column(s) to link on",
                        [o for o in options if o != entity_col],
                        help="The column(s) containing attribute values that would only be shared by closely related entities.",
                    )
                elif link_type == "Entity-Flag":
                    value_cols = st.multiselect(
                        "Flag value column(s)",
                        [o for o in options if o != entity_col],
                        help="The column(s) containing risk flags associated with entities.",
                    )
                    flag_agg = st.selectbox(
                        "Flag format",
                        ["Count", "Instance"],
                        help="How flags are represented: as individual instances or as aggregate counts in a flag value column.",
                    )
                elif link_type == "Entity-Group":
                    value_cols = st.multiselect(
                        "Group value column(s) to group on",
                        [o for o in options if o != entity_col],
                        help="The column(s) containing group values that are shared by groups of broadly related entities.",
                    )
                b1, b2 = st.columns([1, 1])
                with b1:
                    if st.button(
                        "Add links to model",
                        disabled=entity_col == ""
                        or len(value_cols) == 0
                        or link_type == "",
                    ):
                        with st.spinner("Adding links to model..."):
                            selected_df = pl.from_pandas(selected_df)
                            if link_type == "Entity-Attribute":
                                attribute_links = den.add_attribute_links(
                                    selected_df, entity_col, value_cols
                                )
                                sv.network_attribute_links.value = attribute_links
                                node_types = set()
                                for attribute_link in attribute_links:
                                    node_types.add(attribute_link[0][1])

                                sv.network_node_types.value = node_types
                                sv.network_overall_graph.value = den.graph

                            elif link_type == "Entity-Flag":
                                sv.network_flag_links.value = den.add_flag_links(
                                    selected_df,
                                    entity_col,
                                    value_cols,
                                    FlagAggregatorType(flag_agg),
                                )
                                sv.network_integrated_flags.value = den.integrated_flags
                                sv.network_max_entity_flags.value = den.max_entity_flags
                                sv.network_mean_flagged_flags.value = (
                                    den.mean_flagged_flags
                                )

                            elif link_type == "Entity-Group":
                                sv.network_group_links.value = den.add_group_links(
                                    selected_df,
                                    entity_col,
                                    value_cols,
                                )
                            sv.network_attributes_list.value = den.attributes_list
                        # sv.workflow_object.value = den
                with b2:
                    if st.button(
                        "Clear data model",
                        disabled=entity_col == ""
                        or len(value_cols) == 0
                        or link_type == "",
                    ):
                        den.clear_data_model()
                        sv.network_attribute_links.value = []
                        sv.network_flag_links.value = []
                        sv.network_group_links.value = []
                        sv.network_overall_graph.value = None
                        sv.network_entity_graph.value = None
                        sv.network_integrated_flags.value = pl.DataFrame()

            summary_data = den.get_model_summary_data()
            if (
                summary_data.entities > 0
                or summary_data.flags > 0
                or summary_data.groups > 0
            ):
                st.markdown("##### Data model summary")
                st.markdown(
                    f"*Number of entities*: {summary_data.entities}<br/>*Number of attributes*: {summary_data.attributes}<br/>*Number of links*: {summary_data.links}<br/>*Number of flags*: {summary_data.flags}<br/>*Number of groups*: {summary_data.groups}",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Add links to the model to continue.")

    with process_tab:
        index_col, part_col = st.columns([1, 1])
        with index_col:
            st.markdown("##### Match similar nodes (optional)")
            fuzzy_options = den.get_fuzzy_options()

            network_indexed_node_types = st.multiselect(
                "Select node types to fuzzy match",
                default=sv.network_indexed_node_types.value,
                options=fuzzy_options,
                help="Select the node types to embed into a multi-dimensional semantic space for fuzzy matching.",
            )

            if sv.network_overall_graph.value and network_indexed_node_types:
                total_embeddings = sum(
                    1
                    for _, data in sv.network_overall_graph.value.nodes(data=True)
                    if data["type"] in network_indexed_node_types
                )
                st.caption(f"Total of {total_embeddings} nodes to index")
            local_embedding = st.toggle(
                "Use local embeddings",
                sv.network_local_embedding_enabled.value,  # TODO: add option to api
                help="Use local embeddings to index nodes. If disabled, the model will use OpenAI embeddings.",
            )
            index = st.button(
                "Index nodes",
                disabled=len(network_indexed_node_types) == 0,
            )
            network_similarity_threshold = st.number_input(
                "Similarity threshold for fuzzy matching (max)",
                min_value=SIMILARITY_THRESHOLD_MIN,
                max_value=SIMILARITY_THRESHOLD_MAX,
                step=0.001,
                format="%f",
                value=sv.network_similarity_threshold.value,
                help="The maximum cosine similarity threshold for inferring links between nodes based on their embeddings. Higher values will infer more links, but may also infer more false positives.",
            )

            c_1, c_2 = st.columns([3, 1])
            with c_1:
                infer = st.button(
                    "Infer links",
                    disabled=len(sv.network_embedded_texts.value) == 0
                    or len(network_indexed_node_types) == 0,
                )
            with c_2:
                clear_inferring = st.button(
                    "Clear inferred links",
                    disabled=len(sv.network_inferred_links.value) == 0,
                )
            if clear_inferring:
                den.clear_inferred_links()
                sv.network_inferred_links.value = []
                st.rerun()

            if index:
                pb = st.progress(0, "Embedding text batches...")

                def on_embedding_batch_change(
                    current=0, total=0, message="In progress...."
                ):
                    pb.progress(
                        (current) / total,
                        f"{message} {current} of {total}",
                    )

                callback = ProgressBatchCallback()
                callback.on_batch_change = on_embedding_batch_change

                functions_embedder = functions.embedder(local_embedding)
                sv.network_indexed_node_types.value = network_indexed_node_types

                (
                    sv.network_embedded_texts.value,
                    sv.network_nearest_text_distances.value,
                    sv.network_nearest_text_indices.value,
                ) = await index_nodes(
                    network_indexed_node_types,
                    sv.network_overall_graph.value,
                    [callback],
                    functions_embedder,
                    None,
                    sv_home.save_cache.value,
                )
                den.embedded_texts = sv.network_embedded_texts.value
                den.nearest_text_distances = sv.network_nearest_text_distances.value
                den.nearest_text_indices = sv.network_nearest_text_indices.value

                pb.empty()
                st.rerun()

            if infer:
                pb = st.progress(0, "Inferring texts...")

                def on_inferring_batch_change(
                    current=0, total=0, message="In progress...."
                ):
                    pb.progress(
                        (current) / total,
                        f"{message} {current} of {total}",
                    )

                callback_infer = ProgressBatchCallback()
                callback_infer.on_batch_change = on_inferring_batch_change
                sv.network_similarity_threshold.value = network_similarity_threshold

                sv.network_inferred_links.value = den.infer_nodes(
                    network_similarity_threshold,
                    [callback_infer],
                )
                pb.empty()

            inferred_links_count = len(sv.network_inferred_links.value)
            if len(sv.network_embedded_texts.value) > 0:
                st.markdown(
                    f"*Number of nodes indexed*: {len(sv.network_embedded_texts.value)}"
                )
            if inferred_links_count > 0:
                st.markdown(f"*Number of links inferred*: {inferred_links_count}")
                inferred_df = den.inferred_nodes_df()
                st.dataframe(
                    inferred_df.to_pandas(), hide_index=True, use_container_width=True
                )
            else:
                st.markdown(f"*No inferred links*")

        with part_col:
            st.markdown("##### Constrain networks")
            attributes_df = pd.DataFrame(
                sv.network_attributes_list.value, columns=["Attribute"]
            )
            with st.expander("Remove attributes"):
                search = st.text_input("Search for attributes to remove", "")
                if search != "":
                    attributes_df = attributes_df[
                        attributes_df["Attribute"].str.contains(search, case=False)
                    ]

                selected_rows = ui_components.dataframe_with_selections(
                    attributes_df,
                    sv.network_additional_trimmed_attributes.value,
                    "Attribute",
                    "Remove",
                    key="remove_attribute_table",
                )
                sv.network_additional_trimmed_attributes.value = selected_rows[
                    "Attribute"
                ].tolist()
                den.additional_trimmed_attributes = (
                    sv.network_additional_trimmed_attributes.value
                )

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                network_max_attribute_degree = st.number_input(
                    "Maximum attribute degree",
                    min_value=1,
                    value=sv.network_max_attribute_degree.value,
                    help="The maximum number of entities that can share an attribute before it is removed from the network.",
                )
            with c2:
                network_max_network_entities = st.number_input(
                    "Max network entities",
                    min_value=1,
                    value=sv.network_max_network_entities.value,
                    help="Any network with entities >= max_network_entities will be isolated into a subnetwork",
                )
            with c3:
                network_supporting_attribute_types = st.multiselect(
                    "Supporting attribute types",
                    default=sv.network_supporting_attribute_types.value,
                    options=sorted(sv.network_node_types.value),
                    help="Attribute types that should not be used to detect networks (e.g., because of potential noise/unreliability) but which should be added back into detected networks for context.",
                )
            comm_count = 0

            identify = st.button("Identify networks")
            if identify:
                sv.network_max_attribute_degree.value = network_max_attribute_degree
                sv.network_max_network_entities.value = network_max_network_entities
                sv.network_supporting_attribute_types.value = (
                    network_supporting_attribute_types
                )

                with st.spinner("Identifying networks..."):
                    sv.network_table_index.value += 1
                    entity_records = den.identify(
                        sv.network_max_network_entities.value,
                        sv.network_max_attribute_degree.value,
                        sv.network_supporting_attribute_types.value,
                    )
                    # (trimmed_degrees, trimmed_nodes) = trim_nodeset(
                    #     sv.network_overall_graph.value,
                    #     sv.network_max_attribute_degree.value,
                    #     sv.network_additional_trimmed_attributes.value,
                    # )

                    sv.network_trimmed_attributes.value = (
                        den.trimmed_attributes.to_pandas()
                    )

                    # (
                    #     sv.network_community_nodes.value,
                    #     sv.network_entity_to_community_ix.value,
                    # ) = build_networks(
                    #     sv.network_overall_graph.value,
                    #     trimmed_nodes,
                    #     sv.network_inferred_links.value,
                    #     sv.network_supporting_attribute_types.value,
                    #     sv.network_max_network_entities.value,
                    # )

                sv.network_entity_df.value = den.get_entity_df().to_pandas()
                sv.network_table_index.value += 1
                sv.network_community_nodes.value = den.community_nodes
                st.rerun()

            comm_count = len(sv.network_community_nodes.value)

            if comm_count > 0:
                comm_sizes = den.get_community_sizes()
                max_comm_size = max(comm_sizes)
                trimmed_atts = len(sv.network_trimmed_attributes.value)
                st.markdown(
                    f"*Networks identified: {comm_count} ({len(comm_sizes)} with multiple entities, maximum {max_comm_size})*"
                )
                st.markdown(
                    f"*Attributes removed because of high degree*: {trimmed_atts}"
                )

                attributes_df = pd.DataFrame(
                    sv.network_attributes_list.value, columns=["Attribute"]
                )
                if trimmed_atts > 0:
                    st.dataframe(
                        sv.network_trimmed_attributes.value,
                        hide_index=True,
                        use_container_width=True,
                    )
    with view_tab:
        if len(sv.network_entity_df.value) == 0:
            st.warning("Detect networks to continue.")
        else:
            with st.expander("Select entity network", expanded=True):
                c1, c2 = st.columns([1, 5])
                with c1:
                    st.markdown("##### Options")
                    show_entities = st.checkbox(
                        "Show entities", value=sv.network_last_show_entities.value
                    )
                    show_groups = st.checkbox(
                        "Show groups", value=sv.network_last_show_groups.value
                    )
                    flag_paths = st.checkbox(
                        "Show flag paths",
                        value=sv.network_flag_paths.value,
                        disabled=not show_entities
                    )
                    graph_type = st.radio(
                        "Graph type", ["Full", "Simplified"], horizontal=False
                    )
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
                        show_df = den.get_grouped_df().to_pandas()
                    last_df = show_df.copy()
                    if not show_entities:
                        last_df = (
                            last_df.drop(columns=["entity_id", "entity_flags"])
                            .drop_duplicates()
                            .reset_index(drop=True)
                        )
                    dl_button.download_button(
                        "Download network CSV",
                        last_df.to_csv(index=False),
                        "network_data.csv",
                        "download_network_csv",
                    )
                with c2:
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
                        height=250,
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
                response["selected_rows"][0]["entity_id"]
                if len(response["selected_rows"]) > 0
                and "entity_id" in response["selected_rows"][0]
                else ""
            )
            selected_network = (
                response["selected_rows"][0]["network_id"]
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
                trimmed_attr = {t[0] for t in sv.network_trimmed_attributes.value}
                network_entities_graph = build_network_from_entities(
                    sv.network_overall_graph.value,
                    sv.network_entity_to_community_ix.value,
                    sv.network_integrated_flags.value,
                    trimmed_attr,
                    sv.network_inferred_links.value,
                    c_nodes,
                )
                if selected_entity != "":
                    context = "Upload risk flags to see risk exposure report."
                    if len(sv.network_integrated_flags.value) > 0:
                        
                        context = build_exposure_report(
                            sv.network_integrated_flags.value,
                            selected_entity,
                            c_nodes,
                            network_entities_graph,
                            sv.network_inferred_links.value,
                        )
                        sv.network_risk_exposure.value = context
                else:
                    sv.network_risk_exposure.value = ""

                full_links_df = pd.DataFrame(
                    list(network_entities_graph.edges()), columns=["source", "target"]
                )
                full_links_df["attribute"] = full_links_df["target"].apply(
                    lambda x: x.split(ATTRIBUTE_VALUE_SEPARATOR)[0]
                )
                network_entities_simplified_graph = simplify_entities_graph(
                    network_entities_graph
                )
                merged_nodes_df = pd.DataFrame(
                    [
                        (n, d["type"], d["flags"])
                        for n, d in network_entities_simplified_graph.nodes(data=True)
                    ],
                    columns=["node", "type", "flags"],
                )
                merged_links_df = pd.DataFrame(
                    list(network_entities_simplified_graph.edges()),
                    columns=["source", "target"],
                )
                merged_links_df["attribute"] = merged_links_df["target"].apply(
                    lambda x: x.split(ATTRIBUTE_VALUE_SEPARATOR)[0]
                )

                if graph_type == "Full":
                    render_graph = network_entities_graph
                else:
                    render_graph = network_entities_simplified_graph

                if not show_entities or not flag_paths:
                    network_vis = st.container()
                else:
                    network_pane, path_pane = st.columns([2, 1])
                    with network_pane:
                        network_vis = st.container()
                   
                with network_vis:
                    entity_selected = (
                        f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}{selected_entity}"
                    )
                    attribute_types = [
                        ENTITY_LABEL,
                        *list(sv.network_node_types.value),
                    ]

                    if show_entities:
                        nodes, edges = get_entity_graph(
                            render_graph,
                            entity_selected,
                            attribute_types,
                        )

                        nodes_agraph = [Node(**node) for node in nodes]
                        edges_agraph = [Edge(**edge) for edge in edges]
                    else:
                        nodes, edges = get_entity_graph(
                            render_graph,
                            entity_selected,
                            attribute_types,
                        )

                    if selected_entity != "":
                        network_vis.markdown(
                            f"##### Entity {selected_entity} in Network {selected_network} ({graph_type.lower()})"
                        )
                    else:
                        network_vis.markdown(
                            f"##### Network {selected_network} ({graph_type.lower()})"
                        )

                    AgGrid(
                        last_df[last_df["network_id"] == selected_network]
                        if selected_entity == ""
                        else last_df[last_df["entity_id"] == selected_entity],
                        key=f"report_grid_{sv.network_table_index.value}_selected",
                        height=90,
                        enable_enterprise_modules=False,
                        fit_columns_on_grid_load=False,
                        header_checkbox_selection_filtered_only=False,
                        use_checkbox=False,
                        update_mode=GridUpdateMode.NO_UPDATE,
                        reload_data=True,
                        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                    )
                    nodes_agraph = [Node(**node) for node in nodes]
                    edges_agraph = [Edge(**edge) for edge in edges]

                    default_config = rn_variables.agraph_config
                    config = Config(
                        height=default_config['height'],
                        directed=default_config['directed'],
                        physics=default_config['physics'],
                        hierarchical=default_config['hierarchical'],
                        width = 1200 if show_entities and flag_paths else 1400
                    )
                    agraph(
                        nodes=nodes_agraph,
                        edges=edges_agraph,
                        config=config,
                    )
                if show_entities and flag_paths:
                    with path_pane:
                        st.markdown(sv.network_risk_exposure.value)
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
                    connection_bar = st.progress(10, text="Connecting to AI...")

                    def empty_connection_bar(_):
                        connection_bar.empty()

                    callback_bar = ui_components.remove_connection_bar(
                        empty_connection_bar
                    )

                    try:
                        result = ui_components.generate_text(
                            messages, [callback, callback_bar]
                        )

                        sv.network_report.value = result

                        # validation, messages_to_llm = ui_components.validate_ai_report(
                        #     messages, result
                        # )
                        # sv.network_report_validation.value = validation
                        # sv.network_report_validation_messages.value = messages_to_llm
                        st.rerun()
                    except Exception as _e:
                        empty_connection_bar(_e)
                        raise
                else:
                    if len(sv.network_report.value) == 0:
                        gen_placeholder.warning(
                            "Press the Generate button to create an AI report for the current network."
                        )

                report_data = sv.network_report.value
                report_placeholder.markdown(report_data)

                ui_components.report_download_ui(report_data, "network_report")

                # ui_components.build_validation_ui(
                #     sv.network_report_validation.value,
                #     sv.network_report_validation_messages.value,
                #     sv.network_report.value,
                #     workflow,
                # )
    with examples_tab:
        example_outputs_ui.create_example_outputs_ui(examples_tab, workflow)
