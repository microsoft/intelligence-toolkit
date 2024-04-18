from util.session_variable import SessionVariable
import pandas as pd
import polars as pl
from collections import defaultdict
import streamlit as st

import workflows.risk_networks.prompts as prompts

class SessionVariables:

    def __init__(self, prefix):
        self.create_session(prefix)
       
    def create_session(self, prefix):
        self.network_max_rows_to_process = SessionVariable(0, prefix)
        self.network_uploaded_files = SessionVariable([], prefix)
        self.network_selected_file_name = SessionVariable('', prefix)
        self.network_entity_links = SessionVariable([], prefix)
        self.network_directed_entity_links = SessionVariable([], prefix)
        self.network_attribute_links = SessionVariable([], prefix)
        self.network_flag_links = SessionVariable([], prefix)
        self.network_group_links = SessionVariable([], prefix)
        self.network_components = SessionVariable([], prefix)
        self.network_component_to_nodes = SessionVariable(set(), prefix)
        self.network_component_to_communities = SessionVariable({}, prefix)
        self.network_community_nodes = SessionVariable([], prefix)
        self.network_overall_graph = SessionVariable(None, prefix)
        self.network_entity_graph = SessionVariable(None, prefix)
        self.network_merged_graph = SessionVariable(None, prefix)
        self.network_max_network_size = SessionVariable(50, prefix)
        self.network_max_attribute_degree = SessionVariable(10, prefix)
        self.network_trimmed_attributes = SessionVariable([], prefix)
        self.network_similarity_threshold = SessionVariable(0.05, prefix)
        self.network_inferred_links = SessionVariable(defaultdict(set), prefix)
        self.network_embedded_texts = SessionVariable([], prefix)
        self.network_nearest_text_indices = SessionVariable([], prefix)
        self.network_nearest_text_distances = SessionVariable([], prefix)
        self.network_node_types = SessionVariable(set(), prefix)
        self.network_indexed_node_types = SessionVariable([], prefix)
        self.network_flag_types = SessionVariable(set(), prefix)
        self.network_integrated_flags = SessionVariable(pl.DataFrame(), prefix)
        self.network_community_df = SessionVariable(pl.DataFrame(), prefix)
        self.network_table_index = SessionVariable(0, prefix)
        self.network_supporting_attribute_types = SessionVariable([], prefix)
        self.network_flagged_nodes = SessionVariable([], prefix)
        self.network_entity_to_community_ix = SessionVariable({}, prefix)
        self.network_entity_df = SessionVariable(pd.DataFrame(), prefix)
        self.network_table_index = SessionVariable(0, prefix)
        self.network_selected_entity = SessionVariable('', prefix)
        self.network_selected_community = SessionVariable('', prefix)
        self.network_attributes_list = SessionVariable([], prefix)
        self.network_additional_trimmed_attributes = SessionVariable([], prefix)
        self.network_system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.network_report = SessionVariable('', prefix)
        self.network_report_validation_messages = SessionVariable('', prefix)
        self.network_report_validation = SessionVariable({}, prefix)
        self.network_merged_links_df = SessionVariable([], prefix)
        self.network_merged_nodes_df = SessionVariable([], prefix)
        self.network_group_types = SessionVariable(set(), prefix)
        self.network_max_entity_flags = SessionVariable(0, prefix)
        self.network_mean_flagged_flags = SessionVariable(0, prefix)
        self.network_risk_exposure = SessionVariable('', prefix)
        self.network_last_show_entities = SessionVariable(False, prefix)
        self.network_last_show_groups = SessionVariable(False, prefix)
        self.network_attributes_protected = SessionVariable([], prefix)
        self.network_entities_renamed = SessionVariable([], prefix)
        self.network_attributes_renamed = SessionVariable([], prefix)

    def reset_workflow(self, prefix):
        for key in st.session_state.keys():
            if key.startswith(prefix):
                del st.session_state[key]
        self.create_session(prefix)