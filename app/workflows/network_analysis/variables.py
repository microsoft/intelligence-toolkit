from util.session_variable import SessionVariable
import pandas as pd
import polars as pl
from collections import defaultdict

class SessionVariables:

    def __init__(self, prefix):
        self.network_max_rows_to_process = SessionVariable(0, prefix)
        self.network_uploaded_files = SessionVariable([], prefix)
        self.network_selected_file_name = SessionVariable('', prefix)
        self.network_entity_links = SessionVariable([], prefix)
        self.network_directed_entity_links = SessionVariable([], prefix)
        self.network_attribute_links = SessionVariable([], prefix)
        self.network_flag_links = SessionVariable([], prefix)
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
