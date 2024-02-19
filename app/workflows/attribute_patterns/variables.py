from util.session_variable import SessionVariable
import pandas as pd

class SessionVariables:

    def __init__(self, prefix):
        self.attribute_max_rows_to_process = SessionVariable(0, prefix)
        self.attribute_uploaded_files = SessionVariable([], prefix)
        self.attribute_selected_file_name = SessionVariable('', prefix)
        self.attribute_dynamic_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_type_val_sep_in = SessionVariable('', prefix)
        self.attribute_type_val_sep_out = SessionVariable('=', prefix)
        self.attribute_laplacian = SessionVariable(True, prefix)
        self.attribute_diaga = SessionVariable(True, prefix)
        self.attribute_correlation = SessionVariable(True, prefix)
        self.attribute_missing_edge_prop = SessionVariable(0.1, prefix)
        self.attribute_umap_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_retain_target = SessionVariable(0.33, prefix)
        self.attribute_min_edge_weight = SessionVariable(0.001, prefix)
        self.attribute_min_primary_pattern_count = SessionVariable(10, prefix)
        self.attribute_min_secondary_pattern_count = SessionVariable(100, prefix)
        self.attribute_max_secondary_pattern_length = SessionVariable(5, prefix)
        self.attribute_primary_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_secondary_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_umap_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_node_to_centroid = SessionVariable({}, prefix)
        self.attribute_period_embeddings = SessionVariable([], prefix)
        self.attribute_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_sensitivity = SessionVariable(2.0, prefix)
        self.attribute_primary_pattern_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_record_counter = SessionVariable(None, prefix)
        self.attribute_close_pairs = SessionVariable(0, prefix)
        self.attribute_all_pairs = SessionVariable(0, prefix)
        self.attribute_secondary_pattern_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_overall_pattern_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_primary_threshold = SessionVariable(0.2, prefix)


