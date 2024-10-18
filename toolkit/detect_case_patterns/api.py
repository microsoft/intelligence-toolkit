# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import toolkit.detect_case_patterns.model as model
import toolkit.graph.graph_fusion_encoder_embedding as gfee
import toolkit.AI.utils as utils
from toolkit.AI.client import OpenAIClient
import toolkit.detect_case_patterns.prompts as prompts
import pandas as pd
import altair as alt

class DetectCasePatterns:
    def __init__(self):
        self.dynamic_graph_df = pd.DataFrame()
        self.detect_patterns_df = pd.DataFrame()
        self.patterns_df = pd.DataFrame()
        self.period_to_graph = {}
        self.node_to_label = {}
        self.period_col = ""
        self.type_val_sep = ":"

    def set_ai_configuration(self, ai_configuration):
        self.ai_configuration = ai_configuration

    def generate_graph_model(
        self,
        df,
        period_col,
        type_val_sep=":",
        min_edge_weight=0.001,
        missing_edge_prop=0.1,
    ):
        self.input_df = df
        self.period_col = period_col
        self.type_val_sep = type_val_sep
        self.dynamic_graph_df = model.generate_graph_model(df, period_col, type_val_sep)
        self._prepare_graph(
            min_edge_weight,
            missing_edge_prop,    
        )

    def _prepare_graph(
        self,
        min_edge_weight,
        missing_edge_prop,
    ):
        self.detect_patterns_df, self.period_to_graph = model.prepare_graph(
            self.dynamic_graph_df,
            min_edge_weight,
            missing_edge_prop
        )

    def generate_embedding_model(self):
        node_to_label_str = dict(
            self.dynamic_graph_df[
                ["Full Attribute", "Attribute Type"]
            ].values
        )
        # convert string labels to int labels
        sorted_labels = sorted(set(node_to_label_str.values()))
        label_to_code = {v: i for i, v in enumerate(sorted_labels)}
        self.node_to_label = {
            k: {0: label_to_code[v]} for k, v in node_to_label_str.items()
        }
        self.node_to_period_to_pos, self.node_to_period_to_shift = (
            gfee.generate_graph_fusion_encoder_embedding(
                self.period_to_graph,
                self.node_to_label,
                correlation=True,
                diaga=True,
                laplacian=True,
                max_level=0
            )
        )

    def detect_patterns(
        self,
        min_pattern_count,
        max_pattern_length
    ):
        self.min_pattern_count = min_pattern_count
        self.max_pattern_length = max_pattern_length
        (
            self.patterns_df,
            self.close_pairs,
            self.all_pairs
        ) = model.detect_patterns(
            self.node_to_period_to_pos,
            self.dynamic_graph_df,
            self.type_val_sep,
            self.min_pattern_count,
            self.max_pattern_length
        )

    def create_time_series_df(self):
        self.time_series_df = model.create_time_series_df(self.dynamic_graph_df, self.patterns_df)

    def compute_attribute_counts(
        self,
        selected_pattern,
        selected_pattern_period
    ):
        return model.compute_attribute_counts(
            df=self.input_df,
            pattern=selected_pattern,
            period_col=self.period_col,
            period=selected_pattern_period,
            type_val_sep=self.type_val_sep
        )
    
    def create_time_series_chart(
        self,
        selected_pattern,
        selected_pattern_period,
    ):
        selected_pattern_df = self.time_series_df[
            (self.time_series_df["pattern"] == selected_pattern)
        ]
        title = 'Pattern: ' + selected_pattern + ' (' + selected_pattern_period + ')'
        count_ct = (
            alt.Chart(selected_pattern_df)
            .mark_line()
            .encode(x="period:O", y="count:Q", color=alt.ColorValue("blue"))
            .properties(title=title,
                        height=220, width=600)
        )
        return count_ct
    
    def explain_pattern(
        self,
        selected_pattern,
        selected_pattern_period,
        attribute_counts=None,
        ai_instructions=prompts.user_prompt,
        callbacks=[]
    ):
        if attribute_counts is None:
            attribute_counts = self.compute_attribute_counts(
                selected_pattern,
                selected_pattern_period
            )
        variables = {
            "pattern": selected_pattern,
            "period": selected_pattern_period,
            "time_series": self.time_series_df[self.time_series_df["pattern"]==selected_pattern].to_csv(
                index=False
            ),
            "attribute_counts": attribute_counts.to_csv(
                index=False
            ),
        }
        messages = utils.generate_messages(
            ai_instructions,
            prompts.list_prompts['report_prompt'],
            variables,
            prompts.list_prompts['safety_prompt']
        )
        return OpenAIClient(self.ai_configuration).generate_chat(messages, callbacks=callbacks)