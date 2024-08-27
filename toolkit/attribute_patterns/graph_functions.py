# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

def convert_edge_df_to_graph(edge_df):
    G = nx.from_pandas_edgelist(edge_df, "source", "target", "weight")
    # get largest connected component
    lcc = max(nx.connected_components(G), key=len)
    return G, lcc


def create_edge_df_from_atts(all_atts, pdf, mi, min_edge_weight, missing_edge_prop):
    edge_counter = Counter()
    att_counter = Counter()
    for _, row in pdf.iterrows():
        atts = row["Full Attribute"]
        edges = [(a, b) if a < b else (b, a) for a, b in combinations(atts, 2)]
        edge_counter.update(edges)
        att_counter.update(atts)
    edge_df = pd.DataFrame.from_dict(edge_counter, orient="index").reset_index()
    edge_df.rename(columns={"index": "edge", 0: "count"}, inplace=True)
    edge_df["source"] = edge_df["edge"].apply(lambda x: x[0])
    edge_df["target"] = edge_df["edge"].apply(lambda x: x[1])
    att_count = sum(att_counter.values())
    edge_count = sum(edge_counter.values())

    if mi:
        edge_df["weight"] = edge_df.apply(
            lambda x: (edge_counter[x["edge"]] / edge_count)
            * np.log2(
                edge_counter[x["edge"]]
                / edge_count
                / (
                    (att_counter[x["source"]] / att_count)
                    * (att_counter[x["target"]] / att_count)
                )
            ),
            axis=1,
        )
    else:
        edge_df["weight"] = edge_df.apply(lambda x: edge_counter[x["edge"]], axis=1)

    max_w = edge_df["weight"].max()
    min_w = edge_df["weight"].min()
    edge_df["weight"] = edge_df["weight"].apply(
        lambda x: ((x - min_w) / (max_w - min_w)) * (1 - min_edge_weight)
        + min_edge_weight
    )

    null_rows = []
    missing_w = missing_edge_prop * min_edge_weight
    for ix, att1 in enumerate(all_atts):
        for att2 in all_atts[ix + 1 :]:
            edge = (att1, att2) if att1 < att2 else (att2, att1)
            if edge not in edge_counter:
                null_rows.append({"source": att1, "target": att2, "weight": missing_w})
    null_df = pd.DataFrame(null_rows)
    edge_df = pd.concat([edge_df, null_df])
    return edge_df.sort_values("weight", ascending=False)
