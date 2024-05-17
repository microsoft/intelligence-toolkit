# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from .config import min_edge_weight, missing_edge_prop, type_val_sep


def convert_edge_df_to_graph(edge_df):
    G = nx.from_pandas_edgelist(edge_df, 'source', 'target', 'weight')
    # get largest connected component
    lcc = max(nx.connected_components(G), key=len)
    return G, lcc

def create_edge_df_from_atts(all_atts, pdf, mi):
    edge_counter = Counter()
    att_counter = Counter()
    for ix, row in pdf.iterrows():
        atts = row['Full Attribute']
        edges = [(a, b) if a < b else (b, a) for a, b in combinations(atts, 2)]
        edge_counter.update(edges)
        att_counter.update(atts)
    edge_df = pd.DataFrame.from_dict(edge_counter, orient='index').reset_index()
    edge_df.rename(columns={'index' : 'edge', 0 : 'count'}, inplace=True)
    edge_df['source'] = edge_df['edge'].apply(lambda x: x[0])
    edge_df['target'] = edge_df['edge'].apply(lambda x: x[1])
    att_count = sum(att_counter.values())
    edge_count = sum(edge_counter.values())

    if mi:
        edge_df['weight'] = edge_df.apply(lambda x: (edge_counter[x['edge']] / edge_count)  * np.log2(edge_counter[x['edge']] / edge_count / ((att_counter[x['source']] / att_count) * (att_counter[x['target']] / att_count))), axis=1)
    else:
        edge_df['weight'] = edge_df.apply(lambda x: edge_counter[x['edge']], axis=1)

    max_w = edge_df['weight'].max()
    min_w = edge_df['weight'].min()
    min_t = min_edge_weight
    edge_df['weight'] = edge_df['weight'].apply(lambda x: ((x - min_w) / (max_w - min_w)) * (1 - min_t) + min_t)

    null_rows = []
    missing_w = missing_edge_prop * min_edge_weight
    # if sv.attribute_min_edge_weight.value > 0:
    for ix, att1 in enumerate(all_atts):
        for att2 in all_atts[ix + 1:]:
            edge = (att1, att2) if att1 < att2 else (att2, att1)
            if edge not in edge_counter.keys():
                null_rows.append({'source' : att1, 'target' : att2, 'weight' : missing_w})
    null_df = pd.DataFrame(null_rows)
    edge_df = pd.concat([edge_df, null_df])
    edge_df = edge_df.sort_values('weight', ascending=False)
    return edge_df

def calculate_cosine_distance(vec1: np.array, vec2: np.array):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_euclidean_distance(vec1: np.array, vec2: np.array):
    return np.linalg.norm(vec1 - vec2)

def create_centroid_dists(node_to_centroid):
    centroid_dists = {}
    sorted_nodes = sorted(node_to_centroid.keys())
    for ix, node1 in enumerate(sorted_nodes):
        vector1 = np.array(node_to_centroid[node1])
        for node2 in sorted_nodes[ix + 1:]:
            vector2 = np.array(node_to_centroid[node2])
            cosine = calculate_cosine_distance(vector1, vector2)
            euclidean = calculate_euclidean_distance(vector1, vector2)
            centroid_dists[(node1, node2)] = (cosine, euclidean)

    return centroid_dists

def compute_node_pair_distances(period, attribute_period_embeddings, sorted_nodes, node_to_ix):
    distances = {}
    num_nodes = len(sorted_nodes)
    for ix in range(num_nodes):
        node1 = sorted_nodes[ix]
        for jx in range(ix + 1, num_nodes):
            node2 = sorted_nodes[jx]
            n1v = np.array(attribute_period_embeddings[period][node_to_ix[node1]])
            n2v = np.array(attribute_period_embeddings[period][node_to_ix[node2]])
            cosine = calculate_cosine_distance(n1v, n2v)
            euclidean = calculate_euclidean_distance(n1v, n2v)
            distances[(node1, node2)] = (cosine, euclidean)
    return distances

def create_period_shifts(node_to_centroid, attribute_period_embeddings, attribute_dynamic_df) -> dict:
    centroid_dists = create_centroid_dists(node_to_centroid)
    period_shifts = {}
    sorted_nodes = sorted(node_to_centroid.keys())
    node_to_ix = {n: i for i, n in enumerate(sorted_nodes)}
    used_periods = sorted(attribute_dynamic_df['Period'].unique())
    
    for period in used_periods:
        period_shifts[period] = {}
        node_pair_distances = compute_node_pair_distances(period, attribute_period_embeddings, sorted_nodes, node_to_ix)
        for node_pair, (cosine, euclidean) in node_pair_distances.items():
            centroid_cosine, centroid_euclidean = centroid_dists[node_pair]
            period_shifts[period][node_pair] = (centroid_cosine - cosine, centroid_euclidean - euclidean)
    return period_shifts

def create_period_to_close_nodes(used_periods, period_shifts, sorted_nodes, attribute_min_pattern_count, rc):
    period_to_close_nodes = {}
    all_pairs = 0
    close_pairs = 0
    for period in used_periods:
        period_to_close_nodes[period] = []
        for ix, node1 in enumerate(sorted_nodes):
            for node2 in sorted_nodes[ix + 1:]:
                all_pairs += 1
                n1a = node1.split(type_val_sep)[0]
                n2a = node2.split(type_val_sep)[0]
                if n1a != n2a:
                    cosine_shift, euclidean_shift = period_shifts[period][(node1, node2)]
                    if cosine_shift > 0 or euclidean_shift > 0:
                        period_count = rc.count_records([period, node1, node2])
                        if period_count >= attribute_min_pattern_count:
                            close_pairs += 1
                            period_to_close_nodes[period].append((node1, node2))
    return all_pairs, close_pairs, period_to_close_nodes

def create_close_node_rows(used_periods, period_shifts, sorted_nodes, attribute_min_pattern_count, rc):
    all_pairs, close_pairs, period_to_close_nodes = create_period_to_close_nodes(used_periods, period_shifts, sorted_nodes, attribute_min_pattern_count, rc)

    close_node_rows = []
    for period, close_nodes in period_to_close_nodes.items():
        for node1, node2 in close_nodes:
            period_count = rc.count_records([period, node1, node2])
            mean_count, _, _ = rc.compute_period_mean_sd_max([node1, node2])
            if period_count >= attribute_min_pattern_count:
                count_factor = period_count / mean_count
                count_delta = period_count - mean_count
                cosine_shift, euclidean_shift = period_shifts[period][(node1, node2)]
                row = [period, node1, node2, period_count, mean_count, count_delta, count_factor, cosine_shift, euclidean_shift]
                close_node_rows.append(row)
    columns = ['period', 'node1', 'node2', 'period_count', 'mean_count', 'count_delta', 'count_factor', 'cosine_shift', 'euclidean_shift']
    close_node_df = pd.DataFrame(close_node_rows, columns=columns)
    return close_node_df, all_pairs, close_pairs

def create_period_to_patterns(used_periods, close_node_df, attribute_max_pattern_length, attribute_min_pattern_count, rc):
    period_to_patterns = {}
    pattern_to_periods = defaultdict(set)
    for period in used_periods:
        period_pair_counts = close_node_df[close_node_df['period'] == period][['node1', 'node2', 'period_count']].values.tolist()
        period_to_patterns[period] = [([], 0)]
        period_pairs = [tuple(sorted([a, b])) for a, b, c in period_pair_counts]
        print(F'Period {period}')
        for (pattern, _) in period_to_patterns[period]:
            for (a, b) in period_pairs:
                a_in_pattern = a in pattern
                b_in_pattern = b in pattern
                if len(pattern) > 0 and ((a_in_pattern and b_in_pattern) or (not a_in_pattern and not b_in_pattern)):
                    continue
                candidate = None
                if (a_in_pattern and not b_in_pattern):
                    candidate = [b]
                elif (b_in_pattern and not a_in_pattern):
                    candidate = [a]
                elif (not a_in_pattern and not b_in_pattern):
                    candidate = [a, b]

                if candidate is not None:
                    candidate_pattern = sorted(pattern + candidate)
                    if len(candidate_pattern) <= attribute_max_pattern_length:
                        if candidate_pattern not in [p for p, _ in period_to_patterns[period]]:
                            candidate_pairs = combinations(candidate_pattern, 2)
                            exclude = False
                            for pair in candidate_pairs:
                                if pair not in period_pairs:
                                    exclude = True
                                    break
                            if not exclude:
                                pcount = rc.count_records([period] + list(candidate_pattern))
                                if pcount > attribute_min_pattern_count:
                                    period_to_patterns[period].append((candidate_pattern, pcount))
                                    pattern_to_periods[tuple(candidate_pattern)].add(period)
    return period_to_patterns

def create_pattern_rows(period_to_patterns, rc):
    pattern_rows = []
    for period, patterns in period_to_patterns.items():
        for (pattern, count) in patterns:
            if count > 0:
                mean, sd, _ = rc.compute_period_mean_sd_max(pattern)
                score = (count - mean) / sd
                if score >= 0:
                    row = [period, ' & '.join(pattern), len(pattern), count, round(mean, 0), round(score, 2)]
                    pattern_rows.append(row)
    return pattern_rows