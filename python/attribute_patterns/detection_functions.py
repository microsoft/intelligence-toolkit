# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .config import type_val_sep


def _calculate_cosine_distance(vec1: np.array, vec2: np.array):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def _calculate_euclidean_distance(vec1: np.array, vec2: np.array):
    return np.linalg.norm(vec1 - vec2)

def _create_centroid_dists(node_to_centroid):
    centroid_dists = {}
    sorted_nodes = sorted(node_to_centroid.keys())
    for ix, node1 in enumerate(sorted_nodes):
        vector1 = np.array(node_to_centroid[node1])
        for node2 in sorted_nodes[ix + 1:]:
            vector2 = np.array(node_to_centroid[node2])
            cosine = _calculate_cosine_distance(vector1, vector2)
            euclidean = _calculate_euclidean_distance(vector1, vector2)
            centroid_dists[(node1, node2)] = (cosine, euclidean)

    return centroid_dists

def _compute_node_pair_distances(period, period_embeddings, sorted_nodes, node_to_ix):
    distances = {}
    num_nodes = len(sorted_nodes)
    for ix in range(num_nodes):
        node1 = sorted_nodes[ix]
        for jx in range(ix + 1, num_nodes):
            node2 = sorted_nodes[jx]
            n1v = np.array(period_embeddings[period][node_to_ix[node1]])
            n2v = np.array(period_embeddings[period][node_to_ix[node2]])
            cosine = _calculate_cosine_distance(n1v, n2v)
            euclidean = _calculate_euclidean_distance(n1v, n2v)
            distances[(node1, node2)] = (cosine, euclidean)
    return distances

def create_period_shifts(node_to_centroid, period_embeddings, dynamic_df) -> dict:
    centroid_dists = _create_centroid_dists(node_to_centroid)
    period_shifts = {}
    sorted_nodes = sorted(node_to_centroid.keys())
    node_to_ix = {n: i for i, n in enumerate(sorted_nodes)}
    used_periods = sorted(dynamic_df['Period'].unique())
    
    for period in used_periods:
        period_shifts[period] = {}
        node_pair_distances = _compute_node_pair_distances(period, period_embeddings, sorted_nodes, node_to_ix)
        for node_pair, (cosine, euclidean) in node_pair_distances.items():
            centroid_cosine, centroid_euclidean = centroid_dists[node_pair]
            period_shifts[period][node_pair] = (centroid_cosine - cosine, centroid_euclidean - euclidean)
    return period_shifts

def _create_period_to_close_nodes(used_periods, period_shifts, sorted_nodes, min_pattern_count, rc):
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
                        if period_count >= min_pattern_count:
                            close_pairs += 1
                            period_to_close_nodes[period].append((node1, node2))
    return all_pairs, close_pairs, period_to_close_nodes

def create_close_node_rows(used_periods, period_shifts, sorted_nodes, min_pattern_count, rc):
    all_pairs, close_pairs, period_to_close_nodes = _create_period_to_close_nodes(used_periods, period_shifts, sorted_nodes, min_pattern_count, rc)

    close_node_rows = []
    for period, close_nodes in period_to_close_nodes.items():
        for node1, node2 in close_nodes:
            period_count = rc.count_records([period, node1, node2])
            mean_count, _, _ = rc.compute_period_mean_sd_max([node1, node2])
            if period_count >= min_pattern_count:
                count_factor = period_count / mean_count
                count_delta = period_count - mean_count
                cosine_shift, euclidean_shift = period_shifts[period][(node1, node2)]
                row = [period, node1, node2, period_count, mean_count, count_delta, count_factor, cosine_shift, euclidean_shift]
                close_node_rows.append(row)
    columns = ['period', 'node1', 'node2', 'period_count', 'mean_count', 'count_delta', 'count_factor', 'cosine_shift', 'euclidean_shift']
    close_node_df = pd.DataFrame(close_node_rows, columns=columns)
    return close_node_df, all_pairs, close_pairs

def create_period_to_patterns(used_periods, close_node_df, max_pattern_length, min_pattern_count, rc):
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
                    if len(candidate_pattern) <= max_pattern_length:
                        if candidate_pattern not in [p for p, _ in period_to_patterns[period]]:
                            candidate_pairs = combinations(candidate_pattern, 2)
                            exclude = False
                            for pair in candidate_pairs:
                                if pair not in period_pairs:
                                    exclude = True
                                    break
                            if not exclude:
                                pcount = rc.count_records([period] + list(candidate_pattern))
                                if pcount > min_pattern_count:
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