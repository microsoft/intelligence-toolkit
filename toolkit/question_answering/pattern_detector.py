from collections import defaultdict
from toolkit.graph.graph_fusion_encoder_embedding import is_converging_pair

def detect_converging_pairs(period_to_cids, cid_to_concepts, node_to_period_to_pos):
    # print(f'Period to cids: {period_to_cids}')
    # print(f'CID to concepts: {cid_to_concepts}')
    chunk_to_converging_pairs = defaultdict(list)
    for period, chunks in period_to_cids.items():
        for chunk in chunks:
            concepts = cid_to_concepts[chunk]
            for cx, c1 in enumerate(concepts):
                for c2 in concepts[cx + 1 :]:
                    cp = is_converging_pair(period, c1, c2, node_to_period_to_pos)
                    if cp:
                        chunk_to_converging_pairs[chunk].append((c1, c2))
    return chunk_to_converging_pairs

def _get_ranks_from_counts(node_period_counts, node_edge_counts):
    node_period_ranks = defaultdict(lambda: defaultdict(int))
    edge_period_ranks = defaultdict(lambda: defaultdict(int))
    
    def assign_ranks(period_counts):
        sorted_periods = sorted(period_counts.items(), key=lambda x: x[1], reverse=True)
        ranks = {}
        current_rank = 1
        for i, (period, count) in enumerate(sorted_periods):
            if i > 0 and count < sorted_periods[i - 1][1]:
                current_rank = i + 1
            ranks[period] = current_rank
        return ranks

    for node, period_counts in node_period_counts.items():
        ranks = assign_ranks(period_counts)
        for period, rank in ranks.items():
            node_period_ranks[node][period] = rank

    for edge, period_counts in node_edge_counts.items():
        ranks = assign_ranks(period_counts)
        for period, rank in ranks.items():
            edge_period_ranks[edge][period] = rank

    return node_period_ranks, edge_period_ranks

def _get_props_from_counts(node_period_counts, node_edge_counts):
    node_period_props = defaultdict(lambda: defaultdict(float))
    edge_period_props = defaultdict(lambda: defaultdict(float))
    for node, period_counts in node_period_counts.items():
        mc = max(period_counts.values())
        for period, count in period_counts.items():
            node_period_props[node][period] = count / mc
    for edge, period_counts in node_edge_counts.items():
        mc = max(period_counts.values())
        for period, count in period_counts.items():
            edge_period_props[edge][period] = count / mc
    return node_period_props, edge_period_props


def explain_chunk_significance(period_to_cids, cid_to_converging_pairs, node_period_counts, edge_period_counts, pair_limit=5):
    node_period_ranks, edge_period_ranks = _get_ranks_from_counts(node_period_counts, edge_period_counts)
    node_period_props, edge_period_props = _get_props_from_counts(node_period_counts, edge_period_counts)
    cid_to_summary = {}
    for period, cids in period_to_cids.items():
        if period == "ALL":
            continue
        for cid in cids:
            converging_pairs = cid_to_converging_pairs[cid]
            if len(converging_pairs) == 0:
                continue
            summary = f'This text chunk is part of significant changes in how concepts co-occurred in the period {period}:\n'
            sorted_converging_pairs = sorted(converging_pairs, key=lambda x: (edge_period_props[x][period], edge_period_counts[x][period]), reverse=True)        
            for converging_pair in sorted_converging_pairs[:pair_limit]:
                edge_count = edge_period_counts[converging_pair][period]
                edge_rank = edge_period_ranks[converging_pair][period]
                edge_prop = edge_period_props[converging_pair][period]
                overall_edge_count = edge_period_counts[converging_pair]["ALL"]
                period_prop = edge_count / overall_edge_count
                summary += f'- \"{converging_pair[0]}\" and \"{converging_pair[1]}\" co-occurred {edge_count} times in {period} out of {overall_edge_count} times overall, representing {100*period_prop:.0f}% of all co-occurrences and {100*edge_prop:.0f}% of their maximum co-occurrence count in any period (ranking #{edge_rank} across all periods).\n'
            summary = summary.replace('1 times', '1 time')
            cid_to_summary[cid] = summary
    return cid_to_summary