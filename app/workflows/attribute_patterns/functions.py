import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter, defaultdict

from itertools import combinations
from util.SparseGraphEncoder import GraphEncoderEmbed

import workflows.attribute_patterns.classes as classes
import workflows.attribute_patterns.config as config
import util.df_functions

def create_time_series_df(rc, pattern_df):
    rows = []
    for ix, row in pattern_df.iterrows():
        rows.extend(rc.create_time_series_rows(row['pattern'].split(' & ')))
    columns = ['period', 'pattern', 'count']
    ts_df = pd.DataFrame(rows, columns=columns)
    return ts_df

def convert_edge_df_to_graph(edge_df):
    G = nx.from_pandas_edgelist(edge_df, 'source', 'target', 'weight')
    # get largest connected component
    lcc = max(nx.connected_components(G), key=len)
    return G, lcc

def create_edge_df_from_atts(sv, all_atts, pdf, mi):
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
    min_t = config.min_edge_weight
    edge_df['weight'] = edge_df['weight'].apply(lambda x: ((x - min_w) / (max_w - min_w)) * (1 - min_t) + min_t)

    null_rows = []
    missing_w = config.missing_edge_prop * config.min_edge_weight
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
    
def prepare_graph(sv, mi=False):
    G0 = nx.Graph()
    edge_df = pd.DataFrame()
    time_to_graph = {}
    dynamic_lcc = set()
    pdf = sv.attribute_dynamic_df.value.copy()
    atts = sorted(pdf['Full Attribute'].unique())
    pdf['Grouping ID'] = pdf['Subject ID'] + '@' + pdf['Period']
    # adf = pdf[['Grouping ID', 'Full Attribute']].groupby('Grouping ID').agg(list).reset_index()
    # edge_df = create_edge_df_from_atts(sv, atts, adf, mi)
    
    
    periods = sorted(pdf['Period'].unique())

    # G0, full_lcc = convert_edge_df_to_graph(edge_df)
    for ix, period in enumerate(periods):
        print(period)
        tdf = pdf.copy()
        tdf = tdf[tdf['Period'] == period]
        tdf['Grouping ID'] = tdf['Subject ID'] + '@' + tdf['Period']
        tdf = tdf[['Grouping ID', 'Full Attribute']].groupby('Grouping ID').agg(list).reset_index()
        dedge_df = create_edge_df_from_atts(sv, atts, tdf, mi)
        G, lcc = convert_edge_df_to_graph(dedge_df)
        if ix == 0:
            dynamic_lcc.update(lcc)
        else:
            dynamic_lcc.intersection_update(lcc)
        time_to_graph[period] = G
    pdf['Grouping ID'] = pdf['Subject ID'] + '@' + pdf['Period']
    return pdf, time_to_graph

def generate_embedding(sv, df, time_to_graph):
    period_embeddings = {}
    node_list = sorted(df['Full Attribute'].unique().tolist())
    sorted_att_types = sorted(df['Attribute Type'].unique())
    node_to_ix = {n : i for i, n in enumerate(node_list)}
    node_to_label = {n : sorted_att_types.index(n.split(config.type_val_sep)[0]) for n in node_list}
    embedding_dfs = []
    for period, graph in time_to_graph.items():
        edge_list = []
        for s, t, w in graph.edges(data='weight'):
            if s in node_list and t in node_list:
                edge_list.append([node_to_ix[s], node_to_ix[t], w])

        num_nodes = len(node_list)
        # get root/leaf partitions as labels
        labels = [0] * num_nodes
        for node in node_list:
            labels[node_list.index(node)] = node_to_label[node]
        Y = np.array(labels).reshape((num_nodes, 1))
        Z, W = GraphEncoderEmbed().run(edge_list, Y, num_nodes, EdgeList = True, Laplacian = config.laplacian, DiagA = config.diaga, Correlation = config.correlation)
        period_embeddings[period] = Z.toarray()

        # serialize embedding
        embedding_rows = []
        for node_id in range(len(period_embeddings[period])):
            row = [f'{node_id}=={period}', period, period_embeddings[period][node_id].tolist()]
            embedding_rows.append(row)
        columns = ['dynamic_node_id', 'period', 'embedding']
        embedding_df = pd.DataFrame(embedding_rows, columns=columns)
        embedding_dfs.append(embedding_df)
    overall_embedding_df = pd.concat(embedding_dfs)

    # for each node, find the centroid of its embeddings
    node_to_centroid = {}
    for node in node_list:
        node_to_centroid[node] = np.mean([period_embeddings[period][node_to_ix[node]] for period in time_to_graph.keys()], axis=0).tolist()
    
    # add centroid embeddings to overall embedding df
    centroid_rows = []
    for node_id in range(len(node_to_centroid)):
        row = [f'{node_id}==centroid', 'centroid', node_to_centroid[node_list[node_id]]]
        centroid_rows.append(row)
    columns = ['dynamic_node_id', 'period', 'embedding']
    # centroid_df = pd.DataFrame(centroid_rows, columns=columns)
    # overall_embedding_df = pd.concat([overall_embedding_df, centroid_df])
    overall_embedding_df['Full Attribute'] = overall_embedding_df['dynamic_node_id'].apply(lambda x: node_list[int(x.split(config.att_val_sep)[0])])
    return overall_embedding_df, node_to_centroid, period_embeddings


def detect_patterns(sv):
    df = sv.attribute_final_df.value
    node_to_centroid = sv.attribute_node_to_centroid.value
    period_embeddings = sv.attribute_period_embeddings.value
    node_list = sorted(node_to_centroid.keys())
    node_to_ix = {n : i for i, n in enumerate(node_list)}
    centroid_dists = {}
    sorted_nodes = sorted(node_to_centroid.keys())

    for ix, node1 in enumerate(sorted_nodes):
        for node2 in sorted_nodes[ix + 1:]:
            cosine = 1 - np.dot(np.array(node_to_centroid[node1]), np.array(node_to_centroid[node2])) / (np.linalg.norm(np.array(node_to_centroid[node1])) * np.linalg.norm(np.array(node_to_centroid[node2])))
            euclidean = np.linalg.norm(np.array(node_to_centroid[node1]) - np.array(node_to_centroid[node2]))
            centroid_dists[(node1, node2)] = (cosine, euclidean)

    period_shifts = {}
    used_periods = sorted(sv.attribute_dynamic_df.value['Period'].unique())
    for period in used_periods:
        period_shifts[period] = {}
        for ix, node1 in enumerate(sorted_nodes):
            for node2 in sorted_nodes[ix + 1:]:
                n1v = period_embeddings[period][node_to_ix[node1]]
                n2v = period_embeddings[period][node_to_ix[node2]]
                cosine = 1 - np.dot(np.array(n1v), np.array(n2v)) / (np.linalg.norm(np.array(n1v)) * np.linalg.norm(np.array(n2v)))
                euclidean = np.linalg.norm(np.array(n1v) - np.array(n2v))
                centroid_cosine, centroid_euclidean = centroid_dists[(node1, node2)]
                period_shifts[period][(node1, node2)] = (centroid_cosine - cosine, centroid_euclidean - euclidean)

    rc = sv.attribute_record_counter.value
    close_pairs = 0
    all_pairs = 0
    # for each period, find all pairs of nodes close
    period_to_close_nodes = {}
    for period in used_periods:
        period_to_close_nodes[period] = []
        for ix, node1 in enumerate(sorted_nodes):
            for node2 in sorted_nodes[ix + 1:]:
                all_pairs += 1
                n1a = node1.split(config.type_val_sep)[0]
                n2a = node2.split(config.type_val_sep)[0]
                if n1a != n2a:
                    cosine_shift, euclidean_shift = period_shifts[period][(node1, node2)]
                    if cosine_shift > 0 or euclidean_shift > 0:
                        period_count = rc.count_records([period, node1, node2])
                        if period_count >= sv.attribute_min_pattern_count.value:
                            close_pairs += 1
                            period_to_close_nodes[period].append((node1, node2))

    sv.attribute_converging_pairs.value = close_pairs
    sv.attribute_all_pairs.value = all_pairs
    # convert to df
    close_node_rows = []
    for period, close_nodes in period_to_close_nodes.items():
        for node1, node2 in close_nodes:
            period_count = rc.count_records([period, node1, node2])
            mean_count, sd, max = rc.compute_period_mean_sd_max([node1, node2])
            if period_count >= sv.attribute_min_pattern_count.value:
                count_factor = period_count / mean_count
                count_delta = period_count - mean_count
                # if period_count > mean_count:
                cosine_shift, euclidean_shift = period_shifts[period][(node1, node2)]
                row = [period, node1, node2, period_count, mean_count, count_delta, count_factor, cosine_shift, euclidean_shift]
                close_node_rows.append(row)
    columns = ['period', 'node1', 'node2', 'period_count', 'mean_count', 'count_delta', 'count_factor', 'cosine_shift', 'euclidean_shift']
    close_node_df = pd.DataFrame(close_node_rows, columns=columns)
    # correlation between shift and delta
    # corr1 = close_node_df[['count_delta', 'cosine_shift']].corr()
    # corr2 = close_node_df[['count_delta', 'euclidean_shift']].corr()
    # corr3 = close_node_df[['count_factor', 'cosine_shift']].corr()
    # corr4 = close_node_df[['count_factor', 'euclidean_shift']].corr()
    # print(f'count-cos delta corr: {corr1}')
    # print(f'count-euc delta corr: {corr2}')
    # print(f'count-cos factor corr: {corr3}')
    # print(f'count-euc factor corr: {corr4}')

    # for each period, combine overlapping similar pairs of nodes if they are supported by a positive count
    period_to_patterns = {}
    pattern_to_periods = defaultdict(set)
    for period in used_periods:
        period_pair_counts = close_node_df[close_node_df['period'] == period][['node1', 'node2', 'period_count']].values.tolist()
        period_to_patterns[period] = [([], 0)]
        period_pairs = [tuple(sorted([a, b])) for a, b, c in period_pair_counts]
        print(F'Period {period}')
        for (pattern, count) in period_to_patterns[period]:
            for (a, b) in period_pairs:
                if len(pattern) > 0 and ((a in pattern and b in pattern) or (a not in pattern and b not in pattern)):
                    continue
                candidate = None
                if (a in pattern and b not in pattern):
                    candidate = [b]
                elif (b in pattern and a not in pattern):
                    candidate = [a]
                elif (a not in pattern and b not in pattern):
                    candidate = [a, b]
                if candidate is not None:
                    candidate_pattern = sorted(pattern + candidate)
                    if len(candidate_pattern) <= sv.attribute_max_pattern_length.value:
                        if candidate_pattern not in [p for p, c in period_to_patterns[period]]:
                            candidate_pairs = combinations(candidate_pattern, 2)
                            exclude = False
                            for pair in candidate_pairs:
                                if pair not in period_pairs:
                                    exclude = True
                                    break
                            if not exclude:
                                pcount = rc.count_records([period] + list(candidate_pattern))
                                if pcount > sv.attribute_min_pattern_count.value:
                                    period_to_patterns[period].append((candidate_pattern, pcount))
                                    pattern_to_periods[tuple(candidate_pattern)].add(period)
    print('done combining pairs')
    # convert to df
    pattern_rows = []
    for period, patterns in period_to_patterns.items():
        for (pattern, count) in patterns:
            if count > 0:
                mean, sd, mx = rc.compute_period_mean_sd_max(pattern)
                score = (count - mean) / sd
                if score >= 0:
                    row = [period, ' & '.join(pattern), len(pattern), count, round(mean, 0), round(score, 2)]
                    pattern_rows.append(row)
    columns = ['period', 'pattern', 'length', 'count', 'mean', 'z_score']
    pattern_df = pd.DataFrame(pattern_rows, columns=columns)
    # count number of periods per pattern
    detections = pattern_df.groupby('pattern')['period'].count().reset_index().rename(columns={'period' : 'detections'})
    pattern_df = pattern_df.merge(detections, on='pattern')
    pattern_df['overall_score'] = pattern_df['z_score'] * pattern_df['length'] * pattern_df['detections'] * np.log(pattern_df['count'] + 1)
    # normalize overall score
    max_score = pattern_df['overall_score'].max()
    pattern_df['overall_score'] = pattern_df['overall_score'].apply(lambda x: round((x) / (max_score), 2))
    pattern_df = pattern_df.sort_values('overall_score', ascending=False)
    return pattern_df, close_pairs, all_pairs

def compute_attribute_counts(df, pattern, time_col, period):
    atts = pattern.split(' & ')
    counts = []
    fdf = util.df_functions.fix_null_ints(df).astype(str).replace('nan', '').replace('<NA>', '')
    fdf = fdf[fdf[time_col] == period]
    for att in atts:
        if att == 'Subject ID':
            continue
        ps = att.split(config.type_val_sep)
        if len(ps) == 2:
            a, v = ps
            fdf = fdf[fdf[a] == v]
        else:
            print(f'Error parsing attribute {att}')
    melted = pd.melt(fdf, id_vars=['Subject ID'], value_vars=[c for c in fdf.columns if c not in ['Subject ID', time_col]], var_name='Attribute', value_name='Value')
    melted = melted[melted['Value'] != '']
    melted['AttributeValue'] = melted['Attribute'] + config.type_val_sep + melted['Value']
    att_counts = melted.groupby('AttributeValue').agg({'Subject ID' : 'nunique'}).rename(columns={'Subject ID' : 'Count'}).sort_values(by='Count', ascending=False).reset_index()
    return att_counts