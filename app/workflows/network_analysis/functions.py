import streamlit as st
import pandas as pd
import networkx as nx
import colorsys
import numpy as np
from collections import defaultdict
from streamlit_agraph import Config, Edge, Node, agraph

import workflows.network_analysis.config as config

def dataframe_with_selections(df, key, height=250):
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
        use_container_width=True,
        height=height
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    selected_rows = selected_rows.drop('Select', axis=1)

    return selected_rows

def hsl_to_hex(h, s, l):
    rgb = colorsys.hls_to_rgb(h / 360, l / 100, s / 100)
    hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
    return hex_color

def get_entity_graph(G, selected, links_df, width, height, attribute_types, flagged_nodes, show_risk):
    """
    Implements the entity graph visualization after network selection
    """
    node_names = set()
    nodes = []
    edges = []
    all_nodes = set(links_df["source"]).union(set(links_df["target"]))
    if len(all_nodes) > 0:
        max_risk = max([G.nodes[node]['diffused_risk'][-1] for node in all_nodes])
        for node in all_nodes:
            node_names.add(node)
            size = 20 if node == selected else 12 if node.startswith(config.entity_label) else 8
            vadjust = - size - 10

            def get_type_color(node_type, is_flagged, attribute_types):
                if is_flagged:
                    h = 0
                    s = 70
                    l = 80 
                else:
                    start = 230
                    reserve = 35
                    prop = attribute_types.index(node_type) / len(attribute_types)
                    inc = prop * (360 - 2 * reserve)
                    # avoid reds
                    h = (start + inc) % 360
                    if h < reserve:
                        h += 2 * reserve
                    if h > 360 - reserve :
                        h = (h + 2 * reserve) % 360
                    s = 70
                    l = 80 
                return str(hsl_to_hex(h, s, l))
            
            def get_risk_color(risk, initial, max_risk):
                h = 0
                s = 100 * risk/max_risk if max_risk > 0 else 0
                l = 50 if risk == initial and risk > 0 else 80 
                return str(hsl_to_hex(h, s, l))
                
            parts = [p.split(config.att_val_sep) for p in node.split(config.list_sep)]
            atts = [p[0] for p in parts]
            # remove duplicate values while maintaining order
            atts = list(dict.fromkeys(atts))
            color = get_risk_color(G.nodes[node]['diffused_risk'][-1], G.nodes[node]['diffused_risk'][0], max_risk) if show_risk else get_type_color(atts[0], node in flagged_nodes, attribute_types)
            vals = [p[1] for p in parts if len(p) > 1]
            # remove duplicate values while maintaining order
            vals = list(dict.fromkeys(vals))
            comm = G.nodes[node]['network'] if 'network' in G.nodes[node] else ''
            label = '\n'.join(vals) + '\n(' + config.list_sep.join(atts) + ')'
            i_risk = G.nodes[node]['diffused_risk'][0]
            d_risk = G.nodes[node]['diffused_risk'][-1]
            nodes.append(
                Node(
                    title=node + f'\nInitial flags: {i_risk}\nDiffused risk: {d_risk}',
                    id=node,
                    label=label,
                    size=size,
                    color=color,
                    font={'vadjust': vadjust, 'size' : 5} #, 'size' : '8px', 'face': 'arial', 'color' : 'black'},
                )
            )
        for i, row in links_df.iterrows():
            source = row['source']
            target = row['target']
            edges.append(Edge(source=source, target=target, color="mediumgray", size=1))
    g_config = Config(
        width=width,
        height=height,
        directed=False,
        physics=True,
        hierarchical=False
    )
    return_value = agraph(nodes=nodes, edges=edges, config=g_config) # type: ignore
    return return_value

def merge_nodes(G, flagged_nodes, can_merge_fn):
    merged_flags = list(flagged_nodes)

    nodes = list(G.nodes()) # may change during iteration

    for ix, node in enumerate(nodes):

        if node not in G.nodes():
            continue
        neighbours = list(G.neighbors(node))
        merge_list = [node]
        for n in neighbours:
            if n not in G.nodes():
                continue
            if can_merge_fn(node, n):
                merge_list.append(n)
        if len(merge_list) > 1:
            G, new_flags = merge_node_list(G, merge_list, merged_flags)
            merged_flags.extend(new_flags)
            merged = True

    return G, merged_flags

def merge_node_list(G, merge_list, in_flags):
    new_flags = []
    G1 = G.copy()
    m = config.list_sep.join(sorted(merge_list))
    t = config.list_sep.join(sorted([G.nodes[n]['type'] for n in merge_list]))
    merged_risk = max([G.nodes[n]['diffused_risk'][-1] for n in merge_list])
    initial_risk = max([G.nodes[n]['diffused_risk'][0] for n in merge_list])
    G1.add_node(m, type=t)
    G1.nodes[m]['diffused_risk'] = [initial_risk, merged_risk]
    has_flags = False
    for n in merge_list:
        for nn in G.neighbors(n):
            if nn not in merge_list:
                G1.add_edge(m, nn)
        G1.remove_node(n)
        has_flags = has_flags or n in in_flags
    if has_flags:
        new_flags.append(m)
    return G1, new_flags

def simplify_graph(C, flagged_nodes):
    S = C.copy()
    # remove single degree attributes
    for node in list(S.nodes()):
        if S.degree(node) < 2 and not node.startswith(config.entity_label):
            S.remove_node(node)

    merged_flags = list(flagged_nodes)
    S, new_flags = merge_nodes(S, flagged_nodes, lambda x, y : # merge if overlapping types or values
                               len(set([xv.split(config.att_val_sep)[0] for xv in sorted(x.split(config.list_sep))]).intersection(set([yv.split(config.att_val_sep)[0] for yv in sorted(y.split(config.list_sep))]))) > 0 or 
                               len(set([xv.split(config.att_val_sep)[1] for xv in sorted(x.split(config.list_sep))]).intersection(set([yv.split(config.att_val_sep)[1] for yv in sorted(y.split(config.list_sep))]))) > 0)
    merged_flags.extend(new_flags)

    # remove single degree attributes
    for node in list(S.nodes()):
        if S.degree(node) < 2 and not node.startswith(config.entity_label):
            S.remove_node(node)

    return S, merged_flags

def project_entity_graph(sv):
    # Remove high-degree attributes
    trim = [(n, d) for (n, d) in sv.network_overall_graph.value.degree() if not n.startswith(config.entity_label) and d > sv.network_max_attribute_degree.value]
    trimmed_nodeset = set([t[0] for t in trim])
    sv.network_trimmed_attributes.value = pd.DataFrame(trim, columns=['Attribute', 'Linked Entities']).sort_values('Linked Entities', ascending=False).reset_index(drop=True)
    P = nx.Graph()
    sv.network_entity_graph.value = P
    for node in sv.network_overall_graph.value.nodes():
        if node.startswith(config.entity_label):
            ent_neighbors = set(sv.network_overall_graph.value.neighbors(node)).union(sv.network_inferred_links.value[node])
            for ent_neighbor in ent_neighbors:
                if ent_neighbor not in trimmed_nodeset:
                    if ent_neighbor.startswith(config.entity_label):
                        if node != ent_neighbor:
                            P.add_edge(node, ent_neighbor)
                    else: # att
                        if ent_neighbor.split(config.att_val_sep)[0] not in sv.network_supporting_attribute_types.value:
                            att_neighbors = set(sv.network_overall_graph.value.neighbors(ent_neighbor)).union(sv.network_inferred_links.value[ent_neighbor])
                            for att_neighbor in att_neighbors:
                                if att_neighbor.split(config.att_val_sep)[0] not in sv.network_supporting_attribute_types.value:
                                    if att_neighbor not in trimmed_nodeset:
                                        if att_neighbor.startswith(config.entity_label):
                                            if node != att_neighbor:
                                                P.add_edge(node, att_neighbor)
                                        else: # fuzzy att link
                                            fuzzy_att_neighbors = set(sv.network_overall_graph.value.neighbors(att_neighbor)).union(sv.network_inferred_links.value[att_neighbor])
                                            for fuzzy_att_neighbor in fuzzy_att_neighbors:
                                                if fuzzy_att_neighbor.split(config.att_val_sep)[0] not in sv.network_supporting_attribute_types.value:
                                                    if fuzzy_att_neighbor not in trimmed_nodeset:
                                                        if fuzzy_att_neighbor.startswith(config.entity_label):
                                                            if node != fuzzy_att_neighbor:
                                                                P.add_edge(node, fuzzy_att_neighbor)
    return P

def build_undirected_graph(sv):
    G = nx.Graph()
    sv.network_overall_graph.value = G
    value_to_atts = defaultdict(set)
    for link_list in sv.network_attribute_links.value:
        for link in link_list:
            n1 = f'{config.entity_label}{config.att_val_sep}{link[0]}'
            n2 = f'{link[1]}{config.att_val_sep}{link[2]}'
            edge = (n1, n2) if n1 < n2 else (n2, n1)
            G.add_edge(edge[0], edge[1], type=link[1])
            G.add_node(n1, type=config.entity_label)
            G.add_node(n2, type=link[1])
            value_to_atts[link[2]].add(n2)
    for link_list in sv.network_entity_links.value:
        for link in link_list:
            n1 = f'{config.entity_label}{config.att_val_sep}{link[0]}'
            n2 = f'{config.entity_label}{config.att_val_sep}{link[2]}'
            edge = (n1, n2) if n1 < n2 else (n2, n1)
            G.add_edge(edge[0], edge[2], type=link[1])
            G.add_node(n1, type=config.entity_label)
            G.add_node(n2, type=config.entity_label)
    for val, atts in value_to_atts.items():
        att_list = list(atts)
        for i, att1 in enumerate(att_list):
            for att2 in att_list[i+1:]:
                edge = (att1, att2) if att1 < att2 else (att2, att1)
                G.add_edge(edge[0], edge[1], type='equality')
    return G
    
def build_integrated_flags(sv):
    sv.network_integrated_flags.value = pd.concat([pd.DataFrame(link_list, columns=['entity', 'type', 'flag', 'count']) for link_list in sv.network_flag_links.value])
    sv.network_integrated_flags.value = sv.network_integrated_flags.value.groupby(['entity', 'type', 'flag']).sum().reset_index()
    sv.network_integrated_flags.value['qualified_entity'] = sv.network_integrated_flags.value['entity'].apply(lambda x : f'{config.entity_label}{config.att_val_sep}{x}')

def build_network_from_entities(sv, G, nodes):
    N = nx.Graph()
    trimmed_nodeset = sv.network_trimmed_attributes.value['Attribute'].unique().tolist()
    for node in nodes:
        n_c = str(sv.network_entity_to_community_ix.value[node]) if node in sv.network_entity_to_community_ix.value.keys() else ''
        N.add_node(node, type=config.entity_label, network = n_c)
        ent_neighbors = set(G.neighbors(node)).union(sv.network_inferred_links.value[node])
        for ent_neighbor in ent_neighbors:
            if ent_neighbor not in trimmed_nodeset:
                if ent_neighbor.startswith(config.entity_label):
                    if node != ent_neighbor:
                        en_c = sv.network_entity_to_community_ix.value[ent_neighbor] if ent_neighbor in sv.network_entity_to_community_ix.value.keys() else ''
                        N.add_node(ent_neighbor, type=config.entity_label, network = en_c)
                        N.add_edge(node, ent_neighbor)
                else: # att
                    N.add_node(ent_neighbor, type=ent_neighbor.split(config.att_val_sep)[0])
                    N.add_edge(node, ent_neighbor)
                    att_neighbors = set(G.neighbors(ent_neighbor)).union(sv.network_inferred_links.value[ent_neighbor])
                    for att_neighbor in att_neighbors:
                        if att_neighbor not in trimmed_nodeset:
                            if not att_neighbor.startswith(config.entity_label):
                                N.add_node(att_neighbor, type=att_neighbor.split(config.att_val_sep)[0])
                                fuzzy_att_neighbors = set(G.neighbors(att_neighbor)).union(sv.network_inferred_links.value[att_neighbor])
                                for fuzzy_att_neighbor in fuzzy_att_neighbors:
                                    if fuzzy_att_neighbor not in trimmed_nodeset:
                                        N.add_node(fuzzy_att_neighbor, type=fuzzy_att_neighbor.split(config.att_val_sep)[0])
                                        N.add_edge(att_neighbor, fuzzy_att_neighbor)
    return N

def create_super_community(sv, community_nodes):
    super_communities = set()
    super_nodes = set()
    for node in community_nodes:
        print(node)
        neighbours = set(sv.network_entity_graph.value.neighbors(node))
        print(neighbours)
        for n in neighbours:
            if n in sv.network_entity_to_community_ix.value.keys():
                c = sv.network_entity_to_community_ix.value[n]
                super_communities.add(c)
    for c in super_communities:
        super_nodes.update(sv.network_community_nodes.value[c])
    
    return super_nodes

def diffuse_risk(sv, G, iterations, sensitivity=0.5):
    rdf = sv.network_integrated_flags.value
    if len(rdf) > 0:
        rdf = rdf[rdf['count'] > 0]
        for node in G.nodes():
            initial_risk = 0
            if node in rdf['qualified_entity'].tolist():
                initial_risk = rdf[rdf['qualified_entity'] == node]['count'].sum()
            G.nodes[node]['diffused_risk'] = [initial_risk]
        entity_turn = False
        for i in range(iterations):
            for node in G.nodes():
                if entity_turn:
                    if not node.startswith(config.entity_label):
                        continue
                else:
                    if node.startswith(config.entity_label):
                        continue
                own_risk = G.nodes[node]['diffused_risk'][-1]
                different_neighbours = [n for n in G.neighbors(node) if not n.startswith(config.entity_label)] if entity_turn else [n for n in G.neighbors(node) if n.startswith(config.entity_label)]
                neighbour_risks = [G.nodes[n]['diffused_risk'][-1] for n in different_neighbours]
                neighbour_risks = [r for r in neighbour_risks if r > 0]
                mean_neighbour_risk = np.mean(neighbour_risks) if len(neighbour_risks) > 0 else 0
                max_neighbour_risk = max(neighbour_risks) if len(neighbour_risks) > 0 else 0
                diffused_neighbour_risk = mean_neighbour_risk * sensitivity #* (max_neighbour_risk - mean_neighbour_risk)
                new_diffused_risk = round(max(own_risk, diffused_neighbour_risk), 2)
                G.nodes[node]['diffused_risk'].append(new_diffused_risk)
                

            for node in G.nodes():
                group = [node] + [n for n in G.neighbors(node) if (n.startswith(config.entity_label) and node.startswith(config.entity_label)) or (not n.startswith(config.entity_label) and not node.startswith(config.entity_label))]
                group_risk = [G.nodes[n]['diffused_risk'][-1] for n in group]
                group_risk = [r for r in group_risk if r > 0]
                max_group_risk = max(group_risk) if len(group_risk) > 0 else 0
                for n in group:
                    G.nodes[n]['diffused_risk'][-1] = max_group_risk
            entity_turn = not entity_turn
    else:
        for node in G.nodes():
            G.nodes[node]['diffused_risk'] = [0.0]
