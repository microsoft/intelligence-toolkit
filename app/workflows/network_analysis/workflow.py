import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np

from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

import re
import os
import community

import workflows.network_analysis.functions as functions
import workflows.network_analysis.classes as classes
import workflows.network_analysis.config as config
import workflows.network_analysis.prompts as prompts
import workflows.network_analysis.variables as vars
import util.AI_API
import util.ui_components

embedder = util.AI_API.create_embedder(config.cache_dir)

def create():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Network Analysis')
    sv = vars.SessionVariables('network_analysis')

    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)

    uploader_tab, process_tab, view_tab = st.tabs(['Create data model', 'Process data model', 'Explore networks'])
    df = None
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            # st.markdown('##### Upload data for processing')
            # files = st.file_uploader("Upload CSVs", type=['csv'], accept_multiple_files=True)
            # st.number_input('Maximum rows to process (0 = all)', min_value=0, step=1000, key=sv.network_max_rows_to_process.key, value=sv.network_max_rows_to_process.value)
            
            # if files != None:
            #     for file in files:
            #         if file.name not in sv.network_uploaded_files.value:
            #             df = pd.read_csv(file, encoding='unicode_escape')[:sv.network_max_rows_to_process.value] if sv.network_max_rows_to_process.value > 0 else pd.read_csv(file, encoding='unicode_escape')
            #             df.to_csv(os.path.join(config.outputs_dir, file.name), index=False)
            #             sv.network_uploaded_files.value.append(file.name)
            # selected_file = st.selectbox("Select a file", sv.network_uploaded_files.value)
            
            # if selected_file != None:
            #     df = pd.read_csv(os.path.join(config.outputs_dir, selected_file), encoding='unicode_escape')[:sv.network_max_rows_to_process.value] if sv.network_max_rows_to_process.value > 0 else pd.read_csv(os.path.join(config.outputs_dir, selected_file), encoding='unicode_escape')
            #     st.dataframe(df[:config.max_rows_to_show], hide_index=True, use_container_width=True)
            selected_file, df = util.ui_components.multi_csv_uploader('Upload multiple CSVs', sv.network_uploaded_files, config.outputs_dir, 'network_uploader', sv.network_max_rows_to_process)
        with model_col:
            st.markdown('##### Map columns to model')
            if df is None:
                st.markdown('Upload and select a file to continue')
            else:
                options = [''] + df.columns.values.tolist()
                link_type = st.radio('Link type', ['Entity-Attribute', 'Entity-Entity', 'Entity-Flag'], horizontal=True)
                entity_col = st.selectbox("Entity ID column", options)
                model_links = None
                attribute_label = ''
                if link_type == 'Entity-Entity':
                    value_cols = [st.selectbox("Related entity column", options)]
                    attribute_col = st.selectbox("Relationship type", ['Use column name', 'Use custom name', 'Use related column'])
                    if attribute_col == 'Use custom name':
                        attribute_label = st.text_input('Relationship type', '')
                    direction = st.selectbox("Relationship direction", ['Undirected', 'Entity to related', 'Related to entity'])
                    if direction == 'Undirected':
                        model_links = sv.network_entity_links.value
                    elif direction == 'Entity to related':
                        model_links = sv.network_directed_entity_links.value
                    elif direction == 'Related to entity':
                        tmp = entity_col
                        entity_col = value_cols[0]
                        value_cols = [tmp]
                        model_links = sv.network_directed_entity_links.value
                elif link_type == 'Entity-Attribute':
                    value_cols = st.multiselect("Attribute value column(s)", options)
                    attribute_col = st.selectbox("Attribute type", ['Use column name', 'Use custom name', 'Use related column'])
                    if attribute_col == 'Use custom name':
                        attribute_label = st.text_input('Attribute name', '')
                    model_links = sv.network_attribute_links.value
                elif link_type == 'Entity-Flag':
                    value_cols = st.multiselect("Flag value column(s)", options)
                    attribute_col = st.selectbox("Flag type", ['Use column name', 'Use custom name', 'Use related column'])
                    if attribute_col == 'Use custom name':
                        attribute_label = st.text_input('Flag name', '')
                    flag_agg = st.selectbox("Flag format", ['Instance', 'Count', 'List'])
                    model_links = sv.network_flag_links.value

                if st.button("Add links to model", disabled=entity_col == '' or attribute_col == '' or len(value_cols) == 0 or link_type == ''):
                    with st.spinner('Adding links to model...'):
                        for value_col in value_cols:
                            if attribute_col == 'Use column name':
                                attribute_label = value_col
                            # remove punctuation but retain characters and digits in any language
                            # compress whitespace to single space
                            df[entity_col] = df[entity_col].apply(lambda x : re.sub(r'[^\w\s&@\+]', '', str(x)).strip())
                            df[value_col] = df[value_col].apply(lambda x : re.sub(r'[^\w\s&@\+]', '', str(x)).strip())
                            df[entity_col] = df[entity_col].apply(lambda x : re.sub(r'\s+', ' ', str(x)).strip())
                            df[value_col] = df[value_col].apply(lambda x : re.sub(r'\s+', ' ', str(x)).strip())
                            if link_type == 'Entity-Attribute':
                                if attribute_col in ['Use column name', 'Use custom name']:
                                    df['attribute_col'] = attribute_label
                                    sv.network_node_types.value.add(attribute_label)
                                    model_links.append(df[[entity_col, 'attribute_col', value_col]].values.tolist())
                                else:
                                    sv.network_node_types.value.update(df[attribute_label].unique().tolist())
                                    model_links.append(df[[entity_col, attribute_col, value_col]].values.tolist())
                                functions.build_undirected_graph(sv)
                            elif link_type == 'Entity-Flag':
                                # groupby entity and sum flag counts
                                gdf = df.groupby([entity_col]).sum().reset_index()
                                gdf['attribute_col'] = attribute_label
                                gdf['attribute_col2'] = attribute_label
                                if attribute_col in ['Use column name', 'Use custom name']:
                                    sv.network_flag_types.value.add(attribute_label)
                                    if flag_agg == 'Instance':
                                        gdf['count_col'] = 1
                                        model_links.append(gdf[[entity_col, 'attribute_col', value_col, 'count_col']].values.tolist())
                                    elif flag_agg == 'Count':
                                        gdf[value_col] = gdf[value_col].astype(int)
                                        model_links.append(gdf[[entity_col, 'attribute_col', 'attribute_col2', value_col]].values.tolist())
                                    elif flag_agg == 'List':
                                        gdf['count_col'] = 1
                                        model_links.append(gdf[[entity_col, 'attribute_col', 'attribute_col2', 'count_col']].values.tolist())
                                else:
                                    sv.network_flag_types.value.update(df[attribute_label].unique().tolist())
                                    if flag_agg == 'Instance':
                                        model_links.append(gdf[[entity_col, attribute_col, value_col]].values.tolist())
                                    elif flag_agg == 'Count':
                                        gdf[value_col] = gdf[value_col].astype(int)
                                        model_links.append(gdf[[entity_col, attribute_col, 'attribute_col2', value_col]].values.tolist())
                                    elif flag_agg == 'List':
                                        gdf['count_col'] = 1
                                        model_links.append(gdf[[entity_col, 'attribute_col', 'attribute_col2', 'count_col']].values.tolist())
                                functions.build_integrated_flags(sv)
                            elif link_type == 'Entity-Entity':
                                if attribute_col in ['Use column name', 'Use custom name']:
                                    df['attribute_col'] = attribute_label
                                    model_links.append(df[[entity_col, 'attribute_col', value_col]].values.tolist())
                                else:
                                    model_links.append(df[[entity_col, attribute_col, value_col]].values.tolist())
                                functions.build_undirected_graph(sv)
                                # build_directed_graph(sv) TODO

            st.markdown('##### Initial data model summary')
            
            # TODO: add other link types

            num_entities = 0
            num_attributes = 0
            num_edges = 0
            num_flags = 0
            if sv.network_overall_graph.value != None:
                all_nodes = sv.network_overall_graph.value.nodes()
                entity_nodes = [node for node in all_nodes if node.startswith(config.entity_label)]
                num_entities = len(entity_nodes)
                num_attributes = len(all_nodes) - num_entities
                num_edges = len(sv.network_overall_graph.value.edges())
            if len(sv.network_integrated_flags.value) > 0:
                num_flags = sv.network_integrated_flags.value['count'].sum()
            st.markdown(f'*Number of entities*: {num_entities}')
            st.markdown(f'*Number of attributes*: {num_attributes}')
            st.markdown(f'*Number of links*: {num_edges}')
            st.markdown(f'*Number of flags*: {num_flags}')
            st.markdown('Advance to the next tab when you are ready to process the data model. You can always return to this tab to upload more data files and/or add more links from existing files.')

    with process_tab:
        index_col, infer_col, part_col = st.columns([1, 2, 2])
        components = None
        with index_col:
            st.markdown('##### Index nodes')
            st.multiselect('Select node types to fuzzy match', options=sorted([config.entity_label] + list(sv.network_node_types.value)), key=sv.network_indexed_node_types.key)
            if st.button('Index nodes'):
                
                with st.spinner('Indexing nodes...'):

                    text_types = list([(n, d['type']) for n, d in sv.network_overall_graph.value.nodes(data=True) if d['type'] in sv.network_indexed_node_types.value])
                    texts = [t[0] for t in text_types]
                    
                    df = pd.DataFrame(text_types, columns=['text', 'type'])
                    embeddings = embedder.encode_all(texts)
                    vals = [(n, t, e) for (n, t), e in zip(text_types, embeddings)]
                    edf = pd.DataFrame(vals, columns=['text', 'type', 'vector'])

                    edf = edf[edf['text'].isin(texts)]
                    sv.network_embedded_texts.value = edf['text'].tolist()
                    # edf['vector'] = edf['vector'].apply(lambda x : np.array([np.float32(y) for y in x[1:-1].split(' ') if y != '']))
                    # embeddings = np.array(edf['vector'].tolist())
                    nbrs = NearestNeighbors(n_neighbors=20, n_jobs=1, algorithm='auto', leaf_size=20, metric='cosine').fit(embeddings)
                    sv.network_nearest_text_distances.value, sv.network_nearest_text_indices.value = nbrs.kneighbors(embeddings)
            st.markdown(f'*Number of nodes indexed*: {len(sv.network_embedded_texts.value)}')

        with infer_col:
            st.markdown('##### Infer links')
            st.number_input('Similarity threshold', min_value=0.0, max_value=1.0, key=sv.network_similarity_threshold.key, step=0.01, value=sv.network_similarity_threshold.value)
            if st.button('Infer links'):
                sv.network_inferred_links.value = defaultdict(set)
                texts = sv.network_embedded_texts.value
                pb = st.progress(0, text = 'Inferring links...')
                for ix in range(len(texts)):
                    pb.progress(int(ix * 100 / len(texts)), text = 'Inferring links...')
                    near_is = sv.network_nearest_text_indices.value[ix]
                    near_ds = sv.network_nearest_text_distances.value[ix]
                    nearest = zip(near_is, near_ds)
                    for near_i, near_d in nearest:
                        if near_i != ix and near_d <= sv.network_similarity_threshold.value:
                            if texts[ix] != texts[near_i]:
                                sv.network_inferred_links.value[texts[ix]].add(texts[near_i])
                                sv.network_inferred_links.value[texts[near_i]].add(texts[ix])

                pb.empty()
            
            link_list = []
            for text, near in sv.network_inferred_links.value.items():
                for n in near:
                    if text < n:
                        link_list.append((text, n))
            ilc = len(link_list)
            st.markdown(f'*Number of links inferred*: {ilc}')
            if ilc > 0:
                idf = pd.DataFrame(link_list, columns=['text', 'similar'])
                idf['text'] = idf['text'].str.replace(config.entity_label + config.att_val_sep, '')
                idf['similar'] = idf['similar'].str.replace(config.entity_label + config.att_val_sep, '')
                idf = idf.sort_values(by=['text', 'similar']).reset_index(drop=True)
                st.dataframe(idf, hide_index=True, use_container_width=True)
            
        with part_col:
            st.markdown('##### Identify networks')
            # c1, c2 = st.columns([1, 1])
            # with c1:
            #     st.number_input('Maximum network entities', min_value=1, max_value=1000, key=sv.network_max_network_size.key)
            # with c2:
            st.number_input('Maximum attribute degree', min_value=1, key=sv.network_max_attribute_degree.key, value=sv.network_max_attribute_degree.value)
            sv.network_supporting_attribute_types.value = st.multiselect('Supporting attribute types', options=sorted(sv.network_node_types.value))
            comm_count = 0    
            if st.button('Identify networks'):
                with st.spinner('Identifying networks...'):
                    sv.network_table_index.value += 1
                    # Create a new graph P in which entities are connected if they share an attribute
                    P = functions.project_entity_graph(sv)

                    components = sorted(list(nx.components.connected_components(P)), key=lambda x : len(x), reverse=True)
                    comp_count = len(components)
                    sv.network_components.value = range(comp_count)
                    sv.network_component_to_nodes.value = dict(zip(sv.network_components.value, components))
                    sv.network_community_nodes.value = []
                    sv.network_entity_to_community_ix.value = {}
                    for component in sv.network_components.value:
                        nodes = sv.network_component_to_nodes.value[component]
                        if len(nodes) > sv.network_max_network_size.value:
                            S = nx.subgraph(P, nodes)
                            node_to_network = community.best_partition(S, resolution=1.0, randomize=False, weight='weight')
                            network_to_nodes = defaultdict(set)
                            for node, network in node_to_network.items():
                                network_to_nodes[network].add(node)
                            networks = [list(nodes) for nodes in network_to_nodes.values()]
                            for network in networks:
                                sv.network_community_nodes.value.append(network)
                                for node in network:
                                    sv.network_entity_to_community_ix.value[node] = len(sv.network_community_nodes.value) - 1
                        else:
                            sv.network_community_nodes.value.append(nodes)
                            for node in nodes:
                                sv.network_entity_to_community_ix.value[node] = len(sv.network_community_nodes.value) - 1

                    N = functions.build_network_from_entities(sv, sv.network_overall_graph.value, sv.network_overall_graph.value.nodes())
                    if len(sv.network_integrated_flags.value) > 0:
                        fdf = sv.network_integrated_flags.value
                        fdf = fdf[fdf['count'] > 0]
                        sv.network_flagged_nodes.value = fdf['qualified_entity'].unique().tolist()
                    else:
                        sv.network_flagged_nodes.value = []

                entity_records = []
                for ix, entities in enumerate(sv.network_community_nodes.value):
                    attributes = set()
                    
                    community_flags = 0
                    flagged = 0
                    unflagged = 0
                    flaggedPerUnflagged = 0
                    if len(sv.network_integrated_flags.value) > 0:
                        flags_df = sv.network_integrated_flags.value[sv.network_integrated_flags.value['qualified_entity'].isin(entities)]
                        community_flags = flags_df['count'].sum()
                        flagged = len(flags_df[flags_df['count'] > 0])
                        unflagged = len(entities) - flagged
                        flaggedPerUnflagged = flagged / unflagged if unflagged > 0 else 0
                        flaggedPerUnflagged = round(flaggedPerUnflagged, 2)
                    for n in entities:
                        ent_neighbors = [x for x in set(sv.network_overall_graph.value.neighbors(n)).union(sv.network_inferred_links.value[n]) if not x.startswith(config.entity_label) and x not in sv.network_trimmed_attributes.value['Attribute'].tolist()]
                        attributes.update(ent_neighbors)
                        ent_neighbor_neighbors = [x for x in set(sv.network_overall_graph.value.neighbors(n)).union(sv.network_inferred_links.value[n]) if not x.startswith(config.entity_label) and x not in sv.network_trimmed_attributes.value['Attribute'].tolist()]
                        attributes.update(ent_neighbor_neighbors)
                        flags = sv.network_integrated_flags.value[sv.network_integrated_flags.value['qualified_entity'] == n]['count'].sum() if len(sv.network_integrated_flags.value) > 0 else 0
                        entity_records.append((n.split(config.att_val_sep)[1], flags, ix, len(entities), len(attributes), community_flags, flagged, flaggedPerUnflagged))
                sv.network_entity_df.value = pd.DataFrame(entity_records, columns=['Entity ID', 'Entity Flags', 'Network ID', 'Network Entities', 'Network Attributes', 'Network Flags', 'Flagged', 'Flagged/Unflagged'])
                sv.network_entity_df.value = sv.network_entity_df.value.sort_values(by=['Flagged/Unflagged'], ascending=False).reset_index(drop=True)
                sv.network_table_index.value += 1

            comm_count = len(sv.network_community_nodes.value)

            st.markdown(f'*Number of networks identified*: {comm_count}')
            if comm_count > 0:
                comm_sizes = [len(comm) for comm in sv.network_community_nodes.value if len(comm) > 1]
                max_comm_size = max(comm_sizes)
                trimmed_atts = len(sv.network_trimmed_attributes.value)
                st.markdown(f'*Number of multi-entity networks*: {len(comm_sizes)}')
                st.markdown(f'*Maximum network entities*: {max_comm_size}')
                st.markdown(f'*Attributes removed because of high degree*: {trimmed_atts}')
                if trimmed_atts > 0:
                    st.dataframe(sv.network_trimmed_attributes.value, hide_index=True, use_container_width=True)

            
    with view_tab:
        if len(sv.network_entity_df.value) == 0:
            st.markdown('##### No entity networks identified')
        else:
            selected_entities = functions.dataframe_with_selections(sv.network_entity_df.value, key='entity_table')
          
            selected_network = ''
            selected_entity = ''
            if len(selected_entities) > 0:
                selected_network = selected_entities['Network ID'].iloc[0]
                selected_entity = selected_entities['Entity ID'].iloc[0]
            if selected_network != "":
                c_nodes = sv.network_community_nodes.value[selected_network]

                N = functions.build_network_from_entities(sv, sv.network_overall_graph.value, c_nodes)
                functions.diffuse_risk(sv, N, 6, 0.8)

                full_links_df = pd.DataFrame([(u, v) for u, v in N.edges()], columns=['source', 'target'])
                full_links_df['attribute'] = full_links_df['target'].apply(lambda x : x.split(config.att_val_sep)[0])
                N1, new_flags = functions.simplify_graph(N, sv.network_flagged_nodes.value)
                merged_nodes_df = pd.DataFrame([(n, d['type'], d['diffused_risk'][0], d['diffused_risk'][-1]) for n, d in N1.nodes(data=True)], columns=['node', 'type', 'initial_risk', 'diffused_risk'])
                merged_links_df = pd.DataFrame([(u, v) for u, v in N1.edges()], columns=['source', 'target'])
                merged_links_df['attribute'] = merged_links_df['target'].apply(lambda x : x.split(config.att_val_sep)[0])
                c1, c2 = st.columns([2, 1])
                with c1:
                    gtp = st.empty()
                    gp = st.empty()
                with c2:
                    d1, d2 = st.columns([1, 1])
                    with d1:
                        graph_type = st.radio('Graph type', ['Full', 'Simplified'], horizontal=True)
                    with d2:
                        node_color = st.radio('Node color', ['Type', 'Risk'], horizontal=True)
                    if graph_type == 'Full':
                        gtp.markdown(f'##### Entity {selected_entity} in Network {selected_network} (full)')
                        with gp:
                            functions.get_entity_graph(N, f'{config.entity_label}{config.att_val_sep}{selected_entity}', full_links_df, 800, 700, [config.entity_label] + list(sv.network_node_types.value), sv.network_flagged_nodes.value, show_risk=node_color=='Risk')
                    elif graph_type == 'Simplified':
                        gtp.markdown(f'##### Entity {selected_entity} in Network {selected_network} (simplified)')
                        with gp:
                            functions.get_entity_graph(N1, f'{config.entity_label}{config.att_val_sep}{selected_entity}', merged_links_df, 800, 700, [config.entity_label] + list(sv.network_node_types.value), new_flags, show_risk=node_color=='Risk')
                    
                    if st.button('Explain'):
                        placeholder = st.empty()
                        
                        variables = {
                            'network_id': selected_network,
                            'network_nodes': merged_nodes_df.to_csv(index=False),
                            'network_edges': merged_links_df.to_csv(index=False)
                        }
                        util.AI_API.generate_from_message_pair(
                            placeholder=placeholder,
                            system_message=prompts.system_prompt,
                            user_message=prompts.user_prompt,
                            variables=variables
                        )
                    
