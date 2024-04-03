# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st
import polars as pl
import pandas as pd

import re
import io
import os

from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

import workflows.record_matching.functions as functions
import workflows.record_matching.config as config
import workflows.record_matching.variables as vars
import util.Embedder
import util.ui_components

embedder = util.Embedder.create_embedder(config.cache_dir)

def create():
    sv = vars.SessionVariables('record_matching')

    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)

    intro_tab, uploader_tab, process_tab, evaluate_tab = st.tabs(['Record matching workflow:', 'Upload data to match', 'Detect record groups', 'Evaluate record groups'])
    df = None
    with intro_tab:
        st.markdown(config.intro)
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            selected_file, df = util.ui_components.multi_csv_uploader('Upload multiple CSVs', sv.matching_uploaded_files, config.outputs_dir, 'matching_uploader', sv.matching_max_rows_to_process)
        with model_col:
                st.markdown('##### Map columns to data model')
                if df is None:
                    st.markdown('Upload and select a file to continue')
                else:
                    df = pl.from_pandas(df).lazy()
                    cols = [''] + df.columns
                    entity_col = ''
                    ready = False
                    dataset = st.text_input('Dataset name', key=f'{selected_file}_dataset_name')
                    name_col = st.selectbox('Entity name column', cols)
                    entity_col = st.selectbox('Entity ID column (optional)', cols)
                    filtered_cols = [c for c in cols if c not in [entity_col, name_col]]
                    att_cols = st.multiselect('Entity attribute columns', filtered_cols)
                    ready = dataset is not None and len(dataset) > 0 and len(att_cols) > 0 and name_col != ''
                    b1, b2 = st.columns([1, 1])
                    with b1:
                        if st.button("Add records to model", disabled=not ready, use_container_width=True):
                            if entity_col == '':
                                df = df.with_row_count(name="Entity ID")
                                df = df.rename({name_col: 'Entity name'})
                            else:
                                df = df.rename({entity_col: 'Entity ID', name_col: 'Entity name'})
                                df = df.with_columns(pl.col('Entity ID').cast(pl.Utf8))
                            df = df.select([pl.col('Entity ID'), pl.col('Entity name')] + [pl.col(c) for c in sorted(att_cols)]).collect()
                            if sv.matching_max_rows_to_process.value > 0:
                                df = df.head(sv.matching_max_rows_to_process.value)
                            sv.matching_dfs.value[dataset] = df
                    with b2:
                        if st.button('Reset data model', disabled=len(sv.matching_dfs.value) == 0, use_container_width=True):
                            if dataset in sv.matching_dfs.value.keys():
                                del sv.matching_dfs.value[dataset]
                    if len(sv.matching_dfs.value) > 0:
                        recs = sum([len(df) for df in sv.matching_dfs.value.values()])
                        st.markdown(f'Data model has **{len(sv.matching_dfs.value)}** datasets with **{recs}** total records.')
    with process_tab:
        if len(sv.matching_dfs.value) == 0:
            st.markdown('Upload data files to continue')
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('##### Configure text embedding model')
                # max_atts = max([len(df.columns) for df in sv.matching_dfs.value.values()])
                all_atts = []
                for dataset, df in sv.matching_dfs.value.items():
                    all_atts.extend([f'{c}::{dataset}' for c in df.columns if c not in ['Entity ID', 'Entity name']])
                    all_atts = sorted(all_atts)
                options = sorted(all_atts)
                renaming = defaultdict(dict)
                atts_to_datasets = defaultdict(list)
                sv.matching_mapped_atts.value = []
                num_atts = 0
                while True:
                    if f'att{num_atts}_vals' not in st.session_state.keys():
                        break
                    num_atts += 1
                

                def att_ui(i):
                    st.markdown(f'**Attribute {i+1}**')
                    is_assigned = False
                    b1, b2 = st.columns([3, 1])
                    with b1:
                        att_vals = st.multiselect(f'Values', key=f'att{i}_vals', options=options)
                        if len(att_vals) > 0:
                            is_assigned = True
                    with b2:
                        att_name = st.text_input(f'Label (optional)', key=f'att{i}_name')
                    if att_name == '' and len(att_vals) > 0:
                        att_name = att_vals[0].split('::')[0]
                    for val in att_vals:
                        col, dataset = val.split('::')
                        renaming[dataset][col] = att_name
                        atts_to_datasets[att_name].append(dataset)
                    return is_assigned 

                any_empty = False
                for i in range(num_atts):
                    is_assigned = att_ui(i)
                    if not is_assigned:
                        any_empty = True
                if not any_empty:
                    att_ui(num_atts)
                st.markdown('##### Configure similarity thresholds')
                b1, b2 = st.columns([1, 1])
                with b1:
                    st.number_input('Matching record distance (max)', min_value=0.0, max_value=1.0, step=0.01, key=sv.matching_sentence_pair_embedding_threshold.key, value=sv.matching_sentence_pair_embedding_threshold.value)

                with b2:
                    st.number_input('Matching name similarity (min)', min_value=0.0, max_value=1.0, step=0.01, key=sv.matching_sentence_pair_jaccard_threshold.key, value=sv.matching_sentence_pair_jaccard_threshold.value)
                if st.button('Detect record groups', use_container_width=True):
                    with st.spinner('Detecting groups...'):
                        if len(sv.matching_merged_df.value) == 0 or sv.matching_sentence_pair_embedding_threshold.value != sv.matching_last_sentence_pair_embedding_threshold.value:
                            sv.matching_last_sentence_pair_embedding_threshold.value = sv.matching_sentence_pair_embedding_threshold.value
                            aligned_dfs = []
                            for dataset, df in sv.matching_dfs.value.items():
                                rdf = df.clone()
                                rdf = rdf.rename(renaming[dataset])
                                # drop columns that were not renamed
                                for col in df.columns:
                                    if col not in ['Entity ID', 'Entity name'] and col not in renaming[dataset].values():
                                        rdf = rdf.drop(col)
                                for att, datasets in atts_to_datasets.items():
                                    if dataset not in datasets:
                                        rdf = rdf.with_columns(pl.lit('').alias(att))
                                rdf = rdf.with_columns(pl.lit(dataset).alias('Dataset'))
                                rdf = rdf.select(sorted(rdf.columns))
                                aligned_dfs.append(rdf)
                            string_dfs = []
                            for df in aligned_dfs:
                                # convert all columns to strings
                                for col in df.columns:
                                    df = df.with_columns(pl.col(col).cast(pl.Utf8))
                                string_dfs.append(df)
                            sv.matching_merged_df.value = pl.concat(string_dfs)
                            # filter out any rows with no entity name
                            sv.matching_merged_df.value = sv.matching_merged_df.value.filter(pl.col('Entity name') != '')
                            sv.matching_merged_df.value = sv.matching_merged_df.value.with_columns((pl.col('Entity ID').cast(pl.Utf8)) + '::' + pl.col('Dataset').alias('Unique ID'))
                            all_sentences = functions.convert_to_sentences(sv.matching_merged_df.value, skip=['Unique ID', 'Entity ID', 'Dataset'])
                            # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                            embeddings = embedder.encode_all(all_sentences)
                            nbrs = NearestNeighbors(n_neighbors=50, n_jobs=1, algorithm='auto', leaf_size=20, metric='cosine').fit(embeddings)
                            distances, indices = nbrs.kneighbors(embeddings)
                            threshold = sv.matching_sentence_pair_embedding_threshold.value
                            near_map = defaultdict(list)
                            for ix in range(len(all_sentences)):
                                near_is = indices[ix][1:]
                                near_ds = distances[ix][1:]
                                nearest = zip(near_is, near_ds)
                                for near_i, near_d in nearest:
                                    if near_d <= threshold:
                                        near_map[ix].append(near_i)

                            df = sv.matching_merged_df.value
                            
                            sv.matching_sentence_pair_scores.value = []
                            for ix, nx_list in near_map.items():
                                ixs = all_sentences[ix]
                                ixrec = df.row(ix, named=True)
                                for nx in nx_list:

                                    nxs = all_sentences[nx]
                                    nxrec = df.row(nx, named=True)
                                    ixn = ixrec['Entity name'].upper()
                                    nxn = nxrec['Entity name'].upper()
                                    

                                    ixn_c = re.sub(r'[^\w\s]', '', ixn)
                                    nxn_c = re.sub(r'[^\w\s]', '', nxn)
                                    N = 3
                                    igrams = set([ixn_c[i: i + N] for i in range(len(ixn_c) - N + 1)])
                                    ngrams = set([nxn_c[i: i + N] for i in range(len(nxn_c) - N + 1)])
                                    inter = len(igrams.intersection(ngrams))
                                    union = len(igrams.union(ngrams))
                                    score = inter/union if union > 0 else 0

                                    sv.matching_sentence_pair_scores.value.append((ix, nx, score))

                # st.markdown(f'Identified **{len(sv.matching_sentence_pair_scores.value)}** pairwise record matches.')
                                
                        df = sv.matching_merged_df.value
                        entity_to_group = {}
                        group_id = 0
                        matches = set()
                        pair_to_match = {}
                        for ix, nx, score in sorted(sv.matching_sentence_pair_scores.value, key=lambda x: x[2], reverse=True):
                            if score >= sv.matching_sentence_pair_jaccard_threshold.value:
                                ixrec = df.row(ix, named=True)
                                nxrec = df.row(nx, named=True)
                                ixn = ixrec['Entity name']
                                nxn = nxrec['Entity name']
                                ixp = ixrec['Dataset']
                                nxp = nxrec['Dataset']
                                
                                ix_id = f'{ixn}::{ixp}'
                                nx_id = f'{nxn}::{nxp}'
                                                
                                if ix_id in entity_to_group.keys() and nx_id in entity_to_group.keys():
                                    ig = entity_to_group[ix_id]
                                    ng = entity_to_group[nx_id]
                                    if ig != ng:
                                        # print(f'Merging group of {ix_id} into group of {nx_id}')
                                        for k, v in list(entity_to_group.items()):
                                            if v == ig:
                                                # print(f'Updating {k} to group {ng}')
                                                entity_to_group[k] = ng    
                                elif ix_id in entity_to_group.keys():
                                    # print(f'Adding {nx_id} to group {entity_to_group[ix_id]}')
                                    entity_to_group[nx_id] = entity_to_group[ix_id]
                                elif nx_id in entity_to_group.keys():
                                    # print(f'Adding {ix_id} to group {entity_to_group[nx_id]}')
                                    entity_to_group[ix_id] = entity_to_group[nx_id]
                                else:
                                    # print(f'Creating new group {group_id} for {ix_id} and {nx_id}')
                                    entity_to_group[ix_id] = group_id
                                    entity_to_group[nx_id] = group_id
                                    group_id += 1
                                        
                        for ix, nx, score in sorted(sv.matching_sentence_pair_scores.value, key=lambda x: x[2], reverse=True):
                            if score >= sv.matching_sentence_pair_jaccard_threshold.value:                       
                                ixrec = df.row(ix, named=True)
                                nxrec = df.row(nx, named=True)
                                ixn = ixrec['Entity name']
                                nxn = nxrec['Entity name']
                                ixp = ixrec['Dataset']
                                nxp = nxrec['Dataset']
                                
                                ix_id = f'{ixn}::{ixp}'
                                nx_id = f'{nxn}::{nxp}'        
                                matches.add(tuple([entity_to_group[ix_id]] + list(df.row(ix))))
                                matches.add(tuple([entity_to_group[nx_id]] + list(df.row(nx))))

                                pair_to_match[tuple(sorted([ix_id, nx_id]))] = score
     
                        sv.matching_matches_df.value = pl.DataFrame(list(matches), schema=['Group ID'] + sv.matching_merged_df.value.columns).sort(by=['Group ID','Entity name','Dataset'], descending=False)
                        group_to_size = sv.matching_matches_df.value.group_by('Group ID').agg(pl.count('Entity ID').alias('Size')).to_dict()
                        group_to_size = {k: v for k, v in zip(group_to_size['Group ID'], group_to_size['Size'])}
                        sv.matching_matches_df.value = sv.matching_matches_df.value.with_columns(sv.matching_matches_df.value['Group ID'].map_elements(lambda x: group_to_size[x]).alias('Group size'))

                        
                        sv.matching_matches_df.value = sv.matching_matches_df.value.select(['Group ID', 'Group size', 'Entity name', 'Dataset', 'Entity ID'] + [c for c in sv.matching_matches_df.value.columns if c not in ['Group ID', 'Group size', 'Entity name', 'Dataset', 'Entity ID']])
                        sv.matching_matches_df.value = sv.matching_matches_df.value.with_columns(sv.matching_matches_df.value['Entity ID'].map_elements(lambda x: x.split('::')[0]).alias('Entity ID'))
                        # keep only groups larger than 1
                        sv.matching_matches_df.value = sv.matching_matches_df.value.filter(pl.col('Group size') > 1)
                        # iterate over groups, calculating mean score
                        group_to_scores = defaultdict(list)

                        for (ix_id, nx_id), score in pair_to_match.items():
                            if ix_id in entity_to_group.keys() and nx_id in entity_to_group.keys() and entity_to_group[ix_id] == entity_to_group[nx_id]:
                                group_to_scores[entity_to_group[ix_id]].append(score)

                        group_to_mean_similarity = {}
                        for group, scores in group_to_scores.items():
                            group_to_mean_similarity[group] = sum(scores)/len(scores) if len(scores) > 0 else 0
                        sv.matching_matches_df.value = sv.matching_matches_df.value.with_columns(sv.matching_matches_df.value['Group ID'].map_elements(lambda x: group_to_mean_similarity[x] if x in group_to_mean_similarity.keys() else 0).alias('Name similarity'))
                        sv.matching_matches_df.value = sv.matching_matches_df.value.sort(by=['Name similarity', 'Group ID'], descending=[False, False])
                        # # keep all records linked to a group ID if any record linked to that ID has dataset GD or ILM
                        # sv.matching_matches_df.value = sv.matching_matches_df.value.filter(pl.col('Group ID').is_in(sv.matching_matches_df.value.filter(pl.col('Dataset').is_in(['GD', 'ILM']))['Group ID'].unique()))
                if len(sv.matching_matches_df.value) > 0:
                    st.markdown(f'Identified **{len(sv.matching_matches_df.value)}** record groups.')
            with c2:
                st.markdown('##### Record groups')
                if len(sv.matching_matches_df.value) > 0:
                    st.dataframe(sv.matching_matches_df.value, height=700, use_container_width=True, hide_index=True)
                    st.download_button('Download record groups', data=sv.matching_matches_df.value.write_csv(), file_name='record_groups.csv', mime='text/csv')
                           
    with evaluate_tab:
        b1, b2 = st.columns([2, 3])
        with b1:
            batch_size = 100
            data = sv.matching_matches_df.value.drop(['Entity ID', 'Dataset', 'Name similarity']).to_pandas()
            generate, batch_messages = util.ui_components.generative_batch_ai_component(sv.matching_system_prompt, sv.matching_instructions, {}, 'data', data, batch_size)
        with b2:
            st.markdown('##### AI evaluation of record groups')
            prefix = '```\nGroup ID,Relatedness,Explanation\n'
            placeholder = st.empty()
            if generate:
                for messages in batch_messages:
                    response = util.AI_API.generate_text_from_message_list(messages, placeholder, prefix=prefix)
                    if len(response.strip()) > 0:
                        prefix = prefix + response + '\n'
                result = prefix.replace('```\n', '').strip()
                sv.matching_evaluations.value = pl.read_csv(io.StringIO(result))
            placeholder.empty()
            if len(sv.matching_evaluations.value) > 0:
                st.dataframe(sv.matching_evaluations.value.to_pandas(), height=700, use_container_width=True, hide_index=True)
                jdf = sv.matching_matches_df.value.join(sv.matching_evaluations.value, left_on='Group ID', right_on='Group ID', how='inner')
                st.download_button('Download AI match report', data=jdf.write_csv(), file_name='record_groups_evaluated.csv', mime='text/csv')