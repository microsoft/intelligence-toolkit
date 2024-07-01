# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict

import numpy as np
import pandas as pd

from python.AI.metaprompts import do_not_harm
from python.AI.utils import generate_messages
from python.helpers import df_functions

from .config import type_val_sep
from .detection_functions import (
    create_close_node_rows,
    create_pattern_rows,
    create_period_shifts,
    create_period_to_patterns,
)
from .graph_functions import (
    convert_edge_df_to_graph,
    create_edge_df_from_atts,
)
from .prompts import report_prompt, user_prompt
from .record_counter import RecordCounter


def prepare_data(df, identifier_col=None):
    if not identifier_col:
        df['Subject ID'] = [i for i in range(len(df))]
    else:
        df['Subject ID'] = list(df.value[identifier_col])

    # Drop empty Subject ID rows
    filtered = df.dropna(subset=['Subject ID'])
    melted = filtered.melt(id_vars=['Subject ID'], var_name='Attribute', value_name='Value').drop_duplicates()
    att_to_subject_to_vals = defaultdict(lambda: defaultdict(set))
    for i, row in melted.iterrows():
        att_to_subject_to_vals[row['Attribute']][row['Subject ID']].add(row['Value'])

    # define expanded atts as all attributes with more than one value for a given subject
    expanded_atts = []
    for att, subject_to_vals in att_to_subject_to_vals.items():
        max_count = max([len(vals) for vals in subject_to_vals.values()])
        if max_count > 1:
            expanded_atts.append(att)
    if len(expanded_atts) > 0:
        new_rows = []
        for i, row in melted.iterrows():
            if row['Attribute'] in expanded_atts:
                if str(row['Value']) not in ['', '<NA>']:
                    new_rows.append([row['Subject ID'], row['Attribute']+'_'+str(row['Value']), '1'])
            else:
                new_rows.append([row['Subject ID'], row['Attribute'], str(row['Value'])])
        melted = pd.DataFrame(new_rows, columns=['Subject ID', 'Attribute', 'Value'])
        # convert back to wide format
        wdf = melted.pivot(index='Subject ID', columns='Attribute', values='Value').reset_index()
        # wdf = wdf.drop(columns=['Subject ID'])
        
        output_df_var = wdf
    else:
        wdf = df.copy()

    
    output_df_var = wdf
    output_df_var.replace({'<NA>': np.nan}, inplace=True)
    output_df_var.replace({'nan': ''}, inplace=True)
    output_df_var.replace({'1.0': '1'}, inplace=True)
    return output_df_var

def generate_graph_model(df, period_col):
    att_cols = [col for col in df.columns.values if col not in ['Subject ID', period_col]]
    df = df_functions.fix_null_ints(df).astype(str).replace('nan', '').replace('<NA>', '')
    df['Subject ID'] = [str(x) for x in range(1, len(df) + 1)]
    df['Subject ID'] = df['Subject ID'].astype(str)
    pdf = df.copy(deep=True)[[period_col, 'Subject ID'] + att_cols]
    pdf = pdf[pdf[period_col].notna() & pdf['Subject ID'].notna()]
    pdf.rename(columns={period_col : 'Period'}, inplace=True)

    pdf['Period'] = pdf['Period'].astype(str)

    pdf = pd.melt(pdf, id_vars=['Subject ID', 'Period'], value_vars=att_cols, var_name='Attribute Type', value_name='Attribute Value')
    pdf = pdf[pdf['Attribute Value'] != '']
    pdf['Full Attribute'] = pdf.apply(lambda x: str(x['Attribute Type']) + type_val_sep + str(x['Attribute Value']), axis=1)
    pdf = pdf[pdf['Period'] != '']
    return pdf

def compute_attribute_counts(df, pattern, period_col, period):
    atts = pattern.split(' & ')
    fdf = df_functions.fix_null_ints(df).astype(str).replace('nan', '').replace('<NA>', '')
    fdf = fdf[fdf[period_col] == period]
    for att in atts:
        if att == 'Subject ID':
            continue
        ps = att.split(type_val_sep)
        if len(ps) == 2:
            a, v = ps
            fdf = fdf[fdf[a] == v]
        else:
            print(f'Error parsing attribute {att}')
            
    melted = pd.melt(fdf, id_vars=['Subject ID'], value_vars=[c for c in fdf.columns if c not in ['Subject ID', period_col]], var_name='Attribute', value_name='Value')
    melted = melted[melted['Value'] != '']
    melted['AttributeValue'] = melted['Attribute'] + type_val_sep + melted['Value']
    att_counts = melted.groupby('AttributeValue').agg({'Subject ID' : 'nunique'}).rename(columns={'Subject ID' : 'Count'}).sort_values(by='Count', ascending=False).reset_index()
    return att_counts

def create_time_series_df(model, pattern_df):
    record_counter = RecordCounter(model)

    rows = []
    for _, row in pattern_df.iterrows():
        rows.extend(record_counter.create_time_series_rows(row['pattern'].split(' & ')))
    columns = ['period', 'pattern', 'count']
    ts_df = pd.DataFrame(rows, columns=columns)
    return ts_df

def prepare_graph(dynamic_df, mi=False):
    time_to_graph = {}
    dynamic_lcc = set()
    pdf = dynamic_df.copy()
    atts = sorted(pdf['Full Attribute'].unique())
    pdf['Grouping ID'] = str(pdf['Subject ID']) + '@' + str(pdf['Period'])
    
    periods = sorted(pdf['Period'].unique())

    for ix, period in enumerate(periods):
        tdf = pdf.copy()
        tdf = tdf[tdf['Period'] == period]
        tdf['Grouping ID'] = str(tdf['Subject ID']) + '@' + str(tdf['Period'])
        tdf = tdf[['Grouping ID', 'Full Attribute']].groupby('Grouping ID').agg(list).reset_index()
        dedge_df = create_edge_df_from_atts(atts, tdf, mi)
        G, lcc = convert_edge_df_to_graph(dedge_df)
        if ix == 0:
            dynamic_lcc.update(lcc)
        else:
            dynamic_lcc.intersection_update(lcc)
        time_to_graph[period] = G
    return pdf, time_to_graph

def detect_patterns(node_to_centroid, period_embeddings, dynamic_df, min_pattern_count = 5, max_pattern_length = 100) -> tuple[pd.DataFrame, int, int]:
    sorted_nodes = sorted(node_to_centroid.keys())
    record_counter  = RecordCounter(dynamic_df)

    period_shifts = create_period_shifts(node_to_centroid, period_embeddings, dynamic_df)
    used_periods = sorted(dynamic_df['Period'].unique())

    # # for each period, find all pairs of nodes close
    close_node_df, all_pairs, close_pairs = create_close_node_rows(used_periods, period_shifts, sorted_nodes, min_pattern_count, record_counter)

    period_to_patterns = create_period_to_patterns(used_periods, close_node_df, max_pattern_length, min_pattern_count, record_counter)
    # convert to df
    pattern_rows = create_pattern_rows(period_to_patterns, record_counter)
    
    columns = ['period', 'pattern', 'length', 'count', 'mean', 'z_score']
    pattern_df = pd.DataFrame(pattern_rows, columns=columns)

    # Count the number of periods per pattern and merge it into the DataFrame
    detections = pattern_df.groupby('pattern', as_index=False)['period'].count().rename(columns={'period': 'detections'})
    pattern_df = pattern_df.merge(detections, on='pattern')

    # Calculate the overall score
    pattern_df['overall_score'] = (
        pattern_df['z_score'] * 
        pattern_df['length'] * 
        pattern_df['detections'] * 
        np.log1p(pattern_df['count'])  # np.log1p(x) is equivalent to np.log(x + 1)
    )

    # Normalize the overall score
    pattern_df['overall_score'] = pattern_df['overall_score'] / pattern_df['overall_score'].max()
    pattern_df['overall_score'] = pattern_df['overall_score'].round(2)

    # Sort the DataFrame by the overall score in descending order
    pattern_df = pattern_df.sort_values('overall_score', ascending=False)
    return pattern_df, close_pairs, all_pairs

def prepare_for_ai_report(pattern: str, period: str, time_series: pd.DataFrame, attribute_counts: pd.DataFrame, u_prompt: str = user_prompt) -> list[dict[str, str]]:
    variables = {
        'pattern': pattern,
        'period': period,
        'time_series': time_series.to_csv(index=False),
        'attribute_counts': attribute_counts.to_csv(index=False)
    }

    safety_prompt = do_not_harm
    return generate_messages(u_prompt, report_prompt, variables, safety_prompt)
