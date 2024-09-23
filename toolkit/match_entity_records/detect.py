# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import re
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from toolkit.AI.classes import VectorData
from toolkit.AI.utils import hash_text
from toolkit.match_entity_records.config import (
    DEFAULT_COLUMNS_DONT_CONVERT,
    DEFAULT_MAX_RECORD_DISTANCE,
    DEFAULT_SENTENCE_PAIR_JACCARD_THRESHOLD,
)


def convert_to_sentences(
    merged_dataframe: pl.DataFrame,
    skip_columns: list[str] | None = DEFAULT_COLUMNS_DONT_CONVERT,
) -> list[VectorData]:
    sentences: list[VectorData] = []
    skip_columns = skip_columns or []
    cols = merged_dataframe.columns
    for row in merged_dataframe.iter_rows(named=True):
        sentence = ""
        for field in cols:
            if field not in skip_columns:
                val = str(row[field]).upper()
                if val == "NAN":
                    val = ""
                sentence += field.upper() + ": " + val + "; "
        sentence = sentence.strip()
        text_hashed = hash_text(sentence)
        sentences.append({"text": sentence, "hash": text_hashed})

    return sentences


def build_nearest_neighbors(
    embeddings: np.array,
    n_neighbors: int = 50,
    leaf_size: int = 20,
    metric: str = "cosine",
) -> tuple[np.array, np.array]:
    if len(embeddings) < n_neighbors:
        msg = f"Number of neighbors ({n_neighbors}) is greater than number of embeddings ({len(embeddings)})"
        raise ValueError(msg)

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        n_jobs=1,
        algorithm="auto",
        leaf_size=leaf_size,
        metric=metric,
    ).fit(embeddings)

    distances, indices = nbrs.kneighbors(embeddings)
    return distances, indices


def build_near_map(
    distances: np.array,
    indices: np.array,
    all_sentences: list[str],
    max_record_distance: int | None = DEFAULT_MAX_RECORD_DISTANCE,
) -> defaultdict[Any, list]:
    near_map = defaultdict(list)
    for ix in range(len(all_sentences)):
        near_is = indices[ix][1:]
        near_ds = distances[ix][1:]
        nearest = zip(near_is, near_ds, strict=False)
        for near_i, near_d in nearest:
            if near_d <= max_record_distance:
                near_map[ix].append(near_i)

    return near_map


def build_sentence_pair_scores(
    near_map: defaultdict[Any, list], merged_df: pl.DataFrame
) -> list:
    sentence_pair_scores = []
    for ix, nx_list in near_map.items():
        ixrec = merged_df.row(ix, named=True)
        for nx in nx_list:
            nxrec = merged_df.row(nx, named=True)
            ixn = ixrec["Entity name"].upper()
            nxn = nxrec["Entity name"].upper()

            ixn_c = re.sub(r"[^\w\s]", "", ixn)
            nxn_c = re.sub(r"[^\w\s]", "", nxn)
            N = 3
            igrams = {ixn_c[i : i + N] for i in range(len(ixn_c) - N + 1)}
            ngrams = {nxn_c[i : i + N] for i in range(len(nxn_c) - N + 1)}
            inter = len(igrams.intersection(ngrams))
            union = len(igrams.union(ngrams))
            score = inter / union if union > 0 else 0

            sentence_pair_scores.append(
                (
                    ix,
                    nx,
                    score,
                )
            )
    return sentence_pair_scores


def build_matches(
    sentence_pair_scores,
    merged_df: pl.DataFrame,
    sentence_pair_jaccard_threshold: float = DEFAULT_SENTENCE_PAIR_JACCARD_THRESHOLD,
) -> tuple[dict, set, dict]:
    entity_to_group = {}
    group_id = 0
    matches = set()
    pair_to_match = {}

    for ix, nx, score in sorted(
        sentence_pair_scores,
        key=lambda x: x[2],
        reverse=True,
    ):
        if score < sentence_pair_jaccard_threshold:
            continue

        ixrec = merged_df.row(ix, named=True)
        nxrec = merged_df.row(nx, named=True)
        ixn = ixrec["Entity name"]
        nxn = nxrec["Entity name"]
        ixp = ixrec["Dataset"]
        nxp = nxrec["Dataset"]

        ix_id = f"{ixn}::{ixp}"
        nx_id = f"{nxn}::{nxp}"

        if ix_id in entity_to_group and nx_id in entity_to_group:
            ig = entity_to_group[ix_id]
            ng = entity_to_group[nx_id]
            if ig != ng:
                for k, v in list(entity_to_group.items()):
                    if v == ig:
                        entity_to_group[k] = ng
        elif ix_id in entity_to_group:
            entity_to_group[nx_id] = entity_to_group[ix_id]
        elif nx_id in entity_to_group:
            entity_to_group[ix_id] = entity_to_group[nx_id]
        else:
            entity_to_group[ix_id] = group_id
            entity_to_group[nx_id] = group_id
            group_id += 1

        matches.add((entity_to_group[ix_id], *list(merged_df.row(ix))))
        matches.add((entity_to_group[nx_id], *list(merged_df.row(nx))))

        pair_to_match[tuple(sorted([ix_id, nx_id]))] = score

    return entity_to_group, matches, pair_to_match


def _calculate_mean_score(pair_to_match: dict, entity_to_group: dict) -> dict:
    group_to_scores = defaultdict(list)

    for (ix_id, nx_id), score in pair_to_match.items():
        if (
            ix_id in entity_to_group
            and nx_id in entity_to_group
            and entity_to_group[ix_id] == entity_to_group[nx_id]
        ):
            group_to_scores[entity_to_group[ix_id]].append(score)

    group_to_mean_similarity = {}
    for group, scores in group_to_scores.items():
        group_to_mean_similarity[group] = (
            sum(scores) / len(scores) if len(scores) > 0 else 1 # Must be the same value
        )
    return group_to_mean_similarity


def build_matches_dataset(
    matches_df: pl.DataFrame, pair_to_match: dict, entity_to_group: dict
) -> pl.DataFrame:
    if matches_df.is_empty():
        return matches_df

    group_to_size = (
        matches_df.group_by("Group ID")
        .agg(pl.count("Entity ID").alias("Size"))
        .to_dict()
    )
    group_to_size = dict(
        zip(
            group_to_size["Group ID"],
            group_to_size["Size"],
            strict=False,
        )
    )
    matches_df = matches_df.with_columns(
        matches_df["Group ID"].replace(group_to_size).alias("Group size")
    )

    order_first_columns = [
        "Group ID",
        "Group size",
        "Entity name",
        "Dataset",
        "Entity ID",
    ]
    remaining_columns = [c for c in matches_df.columns if c not in order_first_columns]
    new_column_order = order_first_columns + remaining_columns
    matches_df = matches_df.with_columns([matches_df[c] for c in new_column_order])

    # keep only groups larger than 1
    matches_df = matches_df.with_columns(
        matches_df["Entity ID"]
        .map_elements(lambda x: x.split("::")[0])
        .alias("Entity ID")
    ).filter(pl.col("Group size") > 1)

    # iterate over groups, calculating mean score
    group_to_mean_similarity = _calculate_mean_score(pair_to_match, entity_to_group)

    if matches_df.is_empty():
        return matches_df

    matches_df = matches_df.with_columns(
        matches_df["Group ID"]
        .map_elements(lambda x: group_to_mean_similarity.get(x, 0))
        .alias("Name similarity")
    )

    return matches_df.sort(by=["Name similarity", "Group ID"])


def build_attributes_dataframe(
    matching_dfs: dict[pl.DataFrame], atts_to_datasets: defaultdict[dict]
) -> pl.DataFrame:
    if not matching_dfs:
        return pl.DataFrame()

    aligned_dfs = []
    for dataset, merged_df in matching_dfs.items():
        if dataset not in atts_to_datasets:
            continue
        rdf = merged_df.clone()
        rdf = rdf.rename(atts_to_datasets[dataset])
        # drop columns that are not in atts_to_datasets
        for col in matching_dfs[dataset].columns:
            if col not in rdf.columns:
                continue
            if col not in atts_to_datasets[dataset] and col not in [
                "Entity ID",
                "Entity name",
            ]:
                rdf = rdf.drop(col)
                continue

            for dataset1 in atts_to_datasets:
                if dataset1 not in atts_to_datasets and col not in [
                    "Entity ID",
                    "Entity name",
                ]:
                    rdf = rdf.drop(col)

        rdf = rdf.with_columns(pl.lit(dataset).alias("Dataset"))
        rdf = rdf.select(sorted(rdf.columns))
        aligned_dfs.append(rdf)

    string_dfs = []
    for merged_df in aligned_dfs:
        for col in merged_df.columns:
            merged_df = merged_df.with_columns(pl.col(col).cast(pl.Utf8))
        string_dfs.append(merged_df)

    return pl.concat(string_dfs).filter(pl.col("Entity name") != "")