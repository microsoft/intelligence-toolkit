# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import re
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from toolkit.record_matching.config import DEFAULT_COLUMNS_DONT_CONVERT


def convert_to_sentences(
    merged_dataframe: pl.DataFrame,
    skip_columns: list[str] | None = DEFAULT_COLUMNS_DONT_CONVERT,
) -> list[str]:
    sentences = []
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
        sentences.append(sentence.strip())
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
    max_record_distance: int | None = 0.05,
) -> defaultdict[Any, list]:
    near_map = defaultdict(list)
    for ix in range(len(all_sentences)):
        near_is = indices[ix][1:]
        near_ds = distances[ix][1:]
        nearest = zip(near_is, near_ds, strict=False)
        for near_i, near_d in nearest:
            if near_d <= max_record_distance:
                near_map[ix].append(near_i)
                near_map[ix].append(near_i)

    return near_map


def build_sentence_pair_scores(
    near_map: defaultdict[Any, list], merged_df: pl.DataFrame
) -> list:
    matching_sentence_pair_scores = []
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

            matching_sentence_pair_scores.append(
                (
                    ix,
                    nx,
                    score,
                )
            )
    return matching_sentence_pair_scores
