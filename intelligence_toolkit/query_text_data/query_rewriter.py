# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import networkx as nx

import intelligence_toolkit.AI.utils as utils
import intelligence_toolkit.query_text_data.prompts as prompts


async def rewrite_query(ai_configuration, query, concept_graph, top_concepts):
    """
    Rewrite a user query to better match the concepts contained in the dataset.

    The output query should retain all the key phrases from the input query, but may expand on them with additional concepts and phrasing to better match relevant concepts in the dataset. If no concepts are relevant, the output query should be the same as the input query.

    Args:
        query (str): The user query to rewrite.
        concept_graph (nx.Graph): The concept graph representing the dataset.
        top_concepts (int): The number of top concepts to consider.

    Returns:
        str: The rewritten query.
    """
    concepts = sorted(concept_graph.degree(), key=lambda x: x[1], reverse=True)
    concepts = [c for c in concepts if c[0] != "dummynode"]

    concepts = concepts[:top_concepts]
    concepts_str = ", ".join([concept for concept, _ in concepts])
    messages = utils.prepare_messages(
        prompts.query_anchoring_prompt, {"query": query, "concepts": concepts_str}
    )
    return await utils.generate_text_async(ai_configuration, messages, stream=False)