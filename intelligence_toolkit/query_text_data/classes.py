# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import networkx as nx
import graspologic as gc

class ProcessedChunks:
    def __init__(
        self,
        cid_to_text: dict[int, str],
        text_to_cid: dict[str, int],
        period_concept_graphs: dict[str, nx.Graph],
        hierarchical_communities: gc.partition.HierarchicalCluster,
        community_to_label: dict[int, dict[int, str]],
        concept_to_cids: dict[str, list[int]],
        cid_to_concepts: dict[int, list[str]],
        previous_cid: dict[int, int],
        next_cid: dict[int, int],
        period_to_cids: dict[str, list[int]],
        node_period_counts: dict[str, dict[str, int]],
        edge_period_counts: dict[tuple[str, str], dict[str, int]]
    ):
        """
        Represents the results of processing text chunks into concepts and communities.

        Args:
            cid_to_text (dict[int, str]): A dictionary of chunk IDs to text
            text_to_cid (dict[str, int]): A dictionary of text to chunk IDs
            period_concept_graphs (dict[str, nx.Graph]): A dictionary of period to concept graph
            hierarchical_communities (gc.partition.HierarchicalCluster): A hierarchical community structure
            community_to_label (dict[int, dict[int, str]]): A dictionary of community ID to a dictionary of node ID to label
            concept_to_cids (dict[str, list[int]]): A dictionary of concept to chunk IDs
            cid_to_concepts (dict[int, list[str]]): A dictionary of chunk ID to concepts
            previous_cid (dict[int, int]): A dictionary of chunk ID to previous chunk ID
            next_cid (dict[int, int]): A dictionary of chunk ID to next chunk ID
            period_to_cids (dict[str, list[int]]): A dictionary of period to chunk IDs
            node_period_counts (dict[str, dict[str, int]]): A dictionary of period to node to count
            edge_period_counts (dict[tuple[str, str], dict[str, int]]): A dictionary of period to edge to count
        """
        self.cid_to_text = cid_to_text
        self.text_to_cid = text_to_cid
        self.period_concept_graphs = period_concept_graphs
        self.hierarchical_communities = hierarchical_communities
        self.community_to_label = community_to_label
        self.concept_to_cids = concept_to_cids
        self.cid_to_concepts = cid_to_concepts
        self.previous_cid = previous_cid
        self.next_cid = next_cid
        self.period_to_cids = period_to_cids
        self.node_period_counts = node_period_counts
        self.edge_period_counts = edge_period_counts

    def __repr__(self):
        return f"ProcessedChunks(num_chunks={len(self.cid_to_text.keys())})"
    
class ChunkSearchConfig:
    def __init__(
        self,
        adjacent_test_steps: int,
        community_relevance_tests: int,
        community_ranking_chunks: int,
        relevance_test_batch_size: int,
        relevance_test_budget: int,
        irrelevant_community_restart: int,
        analysis_update_interval = 0: int
    ) -> None:
        """
        Represents the configuration used to search for relevant text chunks.

        Args:
            adjacent_test_steps (int): How many chunks before and after each relevant chunk to test, once the relevance test budget is near or the search process has terminated
            community_relevance_tests (int): How many relevance tests to run on each community in turn
            community_ranking_chunks (int): How many chunks to use to rank communities by relevance
            relevance_test_batch_size (int): How many relevance tests to run in parallel at a time
            relevance_test_budget (int): How many relevance tests are permitted per query. Higher values may provide higher quality results at higher cost
            irrelevant_community_restart (int): When to restart testing communities in relevance order
            analysis_update_interval (int): How many chunks to process before updating the analysis. Use 0 to skip analysis updates
        """
        self.adjacent_test_steps = adjacent_test_steps
        self.community_relevance_tests = community_relevance_tests
        self.community_ranking_chunks = community_ranking_chunks
        self.relevance_test_batch_size = relevance_test_batch_size
        self.relevance_test_budget = relevance_test_budget
        self.irrelevant_community_restart = irrelevant_community_restart
        self.analysis_update_interval = analysis_update_interval

    def __repr__(self):
        return f"ChunkSearchConfig(adjacent_test_steps={self.adjacent_test_steps}, community_relevance_tests={self.community_relevance_tests}, relevance_test_batch_size={self.relevance_test_batch_size}, relevance_test_budget={self.relevance_test_budget}, irrelevant_community_restart={self.irrelevant_community_restart}, analysis_update_interval={self.analysis_update_interval})"
    
class AnswerObject:
    def __init__(
        self,
        extended_answer: str,
        references: list[str],
        referenced_chunks: list[int],
        net_new_sources: int,
    ) -> None:
        """
        Represents the answer to a user query.

        Args:
            extended_answer (str): The extended answer to the user query
            references (list[str]): A list of references used in the answer
            referenced_chunks (list[int]): A list of chunk IDs referenced in the answer
            net_new_sources (int): The number of new sources used in the answer
        """
        self.extended_answer = extended_answer
        self.references = references
        self.referenced_chunks = referenced_chunks
        self.net_new_sources = net_new_sources

    def __repr__(self):
        return f"AnswerObject(extended_answer={self.extended_answer[:100]}, references={len(self.references)}, referenced_chunks={len(self.referenced_chunks)}, net_new_sources={self.net_new_sources})"
    