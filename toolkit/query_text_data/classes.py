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
        irrelevant_community_restart: int
    ):
        self.adjacent_test_steps = adjacent_test_steps
        self.community_relevance_tests = community_relevance_tests
        self.community_ranking_chunks = community_ranking_chunks
        self.relevance_test_batch_size = relevance_test_batch_size
        self.relevance_test_budget = relevance_test_budget
        self.irrelevant_community_restart = irrelevant_community_restart

    def __repr__(self):
        return f"ChunkSearchConfig(adjacent_test_steps={self.adjacent_test_steps}, community_relevance_tests={self.community_relevance_tests}, relevance_test_batch_size={self.relevance_test_batch_size}, relevance_test_budget={self.relevance_test_budget}, irrelevant_community_restart={self.irrelevant_community_restart})"
    
class AnswerConfig:
    def __init__(
        self,
        target_chunks_per_cluster: int,
        extract_claims: bool,
        claim_search_depth: int,
    ) -> None:
        self.target_chunks_per_cluster = target_chunks_per_cluster
        self.extract_claims = extract_claims
        self.claim_search_depth = claim_search_depth

    def __repr__(self):
        return f"AnswerConfig(target_chunks_per_cluster={self.target_chunks_per_cluster}, extract_claims={self.extract_claims}, claim_search_depth={self.claim_search_depth})"
    
class AnswerObject:
    def __init__(
        self,
        extended_answer,
        references,
        referenced_chunks,
        net_new_sources,
    ) -> None:
        self.extended_answer = extended_answer
        self.references = references
        self.referenced_chunks = referenced_chunks
        self.net_new_sources = net_new_sources

    def __repr__(self):
        return f"AnswerObject(extended_answer={self.extended_answer[:100]}, references={len(self.references)}, referenced_chunks={len(self.referenced_chunks)}, net_new_sources={self.net_new_sources})"
    