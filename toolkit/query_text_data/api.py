import toolkit.query_text_data.input_processor as input_processor
import toolkit.query_text_data.helper_functions as helper_functions
import toolkit.query_text_data.relevance_assessor as relevance_assessor
import toolkit.query_text_data.graph_builder as graph_builder
import toolkit.query_text_data.answer_builder as answer_builder
from toolkit.query_text_data.classes import ChunkSearchConfig, AnswerConfig, AnswerObject
from toolkit.AI.base_embedder import BaseEmbedder
import networkx as nx
from enum import Enum

class QueryTextDataStage(Enum):
    INITIAL = 0
    CHUNKS_CREATED = 1
    CHUNKS_PROCESSED = 2
    CHUNKS_EMBEDDED = 3
    CHUNKS_MINED = 4
    QUESTION_ANSWERED = 5

class QueryTextData:
    def __init__(self):
        self.reset_workflow()
        
    def set_ai_config(self, ai_configuration, embedding_cache):
        self.ai_configuration = ai_configuration
        self.embedding_cache = embedding_cache

    def reset_workflow(self):
        self.stage = QueryTextDataStage.INITIAL
        self.label_to_chunks = None
        self.processed_chunks = None
        self.cid_to_vector = None
        self.question = None
        self.chunk_search_config = None
        self.relevant_cids = None
        self.search_summary = None
        self.answer_config = None
        self.answer_object = None
        self.level_to_label_to_network = None

    def set_embedder(self, text_embedder) -> BaseEmbedder:
       self.text_embedder = text_embedder

    def process_data_from_df(self, df, label):
        self.label_to_chunks = input_processor.convert_df_to_chunks(df, label)
        self.stage = QueryTextDataStage.CHUNKS_CREATED
        return self.label_to_chunks
    
    def process_data_from_files(self, input_file_bytes, analysis_window_size: input_processor.PeriodOption, callbacks=[]):
        self.label_to_chunks = input_processor.convert_file_bytes_to_chunks(input_file_bytes, analysis_window_size, callbacks)
        self.stage = QueryTextDataStage.CHUNKS_CREATED
        return self.label_to_chunks

    def process_text_chunks(self, max_cluster_size=25, min_edge_weight=2, min_node_degree=1, callbacks=[]):
        self.processed_chunks = input_processor.process_chunks(
            self.label_to_chunks,
            max_cluster_size,
            min_edge_weight,
            min_node_degree,
            callbacks=callbacks
        )
        self.stage = QueryTextDataStage.CHUNKS_PROCESSED
        return self.processed_chunks

    async def embed_text_chunks(self, callbacks=[]):
        self.cid_to_vector = await helper_functions.embed_texts(
            self.processed_chunks.cid_to_text,
            self.text_embedder,
            cache_data=self.embedding_cache,
            callbacks=callbacks
        )
        self.stage = QueryTextDataStage.CHUNKS_EMBEDDED
        return self.cid_to_vector
    
    async def detect_relevant_text_chunks(
        self,
        question: str,
        chunk_search_config: ChunkSearchConfig,
        chunk_progress_callback=None,
        chunk_callback=None
    ) -> tuple[list[int], str]:
        self.question = question
        self.chunk_search_config = chunk_search_config
        self.relevant_cids, self.search_summary = await relevance_assessor.detect_relevant_chunks(
            ai_configuration=self.ai_configuration,
            question=self.question,
            processed_chunks=self.processed_chunks,
            cid_to_vector=self.cid_to_vector,
            embedder=self.text_embedder,
            embedding_cache=self.embedding_cache,
            chunk_search_config=self.chunk_search_config,
            chunk_progress_callback=chunk_progress_callback,
            chunk_callback=chunk_callback

        )
        self.stage = QueryTextDataStage.CHUNKS_MINED
        return self.relevant_cids, self.search_summary
    
    async def answer_question_with_relevant_chunks(self, answer_config: AnswerConfig):
        self.answer_config = answer_config
        self.answer_object: AnswerObject = await answer_builder.answer_question(
            self.ai_configuration,
            self.question,
            self.processed_chunks,
            self.relevant_cids,
            self.cid_to_vector,
            self.text_embedder,
            self.embedding_cache,
            self.answer_config
        )
        self.stage = QueryTextDataStage.QUESTION_ANSWERED
        return self.answer_object
    
    def build_concept_community_graph(self) -> dict[int, dict[str, nx.Graph]]:
        self.level_to_label_to_network = graph_builder.build_meta_graph(self.processed_chunks.period_concept_graphs['ALL'], self.processed_chunks.hierarchical_communities)
        return self.level_to_label_to_network
    
    def __repr__(self):
        return f"QueryTextData()"