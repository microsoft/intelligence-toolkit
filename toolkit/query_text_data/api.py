# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import toolkit.query_text_data.input_processor as input_processor
import toolkit.query_text_data.helper_functions as helper_functions
import toolkit.query_text_data.relevance_assessor as relevance_assessor
import toolkit.query_text_data.query_rewriter as query_rewriter
import toolkit.query_text_data.graph_builder as graph_builder
import toolkit.query_text_data.answer_builder as answer_builder
import toolkit.query_text_data.prompts as prompts
from toolkit.query_text_data.classes import ProcessedChunks, ChunkSearchConfig, AnswerConfig, AnswerObject
from toolkit.AI.base_embedder import BaseEmbedder
from toolkit.AI.openai_configuration import OpenAIConfiguration
from toolkit.AI.client import OpenAIClient
import toolkit.AI.utils as utils
import networkx as nx
import pandas as pd
from enum import Enum

class QueryTextDataStage(Enum):
    """
    Enum for the stages of the QueryTextData workflow.
    
    Attributes:
        INITIAL: The initial stage of the workflow.
        CHUNKS_CREATED: The chunks have been created.
        CHUNKS_PROCESSED: The chunks have been processed.
        CHUNKS_EMBEDDED: The chunks have been embedded.
        CHUNKS_MINED: The chunks have been mined.
        QUESTION_ANSWERED: The query has been answered.
    """
    INITIAL = 0
    CHUNKS_CREATED = 1
    CHUNKS_PROCESSED = 2
    CHUNKS_EMBEDDED = 3
    CHUNKS_MINED = 4
    QUESTION_ANSWERED = 5

class QueryTextData:
    def __init__(self) -> None:
        self.reset_workflow()
        
    def set_ai_config(
            self,
            ai_configuration: OpenAIConfiguration,
            embedding_cache: str
        ) -> None:
        """
        Set the AI configuration and embedding cache for the workflow.

        Args:
            ai_configuration (OpenAIConfiguration): The OpenAI configuration
            embedding_cache (str): The embedding cache
        """
        self.ai_configuration = ai_configuration
        self.embedding_cache = embedding_cache

    def reset_workflow(self) -> None:
        """
        Resets the workflow to its initial state.
        """
        self.stage = QueryTextDataStage.INITIAL
        self.label_to_chunks = None
        self.processed_chunks = None
        self.cid_to_vector = None
        self.query = None
        self.expanded_query = None
        self.chunk_search_config = None
        self.relevant_cids = None
        self.search_summary = None
        self.answer_config = None
        self.answer_object = None
        self.level_to_label_to_network = None

    def set_embedder(
            self, 
            text_embedder: BaseEmbedder
        ) -> None:
        """
        Set the text embedder for the workflow.
        
        Args:
            text_embedder (BaseEmbedder): The text embedder
        """
        self.text_embedder = text_embedder

    def process_data_from_df(
            self,
            df: pd.DataFrame,
            label: str
        ) -> dict[str, list[str]]:
        """
        Process data from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame
            label (str): The label (e.g., filename) used as the prefix for the chunk names

        Returns:
            dict[str, list[str]]: The label to chunks mapping
        """
        self.label_to_chunks = input_processor.convert_df_to_chunks(df, label)
        self.stage = QueryTextDataStage.CHUNKS_CREATED
        return self.label_to_chunks
    
    def process_data_from_files(
            self,
            input_file_bytes: bytes,
            analysis_window_size: input_processor.PeriodOption=input_processor.PeriodOption.NONE,
            callbacks: list=[]
        ) -> dict[str, list[str]]:
        """
        Process data from files.

        Args:
            input_file_bytes (bytes): The input file bytes
            analysis_window_size (input_processor.PeriodOption): The analysis window size
            callbacks (list): The list of callbacks

        Returns:
            dict[str, list[str]]: The label to chunks mapping
        """
        self.label_to_chunks = input_processor.convert_file_bytes_to_chunks(input_file_bytes, analysis_window_size, callbacks)
        self.stage = QueryTextDataStage.CHUNKS_CREATED
        return self.label_to_chunks

    def process_text_chunks(
            self,
            max_cluster_size: int=25,
            min_edge_weight: int=2,
            min_node_degree: int=1,
            callbacks=[]
        ) -> ProcessedChunks:
        """
        Process text chunks by extracting noun-phrase coooccurrences into a concept graph.

        Args:
            max_cluster_size (int): The maximum cluster size
            min_edge_weight (int): The minimum edge weight
            min_node_degree (int): The minimum node degree
            callbacks (list): The list of callbacks

        Returns:
            ProcessedChunks: The processed chunks
        """
        self.processed_chunks = input_processor.process_chunks(
            self.label_to_chunks,
            max_cluster_size,
            min_edge_weight,
            min_node_degree,
            callbacks=callbacks
        )
        self.stage = QueryTextDataStage.CHUNKS_PROCESSED
        return self.processed_chunks

    async def embed_text_chunks(
            self,
            callbacks: list=[]
        ) -> dict[int, list[float]]:
        """
        Embed text chunks.

        Args:
            callbacks (list): The list of callbacks

        Returns:
            dict[int, list[float]]: The chunk ID to vector mapping
        """
        self.cid_to_vector = await helper_functions.embed_texts(
            self.processed_chunks.cid_to_text,
            self.text_embedder,
            cache_data=self.embedding_cache,
            callbacks=callbacks
        )
        self.stage = QueryTextDataStage.CHUNKS_EMBEDDED
        return self.cid_to_vector
    
    async def anchor_query_to_concepts(
        self,
        query: str,
        top_concepts: int=100
    ) -> str:
        """
        Anchor the query to the top concepts in the graph.
        """
        anchored_query = await query_rewriter.rewrite_query(
            self.ai_configuration,
            query,
            self.processed_chunks.period_concept_graphs['ALL'],
            top_concepts
        )
        return anchored_query

    async def detect_relevant_text_chunks(
        self,
        query: str,
        expanded_query: str,
        chunk_search_config: ChunkSearchConfig,
        chunk_progress_callback=None,
        chunk_callback=None
    ) -> tuple[list[int], str]:
        """
        Detect relevant text chunks.

        Args:
            query (str): The query
            chunk_search_config (ChunkSearchConfig): The chunk search configuration
            chunk_progress_callback: The chunk progress callback
            chunk_callback: The chunk callback

        Returns:
            tuple[list[int], str]: The relevant chunk IDs and search summary
        """
        self.query = query
        self.expanded_query = expanded_query
        self.chunk_search_config = chunk_search_config
        self.relevant_cids, self.search_summary = await relevance_assessor.detect_relevant_chunks(
            ai_configuration=self.ai_configuration,
            query=self.expanded_query,
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
    
    async def answer_query_with_relevant_chunks(
            self,
            answer_config: AnswerConfig
        ) -> AnswerObject:
        """
        Answer a query with relevant chunks.

        Args:
            answer_config (AnswerConfig): The answer configuration
            
        Returns:
            AnswerObject: The answer object
        """
        self.answer_config = answer_config
        self.answer_object: AnswerObject = await answer_builder.answer_query(
            self.ai_configuration,
            self.query,
            self.expanded_query,
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
        """
        Build the concept community graph.

        Returns:
            dict[int, dict[str, nx.Graph]]: The community level to community label to community network mapping
        """
        self.level_to_label_to_network = graph_builder.build_meta_graph(
            self.processed_chunks.period_concept_graphs['ALL'],
            self.processed_chunks.hierarchical_communities
        )
        return self.level_to_label_to_network
    
    def condense_answer(
        self,
        ai_instructions=prompts.user_prompt,
        callbacks=[]
    ) -> str:
        """
        Condense the answer.

        Args:
            ai_instructions: The AI instructions
            callbacks: The list of callbacks
        """
        variables = {
            "query": self.query,
            "answer": self.answer_object.extended_answer,
        }
        messages = utils.generate_messages(
            ai_instructions,
            prompts.list_prompts['report_prompt'],
            variables,
            prompts.list_prompts['safety_prompt']
        )
        self.condensed_answer = OpenAIClient(self.ai_configuration).generate_chat(messages, callbacks=callbacks)
        return self.condensed_answer

    def prepare_for_new_query(self) -> None:
        """
        Prepare for a new query.
        """
        self.query = None
        self.expanded_query = None
        self.chunk_search_config = None
        self.relevant_cids = None
        self.search_summary = None
        self.answer_config = None
        self.answer_object = None
        self.level_to_label_to_network = None
        self.stage = QueryTextDataStage.CHUNKS_EMBEDDED

    def prepare_for_new_answer(self) -> None:
        """
        Prepare for a new query.
        """
        self.answer_config = None
        self.answer_object = None
        self.stage = QueryTextDataStage.CHUNKS_MINED

    def __repr__(self):
        return f"QueryTextData()"