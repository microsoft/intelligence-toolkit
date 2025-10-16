# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl
import pytest

from intelligence_toolkit.AI.classes import LLMCallback
from intelligence_toolkit.match_entity_records.api import MatchEntityRecords
from intelligence_toolkit.match_entity_records.classes import (
    AttributeToMatch,
    RecordsModel,
)


class TestMatchEntityRecords:
    @pytest.fixture()
    def api_instance(self) -> MatchEntityRecords:
        """Create a MatchEntityRecords instance for testing."""
        return MatchEntityRecords()

    @pytest.fixture()
    def sample_dataframe(self) -> pl.DataFrame:
        """Create a sample dataframe for testing."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Entity A", "Entity B", "Entity C"],
                "attribute1": ["value1", "value2", "value3"],
                "attribute2": ["val1", "val2", "val3"],
            }
        )

    @pytest.fixture()
    def sample_model(self, sample_dataframe) -> RecordsModel:
        """Create a sample RecordsModel for testing."""
        return RecordsModel(
            dataframe=sample_dataframe,
            dataframe_name="test_dataset",
            id_column="id",
            name_column="name",
            columns=["attribute1", "attribute2"],
        )

    @pytest.fixture()
    def populated_api(self, api_instance, sample_model) -> MatchEntityRecords:
        """Create a populated API instance with data."""
        api_instance.add_df_to_model(sample_model)
        return api_instance


class TestTotalRecords(TestMatchEntityRecords):
    def test_total_records_empty(self, api_instance) -> None:
        """Test total_records with no data."""
        assert api_instance.total_records == 0

    def test_total_records_single_dataset(self, populated_api) -> None:
        """Test total_records with one dataset."""
        assert populated_api.total_records == 3

    def test_total_records_multiple_datasets(self, api_instance, sample_dataframe) -> None:
        """Test total_records with multiple datasets."""
        # Clear existing data first
        api_instance.clear_model_dfs()
        
        model1 = RecordsModel(
            dataframe=sample_dataframe,
            dataframe_name="dataset1",
            id_column="id",
            name_column="name",
            columns=["attribute1"],
        )
        model2 = RecordsModel(
            dataframe=sample_dataframe.head(2),
            dataframe_name="dataset2",
            id_column="id",
            name_column="name",
            columns=["attribute1"],
        )
        api_instance.add_df_to_model(model1)
        api_instance.add_df_to_model(model2)
        assert api_instance.total_records == 5  # 3 + 2


class TestAttributeOptions(TestMatchEntityRecords):
    def test_attribute_options_empty(self, api_instance) -> None:
        """Test attribute_options with no data."""
        # Clear any existing data
        api_instance.clear_model_dfs()
        options = api_instance.attribute_options
        assert isinstance(options, list)
        assert len(options) == 0

    def test_attribute_options_populated(self, populated_api) -> None:
        """Test attribute_options with data."""
        options = populated_api.attribute_options
        assert isinstance(options, list)
        assert len(options) > 0
        # Options should be in format "column::dataset"
        for option in options:
            assert "::" in option


class TestIntegratedResults(TestMatchEntityRecords):
    def test_integrated_results_empty(self, api_instance) -> None:
        """Test integrated_results with empty data creates empty dataframe."""
        # Initialize with empty dataframes to avoid panic
        api_instance.evaluations_df = pl.DataFrame(
            schema={"Group ID": pl.Int64, "Relatedness": pl.Float64, "Explanation": pl.Utf8}
        )
        api_instance.matches_df = pl.DataFrame(
            schema={"Group ID": pl.Int64, "Entity name": pl.Utf8, "Dataset": pl.Utf8}
        )
        result = api_instance.integrated_results
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_integrated_results_with_data(self, api_instance) -> None:
        """Test integrated_results with matches and evaluations."""
        # Create sample matches_df
        api_instance.matches_df = pl.DataFrame(
            {
                "Group ID": [1, 2],
                "Entity name": ["A", "B"],
                "Dataset": ["test", "test"],
            }
        )
        # Create sample evaluations_df
        api_instance.evaluations_df = pl.DataFrame(
            {
                "Group ID": [1, 2],
                "Relatedness": [8, 6],
                "Explanation": ["Similar", "Somewhat similar"],
            }
        )
        result = api_instance.integrated_results
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()
        assert "Group ID" in result.columns
        assert "Relatedness" in result.columns


class TestAddDfToModel(TestMatchEntityRecords):
    def test_add_df_to_model_basic(self, api_instance, sample_model) -> None:
        """Test adding a dataframe to the model."""
        result = api_instance.add_df_to_model(sample_model)
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()
        assert "test_dataset" in api_instance.model_dfs

    def test_add_df_to_model_auto_name(self, api_instance, sample_dataframe) -> None:
        """Test adding a dataframe without a name."""
        # Clear existing data first
        api_instance.clear_model_dfs()
        
        model = RecordsModel(
            dataframe=sample_dataframe,
            dataframe_name="",
            id_column="id",
            name_column="name",
            columns=["attribute1"],
        )
        result = api_instance.add_df_to_model(model)
        assert isinstance(result, pl.DataFrame)
        assert len(api_instance.model_dfs) == 1

    def test_add_df_to_model_with_max_rows(self, api_instance, sample_model) -> None:
        """Test adding a dataframe with max_rows limit."""
        api_instance.max_rows_to_process = 2
        result = api_instance.add_df_to_model(sample_model)
        assert len(result) <= 2


class TestBuildModelDf(TestMatchEntityRecords):
    @pytest.fixture()
    def attributes_list(self) -> list[AttributeToMatch]:
        """Create sample attributes list."""
        return [
            {"label": "Attribute 1", "columns": ["attribute1::test_dataset"]},
            {"label": "Attribute 2", "columns": ["attribute2::test_dataset"]},
        ]

    def test_build_model_df_basic(self, populated_api, attributes_list) -> None:
        """Test building model dataframe."""
        result = populated_api.build_model_df(attributes_list)
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()
        # Check that the model_df was set correctly
        assert hasattr(populated_api, "model_df")
        assert "Dataset" in result.columns

    def test_build_model_df_creates_sentences(
        self, populated_api, attributes_list
    ) -> None:
        """Test that building model df creates sentence vector data."""
        populated_api.build_model_df(attributes_list)
        assert hasattr(populated_api, "sentences_vector_data")
        assert isinstance(populated_api.sentences_vector_data, list)


class TestEmbedSentences(TestMatchEntityRecords):
    @pytest.mark.asyncio()
    async def test_embed_sentences(self, populated_api) -> None:
        """Test embedding sentences."""
        # Setup mock embedder
        mock_embedder = AsyncMock()
        mock_embedder.embed_store_many = AsyncMock(
            return_value=[
                {"text": "sentence1", "vector": [0.1, 0.2, 0.3]},
                {"text": "sentence2", "vector": [0.4, 0.5, 0.6]},
            ]
        )
        populated_api.embedder = mock_embedder
        populated_api.cache_embeddings = True

        # Setup sentences_vector_data
        populated_api.sentences_vector_data = [
            {"text": "sentence1"},
            {"text": "sentence2"},
        ]

        await populated_api.embed_sentences()

        assert hasattr(populated_api, "all_sentences")
        assert hasattr(populated_api, "embeddings")
        assert len(populated_api.all_sentences) == 2
        assert len(populated_api.embeddings) == 2
        assert isinstance(populated_api.embeddings[0], np.ndarray)
        mock_embedder.embed_store_many.assert_called_once()


class TestDetectRecordGroups(TestMatchEntityRecords):
    def test_detect_record_groups(self, populated_api) -> None:
        """Test detecting record groups."""
        # Setup required data with enough embeddings for default n_neighbors (50)
        # Use 60 embeddings to be safe
        embeddings_list = [np.random.rand(3) for _ in range(60)]
        populated_api.embeddings = embeddings_list
        populated_api.all_sentences = [f"sent{i}" for i in range(60)]
        
        # Create model_df with matching number of rows
        entity_ids = [str(i) for i in range(60)]
        entity_names = [f"Entity{i}" for i in range(60)]
        datasets = ["test"] * 60
        unique_ids = [f"{i}::test" for i in range(60)]
        
        populated_api.model_df = pl.DataFrame(
            {
                "Entity ID": entity_ids,
                "Entity name": entity_names,
                "Dataset": datasets,
                "Unique ID": unique_ids,
            }
        )

        result = populated_api.detect_record_groups(
            pair_embedding_threshold=80, pair_jaccard_threshold=50
        )

        assert isinstance(result, pl.DataFrame)
        assert hasattr(populated_api, "matches_df")


class TestEvaluateGroups(TestMatchEntityRecords):
    @pytest.mark.asyncio()
    async def test_evaluate_groups_basic(self, populated_api) -> None:
        """Test evaluating groups."""
        # Setup model_df
        populated_api.model_df = pl.DataFrame(
            {
                "Entity ID": ["1", "2"],
                "Entity name": ["A", "B"],
                "Dataset": ["test", "test"],
                "Name similarity": [0.9, 0.8],
                "Attribute 1": ["val1", "val2"],
            }
        )

        # Mock AI configuration and client
        populated_api.ai_configuration = MagicMock()
        mock_response = "1,9,Very similar entities\n2,7,Somewhat related"

        with patch(
            "intelligence_toolkit.match_entity_records.api.OpenAIClient"
        ) as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate_chat_async = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await populated_api.evaluate_groups()

            assert isinstance(result, str)
            assert hasattr(populated_api, "evaluations_df")
            assert isinstance(populated_api.evaluations_df, pl.DataFrame)

    @pytest.mark.asyncio()
    async def test_evaluate_groups_with_callbacks(self, populated_api) -> None:
        """Test evaluating groups with callbacks."""
        populated_api.model_df = pl.DataFrame(
            {
                "Entity ID": ["1"],
                "Entity name": ["A"],
                "Dataset": ["test"],
                "Name similarity": [0.9],
            }
        )

        populated_api.ai_configuration = MagicMock()
        mock_response = "1,9,Similar"
        callbacks = [MagicMock(spec=LLMCallback)]

        with patch(
            "intelligence_toolkit.match_entity_records.api.OpenAIClient"
        ) as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate_chat_async = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await populated_api.evaluate_groups(callbacks=callbacks)
            assert isinstance(result, str)


class TestClearModelDfs(TestMatchEntityRecords):
    def test_clear_model_dfs_empty(self, api_instance) -> None:
        """Test clearing model dfs when empty."""
        api_instance.clear_model_dfs()
        assert len(api_instance.model_dfs) == 0

    def test_clear_model_dfs_populated(self, populated_api) -> None:
        """Test clearing model dfs with data."""
        assert len(populated_api.model_dfs) > 0
        populated_api.clear_model_dfs()
        assert len(populated_api.model_dfs) == 0
