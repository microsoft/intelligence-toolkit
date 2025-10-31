# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import polars as pl
import pytest

from intelligence_toolkit.detect_entity_networks.api import DetectEntityNetworks
from intelligence_toolkit.detect_entity_networks.classes import (
    FlagAggregatorType,
    SummaryData,
)
from intelligence_toolkit.detect_entity_networks.explore_networks import (
    build_network_from_entities,
)
from intelligence_toolkit.detect_entity_networks.config import ENTITY_LABEL
from intelligence_toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR


class TestDetectEntityNetworks:
    @pytest.fixture()
    def api_instance(self) -> DetectEntityNetworks:
        """Create a DetectEntityNetworks instance for testing."""
        return DetectEntityNetworks()

    @pytest.fixture()
    def sample_dataframe(self) -> pl.DataFrame:
        """Create a sample dataframe for testing."""
        return pl.DataFrame(
            {
                "entity_id": ["E1", "E2", "E3", "E4"],
                "phone": ["555-1111", "555-2222", "555-1111", "555-3333"],
                "email": ["a@test.com", "b@test.com", "c@test.com", None],
                "flag1": [1, 0, 1, 0],
                "flag2": [0, 1, 0, 1],
                "group": ["G1", "G1", "G2", "G2"],
            }
        )

    @pytest.fixture()
    def populated_api(self, api_instance, sample_dataframe) -> DetectEntityNetworks:
        """Create a populated API instance with data."""
        api_instance.add_attribute_links(
            sample_dataframe, "entity_id", ["phone", "email"]
        )
        return api_instance


class TestInitialization(TestDetectEntityNetworks):
    def test_initialization(self, api_instance) -> None:
        """Test that DetectEntityNetworks initializes correctly."""
        assert isinstance(api_instance, DetectEntityNetworks)
        assert api_instance.attribute_links == []
        assert api_instance.attributes_list == []
        assert api_instance.flag_links == []
        assert api_instance.group_links == []
        assert api_instance.additional_trimmed_attributes == []
        assert isinstance(api_instance.graph, nx.Graph)
        assert isinstance(api_instance.integrated_flags, pl.DataFrame)
        assert isinstance(api_instance.node_types, set)
        assert isinstance(api_instance.inferred_links, dict)
        assert api_instance.exposure_report == ""


class TestFormatLinksAdded(TestDetectEntityNetworks):
    def test_format_links_added_basic(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test formatting links with basic data."""
        result = api_instance.format_links_added(
            sample_dataframe, "entity_id", ["phone", "email"]
        )

        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()

    def test_format_links_added_single_column(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test formatting links with single column."""
        result = api_instance.format_links_added(
            sample_dataframe, "entity_id", ["phone"]
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0


class TestGetEntityTypes(TestDetectEntityNetworks):
    def test_get_entity_types_empty(self, api_instance) -> None:
        """Test getting entity types with no data."""
        types = api_instance.get_entity_types()

        assert isinstance(types, list)
        assert "ENTITY" in types  # ENTITY_LABEL is default

    def test_get_entity_types_populated(self, populated_api) -> None:
        """Test getting entity types with data."""
        types = populated_api.get_entity_types()

        assert isinstance(types, list)
        assert len(types) > 0
        assert "ENTITY" in types

    def test_get_entity_types_sorted(self, populated_api) -> None:
        """Test that entity types are sorted."""
        types = populated_api.get_entity_types()

        assert types == sorted(types)


class TestGetAttributes(TestDetectEntityNetworks):
    def test_get_attributes_empty(self, api_instance) -> None:
        """Test getting attributes with no data."""
        attrs = api_instance.get_attributes()

        assert isinstance(attrs, pl.DataFrame)
        assert attrs.is_empty()

    def test_get_attributes_populated(self, populated_api) -> None:
        """Test getting attributes with data."""
        # First add data to attributes_list
        populated_api.get_model_summary_data()  # This populates attributes_list
        
        attrs = populated_api.get_attributes()

        assert isinstance(attrs, pl.DataFrame)


class TestRemoveAttributes(TestDetectEntityNetworks):
    def test_remove_attributes_basic(self, api_instance) -> None:
        """Test removing attributes."""
        attrs_df = pl.DataFrame({"Attribute": ["attr1", "attr2", "attr3"]})
        
        api_instance.remove_attributes(attrs_df)

        assert api_instance.additional_trimmed_attributes == ["attr1", "attr2", "attr3"]

    def test_remove_attributes_empty(self, api_instance) -> None:
        """Test removing attributes with empty dataframe."""
        attrs_df = pl.DataFrame({"Attribute": []})
        
        api_instance.remove_attributes(attrs_df)

        assert api_instance.additional_trimmed_attributes == []


class TestAddAttributeLinks(TestDetectEntityNetworks):
    def test_add_attribute_links_basic(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test adding attribute links."""
        links = api_instance.add_attribute_links(
            sample_dataframe, "entity_id", ["phone"]
        )

        assert isinstance(links, list)
        assert len(links) > 0
        assert len(api_instance.attribute_links) > 0
        assert isinstance(api_instance.graph, nx.Graph)
        assert len(api_instance.graph.nodes()) > 0

    def test_add_attribute_links_multiple_columns(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test adding attribute links with multiple columns."""
        links = api_instance.add_attribute_links(
            sample_dataframe, "entity_id", ["phone", "email"]
        )

        assert isinstance(links, list)
        assert len(links) > 0
        assert len(api_instance.node_types) > 0

    def test_add_attribute_links_filters_nulls(self, api_instance) -> None:
        """Test that add_attribute_links filters out null values."""
        df_with_nulls = pl.DataFrame(
            {
                "entity_id": ["E1", "E2", "E3"],
                "phone": ["555-1111", None, "555-3333"],
            }
        )
        
        links = api_instance.add_attribute_links(df_with_nulls, "entity_id", ["phone"])

        assert isinstance(links, list)
        # Should only have links for non-null values


class TestAddFlagLinks(TestDetectEntityNetworks):
    def test_add_flag_links_basic(self, api_instance, sample_dataframe) -> None:
        """Test adding flag links."""
        links = api_instance.add_flag_links(
            sample_dataframe,
            "entity_id",
            ["flag1", "flag2"],
            FlagAggregatorType.Count,
        )

        assert isinstance(links, list)
        assert len(links) > 0
        assert len(api_instance.flag_links) > 0
        assert not api_instance.integrated_flags.is_empty()
        assert hasattr(api_instance, "max_entity_flags")
        assert hasattr(api_instance, "mean_flagged_flags")

    def test_add_flag_links_instance_format(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test adding flag links with Instance format."""
        links = api_instance.add_flag_links(
            sample_dataframe,
            "entity_id",
            ["flag1"],
            FlagAggregatorType.Instance,
        )

        assert isinstance(links, list)


class TestAddGroupLinks(TestDetectEntityNetworks):
    def test_add_group_links_basic(self, api_instance, sample_dataframe) -> None:
        """Test adding group links."""
        links = api_instance.add_group_links(
            sample_dataframe, "entity_id", ["group"]
        )

        assert isinstance(links, list)
        assert len(links) > 0
        assert len(api_instance.group_links) > 0

    def test_add_group_links_multiple_groups(self, api_instance) -> None:
        """Test adding multiple group columns."""
        df = pl.DataFrame(
            {
                "entity_id": ["E1", "E2"],
                "group1": ["G1", "G2"],
                "group2": ["X", "Y"],
            }
        )
        
        links = api_instance.add_group_links(df, "entity_id", ["group1", "group2"])

        assert isinstance(links, list)
        assert len(links) > 0


class TestGetModelSummaryData(TestDetectEntityNetworks):
    def test_get_model_summary_data_empty(self, api_instance) -> None:
        """Test getting model summary with no data."""
        summary = api_instance.get_model_summary_data()

        assert isinstance(summary, SummaryData)
        assert summary.entities == 0
        assert summary.attributes == 0
        assert summary.flags == 0
        assert summary.groups == 0
        assert summary.links == 0

    def test_get_model_summary_data_with_graph(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test getting model summary with graph data."""
        api_instance.add_attribute_links(
            sample_dataframe, "entity_id", ["phone"]
        )
        
        summary = api_instance.get_model_summary_data()

        assert isinstance(summary, SummaryData)
        assert summary.entities > 0
        assert summary.attributes > 0
        assert summary.links > 0

    def test_get_model_summary_data_with_flags(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test getting model summary with flag data."""
        api_instance.add_flag_links(
            sample_dataframe,
            "entity_id",
            ["flag1"],
            FlagAggregatorType.Count,
        )
        
        summary = api_instance.get_model_summary_data()

        assert isinstance(summary, SummaryData)
        assert summary.flags > 0

    def test_get_model_summary_data_with_groups(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test getting model summary with group data."""
        api_instance.add_group_links(sample_dataframe, "entity_id", ["group"])
        
        summary = api_instance.get_model_summary_data()

        assert isinstance(summary, SummaryData)
        assert summary.groups > 0


class TestGetModelSummaryValue(TestDetectEntityNetworks):
    def test_get_model_summary_value_format(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test that summary value is formatted correctly."""
        api_instance.add_attribute_links(
            sample_dataframe, "entity_id", ["phone"]
        )
        
        summary_str = api_instance.get_model_summary_value()

        assert isinstance(summary_str, str)
        assert "Number of entities:" in summary_str
        assert "Number of attributes:" in summary_str
        assert "Number of flags:" in summary_str
        assert "Number of groups:" in summary_str
        assert "Number of links:" in summary_str


class TestIndexNodes(TestDetectEntityNetworks):
    @pytest.mark.asyncio()
    async def test_index_nodes_basic(self, populated_api) -> None:
        """Test indexing nodes."""
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.index_nodes"
        ) as mock_index:
            mock_index.return_value = (["text1"], [[0.1, 0.2]], [[1, 2]])
            
            await populated_api.index_nodes(["ENTITY"])

            assert hasattr(populated_api, "embedded_texts")
            assert hasattr(populated_api, "nearest_text_distances")
            assert hasattr(populated_api, "nearest_text_indices")
            mock_index.assert_called_once()


class TestInferNodes(TestDetectEntityNetworks):
    def test_infer_nodes_basic(self, api_instance) -> None:
        """Test inferring nodes."""
        # Setup mock data
        api_instance.embedded_texts = ["text1", "text2"]
        api_instance.nearest_text_indices = [[1], [0]]
        api_instance.nearest_text_distances = [[0.1], [0.1]]
        
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.infer_nodes"
        ) as mock_infer:
            mock_result = defaultdict(set)
            mock_result["text1"].add("text2")
            mock_infer.return_value = mock_result
            
            result = api_instance.infer_nodes(0.5)

            assert isinstance(result, defaultdict)
            mock_infer.assert_called_once()

    def test_infer_nodes_with_callbacks(self, api_instance) -> None:
        """Test inferring nodes with progress callbacks."""
        api_instance.embedded_texts = ["text1"]
        api_instance.nearest_text_indices = [[]]
        api_instance.nearest_text_distances = [[]]
        
        callbacks = [MagicMock()]
        
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.infer_nodes"
        ) as mock_infer:
            mock_infer.return_value = defaultdict(set)
            
            api_instance.infer_nodes(0.5, progress_callbacks=callbacks)

            mock_infer.assert_called_once()


class TestClearMethods(TestDetectEntityNetworks):
    def test_clear_inferred_links(self, api_instance) -> None:
        """Test clearing inferred links."""
        api_instance.inferred_links = {"test": {"link"}}
        
        api_instance.clear_inferred_links()

        assert api_instance.inferred_links == {}

    def test_clear_data_model(self, populated_api) -> None:
        """Test clearing the entire data model."""
        # Verify data exists
        assert len(populated_api.attribute_links) > 0
        
        populated_api.clear_data_model()

        assert populated_api.attribute_links == []
        assert populated_api.flag_links == []
        assert populated_api.group_links == []
        assert len(populated_api.graph.nodes()) == 0
        assert populated_api.integrated_flags.is_empty()
        assert len(populated_api.node_types) == 0
        assert populated_api.inferred_links == {}


class TestInferredNodesDf(TestDetectEntityNetworks):
    def test_inferred_nodes_df_empty(self, api_instance) -> None:
        """Test getting inferred nodes dataframe with no data."""
        api_instance.inferred_links = {}
        
        result = api_instance.inferred_nodes_df()

        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_inferred_nodes_df_with_data(self, api_instance) -> None:
        """Test getting inferred nodes dataframe with data."""
        api_instance.inferred_links = {
            "ENTITY::A": {"ENTITY::B"},
            "ENTITY::C": {"ENTITY::D"},
        }
        
        result = api_instance.inferred_nodes_df()

        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()
        assert "text" in result.columns
        assert "similar" in result.columns


class TestIdentify(TestDetectEntityNetworks):
    def test_identify_basic(self, populated_api) -> None:
        """Test identify method with basic parameters."""
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.trim_nodeset"
        ) as mock_trim, patch(
            "intelligence_toolkit.detect_entity_networks.api.build_networks"
        ) as mock_build, patch(
            "intelligence_toolkit.detect_entity_networks.api.build_entity_records"
        ) as mock_records:
            mock_trim.return_value = ([], set())
            mock_build.return_value = ([[]], {})
            mock_records.return_value = []
            
            result = populated_api.identify()

            assert isinstance(result, list)
            mock_trim.assert_called_once()
            mock_build.assert_called_once()
            mock_records.assert_called_once()


class TestGetCommunitySizes(TestDetectEntityNetworks):
    def test_get_community_sizes_empty(self, api_instance) -> None:
        """Test getting community sizes with no data."""
        api_instance.community_nodes = []
        
        sizes = api_instance.get_community_sizes()

        assert isinstance(sizes, list)
        assert len(sizes) == 0

    def test_get_community_sizes_with_data(self, api_instance) -> None:
        """Test getting community sizes with community data."""
        api_instance.community_nodes = [
            ["E1"],  # Single entity - excluded
            ["E2", "E3"],  # 2 entities
            ["E4", "E5", "E6"],  # 3 entities
        ]
        
        sizes = api_instance.get_community_sizes()

        assert isinstance(sizes, list)
        assert len(sizes) == 2  # Only multi-entity communities
        assert 2 in sizes
        assert 3 in sizes


class TestGetRecordsSummary(TestDetectEntityNetworks):
    def test_get_records_summary_empty(self, api_instance) -> None:
        """Test getting records summary with no data."""
        api_instance.community_nodes = []
        
        summary = api_instance.get_records_summary()

        assert isinstance(summary, str)
        assert summary == ""

    def test_get_records_summary_with_data(self, api_instance) -> None:
        """Test getting records summary with community data."""
        api_instance.community_nodes = [
            ["E1", "E2"],
            ["E3", "E4", "E5"],
        ]
        
        summary = api_instance.get_records_summary()

        assert isinstance(summary, str)
        assert "Networks identified:" in summary
        assert "2" in summary  # 2 networks
        assert "3" in summary  # Max size is 3


class TestGetEntityDf(TestDetectEntityNetworks):
    def test_get_entity_df_basic(self, api_instance) -> None:
        """Test getting entity dataframe."""
        api_instance.entity_records = [
            ("E1", 5, 0, 2, 10, 1, 5.0, 1.0),
            ("E2", 0, 0, 2, 10, 0, 5.0, 0.5),
        ]
        
        result = api_instance.get_entity_df()

        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()
        assert "entity_id" in result.columns
        assert "network_id" in result.columns
        assert "entity_flags" in result.columns


class TestGetGroupedDf(TestDetectEntityNetworks):
    def test_get_grouped_df_basic(self, api_instance, sample_dataframe) -> None:
        """Test getting grouped dataframe."""
        api_instance.entity_records = [("E1", 5, 0, 2, 10, 1, 5.0, 1.0)]
        api_instance.add_group_links(sample_dataframe, "entity_id", ["group"])
        
        result = api_instance.get_grouped_df()

        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()


class TestGetExposureReport(TestDetectEntityNetworks):
    def test_get_exposure_report_basic(self, api_instance) -> None:
        """Test getting exposure report."""
        api_instance.community_nodes = [["ENTITY::E1", "ENTITY::E2"]]
        api_instance.integrated_flags = pl.DataFrame(
            {"qualified_entity": ["ENTITY::E1"], "count": [5]}
        )
        api_instance.graph = nx.Graph()
        api_instance.graph.add_edge("ENTITY::E1", "phone::555-1111")
        api_instance.graph.add_edge("ENTITY::E2", "email::test@test.com")
        api_instance.inferred_links = {}
        api_instance.entity_to_community_ix = {"E1": 0, "E2": 0}
        api_instance.trimmed_attributes = pl.DataFrame()
        
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.build_exposure_report"
        ) as mock_report:
            mock_report.return_value = "Test report"
            
            result = api_instance.get_exposure_report("E1", 0)

            assert result == "Test report"
            assert api_instance.exposure_report == "Test report"


class TestGetEntitiesGraph(TestDetectEntityNetworks):
    def test_get_entities_graph_basic(self, api_instance) -> None:
        """Test getting entities graph."""
        api_instance.community_nodes = [["ENTITY::E1"]]
        api_instance.graph = nx.Graph()
        api_instance.entity_to_community_ix = {"E1": 0}
        api_instance.integrated_flags = pl.DataFrame()
        api_instance.trimmed_attributes = pl.DataFrame()
        api_instance.inferred_links = {}
        
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.build_network_from_entities"
        ) as mock_build:
            mock_build.return_value = nx.Graph()
            
            result = api_instance.get_entities_graph(0)

            assert isinstance(result, nx.Graph)
            mock_build.assert_called_once()


class TestGetSingleEntityGraph(TestDetectEntityNetworks):
    def test_get_single_entity_graph_basic(self, api_instance) -> None:
        """Test getting single entity graph."""
        entities_graph = nx.Graph()
        entities_graph.add_node("ENTITY::E1")
        
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.get_entity_graph"
        ) as mock_get:
            mock_get.return_value = ([], [])
            
            result = api_instance.get_single_entity_graph(entities_graph, "E1")

            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_get.assert_called_once()


class TestGetMergedGraphDf(TestDetectEntityNetworks):
    def test_get_merged_graph_df_basic(self, api_instance) -> None:
        """Test getting merged graph dataframe."""
        # Setup minimal required data
        api_instance.community_nodes = [["ENTITY::E1"]]
        api_instance.graph = nx.Graph()
        api_instance.graph.add_node("ENTITY::E1", type="entity", flags=0)
        api_instance.entity_to_community_ix = {}
        api_instance.integrated_flags = pl.DataFrame()
        api_instance.trimmed_attributes = pl.DataFrame()
        api_instance.inferred_links = {}
        
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.build_network_from_entities"
        ) as mock_build, patch(
            "intelligence_toolkit.detect_entity_networks.api.simplify_entities_graph"
        ) as mock_simplify:
            test_graph = nx.Graph()
            test_graph.add_node("node1", type="entity", flags=0)
            test_graph.add_node("attr::value", type="attribute", flags=0)
            test_graph.add_edge("node1", "attr::value")
            mock_build.return_value = test_graph
            mock_simplify.return_value = test_graph
            
            nodes, links = api_instance.get_merged_graph_df(0)

            assert isinstance(nodes, pl.DataFrame)
            assert isinstance(links, pl.DataFrame)


class TestGenerateReport(TestDetectEntityNetworks):
    def test_generate_report_basic(self, api_instance) -> None:
        """Test generating a report."""
        # Setup minimal required data
        api_instance.community_nodes = [["ENTITY::E1"]]
        api_instance.max_entity_flags = 10
        api_instance.mean_flagged_flags = 5.0
        api_instance.exposure_report = "Test exposure"
        api_instance.ai_configuration = MagicMock()
        api_instance.graph = nx.Graph()
        api_instance.graph.add_node("ENTITY::E1", type="entity", flags=0)
        api_instance.entity_to_community_ix = {}
        api_instance.integrated_flags = pl.DataFrame()
        api_instance.trimmed_attributes = pl.DataFrame()
        api_instance.inferred_links = {}
        
        with patch(
            "intelligence_toolkit.detect_entity_networks.api.build_network_from_entities"
        ) as mock_build, patch(
            "intelligence_toolkit.detect_entity_networks.api.simplify_entities_graph"
        ) as mock_simplify, patch(
            "intelligence_toolkit.detect_entity_networks.api.OpenAIClient"
        ) as mock_client:
            test_graph = nx.Graph()
            test_graph.add_node("node1", type="entity", flags=0)
            test_graph.add_node("attr::value", type="attribute", flags=0)
            test_graph.add_edge("node1", "attr::value")
            mock_build.return_value = test_graph
            mock_simplify.return_value = test_graph
            
            mock_instance = MagicMock()
            mock_instance.generate_chat.return_value = "Generated report"
            mock_client.return_value = mock_instance
            
            result = api_instance.generate_report(0, "E1")

            assert result == "Generated report"
            mock_client.assert_called_once()
            mock_instance.generate_chat.assert_called_once()


class TestBuildNetworkFromEntities:
    @pytest.fixture()
    def base_graph(self) -> nx.Graph:
        """Create a base graph for testing."""
        graph = nx.Graph()
        graph.add_node(f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1")
        graph.add_node(f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E2")
        graph.add_node(f"phone{ATTRIBUTE_VALUE_SEPARATOR}555-1111")
        graph.add_node(f"email{ATTRIBUTE_VALUE_SEPARATOR}test@test.com")
        
        graph.add_edge(f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1", f"phone{ATTRIBUTE_VALUE_SEPARATOR}555-1111")
        graph.add_edge(f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1", f"email{ATTRIBUTE_VALUE_SEPARATOR}test@test.com")
        graph.add_edge(f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E2", f"phone{ATTRIBUTE_VALUE_SEPARATOR}555-1111")
        
        return graph

    def test_build_network_empty_selected_nodes(self, base_graph) -> None:
        """Test with empty selected_nodes list."""
        result = build_network_from_entities(
            base_graph,
            {},
            selected_nodes=[],
        )

        assert isinstance(result, nx.Graph)
        assert result.number_of_nodes() == 0
        assert result.number_of_edges() == 0

    def test_build_network_with_single_entity(self, base_graph) -> None:
        """Test with single entity."""
        entity_to_community = {f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1": 0}
        
        result = build_network_from_entities(
            base_graph,
            entity_to_community,
            selected_nodes=[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"],
        )

        assert isinstance(result, nx.Graph)
        assert result.number_of_nodes() > 0
        assert f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1" in result.nodes()

    def test_build_network_with_multiple_entities(self, base_graph) -> None:
        """Test with multiple entities."""
        entity_to_community = {
            f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1": 0,
            f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E2": 0,
        }
        
        result = build_network_from_entities(
            base_graph,
            entity_to_community,
            selected_nodes=[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1", f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E2"],
        )

        assert isinstance(result, nx.Graph)
        assert f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1" in result.nodes()
        assert f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E2" in result.nodes()
        assert f"phone{ATTRIBUTE_VALUE_SEPARATOR}555-1111" in result.nodes()

    def test_build_network_with_trimmed_attributes(self, base_graph) -> None:
        """Test with trimmed attributes that should be excluded."""
        entity_to_community = {f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1": 0}
        trimmed_attributes = pl.DataFrame({
            "Attribute": [f"phone{ATTRIBUTE_VALUE_SEPARATOR}555-1111"],
            "Linked Entities": [5],
        })
        
        result = build_network_from_entities(
            base_graph,
            entity_to_community,
            trimmed_attributes=trimmed_attributes,
            selected_nodes=[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"],
        )

        assert isinstance(result, nx.Graph)
        assert f"phone{ATTRIBUTE_VALUE_SEPARATOR}555-1111" not in result.nodes()
        assert f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1" in result.nodes()

    def test_build_network_with_integrated_flags(self, base_graph) -> None:
        """Test with integrated flags."""
        entity_to_community = {f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1": 0}
        integrated_flags = pl.DataFrame({
            "qualified_entity": [f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"],
            "count": [5],
        })
        
        result = build_network_from_entities(
            base_graph,
            entity_to_community,
            integrated_flags=integrated_flags,
            selected_nodes=[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"],
        )

        assert isinstance(result, nx.Graph)
        assert f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1" in result.nodes()
        assert result.nodes[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"]["flags"] == 5

    def test_build_network_with_inferred_links(self, base_graph) -> None:
        """Test with inferred links."""
        entity_to_community = {
            f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1": 0,
            f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E2": 0,
        }
        inferred_links = {f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1": {f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E2"}}
        
        result = build_network_from_entities(
            base_graph,
            entity_to_community,
            inferred_links=inferred_links,
            selected_nodes=[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"],
        )

        assert isinstance(result, nx.Graph)
        assert f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1" in result.nodes()
        assert f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E2" in result.nodes()

    def test_build_network_node_attributes(self, base_graph) -> None:
        """Test that nodes have correct attributes."""
        entity_to_community = {f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1": 0}
        
        result = build_network_from_entities(
            base_graph,
            entity_to_community,
            selected_nodes=[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"],
        )

        entity_node = f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"
        assert result.nodes[entity_node]["type"] == ENTITY_LABEL
        assert result.nodes[entity_node]["network"] == "0"
        assert result.nodes[entity_node]["flags"] == 0

    def test_build_network_with_none_entity_community(self, base_graph) -> None:
        """Test when entity is not in entity_to_community."""
        result = build_network_from_entities(
            base_graph,
            {},
            selected_nodes=[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"],
        )

        assert isinstance(result, nx.Graph)
        assert f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1" in result.nodes()
        assert result.nodes[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"]["network"] == ""

    def test_build_network_with_all_none_defaults(self, base_graph) -> None:
        """Test with all optional parameters as None."""
        result = build_network_from_entities(
            base_graph,
            {},
            integrated_flags=None,
            trimmed_attributes=None,
            inferred_links=None,
            selected_nodes=[f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}E1"],
        )

        assert isinstance(result, nx.Graph)
        assert result.number_of_nodes() > 0
