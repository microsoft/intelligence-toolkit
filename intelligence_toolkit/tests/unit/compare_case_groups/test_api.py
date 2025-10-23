# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from intelligence_toolkit.compare_case_groups.api import CompareCaseGroups


class TestCompareCaseGroups:
    @pytest.fixture()
    def api_instance(self) -> CompareCaseGroups:
        """Create a CompareCaseGroups instance for testing."""
        return CompareCaseGroups()

    @pytest.fixture()
    def sample_dataframe(self) -> pl.DataFrame:
        """Create a sample dataframe for testing."""
        return pl.DataFrame(
            {
                "group1": ["A", "A", "B", "B", "C", "C"],
                "group2": ["X", "Y", "X", "Y", "X", "Y"],
                "attribute1": ["val1", "val2", "val1", "val3", "val2", "val1"],
                "attribute2": ["a", "b", "c", "d", "e", "f"],
                "temporal": ["2020", "2021", "2020", "2021", "2020", "2021"],
            }
        )

    @pytest.fixture()
    def populated_api(self, api_instance, sample_dataframe) -> CompareCaseGroups:
        """Create a populated API instance with data."""
        api_instance.create_data_summary(
            prepared_df=sample_dataframe,
            filters=[],
            groups=["group1"],
            aggregates=["attribute1"],
            temporal="",
        )
        return api_instance


class TestInitialization(TestCompareCaseGroups):
    def test_initialization(self, api_instance) -> None:
        """Test that CompareCaseGroups initializes correctly."""
        assert isinstance(api_instance, CompareCaseGroups)
        assert api_instance.filters == []
        assert api_instance.groups == []
        assert api_instance.aggregates == []
        assert api_instance.temporal == ""

    def test_dataframes_initialized(self, api_instance) -> None:
        """Test that dataframes are initialized."""
        assert isinstance(api_instance.model_df, pl.DataFrame)
        assert isinstance(api_instance.filtered_df, pl.DataFrame)
        assert isinstance(api_instance.prepared_df, pl.DataFrame)


class TestGetDatasetProportion(TestCompareCaseGroups):
    def test_get_dataset_proportion_empty(self, api_instance) -> None:
        """Test dataset proportion with empty dataframes."""
        api_instance.prepared_df = pl.DataFrame()
        api_instance.filtered_df = pl.DataFrame()
        assert api_instance.get_dataset_proportion() == 0

    def test_get_dataset_proportion_full(self, api_instance) -> None:
        """Test dataset proportion when all records are included."""
        api_instance.prepared_df = pl.DataFrame({"col": [1, 2, 3, 4, 5]})
        api_instance.filtered_df = pl.DataFrame({"col": [1, 2, 3, 4, 5]})
        assert api_instance.get_dataset_proportion() == 100

    def test_get_dataset_proportion_half(self, api_instance) -> None:
        """Test dataset proportion with half the records."""
        api_instance.prepared_df = pl.DataFrame({"col": [1, 2, 3, 4]})
        api_instance.filtered_df = pl.DataFrame({"col": [1, 2]})
        assert api_instance.get_dataset_proportion() == 50

    def test_get_dataset_proportion_rounds(self, api_instance) -> None:
        """Test that dataset proportion rounds correctly."""
        api_instance.prepared_df = pl.DataFrame({"col": [1, 2, 3]})
        api_instance.filtered_df = pl.DataFrame({"col": [1]})
        result = api_instance.get_dataset_proportion()
        assert isinstance(result, (int, float))
        assert result == 33  # 33.33 rounded to 33


class TestGetReportGroupsFilterOptions(TestCompareCaseGroups):
    def test_get_report_groups_filter_options_single_group(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test getting filter options with single group."""
        api_instance.model_df = sample_dataframe
        api_instance.groups = ["group1"]
        
        options = api_instance.get_report_groups_filter_options()
        
        assert isinstance(options, list)
        assert len(options) == 3  # A, B, C
        assert all(isinstance(opt, dict) for opt in options)

    def test_get_report_groups_filter_options_multiple_groups(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test getting filter options with multiple groups."""
        api_instance.model_df = sample_dataframe
        api_instance.groups = ["group1", "group2"]
        
        options = api_instance.get_report_groups_filter_options()
        
        assert isinstance(options, list)
        assert len(options) == 6  # All combinations
        assert all("group1" in opt and "group2" in opt for opt in options)

    def test_get_report_groups_filter_options_sorted(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test that filter options are sorted."""
        api_instance.model_df = sample_dataframe
        api_instance.groups = ["group1"]
        
        options = api_instance.get_report_groups_filter_options()
        
        # Extract group1 values
        values = [opt["group1"] for opt in options]
        assert values == sorted(values)


class TestGetFilterOptions(TestCompareCaseGroups):
    def test_get_filter_options_basic(self, api_instance, sample_dataframe) -> None:
        """Test getting filter options from a dataframe."""
        options = api_instance.get_filter_options(sample_dataframe)
        
        assert isinstance(options, list)
        assert len(options) > 0
        assert all(":" in opt for opt in options)  # Format is "column:value"

    def test_get_filter_options_excludes_nulls(self, api_instance) -> None:
        """Test that filter options exclude null values."""
        df = pl.DataFrame(
            {
                "col1": ["A", "B", "", None, "nan", "NaN"],
                "col2": ["X", "Y", "NULL", "null", "None", "none"],
            }
        )
        
        options = api_instance.get_filter_options(df)
        
        # Should only have A, B, X, Y
        assert "col1:A" in options
        assert "col1:B" in options
        assert "col2:X" in options
        assert "col2:Y" in options
        # Should not have empty string or typical null representations
        assert not any("col1:" == opt or "col2:" == opt for opt in options)

    def test_get_filter_options_sorted(self, api_instance, sample_dataframe) -> None:
        """Test that filter options are sorted."""
        options = api_instance.get_filter_options(sample_dataframe)
        
        # Options should be sorted alphabetically
        assert options == sorted(options)


class TestSelectColumnsRankedDf(TestCompareCaseGroups):
    def test_select_columns_without_temporal(self, api_instance) -> None:
        """Test column selection without temporal data."""
        ranked_df = pl.DataFrame(
            {
                "group1": ["A", "B"],
                "group_count": [10, 20],
                "group_rank": [1, 2],
                "attribute_value": ["val1", "val2"],
                "group_attribute_count": [5, 15],
                "group_attribute_rank": [1, 2],
                "extra_col": ["x", "y"],
            }
        )
        
        api_instance.groups = ["group1"]
        api_instance.temporal = ""
        api_instance._select_columns_ranked_df(ranked_df)
        
        expected_columns = [
            "group1",
            "group_count",
            "group_rank",
            "attribute_value",
            "group_attribute_count",
            "group_attribute_rank",
        ]
        assert api_instance.model_df.columns == expected_columns
        assert "extra_col" not in api_instance.model_df.columns

    def test_select_columns_with_temporal(self, api_instance) -> None:
        """Test column selection with temporal data."""
        ranked_df = pl.DataFrame(
            {
                "group1": ["A"],
                "group_count": [10],
                "group_rank": [1],
                "attribute_value": ["val1"],
                "group_attribute_count": [5],
                "group_attribute_rank": [1],
                "year_window": ["2020"],
                "year_window_count": [3],
                "year_window_rank": [1],
                "year_window_delta": [2],
            }
        )
        
        api_instance.groups = ["group1"]
        api_instance.temporal = "year"
        api_instance._select_columns_ranked_df(ranked_df)
        
        assert "year_window" in api_instance.model_df.columns
        assert "year_window_count" in api_instance.model_df.columns
        assert "year_window_rank" in api_instance.model_df.columns
        assert "year_window_delta" in api_instance.model_df.columns


class TestCreateDataSummary(TestCompareCaseGroups):
    def test_create_data_summary_basic(self, api_instance, sample_dataframe) -> None:
        """Test creating a basic data summary."""
        api_instance.create_data_summary(
            prepared_df=sample_dataframe,
            filters=[],
            groups=["group1"],
            aggregates=["attribute1"],
            temporal="",
        )
        
        assert api_instance.filters == []
        assert api_instance.groups == ["group1"]
        assert api_instance.aggregates == ["attribute1"]
        assert api_instance.temporal == ""
        assert not api_instance.model_df.is_empty()

    def test_create_data_summary_with_filters(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test creating data summary with filters."""
        filters = ["group1:A"]
        
        api_instance.create_data_summary(
            prepared_df=sample_dataframe,
            filters=filters,
            groups=["group1"],
            aggregates=["attribute1"],
            temporal="",
        )
        
        assert api_instance.filters == filters
        # Filtered_df should only contain group1:A rows
        assert len(api_instance.filtered_df) < len(sample_dataframe)

    def test_create_data_summary_with_temporal(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test creating data summary with temporal attribute."""
        api_instance.create_data_summary(
            prepared_df=sample_dataframe,
            filters=[],
            groups=["group1"],
            aggregates=["attribute1"],
            temporal="temporal",
        )
        
        assert api_instance.temporal == "temporal"
        # Should have temporal columns in model_df
        temporal_cols = [col for col in api_instance.model_df.columns if "temporal" in col]
        assert len(temporal_cols) > 0

    def test_create_data_summary_drops_nulls(self, api_instance) -> None:
        """Test that create_data_summary drops null values in group columns."""
        df_with_nulls = pl.DataFrame(
            {
                "group1": ["A", None, "B", "C"],
                "attribute1": ["val1", "val2", "val3", "val4"],
            }
        )
        
        api_instance.create_data_summary(
            prepared_df=df_with_nulls,
            filters=[],
            groups=["group1"],
            aggregates=["attribute1"],
            temporal="",
        )
        
        # Should have 3 rows (null row dropped)
        assert len(api_instance.prepared_df) == 3


class TestFormatList(TestCompareCaseGroups):
    def test_format_list_basic(self, api_instance) -> None:
        """Test formatting a basic list."""
        items = ["item1", "item2", "item3"]
        result = api_instance._format_list(items)
        
        assert result == "[**item1**, **item2**, **item3**]"

    def test_format_list_no_bold(self, api_instance) -> None:
        """Test formatting list without bold."""
        items = ["item1", "item2"]
        result = api_instance._format_list(items, bold=False)
        
        assert result == "[item1, item2]"
        assert "**" not in result

    def test_format_list_escape_colon(self, api_instance) -> None:
        """Test formatting list with escaped colons."""
        items = ["item:1", "item:2"]
        result = api_instance._format_list(items, escape_colon=True)
        
        assert "\\:" in result
        assert result == "[**item\\:1**, **item\\:2**]"

    def test_format_list_empty(self, api_instance) -> None:
        """Test formatting empty list."""
        result = api_instance._format_list([])
        assert result == "[]"


class TestGetSummaryDescription(TestCompareCaseGroups):
    def test_get_summary_description_basic(self, populated_api) -> None:
        """Test getting summary description."""
        description = populated_api.get_summary_description()
        
        assert isinstance(description, str)
        assert "group_count" in description
        assert "group_rank" in description
        assert "attribute_count" in description
        assert "attribute_rank" in description

    def test_get_summary_description_with_filters(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test summary description with filters."""
        api_instance.create_data_summary(
            prepared_df=sample_dataframe,
            filters=["group1:A"],
            groups=["group1"],
            aggregates=["attribute1"],
            temporal="",
        )
        
        description = api_instance.get_summary_description()
        
        assert "group1:A" in description or "group1\\:A" in description
        assert "%" in description  # Should mention percentage

    def test_get_summary_description_with_temporal(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test summary description with temporal data."""
        api_instance.create_data_summary(
            prepared_df=sample_dataframe,
            filters=[],
            groups=["group1"],
            aggregates=["attribute1"],
            temporal="temporal",
        )
        
        description = api_instance.get_summary_description()
        
        assert "temporal_window" in description
        assert "temporal_window_count" in description
        assert "temporal_window_delta" in description


class TestGetReportData(TestCompareCaseGroups):
    def test_get_report_data_no_filter(self, populated_api) -> None:
        """Test getting report data without filters."""
        report_df, description = populated_api.get_report_data()
        
        assert isinstance(report_df, pl.DataFrame)
        assert isinstance(description, str)
        assert description == ""
        assert len(report_df) == len(populated_api.model_df)

    def test_get_report_data_with_selected_groups(
        self, api_instance, sample_dataframe
    ) -> None:
        """Test getting report data with selected groups."""
        api_instance.create_data_summary(
            prepared_df=sample_dataframe,
            filters=[],
            groups=["group1"],
            aggregates=["attribute1"],
            temporal="",
        )
        
        selected_groups = [{"group1": "A"}]
        report_df, description = api_instance.get_report_data(
            selected_groups=selected_groups
        )
        
        assert isinstance(report_df, pl.DataFrame)
        assert len(report_df) < len(api_instance.model_df)
        assert "Filtered to the following groups" in description

    def test_get_report_data_with_top_ranks(self, populated_api) -> None:
        """Test getting report data with top group ranks."""
        report_df, description = populated_api.get_report_data(top_group_ranks=2)
        
        assert isinstance(report_df, pl.DataFrame)
        assert "top 2 groups" in description
        # Should only have groups with rank <= 2
        if "group_rank" in report_df.columns:
            max_rank = report_df["group_rank"].drop_nulls().max()
            if max_rank is not None:
                assert max_rank <= 2


class TestGenerateGroupReport(TestCompareCaseGroups):
    def test_generate_group_report(self, populated_api) -> None:
        """Test generating a group report."""
        populated_api.ai_configuration = MagicMock()
        report_df = populated_api.model_df
        filter_description = "Test filter"
        
        with patch(
            "intelligence_toolkit.compare_case_groups.api.OpenAIClient"
        ) as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate_chat.return_value = "Test report generated"
            mock_client.return_value = mock_instance
            
            result = populated_api.generate_group_report(
                report_data=report_df,
                filter_description=filter_description,
            )
            
            assert result == "Test report generated"
            mock_client.assert_called_once()
            mock_instance.generate_chat.assert_called_once()

    def test_generate_group_report_with_callbacks(self, populated_api) -> None:
        """Test generating report with callbacks."""
        populated_api.ai_configuration = MagicMock()
        report_df = populated_api.model_df
        callbacks = [MagicMock()]
        
        with patch(
            "intelligence_toolkit.compare_case_groups.api.OpenAIClient"
        ) as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate_chat.return_value = "Test report"
            mock_client.return_value = mock_instance
            
            populated_api.generate_group_report(
                report_data=report_df,
                filter_description="",
                callbacks=callbacks,
            )
            
            # Verify callbacks were passed
            call_args = mock_instance.generate_chat.call_args
            assert call_args[1]["callbacks"] == callbacks
