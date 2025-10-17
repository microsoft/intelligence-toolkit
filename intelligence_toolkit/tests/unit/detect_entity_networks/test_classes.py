# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import pytest

from intelligence_toolkit.detect_entity_networks.classes import (
    FlagAggregatorType,
    SummaryData,
)


class TestFlagAggregatorType:
    def test_flag_aggregator_type_enum_values(self) -> None:
        """Test that FlagAggregatorType has expected values."""
        assert hasattr(FlagAggregatorType, "Instance")
        assert hasattr(FlagAggregatorType, "Count")

    def test_flag_aggregator_instance_value(self) -> None:
        """Test Instance enum value."""
        assert FlagAggregatorType.Instance.value == "Instance"

    def test_flag_aggregator_count_value(self) -> None:
        """Test Count enum value."""
        assert FlagAggregatorType.Count.value == "Count"

    def test_flag_aggregator_type_is_enum(self) -> None:
        """Test that FlagAggregatorType is an enum."""
        from enum import Enum

        assert issubclass(FlagAggregatorType, Enum)


class TestSummaryData:
    def test_summary_data_initialization(self) -> None:
        """Test SummaryData initialization with all parameters."""
        summary = SummaryData(
            entities=100, attributes=50, flags=25, groups=10, links=200
        )

        assert summary.entities == 100
        assert summary.attributes == 50
        assert summary.flags == 25
        assert summary.groups == 10
        assert summary.links == 200

    def test_summary_data_zero_values(self) -> None:
        """Test SummaryData with zero values."""
        summary = SummaryData(entities=0, attributes=0, flags=0, groups=0, links=0)

        assert summary.entities == 0
        assert summary.attributes == 0
        assert summary.flags == 0
        assert summary.groups == 0
        assert summary.links == 0

    def test_summary_data_attributes_accessible(self) -> None:
        """Test that all SummaryData attributes are accessible."""
        summary = SummaryData(entities=5, attributes=3, flags=2, groups=1, links=10)

        # All attributes should be accessible
        assert hasattr(summary, "entities")
        assert hasattr(summary, "attributes")
        assert hasattr(summary, "flags")
        assert hasattr(summary, "groups")
        assert hasattr(summary, "links")

    def test_summary_data_with_large_numbers(self) -> None:
        """Test SummaryData with large numbers."""
        summary = SummaryData(
            entities=10000, attributes=5000, flags=2500, groups=100, links=50000
        )

        assert summary.entities == 10000
        assert summary.attributes == 5000
        assert summary.flags == 2500
        assert summary.groups == 100
        assert summary.links == 50000
