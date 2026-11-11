# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import pytest
from itk_patterns import DetectionConfig, PatternConfig
from itk_patterns.models import join_pattern, split_pattern
from pydantic import ValidationError


def test_pattern_config_defaults_match_legacy_config():
    config = PatternConfig(period_column="period")
    assert config.type_value_separator == "="
    assert config.min_edge_weight == 0.001
    assert config.missing_edge_prop == 0.1
    assert (config.laplacian, config.diagonal, config.correlation) == (True, True, True)
    assert config.convergence == "legacy"
    assert config.attribute_columns is None


def test_pattern_config_is_frozen():
    config = PatternConfig(period_column="period")
    with pytest.raises(ValidationError):
        config.period_column = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("period_column", ""),
        ("type_value_separator", ""),
        ("min_edge_weight", 0.0),
        ("min_edge_weight", 1.0),
        ("missing_edge_prop", 1.5),
        ("convergence", "sideways"),
    ],
)
def test_pattern_config_rejects_bad_values(field, value):
    kwargs = {"period_column": "period", field: value}
    with pytest.raises(ValidationError):
        PatternConfig(**kwargs)


def test_detection_config_defaults_and_bounds():
    detection = DetectionConfig()
    assert (detection.min_pattern_count, detection.max_pattern_length) == (5, 5)
    with pytest.raises(ValidationError):
        DetectionConfig(min_pattern_count=0)
    with pytest.raises(ValidationError):
        DetectionConfig(max_pattern_length=0)


def test_pattern_text_round_trips():
    attributes = ["city=A", "product=X"]
    assert split_pattern(join_pattern(attributes)) == attributes
    assert join_pattern(attributes) == "city=A & product=X"
