# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from itk_extract.models import Conflict, ExtractConfig, ExtractionError, ExtractResult
from itk_extract_testing import chunk

if TYPE_CHECKING:
    from itk_extract.schema import RecordSchema


def test_config_defaults_match_legacy_behaviour() -> None:
    config = ExtractConfig()
    assert config.chunk_tokens is None
    assert config.merge_policy == "last"
    assert config.validate_output is True
    assert config.fail_fast is False


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="chunk_tokens"):
        ExtractConfig(chunk_tokens=0)
    with pytest.raises(ValueError, match="merge_policy"):
        ExtractConfig(merge_policy="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_sources"):
        ExtractConfig(max_sources=0)


def test_config_is_frozen() -> None:
    config = ExtractConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.guidance = "x"  # type: ignore[misc]


def test_result_frames_render_non_string_values(schema: RecordSchema) -> None:
    result = ExtractResult(
        schema=schema,
        config=ExtractConfig(),
        extractions=[],
        errors=[ExtractionError(chunk("a#0"), "boom", "raw")],
        object={},
        conflicts=[Conflict("founded", 2000, (("a#0", 1999), ("b#0", 2000)))],
    )
    conflicts = result.conflicts_frame
    assert conflicts.get_column("chosen").to_list() == ["2000"]
    assert conflicts.get_column("values").to_list() == [["1999", "2000"]]
    assert conflicts.get_column("chunk_id").to_list() == [["a#0", "b#0"]]
    errors = result.errors_frame
    assert errors.to_dicts() == [
        {"chunk_id": "a#0", "source_id": "a", "message": "boom"}
    ]
