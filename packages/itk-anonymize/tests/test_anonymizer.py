# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from itk_anonymize import AnonymizeConfig, Anonymizer, AnonymizeResult
from itk_anonymize.aggregate import count_combinations
from itk_anonymize.models import RawTable
from itk_anonymize_testing import FakeSynthesizer, av

if TYPE_CHECKING:
    import polars as pl


def test_runs_synthesizer_once_on_prepared_table(
    complaints: pl.DataFrame, fake: FakeSynthesizer, config: AnonymizeConfig
) -> None:
    result = Anonymizer(fake).anonymize(complaints, config)
    assert len(fake.calls) == 1
    table, passed = fake.calls[0]
    assert passed is config
    assert table.header == tuple(complaints.columns)
    assert table.height == complaints.height
    assert result.sensitive.equals(table.to_frame())


def test_result_carries_counts_and_delta(result: AnonymizeResult) -> None:
    assert result.protected_record_count == 8
    assert result.delta == pytest.approx(1.0 / (math.log(8) * 8))
    assert result.sensitive_aggregates == count_combinations(
        RawTable.from_frame(result.sensitive), 3
    )
    # the fake returns the plain counts, so DP equals sensitive here
    assert result.dp_aggregates == result.sensitive_aggregates
    assert result.synthetic.height == 8
    assert set(result.synthetic.rows()) == set(result.sensitive.rows())
    assert result.synthetic_aggregates == result.sensitive_aggregates


def test_error_reports_are_exact_for_a_perfect_fake(result: AnonymizeResult) -> None:
    assert result.aggregate_error_report is not None
    assert result.synthetic_error_report is not None
    for report in (result.aggregate_error_report, result.synthetic_error_report):
        assert report.overall.mean_error == 0.0
        assert report.overall.suppressed_pct == 0.0
        assert report.overall.fabricated_pct == 0.0
        assert [r.length for r in report.rows] == [1, 2, 3, None]


def test_noisy_fake_shows_suppression_and_fabrication(
    complaints: pl.DataFrame, config: AnonymizeConfig
) -> None:
    result = Anonymizer(FakeSynthesizer(noise=True, drop_last=True)).anonymize(
        complaints, config
    )
    report = result.aggregate_error_report
    assert report is not None
    overall = report.overall
    assert overall.mean_error is not None
    assert overall.mean_error > 0
    assert overall.suppressed_pct is not None
    assert overall.suppressed_pct > 0
    assert overall.fabricated_pct is not None
    assert overall.fabricated_pct > 0
    assert ("fabricated:yes",) in result.dp_aggregates
    assert result.synthetic.height == 7
    assert result.raw_dp_aggregates["fabricated:yes"] == 2


def test_aggregate_frames(result: AnonymizeResult) -> None:
    frame = result.aggregates_frame
    assert frame.columns == ["selections", "protected_count"]
    assert frame.row(0) == ("record_count", 8)
    assert frame.height == len(result.dp_aggregates) + 1
    assert (
        "city:Lakeside;price_issue:True;product:A"
        in frame.get_column("selections").to_list()
    )
    assert (
        result.synthetic_aggregates_frame.height == len(result.synthetic_aggregates) + 1
    )


def test_evaluate_reruns_without_the_synthesizer(
    result: AnonymizeResult, fake: FakeSynthesizer
) -> None:
    calls = len(fake.calls)
    result.dp_aggregates = {}
    again = Anonymizer.evaluate(result)
    assert again is result
    assert again.aggregate_error_report is not None
    assert again.aggregate_error_report.overall.suppressed_pct == 100.0
    assert len(fake.calls) == calls


def test_delta_is_none_for_tiny_protected_counts(complaints: pl.DataFrame) -> None:
    class One:
        def synthesize(self, table, config):
            from itk_anonymize.models import SynthesisOutput

            return SynthesisOutput({}, 1, table)

    result = Anonymizer(One()).anonymize(complaints, AnonymizeConfig(epsilon=1.0))
    assert result.delta is None


def test_links_choose_aggregates_within_reporting_length(
    result: AnonymizeResult,
) -> None:
    exact = Anonymizer.links(result, [], "city", "product")
    assert exact.get_column("Dataset").unique().to_list() == ["Aggregate"]
    # 2 edge attributes + 1 highlight = 3 fits reporting_length 3
    highlighted = Anonymizer.links(
        result, [], "city", "product", av("price_issue", "True")
    )
    assert highlighted.get_column("Dataset").unique().to_list() == ["Aggregate"]
    # ...but adding a selection term makes 4, so synthetic rows are counted
    selected = Anonymizer.links(
        result, [av("period", "2023-H1")], "city", "product", av("price_issue", "True")
    )
    assert selected.get_column("Dataset").unique().to_list() == ["Synthetic"]


def test_links_agree_between_sources_for_a_perfect_fake(
    result: AnonymizeResult,
) -> None:
    from itk_anonymize.queries import aggregate_links, synthetic_links

    exact = aggregate_links(
        result.dp_aggregates, [], "city", "product", av("price_issue", "True")
    )
    counted = synthetic_links(
        result.synthetic, [], "city", "product", av("price_issue", "True")
    )
    strip = lambda f: set(f.drop("Dataset").rows())  # noqa: E731
    assert strip(exact) == strip(counted)


def test_from_pacsynth_builds_the_adapter() -> None:
    anonymizer = Anonymizer.from_pacsynth()
    assert type(anonymizer.synthesizer).__name__ == "PacsynthSynthesizer"


def test_synthesizability_passthrough(complaints: pl.DataFrame) -> None:
    stats = Anonymizer.analyze_synthesizability(complaints)
    assert stats.num_cols == 4
    assert stats.overall_att_count == 3 + 2 + 1 + 2
