# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Facade that runs the pipeline end to end."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from itk_anonymize.aggregate import count_combinations, parse_aggregates
from itk_anonymize.evaluate import error_report
from itk_anonymize.models import AnonymizeResult, AttributeValue
from itk_anonymize.prepare import prepare_frame, to_raw_table
from itk_anonymize.queries import aggregate_links, synthetic_links
from itk_anonymize.synthesizability import analyze_synthesizability

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from itk_anonymize.engine import Synthesizer
    from itk_anonymize.models import AnonymizeConfig, SynthesizabilityStatistics


class Anonymizer:
    """Holds the synthesizer; everything else is passed in and returned."""

    def __init__(self, synthesizer: Synthesizer) -> None:
        self.synthesizer = synthesizer

    @classmethod
    def from_pacsynth(cls) -> Anonymizer:
        from itk_anonymize.pacsynth_engine import PacsynthSynthesizer

        return cls(PacsynthSynthesizer())

    @staticmethod
    def analyze_synthesizability(frame: pl.DataFrame) -> SynthesizabilityStatistics:
        return analyze_synthesizability(frame)

    def anonymize(
        self, frame: pl.DataFrame, config: AnonymizeConfig
    ) -> AnonymizeResult:
        """Prepare, synthesize, count combinations on both sides and evaluate.

        The synthesizer runs exactly once; the result carries everything needed
        to re-evaluate and query without it.
        """
        sensitive = prepare_frame(frame)
        table = to_raw_table(sensitive)
        output = self.synthesizer.synthesize(table, config)
        n = output.protected_record_count
        delta = 1.0 / (math.log(n) * n) if n > 1 else None
        result = AnonymizeResult(
            config=config,
            sensitive=sensitive,
            synthetic=output.synthetic.to_frame(),
            protected_record_count=n,
            delta=delta,
            sensitive_aggregates=count_combinations(
                table,
                config.reporting_length,
                config.value_separator,
                config.sensitive_zeros,
            ),
            dp_aggregates=parse_aggregates(
                output.dp_aggregates, config.attribute_separator
            ),
            synthetic_aggregates=count_combinations(
                output.synthetic,
                config.reporting_length,
                config.value_separator,
                config.sensitive_zeros,
            ),
            raw_dp_aggregates=dict(output.dp_aggregates),
        )
        return self.evaluate(result)

    @staticmethod
    def evaluate(result: AnonymizeResult) -> AnonymizeResult:
        """Recompute both error reports from the kept aggregates."""
        result.aggregate_error_report = error_report(
            result.sensitive_aggregates, result.dp_aggregates
        )
        result.synthetic_error_report = error_report(
            result.sensitive_aggregates, result.synthetic_aggregates
        )
        return result

    @staticmethod
    def links(
        result: AnonymizeResult,
        selection: Sequence[AttributeValue],
        source_attribute: str,
        target_attribute: str,
        highlight: AttributeValue | None = None,
    ) -> pl.DataFrame:
        """Flow-chart links, exact from DP aggregates when they can be, else synthetic.

        DP aggregates cover the edge pair, the selection and the highlight only
        while that many attributes fit within the configured reporting length.
        """
        attribute_count = 2 + len(selection) + (1 if highlight is not None else 0)
        if attribute_count <= result.config.reporting_length:
            return aggregate_links(
                result.dp_aggregates,
                selection,
                source_attribute,
                target_attribute,
                highlight,
                result.config.value_separator,
            )
        return synthetic_links(
            result.synthetic, selection, source_attribute, target_attribute, highlight
        )
