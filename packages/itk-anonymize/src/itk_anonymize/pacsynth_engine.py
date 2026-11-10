# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""pacsynth behind the Synthesizer protocol. The only module importing pacsynth."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from itk_anonymize.models import RawTable, SynthesisOutput

if TYPE_CHECKING:
    from itk_anonymize.models import AnonymizeConfig


def build_parameters(config: AnonymizeConfig, pacsynth: Any) -> Any:
    """Map the config onto ``DpAggregateSeededParametersBuilder``.

    ``pacsynth`` is the imported module, passed in so the mapping can be
    tested against a stand-in without the real library.
    """
    accuracy = getattr(pacsynth.AccuracyMode, config.accuracy_mode)()
    fabrication_name = (
        "minimize"
        if config.fabrication_mode == "minimized"
        else config.fabrication_mode
    )
    fabrication = getattr(pacsynth.FabricationMode, fabrication_name)()
    return (
        pacsynth.DpAggregateSeededParametersBuilder()
        .reporting_length(config.reporting_length)
        .epsilon(config.epsilon)
        .percentile_percentage(config.percentile_percentage)
        .percentile_epsilon_proportion(config.percentile_epsilon_proportion)
        .accuracy_mode(accuracy)
        .number_of_records_epsilon_proportion(
            config.number_of_records_epsilon_proportion
        )
        .fabrication_mode(fabrication)
        .empty_value(config.empty_value)
        .weight_selection_percentile(config.weight_selection_percentile)
        .use_synthetic_counts(config.use_synthetic_counts)
        .aggregate_counts_scale_factor(config.aggregate_counts_scale_factor)
        .build()
    )


class PacsynthSynthesizer:
    """``DpAggregateSeededSynthesizer`` fit and sample, nothing else."""

    def synthesize(self, table: RawTable, config: AnonymizeConfig) -> SynthesisOutput:
        import pacsynth

        dataset = pacsynth.Dataset(
            table.as_lists(), sensitive_zeros=list(config.sensitive_zeros) or None
        )
        synth = pacsynth.DpAggregateSeededSynthesizer(
            build_parameters(config, pacsynth)
        )
        synth.fit(dataset)
        return SynthesisOutput(
            dp_aggregates=synth.get_dp_aggregates(config.attribute_separator),
            protected_record_count=synth.get_dp_number_of_records(),
            synthetic=RawTable.from_lists(synth.sample()),
        )
