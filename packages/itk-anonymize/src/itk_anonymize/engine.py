# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""The one external dependency of the pipeline, as a protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from itk_anonymize.models import AnonymizeConfig, RawTable, SynthesisOutput


class Synthesizer(Protocol):
    """Fits a differentially private model on a text table and samples from it.

    ``dp_aggregates`` keys must be ``attribute<value_separator>value`` parts
    joined by ``config.attribute_separator``; any part order is accepted.
    """

    def synthesize(
        self, table: RawTable, config: AnonymizeConfig
    ) -> SynthesisOutput: ...
