# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""The pacsynth adapter, exercised against a recording stand-in module.

The real library is never imported here; the step-3 comparison script and
the repository smoke tests cover it.
"""

from __future__ import annotations

import sys
import types

import pytest
from itk_anonymize.models import AnonymizeConfig, RawTable
from itk_anonymize.pacsynth_engine import PacsynthSynthesizer, build_parameters


class RecordingBuilder:
    def __init__(self) -> None:
        self.calls: dict[str, object] = {}

    def __getattr__(self, name: str):
        def setter(value):
            self.calls[name] = value
            return self

        return setter

    def build(self):
        return ("params", self.calls)


class Mode:
    def __init__(self, family: str, name: str) -> None:
        self.family, self.name = family, name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mode) and (self.family, self.name) == (
            other.family,
            other.name,
        )

    def __hash__(self) -> int:
        return hash((self.family, self.name))


def mode_class(family: str, names: list[str]) -> type:
    return type(
        family, (), {n: staticmethod(lambda n=n: Mode(family, n)) for n in names}
    )


class FakeDataset:
    def __init__(self, raw, sensitive_zeros=None) -> None:
        self.raw = raw
        self.sensitive_zeros = sensitive_zeros


class FakeSynth:
    def __init__(self, params) -> None:
        self.params = params
        self.fitted = None

    def fit(self, dataset) -> None:
        self.fitted = dataset

    def get_dp_aggregates(self, sep: str) -> dict[str, int]:
        return {f"A{self.params[1]['empty_value'] or ':'}x{sep}B:y": 3}

    def get_dp_number_of_records(self) -> int:
        return 42

    def sample(self):
        header = list(self.fitted.raw[0])
        return [header, ["x", "y"][: len(header)]]


@pytest.fixture()
def fake_pacsynth(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("pacsynth")
    module.AccuracyMode = mode_class(
        "AccuracyMode",
        ["prioritize_long_combinations", "prioritize_short_combinations", "balanced"],
    )
    module.FabricationMode = mode_class(
        "FabricationMode", ["balanced", "progressive", "minimize", "uncontrolled"]
    )
    module.DpAggregateSeededParametersBuilder = RecordingBuilder
    module.Dataset = FakeDataset
    module.DpAggregateSeededSynthesizer = FakeSynth
    monkeypatch.setitem(sys.modules, "pacsynth", module)
    return module


def test_build_parameters_maps_every_knob(fake_pacsynth) -> None:
    config = AnonymizeConfig(
        epsilon=3.5,
        reporting_length=2,
        percentile_percentage=90,
        percentile_epsilon_proportion=0.02,
        number_of_records_epsilon_proportion=0.01,
        weight_selection_percentile=80,
        accuracy_mode="balanced",
        fabrication_mode="minimized",
        empty_value="-",
        use_synthetic_counts=False,
        aggregate_counts_scale_factor=2.0,
    )
    _, calls = build_parameters(config, fake_pacsynth)
    assert calls == {
        "reporting_length": 2,
        "epsilon": 3.5,
        "percentile_percentage": 90,
        "percentile_epsilon_proportion": 0.02,
        "accuracy_mode": Mode("AccuracyMode", "balanced"),
        "number_of_records_epsilon_proportion": 0.01,
        "fabrication_mode": Mode("FabricationMode", "minimize"),
        "empty_value": "-",
        "weight_selection_percentile": 80,
        "use_synthetic_counts": False,
        "aggregate_counts_scale_factor": 2.0,
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("balanced", "balanced"),
        ("progressive", "progressive"),
        ("minimized", "minimize"),
        ("uncontrolled", "uncontrolled"),
    ],
)
def test_fabrication_mode_names(fake_pacsynth, name: str, expected: str) -> None:
    config = AnonymizeConfig(epsilon=1, fabrication_mode=name)  # type: ignore[arg-type]
    _, calls = build_parameters(config, fake_pacsynth)
    assert calls["fabrication_mode"] == Mode("FabricationMode", expected)


def test_synthesize_round_trip(fake_pacsynth) -> None:
    table = RawTable(("A", "B"), (("x", "y"), ("", "y")))
    config = AnonymizeConfig(epsilon=1, sensitive_zeros=("B",), attribute_separator="|")
    output = PacsynthSynthesizer().synthesize(table, config)
    assert output.protected_record_count == 42
    assert output.dp_aggregates == {"A:x|B:y": 3}
    assert output.synthetic == RawTable(("A", "B"), (("x", "y"),))


def test_sensitive_zeros_default_to_none(fake_pacsynth) -> None:
    synthesizer = PacsynthSynthesizer()
    seen = {}
    original = fake_pacsynth.Dataset

    def spy(raw, sensitive_zeros=None):
        seen["zeros"] = sensitive_zeros
        return original(raw, sensitive_zeros)

    fake_pacsynth.Dataset = spy
    synthesizer.synthesize(RawTable(("A",), (("x",),)), AnonymizeConfig(epsilon=1))
    assert seen["zeros"] is None
