# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
import pytest
from itk_networks.exposure import exposure_data, exposure_nodes, render_exposure
from itk_networks.models import (
    Exposure,
    ExposureNode,
    ExposurePath,
    ExposureSummary,
    FlagTable,
)
from itk_networks.prepare import flag_links, integrate_flags
from itk_networks_testing import A, E, path_graph


def _flags(**counts: int) -> pl.DataFrame:
    frame = pl.DataFrame({"name": list(counts), "f": list(counts.values())})
    return integrate_flags(
        flag_links(FlagTable(frame=frame, entity_column="name", columns=["f"]))
    )[0]


@pytest.fixture()
def chain():
    # A -phone- B -email- C ; D also on the phone
    graph = path_graph(E("A"), A("phone", "1"), E("B"), A("email", "x"), E("C"))
    graph.add_node(E("D"), type="ENTITY")
    graph.add_edge(E("D"), A("phone", "1"))
    return graph


def test_exposure_nodes_add_inferred_endpoints_present_in_graph(chain):
    links = {E("A"): frozenset({E("D"), E("Z")})}
    assert exposure_nodes([E("A")], chain, links) == [E("A"), E("D")]


def test_exposure_summary_and_paths(chain):
    flags = _flags(A=1, B=2, C=0, D=3)
    exposure = exposure_data(flags, [E("A"), E("B"), E("C"), E("D")], "C", chain)
    assert exposure.summary == ExposureSummary(
        direct=0, indirect=6, paths=2, entities=3
    )
    assert exposure.paths == (
        ExposurePath((
            (E("A"), E("D")),
            (A("phone", "1"),),
            (E("B"),),
            (A("email", "x"),),
            (E("C"),),
        )),
        ExposurePath(((E("B"),), (A("email", "x"),), (E("C"),))),
    )
    assert exposure.nodes == (
        ExposureNode(E("A"), flags=1),
        ExposureNode(A("phone", "1"), entities=3),
        ExposureNode(E("B"), flags=2),
        ExposureNode(A("email", "x"), entities=2),
        ExposureNode(E("D"), flags=3),
    )


def test_exposure_counts_direct_flags_and_excludes_self_from_related(chain):
    flags = _flags(A=1, C=4)
    exposure = exposure_data(flags, [E("A"), E("C")], "C", chain)
    assert exposure.summary == ExposureSummary(
        direct=4, indirect=1, paths=1, entities=1
    )


def test_exposure_without_flags_is_empty(chain):
    exposure = exposure_data(_flags().clear(), [E("A")], "A", chain)
    assert exposure == Exposure("A", ExposureSummary(0, 0, 0, 0), (), ())
    assert "has **0** direct flags" in exposure.markdown


def test_exposure_for_entity_outside_graph_has_no_paths(chain):
    exposure = exposure_data(_flags(A=1), [E("A")], "nope", chain)
    assert exposure.summary == ExposureSummary(
        direct=0, indirect=1, paths=0, entities=1
    )


def test_exposure_ignores_flagged_entities_outside_the_network(chain):
    flags = _flags(A=1, D=5)
    exposure = exposure_data(flags, [E("A"), E("B")], "B", chain)
    assert exposure.summary.indirect == 1


def test_render_matches_legacy_markdown(chain):
    flags = _flags(A=1, B=2, D=3)
    text = render_exposure(
        exposure_data(flags, [E("A"), E("B"), E("C"), E("D")], "C", chain)
    )
    assert text == (
        "##### Flag Exposure Paths\n\n"
        "The selected entity **C** has **0** direct flags and is linked to **6** "
        "indirect flags via **2** paths from **3** related entities:\n\n"
        "**Path 1**\n\n```\n"
        "ENTITY==A [linked to 1 flags]\n"
        "ENTITY==D [linked to 3 flags]\n"
        "--->\n"
        "  phone==1 [linked to 3 entities]\n"
        "  --->\n"
        "    ENTITY==B [linked to 2 flags]\n"
        "    --->\n"
        "      email==x [linked to 2 entities]\n"
        "      --->\n"
        "        ENTITY==C [linked to 0 flags]\n"
        "```\n\n"
        "**Path 2**\n\n```\n"
        "ENTITY==B [linked to 2 flags]\n"
        "--->\n"
        "  email==x [linked to 2 entities]\n"
        "  --->\n"
        "    ENTITY==C [linked to 0 flags]\n"
        "```\n\n"
    )
