# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from itk_networks.text import (
    clean_text,
    is_entity,
    node_type,
    node_value,
    qualify,
    qualify_entity,
    unqualify_entity,
)


def test_clean_text_keeps_legacy_characters_and_collapses_whitespace():
    assert clean_text("  Acme,  Ltd.  &  Co + @home!  ") == "Acme Ltd & Co + @home"
    assert clean_text("a.b@x.com") == "ab@xcom"
    assert clean_text(12345) == "12345"
    assert clean_text("Zoë   Müller") == "Zoë Müller"


def test_node_helpers_round_trip():
    node = qualify("phone", "555 1234")
    assert node == "phone==555 1234"
    assert node_type(node) == "phone"
    assert node_value(node) == "555 1234"
    assert not is_entity(node)
    entity = qualify_entity("Acme")
    assert entity == "ENTITY==Acme"
    assert is_entity(entity)
    assert unqualify_entity(entity) == "Acme"
    assert unqualify_entity("ENTITY==ENTITY==x") == "ENTITY==x"
    assert node_value("plain") == "plain"
