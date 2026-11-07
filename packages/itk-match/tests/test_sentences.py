# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from itk_match.sentences import to_sentence, to_sentences
from itk_match.unify import unify


def test_to_sentence_uppercases_and_skips_blank_values():
    sentence = to_sentence("Acme", {"address": "1 Main St", "city": None, "phone": ""})
    assert sentence == "NAME: ACME; ADDRESS: 1 MAIN ST"


def test_to_sentences_follows_row_order_and_label_order(dataset_one, dataset_two):
    records = unify([dataset_one, dataset_two])
    sentences = to_sentences(records)

    assert len(sentences) == records.height
    assert sentences[0] == (
        "NAME: ADVENTURE GEAR; ADDRESS: 1010 TREKKING TRL; CITY: ADVENTURE CITY"
    )
    assert sentences[3] == (
        "NAME: ADVENTURE GEAR; ADDRESS: 1010 TREKKING TRL; "
        "CITY: ADVENTURE CITY; PHONE: 111"
    )
