# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from unittest.mock import patch

import pandas as pd

from python.risk_networks.text_format import clean_text, format_data_columns


class TestCleanText:
    def test_remove_punctuation(self):
        assert clean_text("Hello, world!") == "Hello world"

    def test_remove_special_characters(self):
        assert clean_text("Hello, world!") == "Hello world"

    def test_reduce_multiple_spaces_to_single(self):
        assert clean_text("Hello    world") == "Hello world"

    def test_allow_special_characters(self):
        assert (
            clean_text("Email me@home.com & bring snacks+")
            == "Email me@homecom & bring snacks+"
        )

    def test_combined_scenarios(self):
        assert (
            clean_text("Hello,  world! Email me@home.com & bring snacks+")
            == "Hello world Email me@homecom & bring snacks+"
        )


class TestFormatDataColumns:
    def test_multiple_columns(self):
        initial_df = pd.DataFrame({
            "entity_id": ["123 ", " 456"],
            "name": ["John Doe", "Jane Doe"],
            "email": ["john@doe.com", "jane@doe.com"],
        })
        expected_df = pd.DataFrame({
            "entity_id": ["123", "456"],
            "name": ["John Doe", "Jane Doe"],
            "email": ["john@doecom", "jane@doecom"],
        })
        columns_to_link = ["name", "email"]
        entity_id_column = "entity_id"

        # Exercise
        result_df = format_data_columns(initial_df, columns_to_link, entity_id_column)

        # Verify
        assert result_df.equals(expected_df)  # Assuming clean_text returns input as-is

    @patch("re.sub")
    def test_empty_dataframe(self, mock_clean_text):
        # Setup
        mock_clean_text.side_effect = lambda x: x
        initial_df = pd.DataFrame({"entity_id": [], "name": [], "email": []})
        columns_to_link = ["name", "email"]
        entity_id_column = "entity_id"

        # Exercise
        result_df = format_data_columns(initial_df, columns_to_link, entity_id_column)

        assert mock_clean_text.call_count == 0
        assert result_df.equals(initial_df)

    @patch("re.sub")
    def test_special_characters_in_entity_id(self, mock_clean_text):
        # Setup
        mock_clean_text.side_effect = lambda _x, _y, _z: "cleaned"
        initial_df = pd.DataFrame({
            "entity_id": ["@123!", "#456$"],
            "name": ["John Doe", "Jane Doe"],
        })
        columns_to_link = ["name"]
        entity_id_column = "entity_id"

        # Exercise
        result_df = format_data_columns(initial_df, columns_to_link, entity_id_column)

        # Verify
        assert mock_clean_text.call_count == 8  # 4 for entity_id + 4 for name
        for val in result_df[entity_id_column]:
            assert val == "cleaned"
            assert val == "cleaned"
            assert val == "cleaned"
            assert val == "cleaned"
