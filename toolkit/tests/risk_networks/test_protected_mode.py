# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from unittest.mock import patch

import pandas as pd

from toolkit.risk_networks.protected_mode import is_numeric_column, protect_data


class TestProtectData:
    @patch("re.match")
    def test_is_numeric_column(self, mock_re_match):
        # Setup
        mock_re_match.side_effect = [True, True, False, True]

        # Exercise
        result = is_numeric_column(pd.Series([1, 2, "3", 4.5]))

        # Verify
        assert result is False

    @patch("re.match")
    def test_protect_data_with_numeric_column(self, mock_re_match):
        # Setup
        mock_re_match.side_effect = [True, True, True]
        data_df = pd.DataFrame(
            {
                "entity_id": ["123", "456", "789"],
                "value": [1, 2, 3],
            }
        )
        value_cols = ["value"]
        entity_col = "entity_id"
        entities_renamed = []

        # Exercise
        result_df, result_entities_renamed, result_attributes_renamed = protect_data(
            data_df, value_cols, entity_col, entities_renamed
        )

        # Verify
        expected_df = pd.DataFrame(
            {
                "entity_id": [
                    "Protected_Entity_1",
                    "Protected_Entity_2",
                    "Protected_Entity_3",
                ],
                "value": [1, 2, 3],
            }
        )
        expected_entities_renamed = [
            ("123", "Protected_Entity_1"),
            ("456", "Protected_Entity_2"),
            ("789", "Protected_Entity_3"),
        ]

        expected_attributes_renamed = [(1, 1), (2, 2), (3, 3)]

        assert result_df.equals(expected_df)
        assert result_entities_renamed == expected_entities_renamed
        assert result_attributes_renamed == expected_attributes_renamed

    @patch("re.match")
    def test_protect_data_with_existing_entity_name(self, mock_re_match):
        # Setup
        mock_re_match.side_effect = [True, True, True]
        data_df = pd.DataFrame(
            {
                "entity_id": ["123", "456", "789"],
                "value": [1, 2, 3],
            }
        )
        value_cols = ["value"]
        entity_col = "entity_id"
        entities_renamed = [("123", "Protected_Entity_1")]

        # Exercise
        result_df, result_entities_renamed, result_attributes_renamed = protect_data(
            data_df, value_cols, entity_col, entities_renamed
        )

        # Verify
        expected_df = pd.DataFrame(
            {
                "entity_id": [
                    "Protected_Entity_1",
                    "Protected_Entity_2",
                    "Protected_Entity_3",
                ],
                "value": [1, 2, 3],
            }
        )
        expected_entities_renamed = [
            ("123", "Protected_Entity_1"),
            ("456", "Protected_Entity_2"),
            ("789", "Protected_Entity_3"),
        ]
        expected_attributes_renamed = [(1, 1), (2, 2), (3, 3)]

        assert result_df.equals(expected_df)
        assert result_entities_renamed == expected_entities_renamed
        assert result_attributes_renamed == expected_attributes_renamed

    @patch("re.match")
    def test_protect_data_with_existing_attribute_name(self, mock_re_match):
        # Setup
        mock_re_match.side_effect = [True, True, False, True]
        data_df = pd.DataFrame(
            {
                "entity_id": ["123", "456", "789"],
                "value": [1, 2, 3],
            }
        )
        value_cols = ["value"]
        entity_col = "entity_id"
        entities_renamed = []
        attributes_renamed = [(1, "value_1")]

        # Exercise
        result_df, result_entities_renamed, result_attributes_renamed = protect_data(
            data_df, value_cols, entity_col, entities_renamed, attributes_renamed
        )

        # Verify
        expected_df = pd.DataFrame(
            {
                "entity_id": [
                    "Protected_Entity_1",
                    "Protected_Entity_2",
                    "Protected_Entity_3",
                ],
                "value": ["value_1", "value_2", "value_3"],
            }
        )
        expected_entities_renamed = [
            ("123", "Protected_Entity_1"),
            ("456", "Protected_Entity_2"),
            ("789", "Protected_Entity_3"),
        ]
        expected_attributes_renamed = [
            (1, "value_1"),
            (2, "value_2"),
            (3, "value_3"),
        ]

        assert result_df.equals(expected_df)
        assert result_entities_renamed == expected_entities_renamed
        assert result_attributes_renamed == expected_attributes_renamed
