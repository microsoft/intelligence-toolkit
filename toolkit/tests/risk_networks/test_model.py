# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import pandas as pd
import pytest

from toolkit.risk_networks.config import AttributeColumnType
from toolkit.risk_networks.model import prepare_entity_attribute


class TestPrepareEntityAttribute:
    @pytest.fixture(autouse=True)
    def _setup_method(self):
        self.data = pd.DataFrame(
            {
                "entity_id": [1, 2, 3],
                "attribute1": ["A", "B", "A"],
                "attribute2": ["X", "Y", "X"],
            }
        )
        self.entity_id_column = "entity_id"
        self.columns_to_link = ["attribute1", "attribute2"]

    def test_column_name(self):
        entity_links = prepare_entity_attribute(
            self.data,
            self.entity_id_column,
            AttributeColumnType.ColumnName.value,
            self.columns_to_link,
        )
        assert len(entity_links) == 2  # One for each attribute column

    def test_custom_name(self):
        custom_name = "custom_attribute"
        entity_links = prepare_entity_attribute(
            self.data,
            self.entity_id_column,
            AttributeColumnType.CustomName.value,
            self.columns_to_link,
            attribute_name=custom_name,
        )
        assert (
            len(entity_links) == 2
        )  # One for each attribute column, but with custom name

    def test_other(self):
        entity_links = prepare_entity_attribute(
            self.data,
            self.entity_id_column,
            None,  # Simulating other attribute column types
            self.columns_to_link,
        )
        assert len(entity_links) == 2  # One for each attribute column
