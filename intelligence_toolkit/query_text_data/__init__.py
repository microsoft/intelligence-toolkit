# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import os
from intelligence_toolkit.query_text_data.api import QueryTextData
from intelligence_toolkit.query_text_data.classes import (
    ProcessedChunks,
    ChunkSearchConfig,
    AnswerConfig,
    AnswerObject,
)


def get_readme() -> str:
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()
