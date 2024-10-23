# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import os
from .api import GenerateMockData
from .schema_builder import create_boilerplate_schema

def get_readme():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()
    
__all__ = [
    "GenerateMockData",
    "create_boilerplate_schema"
]