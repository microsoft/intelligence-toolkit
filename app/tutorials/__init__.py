# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os


def get_tutorial(workflow_name: str) -> str:
    file_path = os.path.join(os.path.dirname(__file__), f"{workflow_name}.md")
    with open(file_path) as file:
        return file.read()
