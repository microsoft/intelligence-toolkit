# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


class IntelligenceWorkflow:
    def __init__(self, ai_configuration=None):
        self.ai_configuration = ai_configuration

    def set_ai_configuration(self, ai_configuration) -> None:
        self.ai_configuration = ai_configuration