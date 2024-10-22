# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


from toolkit.AI.openai_configuration import OpenAIConfiguration


class IntelligenceWorkflow:
    # Base class for all AI workflows
    def __init__(self, ai_configuration: OpenAIConfiguration | None = None) -> None:
        self.ai_configuration = ai_configuration

    def set_ai_configuration(self, ai_configuration: OpenAIConfiguration) -> None:
        self.ai_configuration = ai_configuration