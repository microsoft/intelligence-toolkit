# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Prompt text for the group comparison report, unchanged from the legacy workflow."""

from itk_core.metaprompts import do_not_harm

REPORT_PROMPT = """\
You are a data analyst preparing a detailed "group comparison report" on a given dataset.

Your report should clearly describe the focus of the data in terms of filters applied, both to create the initial data summary and to select the groups for the final report.

All claims should be supported by examples drawn from the data, including comparison to related groups/attribute values etc.

=== TASK ===

Dataset description:

{description}

Group filters:

{filters}

Dataset:

{dataset}

Note: The group_attribute_count field represents the overall count for each group across all temporal windows.

Where possible, the text should add numeric counts, ranks, and deltas in parentheses to support its claims, but should avoid using complex column names directly.

"""

USER_PROMPT = """\
The report should be structured in markdown and use plain English accessible to non-native speakers and non-technical audiences.
"""

SAFETY_PROMPT = do_not_harm
