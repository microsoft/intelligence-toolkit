# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from intelligence_toolkit.AI.metaprompts import do_not_harm

report_prompt = """\
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

Where possible, the text should add numeric counts, ranks, and deltas in parentheses to support its claims, but should avoid using complex column names directly.

"""

user_prompt = """\
The report should be structured in markdown and use plain English accessible to non-native speakers and non-technical audiences.
"""

list_prompts = {
    "report_prompt": report_prompt,
    "user_prompt": user_prompt,
    "safety_prompt": f"{do_not_harm}",
}
