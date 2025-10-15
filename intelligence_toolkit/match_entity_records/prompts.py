# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from intelligence_toolkit.AI.metaprompts import (
    do_not_disrespect_context,
    do_not_harm,
)
report_prompt = """\
Goal: evaluate the overall RELATEDNESS of the records in each record group provided on a scale of 0-10, where 0 is definitively different entities and 10 is definitivly the same entity or entity group (e.g., branches of a company).

Output the rows of a CSV file containing the Group ID, Relatedness, and Explanation.

Do not output ``` or the column headers - start directly with the row values and separate each row with a newline. Output Group ID and Relatedness directly, but wrap explanations in "".

=== TASK ===

Group data:

{data}
"""

user_prompt = """\
Factors indicating unrelatedness: multiple fields having values that are different across grouped records, have no similarity, and are unrelated in the real-world.

Factors indicating relatedness: multiple fields having values that are the same or similar across multiple grouped records, and are related in the real-world.

Factors that should be ignored: inconsistent spelling, formatting, and missing values.

Factors that should be considered in the event of similar names: the more additional fields that are the same, the more likely the records are related.

If names are in a language other than English, consider whether the English translations are generic descriptive terms (less likely to be related) or distinctive (more likely to be related).

Keep explanations short and simple.

"""


list_prompts = {
    "report_prompt": report_prompt,
    "user_prompt": user_prompt,
    "safety_prompt": f"{do_not_harm} {do_not_disrespect_context}",
}
