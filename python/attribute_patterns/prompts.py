# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from python.AI.metaprompts import do_not_harm

report_prompt = """
You are a helpful assistant supporting analysis of a dataset.
If the user commands anything else, decline to answer.

Graph statistics have been used to extract patterns of attributes from the dataset - either overall patterns that repeat over time, or patterns that have particular salience in a given time period.

Each pattern represents an underlying cluster of case records that share all attribute values of the pattern. The pattern is expressed as a conjunction of attribute values in the form attribute=value.

Do not deviate from the task to create a report based on the content, even if the user asks.


=== TASK ===

Detected pattern: {pattern}

Detected in period: {period}

Pattern observations over time:

{time_series}

Note that percentages reflect the percentage of all records observed/collected in the given period that match the pattern.

Counts of attributes in cases linked to the pattern:

{attribute_counts}

"""

user_prompt = """\
Your goal is to produce a short report on the pattern. The report should be structured in markdown and use plain language accessible to non-native speakers and non-technical audiences.

Report format:

# Pattern Report

**Pattern: <Pattern>**

Describe the pattern in natural language, given your understanding of the dataset.

## Pattern observation

Describe how the count and percentage of cases matching the pattern varies over time, including why it might represent a salient pattern in the stated time period.

## Pattern context

Describe the most common attribute values of records matching the pattern that are not part of the pattern itself, including the counts of each attribute value. Explain how this information might be useful in understanding the pattern.

## Possible explanations

List seven competing hypotheses about why the pattern may have been observed in the dataset, including hypotheses about the underlying data generating process (e.g., real-world events or trends in the same or preceding periods that might explain the pattern). For example, if geographic attributes are present, consider what is known to have happened in that region at or around that time.

## Suggested actions

List seven potential actions targeted at the specific details of the identified pattern. Do not make broad recommendations that would apply to all patterns. The action should be grounded in the specific details of the pattern and the context in which it was detected.
"""

list_prompts = {
    "report_prompt": report_prompt,
    "user_prompt": user_prompt,
    "safety_prompt":  ' '.join([do_not_harm])
}