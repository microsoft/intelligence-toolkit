# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

from app.workflows.security.metaprompts import (
    do_not_disrespect_context,
    do_not_harm_question_answering,
)

chunk_relevance_prompt = """\
You are a helpful assistant determining whether a given chunk of text has the potential to be relevant to a user query.

Text chunk:

{chunk}

User query:

{query}

--Task--
Answer "Yes" if the text chunk contains information that is potentially relevant to the user query, either directly or indirectly, and "No" if it does not.

The text chunk does not need to answer the query, but should contain information that could be used to construct an answer. For example, the following types of information should all be considered relevant:

- specific facts, claims, or statistics
- background information or thematic context
- applicable rules, guidelines, or policies

Answer with a single word, "Yes" or "No", with no additional output.
"""

user_prompt = """\
The final report should:
- be formatted in markdown, with headings indicated by # symbols and subheadings indicated by additional # symbols
- use plain English accessible to non-native speakers and non-technical audiences
- support each sentence with a source reference in the same format as the input content, "[source: <file> (<chunk>), ...]"
- include ALL source references from the extended answer, combining them as needed if they support the same claim
"""
report_prompt = """\
You are a helpful assistant tasked with generating an output from an extended report.

=== TASK ===

User query:

{query}

Extended answer:

{answer}

"""

query_anchoring_prompt = """\
You are a helpful assistant tasked with rewriting a user query in a way that better matches the concepts contained in the dataset.

The output query should retain all the key phrases from the input query, but may expand on them with additional concepts and phrasing to better match relevant concepts in the dataset. Include only the most relevant concepts, especially as specific examples or general categories of concepts present in the user query. If no concepts are relevant, the output query should be the same as the input query. Keep the output query at 1-2 sentences.

The output query should not be more specific than the input query. For example, if the input query is "What are the effects of climate change on the environment?" and "Arctic" is a provided concept, the output query should not be "What are the effects of climate change on the environment in the Arctic?", but could be "What are the effects of climate change on the environment, for example in the Arctic?".

User query:

{query}

Data concepts:

{concepts}

Output query:
"""

theme_summarization_prompt = """\
You are a helpful assistant tasked with creating a JSON object that summarizes a theme relevant to a given user query.

When presenting source evidence, support each sentence with a source reference to the file and text chunk: "[source: <source_id_1>, <source_id_2>, ...]". Include source IDs only - DO NOT include the chunk ID within the source ID - DO NOT repeat the same source ID within a single sentence.

The output object should summarize the theme as follows:

- "theme_title": a title for the theme, in the form of a claim statement supported by the points to follow
- "point_title": a title for a specific point within the theme, in the form of a claim statement
- "point_evidence": a paragraph, starting with "**Source evidence**:", describing evidence from sources that support or contradict the point, without additional interpretation
- "point_commentary": a paragraph, starting with "**AI commentary**:", suggesting inferences, implications, or conclusions that could be drawn from the source evidence

--Query--

{query}

--Theme hint--

{theme}

--Source text chunks--

Input text chunks JSON, in the form "<source_id>: <text_chunk>":

{chunks}

Output JSON object:
"""

theme_integration_prompt = """\
You are a helpful assistant tasked with creating a JSON object that organizes content relevant to a given user query.

The output object should integrate the theme summaries provided as input as follows:

- "answer": a standalone and detailed answer to the user query, derived from the points and formatted according to the user query/prompt. Quote directly from source text where appropriate, and provide a source reference for each quote
- "report_title": a title for the final report that reflects the overall theme of the content and the user query it answers, in the form of a claim statement. Should not contain punctuation or special characters beyond spaces and hyphens
- "report_overview": an introductory paragraph that provides an overview of the report themes in a neutral way without offering interpretations or implications
- "report_implications": a concluding paragraph that summarizes the implications of the themes and their specific points

When presenting evidence, support each sentence with one or more source references: "[source: <source_id_1>, <source_id_2>,...]". Include source IDs only - DO NOT include the chunk ID within the source ID - DO NOT repeat the same source ID within a single sentence.


--Theme summaries--

{content}

--User query--

{query}

Output JSON object:
"""

commentary_prompt = """\
You are a helpful assistant tasked with providing commentary on a set of themes derived from source texts.

Provide commentary both on the overall thematic structure and specific examples drawn from the sample source texts.

When presenting evidence, support each sentence with one or more source references: "[source: <source_id_1>, <source_id_2>, ...]". Include source IDs only - DO NOT include the chunk ID within the source ID - DO NOT repeat the same source ID within a single sentence.

--User query--

{query}

--Themes--

{structure}

--Sample source texts--

{chunks}

--Output commentary--

"""

thematic_update_prompt = """\
You are a helpful assistant tasked with creating a JSON object that updates a thematic organization of points relevant to a user query.

The output object should capture new themes, points, and source references that should be added to or modify the existing thematic structure:

- "updates": an array of objects, each representing an update to a point derived from the input text chunks
- "point_id": the ID of the point to update, else the next available point ID if creating a new point
- "point_title": the title of the point to update or create, expressed as a full and detailed sentence. If the existing point title is unchanged, the field should be left blank
- "source_ids": an array of source IDs that support the point, to be added to the existing source IDs for the point
- "theme_title": the title of a theme that organizes a set of related points.

--Rules--

- Each point MUST contain sufficient concrete details to capture the specific source information only, and not related information
- If a source relates to an existing point, the source ID MUST be assigned to the existing point ID, rather than creating a new point
- If the addition of a source to a point warrants a change in point title, the point title MUST be updated
- Aim for 3-7 themes overall, with an even distribution of points across themes
- Points should be assigned to a single theme in a logical sequence that addresses the user query
- Themes should contain at least two points if possible
- Order themes in a logical sequence that addresses the user query
- Output themes need not be the same as input themes and should be regenerated as needed to maintain 3-7 themes overall

--User query--

{query}

--Existing thematic structure--

{structure}

--New sources by source ID--

{sources}

--Output JSON object--

"""

list_prompts = {
    "report_prompt": report_prompt,
    "user_prompt": user_prompt,
    "safety_prompt": " ".join([
        do_not_harm_question_answering,
        do_not_disrespect_context,
    ]),
}
