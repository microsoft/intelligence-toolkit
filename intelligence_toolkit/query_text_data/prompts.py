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

The output query should retain all the key phrases from the input query, but may expand on them with additional concepts and phrasing to better match relevant concepts in the dataset. Include as many relevant concepts as possible, especially as specific examples or general categories of concepts present in the user query. If no concepts are relevant, the output query should be the same as the input query.

User query:

{query}

Data concepts:

{concepts}

Output query:
"""

claim_extraction_prompt = """\
You are a helpful assistant tasked with creating a JSON object that extracts relevant claims from a collection of input text chunks.

Given a query, the output object should extract claims from the input text chunks as follows:

- "claim_context": an overall description of the context in which claims are made
- "claim_statement": a statement-based formatting of a claim that is relevant to the user query and includes relevant contextual information (e.g., time and place)
- "claim_attribution": any named source or author of a claim, beyond the title of the text 
- "supporting_sources": a list of source IDs that support the claim
- "contradicting_sources": a list of source IDs that contradict the claim

--TASK--

Question for which claims are being extracted:

{query}

Input text chunks JSON, in the form "<source_id>: <text_chunk>":

{chunks}

Output JSON object:
"""

claim_summarization_prompt = """\
You are a helpful assistant tasked with creating a JSON object that summarizes claims relevant to a given user query.

When presenting source evidence, support each sentence with a source reference to the file and text chunk: "[source: <source_id>, <source_id>]". Include source IDs only - DO NOT include the chunk ID within the source ID.

The output object should summarize all claims from input text chunks as follows:

- "content_title": a title for a specific content item spanning related claims, in the form of a derived claim statement
- "content_summary": a paragraph, starting with "**Source evidence**:", describing each of the individual claims and the balance of evidence supporting or contradicting them
- "content_commentary": a paragraph, starting with "**AI commentary**:", suggesting inferences, implications, or conclusions that could be drawn from the source evidence

--TASK--

Input text chunks JSON, in the form "<source_id>: <text_chunk>":

{chunks}

Input claim analysis JSON:

{analysis}

Output JSON object:
"""

content_integration_prompt = """\
You are a helpful assistant tasked with creating a JSON object that organizes content relevant to a given user query.

The output object should summarize all claims from input text chunks as follows:

- "query": the user query/prompt that the report answers
- "answer": a standalone and detailed answer to the user query, derived from the content items and formatted according to the user query/prompt. Quote directly from source text where appropriate, and provide a source reference for each quote
- "report_title": a title for the final report that reflects the overall theme of the content and the user query it answers, in the form of a derived claim statement. Should not contain punctuation or special characters beyond spaces and hyphens
- "report_summary": an introductory paragraph describes the themes and content items in the report, without offering additional interpretation beyond the content items
- "theme_order": the order in which to present the themes in the final report
- "theme_title": a title for a specific theme spanning related content items, in the form of a derived claim statement
- "theme_summary": an introductory paragraph that links the content items in the theme to the user query, without additional intepretation
- "content_id_order": the order in which to present the content items within the theme, specified by content item id. If content items make the same overall point, include only the more comprehensive content item
- "theme_commentary": a concluding paragraph that summarizes the content items in the theme and their relevance to the user query, with additional interpretation
- "report_commentary": a concluding paragraph that summarizes the themes and their relevance to the user query, with additional interpretation

When presenting evidence, support each sentence with one or more source references: "[source: <source_id>, <source_id>]". Include source IDs only - DO NOT include the chunk ID within the source ID.

Each content item can only be used once, under the most appropriate theme. If a content item is relevant to multiple themes, choose the theme that best captures the main point of the content item.

--TASK--

Content items indexed by id:

{content}

User query:

{query}

Output JSON object:
"""

claim_requery_prompt = """\
You are a helpful assistant tasked with creating a JSON object that analyzes input text chunks for their support or contradiction of given claim.

The output object should summarize all claims from input text chunks as follows:

- "supporting_source": the IDs of the input text chunks that support the claim
- "contradicting_sources": the IDs of the input text chunks that contradict the claim

--TASK--

Claim:

{claim}

Input text chunks JSON, in the form "<source_id>: <text_chunk>":

{chunks}

Output JSON object:
"""

chunk_commentary_prompt = """\
You are a helpful assistant providing a thematic analysis of an information stream with respect to a user query.

---Task---

Output a nested thematic structure that organizes low-level titles of events/insights into higher-level themes. Each theme should be a concise, high-level summary of the events/insights that fall under it.

Themes should clearly related to the user query and the new information provided. Each theme should contain at least one point.

Example:

- **Theme 1**
  - Point 1
  - Point 2
- **Theme 2**
  - Point 3
  - Point 4
- **Theme 3**
  - Point 5
  - Point 6 

---User query---

{query}

---New information---

{information}

---Existing thematic structure---

{commentary}

---New thematic structure---

"""

thematic_update_prompt = """\
You are a helpful assistant tasked with creating a JSON object that updates a thematic organization of points relevant to a user query.

The output object should capture new themes, points, and source references that should be added to or modify the existing thematic structure:

- "updates": an array of objects, each representing an update to a point derived from the input text chunks
- "point_id": the ID of the point to update, else the next available point ID if creating a new point
- "point_title": the title of the point to update or create. If the existing point title is unchanged, the field should be left blank
- "source_ids": an array of source IDs that support the point, to be added to the existing source IDs for the point
- "theme_title": the title of a theme that organizes a set of related points.

--Rules--

- Each point MUST be sufficiently detailed to stand along without ambiguity
- If a source relates to an existing point, the source ID MUST be assigned to the existing point ID, rather than creating a new point
- If the addition of a source to a point warrants a change in point title, the point title MUST be updated
- Aim for 3-7 themes overall, with an even distribution of points across themes
- Points should be assigned to a single theme in a logical sequence that addresses the user query
- Order themes in a logical sequence that addresses the user query

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
