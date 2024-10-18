# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

from app.workflows.security.metaprompts import (
    do_not_disrespect_context,
    do_not_harm_question_answering,
)

chunk_relevance_prompt = """\
You are a helpful assistant determining whether a given chunk of text is relevant to a user question.

Text chunk:

{chunk}

User question:

{question}

--Task--
Answer "Yes" if the text chunk contains information that could be included and referenced in an answer to the user question, and "No" if it does not.

The text chunk does not need to answer the question directly, but should contain information that could be used to construct an answer. For example, a chunk that provides background information or thematic context for the question would be considered relevant, even if it does not contain the answer itself.

Answer with a single word, "Yes" or "No", with no additional output.
"""

user_prompt = """\
The final report should:
- be formatted in markdown, with headings indicated by # symbols and subheadings indicated by additional # symbols
- use plain English accessible to non-native speakers and non-technical audiences
"""
report_prompt = """\
You are a helpful assistant tasked with generating a single coherent report from single extended report.

The final report should:

- have a title that reflects the overall theme of the report and the high-level question it answers
- represent a single coherent narrative, rather than a collection of unrelated facts
- use structured headings as appropriate, but not necessarily in the same order or question format as the extended answer
- include prose between all headings to explain the context and significance of the questions and answers
- include bridging text that makes connections between topic areas, explaining how they relate to each other
- remove redundancy and repetition from the extended answer, providing concise but complete information
- aim for paragraphs of at least 3 sentences each
- support each sentence with a source reference in the same format as the input answers, "[source: <file> (<chunk>), ...]"
- include ALL source references from the extended answer, combining them as needed if they support the same claim

=== TASK ===

User question:

{question}

Extended answer:

{answer}

"""

claim_extraction_prompt = """\
You are a helpful assistant tasked with creating a JSON object that extracts relevant claims from a collection of input text chunks.

Given a question, the output object should extract claims from the input text chunks as follows:

- "claim_context": an overall description of the context in which claims are made
- "claim_statement": a statement-based formatting of a claim that is relevant to the user question
- "claim_attribution": any named source or author of a claim, beyond the title of the text 
- "text_title": the title of the text from which the chunk was exracted
- "chunk_id": the id of the chunk within the text

--TASK--

Question for which claims are being extracted:

{question}

Input text chunks JSON:

{chunks}

Output JSON object:
"""

claim_summarization_prompt = """\
You are a helpful assistant tasked with creating a JSON object that summarizes claims relevant to a given user question.

The output object should summarize all claims from input text chunks as follows:

- "content_title": a title for a specific content item spanning related claims, in the form of a derived claim statement
- "content_summary": a paragraph, starting with "**Source evidence**:", describing each of the individual claims and the balance of evidence supporting or contradicting them
- "content_commentary": a paragraph, starting with "**AI commentary**:", suggesting inferences, implications, or conclusions that could be drawn from the source evidence

When presenting source evidence, support each sentence with a source reference to the file and text chunk: "[source: <file> (<chunk_id>), <file> (<chunk_id>)]. Always use the full name of the file - do not abbreviate - and enter the full filename before each chunk id, even if the same file contains multiple relevant chunks.

--TASK--

Input text chunks:

{data}

Input claim analysis JSON:

{analysis}

Output JSON object:
"""

content_integration_prompt = """\
You are a helpful assistant tasked with creating a JSON object that organizes content relevant to a given user question.

The output object should summarize all claims from input text chunks as follows:

- "question": the user question/prompt that the report answers
- "answer": a standalone and detailed answer to the user question, derived from the content items and formatted according to the user question/prompt. Quote directly from source text where appropriate, and provide a source reference for each quote
- "report_title": a title for the final report that reflects the overall theme of the content and the user question it answers, in the form of a derived claim statement. Should not contain punctuation or special characters beyond spaces and hyphens
- "report_summary": an introductory paragraph describes the themes and content items in the report, without offering additional interpretation beyond the content items
- "theme_order": the order in which to present the themes in the final report
- "theme_title": a title for a specific theme spanning related content items, in the form of a derived claim statement
- "theme_summary": an introductory paragraph that links the content items in the theme to the user question, without additional intepretation
- "content_id_order": the order in which to present the content items within the theme, specified by content item id. If content items make the same overall point, include only the more comprehensive content item
- "theme_commentary": a concluding paragraph that summarizes the content items in the theme and their relevance to the user question, with additional interpretation
- "report_commentary": a concluding paragraph that summarizes the themes and their relevance to the user question, with additional interpretation

When presenting evidence, support each sentence with a source reference to the file and text chunk: "[source: <file> (<chunk_id>), <file> (<chunk_id>)]. Always use the full name of the file - do not abbreviate - and enter the full filename before each chunk id, even if the same file contains multiple relevant chunks.

--TASK--

Content items indexed by id:

{content}

User question:

{question}

Output JSON object:
"""

claim_requery_prompt = """\
You are a helpful assistant tasked with creating a JSON object that analyzes input text chunks for their support or contradiction of given claim.

The output object should summarize all claims from input text chunks as follows:

- "supporting_source_indicies": the indices of the input text chunks that support the claim (starting at 0)
- "contradicting_source_indicies": the indices of the input text chunks that contradict the claim (starting at 0)

--TASK--

Claim:

{claim}

Input text chunks JSON:

{chunks}

Output JSON object:
"""


list_prompts = {
    "report_prompt": report_prompt,
    "user_prompt": user_prompt,
    "safety_prompt": " ".join([
        do_not_harm_question_answering,
        do_not_disrespect_context,
    ]),
}
