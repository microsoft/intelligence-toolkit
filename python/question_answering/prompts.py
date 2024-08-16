# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
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

chunk_summarization_prompt = """\
You are a helpful assistant tasked with generating a report that answers a given user question.

The final report should:

- answer the user question as directly and comprehensively as possible using relevant information from the text chunks
- represent a single coherent narrative, rather than a collection of unrelated facts. Input questions that are clearly unrelated should be ignored
- incude a title and structured headings as appropriate
- include prose between all headings to explain the context and significance of the information presented
- include bridging text that makes connections between topic areas, explaining how they relate to each other
- support each sentence with a source reference to the file and text chunk: "[source: <file> (<chunk_id>), ...]"

--TASK--

User question:

{question}

Text chunks:

{chunks}

"""

user_prompt = """\
The final report should:
- be formatted in markdown, with headings indicated by # symbols and subheadings indicated by additional # symbols
- use plain English accessible to non-native speakers and non-technical audiences
"""
report_prompt = """\
You are a helpful assistant tasked with generating a single coherent report from multiple partial reports.

The final report should:

- have a title that reflects the overall theme of the report and the high-level question it answers
- represent a single coherent narrative, rather than a collection of unrelated facts
- use structured headings as appropriate, but not necessarily in the same order or question format as the partial answers
- include prose between all headings to explain the context and significance of the questions and answers
- include bridging text that makes connections between topic areas, explaining how they relate to each other
- remove redundancy and repetition from the partial answers, providing concise but complete information
- aim for paragraphs of at least 3 sentences each
- support each sentence with a source reference in the same format as the input answers, "[source: <file> (<chunk>), ...]"
- include ALL source references from the partial answers, combining them as needed if they support the same claim

=== TASK ===

User question:

{question}

Partial answers:

{answers}

"""

list_prompts = {
    "report_prompt": report_prompt,
    "user_prompt": user_prompt,
    "safety_prompt": " ".join([
        do_not_harm_question_answering,
        do_not_disrespect_context,
    ]),
}
