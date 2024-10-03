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
You are a helpful assistant tasked with creating a JSON object that answers a given user question.

The output object should update the input JSON object in a way that:

- answers the user question as directly and comprehensively as possible using relevant information from the text chunks
- represents a single coherent narrative, rather than a collection of unrelated facts. Input texts that are clearly unrelated should be ignored
- incudes a title and structured headings as appropriate
- includes prose between all headings to explain the context and significance of the information presented
- includes bridging text that makes connections between topic areas, explaining how they relate to each other
- support each sentence with a source reference to the file and text chunk: "[source: <file> (<chunk_id>), <file> (<chunk_id>)]. Always use the full name of the file - do not abbreviate - and enter the full filename before each chunk id, even if the same file contains multiple relevant chunks."

--FORMAT--

The JSON object should be structured as follows:

{{
    "question": "<user question, which must not be modified>",
    "title": "<title of report that answers question, which may be updated based on new and updated content items>",
    "introduction": "<introduction to the report, including a bulleted list of content item titles divided into thematic groups (not just the new or updated ones)>",
    "content_id_sequence": ["<sequence of content item ids matching their list order in the introduction>"],
    "content_items":
    {{
        "<content item id>":
        {{
            "title": "<title>",
            "content": "<content, including any commentary on the relevant concepts if provided in the analysis field>"
        }},
        ...
    }}
    "conclusion": "<conclusion to the report, summarizing all major claims and their implications>"
}}

The content_sequence field should be a list of content item ids that represents the order in which the content items should be presented in the final report.

Items may be added by creating a new id and removed by not including the corresponding id in the content_sequence.

The content items field should include any new or updated content, but does not need to duplicate content from the input JSON object.

The introduction and conclusion fields must be updated to reflect the new and updated content items.

Content items may be reordered, merged, or split as needed to create a coherent narrative.

ALL new text chunks MUST be included and referenced in the final report PROVIDED they are relevant to the question.

When generating the text of content items, do not mix factual content and commentary in the same sentence. Instead, separate factual content from commentary with a new sentence.

--TASK--

Input JSON object:

{answer_object}

New text chunks:

{chunks}

Output JSON object:
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

intermediate_answer_prompt = """\
You are a helpful assistant tasked with creating a JSON object that organizes content relevant to a given user question.

The output object should extract relevant claims from input text chunks that will help to answer the user question as comprehensively as possible, as follows:

- "text_title": the title of the text from which the chunk was exracted
- "chunk_id": the id of the chunk within the text
- "claim_context": the context in which claims are made
- "claim_statement": a statement-based formatting of a claim
- "claim_attribution": any named source or author of a claim, beyond the title of the text 

--TASK--

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
