extraction_system_prompt = """\
You are a helpful assistant extracting question-answer pairs and related entities from an input text.

Identified questions should:

- reflect questions of interest to users who have not read the input text, rather than assume knowledge of the input text
- pose a a single question rather than a compund question with multiple parts
- combine as many related details as possible into a comprehensive answer, rather than asking many questions that each require a small amount of information
- stand alone as a question and answer pair, rather than relying on previous questions or answers for context
- cover all entities mentioned in the input text, and include the names of these entities in the question and/or answer as appropriate

Output the response as a JSON list of "question" and "answer" strings, with all entities listed in an "entities" field and all references listed in a "source" field:
- Entity references should be listed inside square brackets (even if there is only one entity) using the full name of the entity. Examples of names entities include people, places, and organizations.
- Source references should be listed inside square brackets (even if there is only one source) and indicate the file IDs and pages used to derive the answer and be formatted as "F<X>;P<Y>", where X is the file ID and Y is the page number. Do not combine pages in the style style "F1;P4-5"; rather, list a different reference for each page, as in "F1;P4, F1;P5".
Do not begin the response with ```json or end it with ```.
"""
                        
extraction_user_prompt = """\
File ID: {file_id}

Input text: {text}
"""

answering_system_prompt = """\
You are a helpful assistant generating reports that answer user questions using questions and answers provided.

The final report should:

- answer the user question as directly as possible using the questions and answers provided
- have a title that reflects the overall theme of the report and the high-level question it answers
- include references to at least {source_diversity} source documents
- represent a single coherent narrative, rather than a collection of unrelated facts. Input questions that are clearly unrelated should be ignored
- use structured headings as appropriate, but not necessarily in the same order or question format as the input questions
- include prose between all headings to explain the context and significance of the questions and answers
- include bridging text that makes connections between topic areas, explaining how they relate to each other
- remove redundancy and repetition from the input questions and answers, providing concise but complete information
- aim for paragraphs of at least 3 sentences each
- support each sentence with a source reference in the same format as the input answers, "[source: <file> (p<page>), ...]". If a claim is supported by existing knowledge rather than a source reference, it should be followed by the marker "[source: LLM]"
- include ALL source references from the input questions and answers, combining them as needed if they support the same claim
- end with a ## Conclusion section that summarizes the key findings of the report and answers the high-level question posed in the title, while retaining all relevant source references
- be formatted in markdown, with headings indicated by # symbols and subheadings indicated by additional # symbols
"""
                
answering_user_prompt = """\
User question:

{question}

Relevant questions:

{outline}
"""