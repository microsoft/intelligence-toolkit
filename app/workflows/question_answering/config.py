import os

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "cache"), "question_answering")

chunk_size = 5000
chunk_overlap = 0

intro = """ \
# Question Answering

The **Question Answering** workflow generates intelligence reports from an entity-rich document collection.

## How it works

1. [**Input**] PDF documents (e.g., scientific papers, policy articles, industry reports) covering a domain of interest. 
2. [**Process**] The PDFs are parsed into text and split into pages and chunks.
3. [**Embedding Calls**] The chunks are embedded into a multi-dimensional semantic space, with similar ideas close to one another.
4. [**Process**] The user's question is embedded into the same space, and the chunk closest to the question is selected.
5. [**AI Calls**] The system uses generative AI to mine questions from the matched chunk, without knowing the user's question.
6. [**AI Calls**] The system uses generative AI to insert any partial answers from the mined questions into the user's question.
7. [**Embedding Calls**] The system embeds the questions, answers, and augmented question into the same space.
8. [**Processing**] The cycle (steps 4-7) repeats until the augmented question matches a target number of questions/answers before an unmined chunk.
9. [**AI Calls**] The system uses generative AI to produce an answer report from the augmented question and matching mined question-answer pairs.
10. [**Output**] AI answer report MD file(s) including both a structured answer to the user's question and an FAQ of relevant question-answer pairs.

"""