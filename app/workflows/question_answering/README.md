# Question Answering

The **Question Answering** workflow generates intelligence reports from a collection of text documents.

## How it works

1. [**Input**] Input documents (PDF, TXT, or JSON) covering a domain of interest (e.g., scientific papers, policy articles).
2. [**Process**] The documents are parsed into text and split into chunks.
3. [**Process**] Concepts are extracted from each chunk and used to create a concept-cooccurrence graph.
4. [**Embedding Calls**] Text chunks are embedded into a multi-dimensional semantic space, with similar ideas close to one another.
5. [**Process**] The user's question is embedded into the same space, and the text chunks closest to the question are selected.
6. [**AI Calls**] The system uses generative AI in an iterative cycle to evaluate the relevance of matching text chunks, top chunks from related communities in the concept graph, and neighbouring chunks of any chunks evaluated as relevant.
7. [**AI Calls**] The system uses generative AI to build an answer report progressively from batches of relevant text chunks.
8. [**Output**] AI answer report MD/PDF file(s) including a concise answer to the user's question and the complete answer report.

## Input requirements

- The input files should be in PDF, TXT, or JSON format and contain text of interest.
- The text extracted from input files will be sent to external APIs for embedding and text generation. Using the text in such a way must comply with all applicable laws, regulations, and policies governing their source documents, including those pertaining to privacy and security.
