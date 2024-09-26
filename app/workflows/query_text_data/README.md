# Query Text Data

The [`Query Text Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/query_text_data/README.md) workflow generates intelligence reports from a collection of text documents.

Select the `View example outputs` tab (in app) or navigate to [example_outputs/query_text_data](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/query_text_data) (on GitHub) for examples.

## How it works

1. [**Input**] Input documents (PDF, TXT, JSON) or multi-text table (CSV) covering a domain of interest (e.g., scientific papers, policy articles).
2. [**Process**] The documents are parsed into text and split into chunks.
3. [**Process**] Concepts are extracted from each chunk and used to create a concept-cooccurrence graph.
4. [**Process**] Communities of closely-related concepts are extracted from the graph as topics.
5. [**Embedding Calls**] Text chunks are embedded into a multi-dimensional semantic space, with similar ideas close to one another.
6. [**Process**] The user's question is embedded into the same space, and the text chunks ranked by similarity.
7. [**Process**] The ranking of text chunks is used to determine a ranking of topics, which span the entire dataset.
8. [**AI Calls**] The system uses generative AI to evaluate the relevance of the top-ranked text chunks from each community in turn, until either a relevance test budget is reached, there are no more communities yielding relevant chunks, or there are no more chunks to test.
9. [**AI Calls**] The system uses generative AI to build an answer report progressively from batches of relevant text chunks.
10. [**Output**] AI answer report MD/PDF file(s) including a concise answer to the user's question and the extended answer report.

## Input requirements

- The input files should be in PDF, TXT, JSON, or CSV format and contain text of interest.
- The text extracted from input files will be sent to external APIs for embedding and text generation. Using the text in such a way must comply with all applicable laws, regulations, and policies governing their source documents, including those pertaining to privacy and security.

## Tutorial

The task for this tutorial is querying the `news_articles` dataset available for download either:

- in app, via `View example outputs` tab &rarr; `Input texts` tab
- on GitHub, at [example_outputs/query_text_data/news_articles/news_articles_texts.csv](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/query_text_data/news_articles/news_articles_texts.csv)

This dataset contains mock news articles spanning a range of categories including world events, local events, sports, politics, lifestyle, and culture.

Begin by navigating to the `Prepare data` tab, pressing `Browse files`, and uploading the `news_articles_texts.csv` file.

This file contains one news article per row, stored in the single column `mock_text` (the column name is not important).

Press `Process files` to prepare the data for analysis. After successfully processing the data, you will see a status message like the following:

`Chunked 500 files into 501 chunks of up to 500 tokens. Extracted concept graph with 1327 concepts and 3593 cooccurrences.`

### Query method

Chunks of input text documents represent the fundamental units of the data index used to answer user queries. Each chunk is preprocessed in two distinct ways:

1. **Text embedding**. Text chunks are "embedded" into a vector space that clusters similar texts in similar locations. Given a user query, text chunks can then be ranked according to their similarity to the query (and thus their likelihood of providing relevant information). This similarity-based ranking provides a *best-first* view of the dataset with respect to a user query.
2. **Concept graph extraction**. Noun phrase "concepts" are extracted from each text chunk using NLP techniques and concept cooccurrence counts across all chunks are used to construct a concept graph. The "communities" of closely-related concepts detected within this graph provide a *breadth-first* view of the dataset with respect to a user query.

The method used to identify relevant chunks prior to answering the user query is designed to provide a balance between these approaches, prioritizing the best-matching chunks from across the breadth of the dataset. It does this by:

1. using the concepts associated with a text chunk to map each text chunk to a topic-based community; then
2. using the ranking of text chunks to create a corresponding ranking of communities/topics.

The best-matching text chunks from each topic in turn can then be passed to an LLM for *rapid relevance tests* costing only a single yes/no output token per candidate text chunk, with all relevant text chunks used as context for answer generation.

### Exploring the concept graph

Navigate to the `Explore concept graph` tab to view the results of concept graph extraction. Since the concept graph can be very, it is best viewed one "community" or conceptual topic area at a time. These topic areas are described by their top concepts in the `Select topic area` selection box.

Select a topic to view the graph of associated concepts. In the graph, concept nodes are sized according to their degree (i.e., number of cooccuring concepts) and coloured according to their subcommunity.

Select a concept node in the graph to view a list of matching text chunks on the right-hand side.

### Generating AI extended answers

Navigate to the `Generate AI extended answers` tab to query the data index (i.e., text embeddings plus concept graph) in a way that generates a long-form text answer.

Click on `Options` to expand the available controls, which are as follows:

- `Relevance test budget`. The query method works by asking an LLM to evaluate the relevance of potentially-relevant text chunks, returning a single token, yes/no judgement. This parameter allows the user to cap the number of relvance tests that may be performed prior to generating an answer using all relevant chunks. Larger budgets will generally give better answers for a greater cost.
- `Tests/topic/round`. How many relevant tests to perform for each topic in each round. Larger values reduce the likelihood of prematurely discarding topics whose relevant chunks may not be at the top of the similarity-based ranking, but may result in smaller values of `Relevance test budget` being spread across fewer topics and thus not capturing the full breadth of the data.
- `Restart on irrelevant topics`. When this number of topics in a row fail to return any relevant chunks in their `Tests/topic/round`, return to the start of the topic ranking and continue testing `Tests/topic/round` text chunks from each topic with (a) relevance in the previous round and (b) previously untested text chunks. Higher values can avoid prematurely discarding topics that are relevant but whose relevant chunks are not at the top of the similarity-based ranking, but may result in a larger number of irrelevant topics being tested multiple times.
- `Test relevant neighbours`. If a text chunk is relevant to the query, then adjacent text chunks in the original document may be able to add additional context to the relevant points. The value of this parameter determines how many chunks before and after each relevant text chunk will be evaluated at the end of the process (or `Relevance test budget`) if they are yet to be tested.
- `Relevant chunks/answer update`. Determines how many relevant chunks at a time are incorporated into the extended answer in progress. Higher values may require fewer updates, but may miss more details from the chunks provided.

Enter a query in the `Question` field and press `Ask` to begin the process of answer generation.

For example, try `What are the main political events discussed?`.

The system will first identify relevant chunks before using batches of relevant chunks to update an extended answer in progress. Once this process has completed, a download button will appear after the contents of the extended report text.

### Generating AI answer reports

The extended answer report may be more detailed than required for reviewing and reporting. Navigate to `Generate AI answer reports` to condense the extended answer into a short-form report.

Modify the `Prompt text` accordingly to specify which kinds of details to retain and what kind of interpretation to provide in the final report, before pressing `Generate` to have generative AI write the text of this report in real time.
