# Attribute Patterns

The **Attribute Patterns** workflow generates intelligence reports on attribute patterns detected in streams of case records.

## How it works

1. [**Input**] Case records representing categorical attributes of data subjects observed at a point time. Units are treated as anonymous and independent.
2. [**Process**] Categorical attributes are modelled as a dynamic graph, where nodes represent attribute values in a given time window and edges represent the co-occurrences of attribute values.
3. [**Process**] A technique called [Graph Fusion Encoder Embedding](https://arxiv.org/abs/2303.18051) is used to embed the dynamic attribute graph into a multi-dimensional space.
4. [**Process**] Within each time period, attribute patterns are detected as combinations of attributes all moving towards one another in the embedding space.
5. [**Output**] Attribute patterns CSV file. Can be created and used independently without any AI or embedding calls.
6. [**AI Calls**] For patterns of interest selected by the user, generative AI is used to create AI pattern reports.
7. [**Output**] AI pattern report MD/PDF file(s) describing the nature of the pattern, its progression over time, top co-occurring attribute values, possible explanations, and suggested actions.

## Input requirements

- The input data file should be in CSV format and represent individual data subjects.
- Individual data subjects may be represented by a single row, in which case no identifier is required, or by multiple rows, in which case an identifier is required to link these rows into a single record.
- For attribute pattern detection, each individual must be represented as a collection of discrete (i.e., categorical or binary) attributes. Any continuous attributes must first be quantized via the user interface.
- Given the goal of identifying attribute patterns, no direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) should be included in data outputs. Following the principle of [data minimization](https://en.wikipedia.org/wiki/Data_minimization), such direct identifiers should be removed from data inputs because they are not required for the processing purpose and create unnecessary risks for the data subject. Tools such as Microsoft Excel can be used to delete any direct identifier columns prior to use in Intelligence Toolkit.
- First converting any sensitive input dataset to a synthetic dataset using the Data Synthesis workflow will ensure that any detected attribute patterns can be safely shared without compromising the privacy of data subjects.
