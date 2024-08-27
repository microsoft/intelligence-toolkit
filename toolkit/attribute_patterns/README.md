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
