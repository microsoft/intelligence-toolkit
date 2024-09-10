# Compare Case Groups

The **Compare Case Groups** workflow generates intelligence reports by defining and comparing groups of case records.

## How it works

1. [**Input**] Case records representing observations on data subjects that fall into different groups. 
2. [**Process**] The user defines the groups of interest by specifying a prefilter, grouping attributes, between-subjects comparison attributes, and an optional within-subjects temporal/ordinal attribute.
3. [**Output**] Group summary CSV file containing group time/level deltas and group, group-attribute, and group-attribute-time/level rankings. Can be created and used independently without any AI or embedding calls.
4. [**AI Calls**] For groups of interest selected by the user, generative AI is used to create AI group reports.
5. [**Output**] AI group report MD/PDF file(s) comparing the counts of group records and their comparison attributes, both overall and over time/levels.

## Input requirements

- The input data file should be in CSV format and represent individual data subjects.
- Individual data subjects may be represented by a single row, in which case no identifier is required, or by multiple rows, in which case an identifier is required to link these rows into a single record.
- For group comparisons, each individual must be represented as a collection of discrete (i.e., categorical or binary) attributes. Any continuous attributes must first be quantized via the user interface.
- Given the goal of creating group-level data narratives, no direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) should be included in data outputs. Following the principle of [data minimization](https://en.wikipedia.org/wiki/Data_minimization), such direct identifiers should be removed from data inputs because they are not required for the processing purpose and create unnecessary risks for the data subject. Tools such as Microsoft Excel can be used to delete any direct identifier columns prior to use in Intelligence Toolkit.
- First converting any sensitive input dataset to a synthetic dataset using the Anonymize Case Data workflow will ensure that any group summaries can be safely shared without compromising the privacy of data subjects.
