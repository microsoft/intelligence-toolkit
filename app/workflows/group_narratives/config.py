intro = """ \
# Group Narratives

The **Group Narratives** workflow generates intelligence reports by defining and comparing groups of case records.

## How it works

1. [**Input**] Case records representing observations on data subjects that fall into different groups. 
2. [**Process**] The user defines the groups of interest by specifying a prefilter, grouping attributes, between-subjects comparison attributes, and an optional within-subjects temporal/ordinal attribute.
3. [**Output**] Group summary CSV file containing group time/level deltas and group, group-attribute, and group-attribute-time/level rankings. Can be created and used independently without any AI or embedding calls.
4. [**AI Calls**] For groups of interest selected by the user, generative AI is used to create AI group reports.
5. [**Output**] AI group report MD/PDF file(s) comparing the counts of group records and their comparison attributes, both overall and over time/levels.

"""