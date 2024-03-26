import os

att_val_sep = '=='
list_sep = '; '
max_rows_to_show = 1000
entity_label = 'Entity'
cache_dir = os.path.join(os.environ.get("CACHE_DIR", "cache"), "record_matching")
outputs_dir = os.path.join(cache_dir,"outputs")

intro = """ \
# Record Matching

The **Record Matching** workflow generates intelligence reports on record matches detected across entity datasets.

## How it works

1. [**Input**] Multiple datasets representing overlapping entities with inconsistent record formats/naming conventions. 
2. [**Embedding Calls**] The records are harmonized and embedded into a multi-dimensional semantic space, with similar records close to one another.
3. [**Process**] Clusters of matching records are converted into groups, with user-controllable similarity thresholds.
5. [**Ouput**] Record groups CSV file containing all records matched to the same group. Can be created and used independently without any AI or embedding calls.
6. [**AI Calls**] On request from the user, the system uses generative AI to evaluate the likelihood of the records representing a real-world match.
7. [**Output**] AI match report CSV file evaluating the likelihood of a real-world match for each of identified record groups.

"""