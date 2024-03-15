att_val_sep = '=='
list_sep = '; '
max_rows_to_show = 1000
entity_label = 'ENTITY'
cache_dir = 'cache/risk_networks'
outputs_dir = 'outputs/risk_networks'

intro = """ \
# Risk Networks

The **Risk Networks** workflow generates intelligence reports on risk exposure for networks of related entities.

## How it works

1. [**Input**] Datasets describing entity attributes (used to infer relationships) and entity risks (used to infer risk exposure).
2. [**Process**] The user selects the data columns representing entity attributes, entity relationships, entity risk flags, and entity groups.
3. [**Embedding Calls**] Optional: the user selects node types (entity labels/attributes) to fuzzily match by embedding them into a multi-dimensional semantic space.
4. [**Process**] The system uses the entity relationships and shared attributes to infer networks of closely-related entities.
5. [**Output**] The system shows (a) the structure of any selected network, and (b) risk exposure paths for any selected entity. Can be created and used independently without any AI or embedding calls.
6. [**AI Calls**] For entities or networks of interest selected by the user, generative AI is used to create AI network reports.
7. [**Output**] AI network report MD file(s) describing the structure of the network and the nature of risk propagation within it.

"""