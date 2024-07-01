# Risk Networks

The **Risk Networks** workflow generates intelligence reports on risk exposure for networks of related entities.

## How it works

1. [**Input**] Datasets describing entity attributes (used to infer relationships) and entity risks (used to infer risk exposure).
2. [**Process**] The user selects the data columns representing entity attributes, entity relationships, entity risk flags, and entity groups.
3. [**Embedding Calls**] Optional: the user selects node types (entity labels/attributes) to fuzzily match by embedding them into a multi-dimensional semantic space.
4. [**Process**] The system uses the entity relationships and shared attributes to infer networks of closely related entities.
5. [**Output**] The system shows (a) the structure of any selected network, and (b) risk exposure paths for any selected entity. Can be created and used independently without any AI or embedding calls.
6. [**AI Calls**] For entities or networks of interest selected by the user, generative AI is used to create AI network reports.
7. [**Output**] AI network report MD/PDF file(s) describing the structure of the network and the nature of risk propagation within it.

## Input requirements

- The input data files should be in CSV format and represent (a) attributes of individual entities, (b) relationships between pairs of entities, or (c) individual or aggregated flags linked to individual entities.
- Entities may be represented using a variety of attribute types, including unstructured text (e.g., street addresses) in different formats.
- Given the goal of linking closely-related entities based on their shared attributes, direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) of the respective entities should be included in data inputs as they increase the specificity of the links detected.
- Text representations of input data will be sent to external APIs for embedding and text generation. Using the entity data in such a way must comply with all applicable laws, regulations, and policies governing their source documents, including those pertaining to privacy and security.
