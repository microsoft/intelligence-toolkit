# Match Entity Records

The **Match Entity Records** workflow generates intelligence reports on record matches detected across entity datasets.

## How it works

1. [**Input**] Multiple datasets representing overlapping entities with inconsistent record formats/naming conventions. 
2. [**Embedding Calls**] The records are harmonized and embedded into a multi-dimensional semantic space, with similar records close to one another.
3. [**Process**] Clusters of matching records are converted into groups, with user-controllable similarity thresholds.
5. [**Output**] Record groups CSV file containing all records matched to the same group. Can be created and used independently without any AI calls.
6. [**AI Calls**] On request from the user, the system uses generative AI to evaluate the likelihood of the records representing a real-world match.
7. [**Output**] AI match report CSV file evaluating the likelihood of a real-world match for each of the identified record groups.

## Input requirements

- The input data files should be in CSV format and represent individual entities to be matched against one another, with one entity per row.
- Entities may be represented using a variety of attribute types, including unstructured text (e.g., street addresses) in different formats.
- Given the goal of matching entity records, direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) of the respective entities should be included in data inputs as they increase the specificity of the matches detected.
- Text representations of input records will be sent to external APIs for embedding and text generation. Using the entity records in such a way must comply with all applicable laws, regulations, and policies governing their source documents, including those pertaining to privacy and security.
