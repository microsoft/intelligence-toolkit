# Generate Record Data [in progress]

The **Generate Record Data** workflow generates structured data following a JSON schema defined by the user.

## How it works

1. [**Input**] A JSON file containing the JSON schema with which to generate output records (optional).
2. [**Process**] The user edits the uploaded JSON schema or creates one interactively.
3. [**AI Calls**] The system uses generative AI to create a dataset of mock records following the data schema (optional).
4. [**Output**] A JSON schema defining structured data records and a dataset of mock records following this data schema.

## Input requirements

- The input schema, if any, should be a JSON file.
