# Extract Record Data

The [`Extract Record Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/extract_record_data/README.md) workflow translates unstructured text into structured records via schema-aligned JSON objects uploaded or defined by the user.

Navigate to [example_outputs/extract_record_data](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/extract_record_data) (on GitHub) for examples.

## How it works

1. [**Input**] An instance or collection of unstructured text and (optionally) an existing JSON file containing the JSON schema with which to generate output records.
2. [**Process**] The user edits the uploaded JSON schema or creates one interactively.
3. [**AI Calls**] The system uses generative AI to extract a JSON object from the text following the JSON schema.
4. [**Output**] A dataset of structured records following the JSON schema and (optionally) a newly-defined JSON schema.

## Input requirements

- The input schema, if provided, should be a JSON file conforming to the [JSON schema standard](https://json-schema.org/).

## Use with other workflows

[`Extract Record Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/extract_record_data/README.md) can be used to create structured data suitable for input to any of the following workflows:

- [`Anonymize Case Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/anonymize_case_data/README.md)
- [`Detect Case Patterns`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/detect_case_patterns/README.md)
- [`Compare Case Groups`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/compare_case_groups/README.md)
- [`Match Entity Records`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/match_entity_records/README.md)
- [`Detect Entity Networks`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/detect_entity_networks/README.md)
