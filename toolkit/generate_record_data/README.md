# Generate Record Data

The `Generate Record Data` workflow generates mock data following a JSON schema defined by the user.

Navigate to [example_outputs/generate_record_data](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/generate_record_data) (on GitHub) for examples.

## How it works

1. [**Input**] A JSON file containing the JSON schema with which to generate output records (optional).
2. [**Process**] The user edits the uploaded JSON schema or creates one interactively.
3. [**AI Calls**] The system uses generative AI to create a dataset of mock records following the data schema.
4. [**Output**] A JSON schema defining structured data records and a dataset of mock records following this data schema.

## Input requirements

- The input schema, if provided, should be a JSON file conforming to the [JSON schema standard](https://json-schema.org/).

## Use with other workflows

`Generate Record Data` can be used to create mock data for demonstration or evaluation of any other workflow accepting structured records as input:

- `Anonymize Case Data`
- `Detect Case Patterns`
- `Compare Case Groups`
- `Match Entity Records`
- `Detect Entity Networks`

Mock data is particularly helpful when working in sensitive domains and/or with personally identifiable information (PII).
