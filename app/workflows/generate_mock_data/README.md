# Generate Mock Data

The [`Generate Mock Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/generate_mock_data/README.md) workflow generates (a) structured records and (b) synthetic text notes aligned with each record, each according to a JSON schema uploaded or defined by the user.

Navigate to [example_outputs/generate_mock_data](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/generate_mock_data) (on GitHub) for examples.

## How it works

1. [**Input**] A JSON file containing the JSON schema with which to generate output records (optional).
2. [**Process**] The user edits the uploaded JSON schema or creates one interactively.
3. [**AI Calls**] The system uses generative AI to create a dataset of mock records following the JSON schema.
4. [**Output**] A JSON schema defining structured data records and a dataset of mock records following this schema.

## Input requirements

- The input schema, if provided, should be a JSON file conforming to the [JSON schema standard](https://json-schema.org/).

## Use with other workflows

[`Generate Mock Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/generate_mock_data/README.md) can be used to create mock data for demonstration or evaluation of any other workflow.

Mock data is particularly helpful when working in sensitive domains and/or with personally identifiable information (PII).

## Tutorial

The task for this tutorial is creating a mock case dataset of customer complaints, where each case record describes an identified individual and the nature of their complaint regarding a specific product. This is a useful proxy for any individual-level case data or "microdata" where the privacy of data subjects must be respected.

From the [`Generate Mock Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/generate_mock_data/README.md) homepage in a running instance of Intelligence Toolkit, select `Prepare data schema`.

### Modifying the default schema

In the top left, you have the option to upload an existing JSON schema. This is helpful when the data schema represents an existing data standard or has been saved from previous use of the workflow. In this tutorial though, we will begin by defining a new schema using the `Edit data schema` form. Any changes made using this form will be reflected in the `Preview` pane to the right.

The initial JSON schema contains some boilerplate metadata fields representing best practices for schema design. The metadata fields are as follows:

- `$schema`: Indicates the version of the json-schema standard that the current schema follows. Leave this field as it is.
- `$id`: Provides a global ID for this schema anchored in a web domain, e.g., that of your organization. You may wish to edit this if you expect your schema to be widely used, but it can be left as it is for use inside Intelligence Toolkit.
- `title`: A short title indicating the kind of data that the schema represents.
- `description`: A longer description of the kind of data that the schema represents.

Try editing some of these metadata fields now, and see them reflected in the `Preview` of the `JSON schema` to the right. In particular, set the title field to `Customer Complaints`.

The schema in progress is validated after every change, with the message `Schema is valid` confirming that the current schema conforms to the standard specified in the `$schema` field.

Try downloading an edited schema using the download button, uploading it via the `Upload schema` control, then continuing as below.

### Creating the record collection

Now select the `Sample object` tab, and notice how none of these fields are contained in the sample object itself. We can understand this by going back to the `JSON schema` tab and seeing that the schema is of type `object` and that the `properties` of the object are currently empty, indicated by the empty braces `{}`. Whatever we add to the `properties` of the top-level object in the schema gets added to the `Sample object` (and indeed to any objects that conform to the schema).

Let's now add some fields to the object using the buttons under `Add top-level field` in the form to the left.

To create a dataset of records rather than a single object, the schema needs to contain an object array field. Press the `obj[]` button to add an object array field at the top level (i.e., level 0). The new field will be given a generic name by default: `object_array_1`. Rename this to `complaint_records` and see on the right how this creates an array of objects whose properties you can define next.

Note that all new fields have the `Required?` checkbox checked by default, placing all field names in the `required` field of the object. This is a requirement for the [OpenAI Structured Outputs API](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas), which we'll later use to generate mock data that follows the schema. Similarly, all objects must also have `additionalProperties` set to `false`, so the `Additional?` checkbox is left unchecked by default.

### Defining record attributes

Next, we need to add fields to the objects of `complaint_records` for each attribute of the records we want to create.

Using the controls under `Add field to complaint_records`, press the `str` button to add a string (i.e., text) field. This field appears as the level 1 string `string_1` (level 1 because the field is nested one level down from the `complaint_records` array at the top level, i.e., level 0). Edit the text label from `string_1` to `name`.

As further string fields within `complaint_records`, now add:

- `street` and `city` as string fields
- `age` as a numeric field using the `num` button
- `email` as string field
- `price_issue`, `quality_issue`, `service_issue`, `delivery_issue`, `description_issue` as boolean (`true`/`false`) fields using the `bool` button

Next, we want to add a `product_code` string field, but limit the possible values of the field to a predefined list called an "enumeration". Do this by checking the `Enum?` checkbox and observing the default values `A`, `B`, and `C` added to the enumeration. These values can be edited, deleted, and expanded as desired. For this tutorial, simply add further enum values alphabetically from `D` to `H`.

Note that boolean attributes of the record could also have been created using the `str[]` button to create a string array, checking `Enum?`, and specifying `price_issue`, `quality_issue`, `service_issue`, `delivery_issue`, `description_issue` as possible values. However, by using independent boolean fields we simplify the overall record structure and avoid the challenges of nested arrays in the final data object.

Finally, add a `quarter` string indicating in which calendar quarter the complaint was made. In the description, you can add a hint about the field structure and contents, e.g., "The quarter in which the complaint was made (since 2020-Q1)".

The schema is now complete and can be downloaded using the `Download complaint_records_[schema].json` button below the schema preview. Selecting the `Sample object` tab shows a minimal JSON object confirming to the schema.

### Generating mock data

Navigate to the `Generate mock data` tab to generate a mock dataset conforming to the `Customer Complaints` schema.

You will see that `Primary record array` has already been set to `complaint_records`, since this is the only array field in the schema. In the presence of multiple arrays, select the one that represents the primary record type whose records should be counted towards the `Total records to generate` target.

All `Data configuration controls` are as follows:

- `Records per batch`: How many records to generate in a single LLM call
- `Parallel batches`: In a single iteraion, how many batches to generate via parallel LLM calls
- `Total records to generate`: How many records to generate. Must be a multiple of `Records per batch` x `Parallel batches`
- `Duplicate records per batch`: Within each batch, how many records should be near-duplicates of a seed record randomly selected from existing records
- `Related records per batch`: Within each batch, how many records should appear closely related to (but not the same as) a seed record randomly selected from existing records
- `A data generation guidance`: Guidance to the generative AI model about how mock data should be generated (e.g., targeted a specific region, time period, industry, etc.)

Press `Generate mock data` to generate mock data according to the schema and configuration. Each record array in the JSON schema will be converted into CSV format and shown in its own tab (in this example, there will be just a single tab for `complaint_records`). Both the JSON object for the entire dataset and CSV files for each record array may be downloaded using the buttons provided.

The outputs of this tutorial are available for download either:

- in app, via [`Generate Mock Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/generate_mock_data/README.md) workflow &rarr; `View example outputs` tab &rarr; `Mock data` tab
- on GitHub, at [example_outputs/generate_mock_data/customer_complaints](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/generate_mock_data/customer_complaints).
