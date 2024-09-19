# Extract Record Data

The [`Extract Record Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/extract_record_data/README.md) workflow translates unstructured text into structured records via schema-aligned JSON objects uploaded or defined by the user.

Select the `View example outputs` tab (in app) or navigate to [example_outputs/extract_record_data](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/extract_record_data) (on GitHub) for examples.

## How it works

1. [**Input**] An instance or collection of unstructured text and (optionally) an existing JSON file containing the JSON schema with which to generate output records.
2. [**Process**] The user edits the uploaded JSON schema or creates one interactively.
3. [**AI Calls**] The system uses generative AI to extract a JSON object from the text following the JSON schema.
4. [**Output**] A dataset of structured records following the JSON schema and (optionally) a newly-defined JSON schema.

## Input requirements

- The input schema, if provided, should be a JSON file conforming to the [JSON schema standard](https://json-schema.org/) and following the restrictions of the [OpenAI Structured Outputs API](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas).

## Use with other workflows

[`Extract Record Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/extract_record_data/README.md) can be used to create structured data suitable for input to any of the following workflows:

- [`Anonymize Case Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/anonymize_case_data/README.md)
- [`Detect Case Patterns`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/detect_case_patterns/README.md)
- [`Compare Case Groups`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/compare_case_groups/README.md)
- [`Match Entity Records`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/match_entity_records/README.md)
- [`Detect Entity Networks`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/detect_entity_networks/README.md)

## Tutorial

The task for this tutorial is extracting structured data records from transcripts of customer complaint calls (mock data).

From the [`Extract Record Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/extract_record_data/README.md) homepage in a running instance of Intelligence Toolkit, select `Prepare data schema`.

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

### Extracting structured records

Navigate to the `Extract structured records` tab to use this schema to extract structured records from input text provided.

With `Mode` set to `Extract from single text`, you can enter arbitrary text into the `Unstructured text input` field as desired. For example, try copying and pasting the following mock call transcript:

```code
**Customer Service Representative:** Good afternoon, thank you for calling our customer service hotline. My name is Sarah. How can I assist you today?

**Bob Johnson:** Hi Sarah, my name is Bob Johnson. I'm calling to report an issue with a product I purchased recently.

**Customer Service Representative:** I'm sorry to hear that, Bob. Could you please provide me with some more details about the issue?

**Bob Johnson:** Sure, I live at 123 Maple Street in Springfield, and I recently bought a product from your company. The product code is A, and I received it in the second quarter of 2023.

**Customer Service Representative:** Thank you for that information, Bob. Could you tell me what specific issue you're experiencing with the product?

**Bob Johnson:** Yes, the problem is with the quality of the product. It just doesn't meet the standards I was expecting.

**Customer Service Representative:** I understand how frustrating that can be. Just to confirm, are there any other issues such as price, service, delivery, or description that you're experiencing?

**Bob Johnson:** No, it's just the quality issue. Everything else, like the price and delivery, was fine.

**Customer Service Representative:** Thank you for clarifying that. I see that you're 36 years old, and I have your email as bob.johnson@example.com. Is that correct?

**Bob Johnson:** Yes, that's correct.

**Customer Service Representative:** Great, Bob. I will escalate this quality issue to our product team for further investigation. We will get back to you via email with a resolution as soon as possible.

**Bob Johnson:** Thank you, Sarah. I appreciate your help.

**Customer Service Representative:** You're welcome, Bob. Thank you for bringing this to our attention, and we apologize for any inconvenience. Have a great day!

**Bob Johnson:** You too, bye.
```

Next, try the same extraction process over a table containing unstructured text fields.

From the `View example outputs` tab, select the `customer_complaints` example and navigate to the `Unstructured texts` tab. Download the CSV file, then return to the `Extract structured records` tab.

Set the `Mode` to `Extract from rows of CSV file` and upload the `customer_complaints_texts.csv` file just downloaded.

Press `Extract record data` to see the `Extracted records` generated on the right.
