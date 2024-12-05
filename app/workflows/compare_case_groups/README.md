# Compare Case Groups

The [`Compare Case Groups`]((https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/compare_case_groups/README.md)) workflow generates intelligence reports by defining and comparing groups of case records.

Select the `View example outputs` tab (in app) or navigate to [example_outputs/compare_case_groups](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/compare_case_groups) (on GitHub) for examples.

## How it works

1. [**Input**] Case records representing observations on data subjects that fall into different groups.
2. [**Process**] The user defines the groups of interest by specifying a prefilter, grouping attributes, between-subjects comparison attributes, and an optional within-subjects temporal/ordinal attribute.
3. [**Output**] Group summary CSV file containing group time/level deltas and group, group-attribute, and group-attribute-time/level rankings. Can be created and used independently without any AI or embedding calls.
4. [**AI Calls**] For groups of interest selected by the user, generative AI is used to create AI group reports.
5. [**Output**] AI group report MD/PDF file(s) comparing the counts of group records and their comparison attributes, both overall and over time/levels.

## Input requirements

- The input data file should be in CSV format with each row representing a different case (i.e., individual person or data subject).
- For pattern detection, each case must be represented as a collection of discrete (i.e., categorical or binary) attributes. Any continuous attributes must first be quantized via the user interface.
- Given the goal of creating group-level data narratives, no direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) should be included in data outputs. Following the principle of [data minimization](https://en.wikipedia.org/wiki/Data_minimization), such direct identifiers should be removed from data inputs because they are not required for the processing purpose and create unnecessary risks for the data subject. Tools such as Microsoft Excel can be used to delete any direct identifier columns prior to use in Intelligence Toolkit.
- First converting any sensitive input dataset to a synthetic dataset using the [`Anonymize Case Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/anonymize_case_data/README.md) workflow will ensure that any group summaries can be safely shared without compromising the privacy of data subjects.

## Use with other workflows

[`Compare Case Groups`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/compare_case_groups/README.md) can be used to compare groups of cases in any set of case records, including mock records created using [`Generate Mock Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/generate_mock_data/README.md) and synthetic case records anonymized using [`Anonymize Case Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/anonymize_case_data/README.md).

## Tutorial

The task for this tutorial is detecting patterns in the cooccurrences of attribute values in the `customer_complaints` dataset available for download either:

- in app, via `View example outputs` tab &rarr; `Input data` tab
- on GitHub, at [example_outputs/detect_case_patterns/customer_complaints/customer_complaints_input.csv](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/detect_case_patterns/customer_complaints/customer_complaints_input.csv).

The format of this dataset is as follows, with each row representing an individual customer and their complaint:

| name: *string*  | street: *string* | city: *string* | age: *number* | email: *string*             | \*_issue: *boolean* | product_code: *string* | quarter: *string* |
|-----------------|------------------|----------------|---------------|-----------------------------|---------------------|------------------------|-------------------|
| Bob Johnson     | 123 Maple Street | Springfield    | 36            | <bob.johnson@example.com>   | False               | A                      | 2023-Q2           |
| Charlie Brown   | 321 Elm Street   | Shelbyville    | 28            | <charlie.brown@example.com> | True                | B                      | 2023-Q1           |
| Laura Palmer    | 777 Twin Peaks   | Twin Peaks     | 27            | <laura.palmer@example.com>  | False               | C                      | 2023-Q2           |
| ...             | ...              | ...            | ...           | ...                         | ...                 | ...                    | ...               |

where \*_issue: *boolean* represents five different boolean attributes covering different kinds of complaint: `price_issue`, `quality_issue`, `service_issue`, `delivery_issue`, and `description_issue`. Each complaint may relate to multiple issues.

### Understanding the data preparation process

The approach to comparing groups of cases relies on comparing counts of cases with specific attributes both across groups and over time. It is therefore important that attribute values represent broad categories that are likely to be observed in multiple cases across multiple groups and time periods. This typically means:

1. Quantizing continuous numeric values (e.g., `age`) into discrete ranges (e.g., `age_range`)
2. Suppressing insignificant values that hold little analytical value (e.g., binary 0 or boolean false indicating absence of an attribute)
3. Selecting the minimum set of data columns (i.e., attributes) necessary to support downstream analysis and reporting tasks.

### Preparing data for group comparisons

The following steps show how to prepare a typical sensitive dataset for group comparisons. To skip these steps and go straight to the definition of case groups, download an already-prepared dataset either:

- in app, via `View example outputs` tab &rarr; `Prepared data` tab
- on GitHub, at [example_outputs/compare_case_groups/customer_complaints/customer_complaints_prepared.csv](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/compare_case_groups/customer_complaints/customer_complaints_prepared.csv).

In either case, first navigate to the `Prepare case data` tab, select `Browse files`, and upload the `customer_complaints_input.csv` or `customer_complaints_prepared.csv` dataset. A preview of the dataset should appear below.

If using prepared data directly:

- under `Select attribute columns to include`, press `Select all`
- advance to the [Specifying group comparisons](#specifying-group-comparisons) section

Otherwise, continue by working though the steps of case data preparation below using the `customer_complaints_input.csv` dataset.

#### Select attribute columns to include

Under `Select attribute columns to include`, press `Select all` then deselect the identifying attributes `name`, `street`, and `email`.

#### Quantize datetime attributes

Under `Quantize datetime attributes`, select the `quarter` attribute and `Half` bin size to combine quarters into half years. After pressing `Quantize selected columns`, switching from the `Input data` to the `Prepared data` view of the loaded data will show `quarter` now encoded into half years. The column name can be updated in a later step.

#### Quantize numeric attributes

Under `Quantize numeric attributes`, select `age`, set `Target bins` to `5`, and leave `Trim percent` at `0.00`. After pressing `Quantize selected columns`, the `Prepared data` view of the loaded data will show `age` now encoded into five age ranges represented as (exclusive mininum value-inclusive maximum value]:

- `(0-20]`
- `(20-40]`
- `(40-60]`
- `(60-80]`
- `(80-100]`

Looking at the actual distribution of `age` values in the dataset, we see that there are very few data points in the `(0-20]` and `(80-100]` age ranges. We might therefore decide to trim some of these extreme values before determining appropriate bin sizes for the quantized data. Setting `Trim percent` to `0.01` and pressing `Quantize selected columns` again results in new age ranges as follows:

- `(20-30]`
- `(30-40]`
- `(40-50]`
- `(50-60]`

In general, the fewer the values of a data attribute and the more even the distribution of values across a dataset, the better.

#### Suppress insignificant attribute values

Removing low-frequency values from the sensitive dataset by specifying a `Minimum value count` of say `5` will reduce the number of number of attribue values and thus the pairs of attributes whose cooccurrence patterns should be analyzed.

The checkbox `Suppress boolean False / binary 0` is also selected by default. This should be unchecked if `False` or `0` values are important for analysis. In many cases, however, patterns of missing attributes are less useful for analysis than patterns of present attributes. It is therefore recommended to leave this box checked and recode any meaningful alternatives using `Yes` and `No` values.

#### Rename attributes

Rename `age` to `age_range` and `quarter` to `period`.

### Specifying group comparisons

Navigate to the `Specify group comparions` tab to continue.

In the `Compare groups of records with different combinations of these attributes` field, select the attributes whose combinations define a group of interest.

For this example, we could select `city`, `product_code`, or both of these as defining groups of interest. For simplicity, select `city` here.

Under `Using counts of these attributes`, select the attributes whose counts should form the basis of comparisons. For this dataset, select `product_code` and all of the "issue" attributes: `delivery_issue`, `description_issue`, `price_issue`, `quality_issue`, and `service_issue`.

The `Across windows of this temporal/ordinal attribute (optional)` provides an opportunity to track the period-by-period or level-by-level changes in these counts for temporal or ordinal attributes respectively. In the context of the example dataset, `period` would be an appropriate temporal attribute, while `age-range` would be an appropriate ordinal attribute. Select `period` for this example.

The `After filtering to records matching these values (optional)` allows the dataset to be filtered to specific attribute values before performing group comparisons. Leave this empty for now and press `Create data summary` to build the group comparison table, shown on the right.

Here is a sample row from the resulting data summary and how to interpret it:

| city      | group_count | group_rank | attribute_value    | attribute_count | attribute_rank | period_window | period_window_count | period_window_rank | period_window_delta |
|-----------|-------------|------------|--------------------|-----------------|----------------|---------------|---------------------|--------------------|---------------------|
| Rivertown | 204         | 4          | quality_issue:True | 92              | 3              | 2022-H1       | 7                   | 1                  | 5                   |

- The `city` of `Rivertown` defines a group of case records
- There are `204` case records in the group
- This group count is ranked `4` overall across all groups
- An attribute value within this group of records is `quality_issue:True`
- `92` of the `204` case records in the group have this attribute value
- This attribute count is ranked `3` across all groups for that attribute value
- `2022-H1` is a value of the `period` attribute used to compare groups over successive windows
- There are `7` cases matching the attribute value `quality_issue:True` in this period
- This period is ranked `1` by count for the specific attribute (`quality_issue:True`) and group (`Rivertown`)
- This value represents a delta of `5` compared to the previous window (i.e., `period`)

The user interface also presents an overall summary, as follows:

```code
This table shows:

- A summary of all 2769 data records with values for all grouping attributes
- The group_count of records for all [city] groups, and corresponding group_rank
- The attribute_count of each attribute_value for all [city] groups, and corresponding attribute_rank
- The period_window_count of each attribute_value for each period_window for all [city] groups, and corresponding period_window_rank
- The period_window_delta, or change in the attribute_value_count for successive period_window values, within each [city] group
```

### Generating AI group reports

Navigate to `Generate AI group reports` to translate the table of group comparisons into text-based reports.

Since summaries of large datasets have the potential to exceed the input context window of the generative AI model, specific groups may need to be selected as the focus of the report.

You can either select these groups explicitly under `Select specific groups to report on`, or simply select the groups with the top record counts using the `OR Select top group ranks to report on` field.

For this example, set `OR Select top group ranks to report on` to `10`. This brings the data size within the limits of the generative AI model's input context window and allows the group comparison report to be generated.10
