# Detect Case Patterns

The [`Detect Case Patterns`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/detect_case_patterns/README.md) workflow generates intelligence reports on patterns of attribute values detected in streams of case records.

## How it works

1. [**Input**] Case records representing categorical attributes of data subjects observed at a point time. Units are treated as anonymous and independent.
2. [**Process**] Categorical attributes are modelled as a dynamic graph, where nodes represent attribute values in a given time window and edges represent the co-occurrences of attribute values.
3. [**Process**] A technique called [Graph Fusion Encoder Embedding](https://arxiv.org/abs/2303.18051) is used to embed the dynamic attribute graph into a multi-dimensional space.
4. [**Process**] Within each time period, attribute patterns are detected as combinations of attributes all moving towards one another in the embedding space.
5. [**Output**] Attribute patterns CSV file. Can be created and used independently without any AI or embedding calls.
6. [**AI Calls**] For patterns of interest selected by the user, generative AI is used to create AI pattern reports.
7. [**Output**] AI pattern report MD/PDF file(s) describing the nature of the pattern, its progression over time, top co-occurring attribute values, possible explanations, and suggested actions.

## Input requirements

- The input data file should be in CSV format with each row representing a different case (i.e., individual person or data subject).
- For pattern detection, each case must be represented as a collection of discrete (i.e., categorical or binary) attributes. Any continuous attributes must first be quantized via the user interface.
- Given the goal of detecting attribute-level case patterns, no direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) should be included in data outputs. Following the principle of [data minimization](https://en.wikipedia.org/wiki/Data_minimization), such direct identifiers should be removed from data inputs because they are not required for the processing purpose and create unnecessary risks for the data subject. Tools such as Microsoft Excel can be used to delete any direct identifier columns prior to use in Intelligence Toolkit.
- First converting any sensitive input dataset to a synthetic dataset using the [`Anonymize Case Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/anonymize_case_data/README.md) workflow will ensure that any pattern summaries can be safely shared without compromising the privacy of data subjects.

## Use with other workflows

[`Detect Case Patterns`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/detect_case_patterns/README.md) can be used to detect patterns of attributes in any timestamped set of case records, including mock records created using [`Generate Mock Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/generate_mock_data/README.md) and synthetic case records anonymized using [`Anonymize Case Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/anonymize_case_data/README.md).

## Tutorial

The task for this tutorial is detecting patterns in the cooccurrences of attribute values in the `customer_complaints` dataset available for download either:

- in app, via [`Generate Mock Data`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/generate_mock_data/README.md) workflow &rarr; `View example outputs` tab &rarr; `Mock data` tab
- on GitHub, at [example_outputs/generate_mock_data/customer_complaints](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/generate_mock_data/customer_complaints).

The format of this dataset is as follows, with each row representing an individual customer and their complaint:

| names: *string* | street: *string* | city: *string* | age: *number* | email: *string*           | \*_issue: *boolean* | product_code: *string* | quarter: *string* |
|-----------------|------------------|----------------|---------------|---------------------------|---------------------|------------------------|-------------------|
| Bob Johnson     | 123 Maple Street | Springfield    | 36            | bob.johnson@example.com   | False               | A                      | 2023-Q2           |
| Charlie Brown   | 321 Elm Street   | Shelbyville    | 28            | charlie.brown@example.com | True                | B                      | 2023-Q1           |
| Laura Palmer    | 777 Twin Peaks   | Twin Peaks     | 27            | laura.palmer@example.com  | False               | C                      | 2023-Q2           |
| ...             | ...              | ...            | ...           | ...                       | ...                 | ...                    | ...               |

where \*_issue: *boolean* represents five different boolean attributes covering different kinds of complaint: `price_issue`, `quality_issue`, `service_issue`, `delivery_issue`, and `description_issue`. Each complaint may relate to multiple issues.

### Defining attribute patterns

We define an attribute pattern as a combination of two or more attribute values that are strongly associated in a given time period, compared with both (a) other combinations of attributes in the same time period, and (b) the same combination of attributes in other time periods.

The foundation of the attribute pattern detection algorithm is the ability to detect which *pairs* of attribute values are strongly associated in each time period. We call these "converging pairs". From all converging pairs in each time period, we can build up combinations of attribute values whose pairwise relationships are all converging. This is what we mean by an attribute pattern, and it typically coincides with the time series of record counts for that attribute combination reaching a local or global maximum.

In other words, our pattern detection algoritm identifies groups of records characterized by combinations of attributes that "stand out" in a given time period, potentially warranting further review.

### Preparing data for pattern detection

The statistical approach for detecting attribute patterns measures changes in the cooccurrence of pairs of attribute values over time, relative to both other time periods and other pairs. It is therefore important that attribute values represent broad categories that are likely to be observed in multiple cases across multiple time periods. This typically means:

1. Quantizing continuous numeric values (e.g., `age`) into discrete ranges (e.g., `age_range`)
2. Suppressing insignificant values that hold little analytical value (e.g., binary 0 or boolean false indicating absence of an attribute)
3. Selecting the minimum set of data columns (i.e., attributes) necessary to support downstream analysis and reporting tasks.

We can now work though the steps of senstive preparation using the `customer_complaints_data.csv` dataset above.

First, navigate to the `Prepare case data` tab, select `Browse files`, and upload the `customer_complaints_data.csv` dataset. A preview of the dataset should appear below.

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

#### Create the attribute model

In `Period column`, select the new `period` column and press `Generate attribute model`. You will see a message confirming the statistics of the model and readiness to proceed to the next stage.

### Detecting attribute patterns

Move on to the `Detect attribute patterns` tab and select `Detect patterns`. You will see a small number of detected patterns (around 14) that meet the `Minimum pattern count` of `100` (the default value). Reduce `Minimum pattern count` to `10` and press `Detect patterns` again. This time more patterns will be detected, since the pattern only has to match `10` cases in a period rather than `100`.

The message box shows statistics from the detection process. It will look something like this:

`Over 8 periods, detected 900 attribute patterns (609 unique) from 529/46992 converging attribute pairs (1.13%). Patterns ranked by overall_score = normalize(length * ln(count) * z_score * detections).`

We can interpret this as follows:

- `Over 8 periods`: the number of time periods covered by the input data column assigned to the `Period column` above
- `detected 900 attribute patterns (609 unique)`: the number of attribute patterns detected over all periods, and how many are unique (the same pattern may be detected in multiple periods)
- `from 529/46992 converging attribute pairs (1.13%)`: there are 46992 pairs of attribute values counting each pair in each time period; of these, 529 (1.13%) show a statistical "convergence" in a particular period indicating strong association with respect to other pairs and time periods
- `Patterns ranked by overall_score = normalize(length * ln(count) * z_score * detections)`: the default table ordering is by `overall_score`, which is higher for patterns that are longer (`length`), more frequent (`ln(count)`), more anomalous (`z_score`), and detected in more periods (`detections`)

Select a pattern to view its time series. Note how the period of detection corresponds to a spike in the counts of matching case records over time.

Different kinds of pattern may be important in different contexts, and the table of patterns allows different kinds of ranking and filtering to find patterns of interest.

Double clicking on a column header will sort the values of that column in descending order. Doing this for each column gives priority to different kinds of pattern:

- `length`: prioritises longer patterns, which will be richer but less frequent than shorter patterns
- `count`: prioritises patterns with higher counts in their period of detection, regardless of their counts in other periods
- `mean`:  prioritises patterns with the highest average counts across all time periods
- `z_score`: prioritises patterns with most anomalous count increase (defined as the number of standard deviations above the mean)
- `detections`: prioritises patterns detected in the greatest number of time periods
- `overall_score`: prioritises patterns that are extreme in multiple of the above dimensions, normalized to a [0,1] score

Filtering is also possible by selecting the menu (&#9776;) icon in the relevant column header, or by typing values (for any column) into the `quickfilter...` field. For example, enter `2023-H1 product_code=F` to view all patterns linked to product code F in the first half of 2023.

Press `Download patterns` to download the detected patterns as a CSV.

### Generating AI pattern reports

Select any pattern of interest and navigate to the `Generate AI pattern reports` tab.

On the left side under `Generative AI instructions`, you can view and edit the text of the prompt that determines how the AI reports on the selected pattern.

Press the `Generate` button to generate the report, which will update in real-time on the panel to the right.

The generated report may be downloaded as an MD (markdown) of PDF file using the buttons provided below the report text.

The chart image may also be saved as a PNG by right-clicking on the chart area and selecting `Save image as`.