# Anonymize Case Data

The `Anonymize Case Data` workflow generates differentially-private datasets and data summaries from sensitive case records.

Select the `View example outputs` tab (in app) or navigate to [example_outputs/anonymize_case_data](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/anonymize_case_data) (on GitHub) for examples.

## How it works

1. [**Input**] Sensitive case records representing attributes of individual data subjects, where there is risk that even deidentified subjects may be reidentified using distinctive combinations of attribute values.
2. [**Process**] [Private Accurate Combination Synthesizers (PAC-Synth)](https://github.com/microsoft/synthetic-data-showcase/blob/main/packages/lib-pacsynth/README.md) is used to generate synthetic datasets that are statistically similar to the original data, but with [differential privacy (DP)](https://en.wikipedia.org/wiki/Differential_privacy) ([proof](https://github.com/microsoft/synthetic-data-showcase/blob/main/docs/dp/dp_marginals.pdf)).
3. [**Process**] The synthesizer first creates a DP aggregate dataset in which the counts of all attribute combinations up to length 4 are precomputed and then adjusted with calibrated noise that provides DP.
4. [**Process**] The synthesizer then uses these counts to generate a synthetic dataset in which the synthetic records are consistent with these DP aggregates and thus inherit their DP properties.
5. [**Output**] A DP synthetic dataset CSV file with one row per synthetic record and a DP aggregate dataset CSV file with one row per count of attribute value combinations up to length 4. Can be created and used independently without any AI or embedding calls.
6. [**Process**] The user can query the DP datasets to explore summary datasets and charts.
7. [**Output**] Summary dataset CSVs, chart PNGs, and Plotly JSONs that can be readily shared with differential privacy for data subjects.

## Input requirements

- The input data file should be in CSV format and represent individual data subjects.
- Individual data subjects may be represented by a single row, in which case no identifier is required, or by multiple rows, in which case an identifier is required to link these rows into a single record.
- For data synthesis, each individual must be represented as a collection of discrete (i.e., categorical or binary) attributes. Any continuous attributes must first be quantized via the user interface.
- Given the goal of creating an anonymous dataset, no direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) should be included in data outputs. Following the principle of [data minimization](https://en.wikipedia.org/wiki/Data_minimization), such direct identifiers should be removed from data inputs because they are not required for the processing purpose and create unnecessary risks for the data subject. Tools such as Microsoft Excel can be used to delete any direct identifier columns prior to use in Intelligence Toolkit.
- The nature of differential privacy means that indirect identifiers (e.g., age range, year of birth, gender, country, city) may be freely included in the data inputs. None of the combinations of these identifiers (or of any attributes) in the output data allow the presence of individuals to be inferred with any degree of certainty.

## Use with other workflows

`Anonymize Case Data` can be used to anonymize case data for privacy-preserving analysis in any other workflow accepting structured records as input:

- `Detect Case Patterns`
- `Compare Case Groups`

`Generate Record Data` can also be used to generate mock data for demonstration or evaluation of the `Anonymize Case Data` workflow.

## Tutorial

The task for this tutorial is creating an anonymous version of the `customer_complaints_3k.csv` dataset available for download either from the `View example outputs` tab of the `Generate Record Data` workflow, or from the GitHub repo [here](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/generate_record_data/customer_complaints).

The format of this dataset is as follows, with each row representing an individual customer and their complaint:

| names: *string* | street: *string* | city: *string* | age: *number* | email: *string*           | \*_issue: *boolean* | product_code: *string* | quarter: *string* |
|-----------------|------------------|----------------|---------------|---------------------------|---------------------|------------------------|-------------------|
| Bob Johnson     | 123 Maple Street | Springfield    | 36            | bob.johnson@example.com   | False               | A                      | 2023-Q2           |
| Charlie Brown   | 321 Elm Street   | Shelbyville    | 28            | charlie.brown@example.com | True                | B                      | 2023-Q1           |
| Laura Palmer    | 777 Twin Peaks   | Twin Peaks     | 27            | laura.palmer@example.com  | False               | C                      | 2023-Q2           |
| ...             | ...              | ...            | ...           | ...                       | ...                 | ...                    | ...               |

where \*_issue: *boolean* represents five different boolean attributes covering different kinds of complaint: `price_issue`, `quality_issue`, `service_issue`, `delivery_issue`, and `description_issue`. Each complaint may relate to multiple issues.

### De-identification vs anonymization

To de-identify this dataset, we would need to remove the direct identifiers `name` and `email` as well as any indirect identifiers that may be identifying in combination with one another or with additional data sources. For example, the combination of `street`, `city`, and `age` may uniquely identify a person at an address, whereas `city` and `age` are unlikely to do so. We would therefore remove `street` from the de-identified dataset to create a de-identified dataset.

However, just because a dataset is de-identified does not make it anonymous. For example, if you overheard a neighbour complaining about certain aspects of a product at a given point in time, then observed a single matching record for the associated city, age_range, issues, product code, and quarter, then you could reasonably assume to have re-identified your neighbour in the dataset using your background knowledge. This could allow you to learn new and potentially sensitive information about your neighbour from the additional attributes of their data record &ndash; a privacy threat known as *attribute inference*.

Regulations including GDPR [still consider de-identified or pseudonymized data as personal data](https://gdpr-info.eu/recitals/no-26/) unless "all the means reasonably likely to be used" to identify natural persons, such as "singling out", can be shown to be ineffective. In this case, the data is considered to be anonymous and no longer subject to the principles of data protection.

### Differential privacy (DP)

From a mathematical and technological perspective, [differential privacy (DP)](https://en.wikipedia.org/wiki/Differential_privacy) aims to go beyond the prevention of "singling out" attacks to the prevention of *membership inference* in general. This occurs when it is possible to infer the presence of an individual in a dataset without any background information indicating their membership.

Continuing the above example, imagine the neighbour in question lives in a very small city and has reached a very advanced age. Knowing the individual and observing their combination of city and age in the dataset would be sufficient make the connection between the two. The same reasoning applies to small groups of distinctive individuals, e.g., imagine the neighbour has a twin. It may not be possible to determine which of two matching records belongs to each of the twins, but the presence of both individuals in the dataset could easily be inferred.

### Anonymization via DP data synthesis

The `Anonymize Case Data` workflow uses the differentially-private data synthesizer of [Synthetic Data Showcase](https://github.com/microsoft/synthetic-data-showcase) to create a synthetic dataset with the same structure and statistics as a sensitive input dataset while providing strong statistical protection against all possible privacy attacks. It achieves this through a two-step process:

1. All possible combinations of up to four attribute values are considered (even those not present in the data), and calibrated noise is added to the associated counts of records such that the presence or absense of any arbitrary record is not detectable (i.e., the counts are differentially private).
2. These DP aggregate counts or "marginals" are iteratively sampled to create synthetic data records consistent with the counts of individual attribute values in the original sensitive dataset, reproducing the DP aggregate counts as accurately as possible while retaining differential privacy.

### Preparing sensitive data for anonymization

Since the above approach to DP data synthesis works by controlling the release of attribute combinations, it is important that the sensitive input dataset has as few unique or rare attribute combinations as possible before attempting the anonymization process. This typically means:

1. Quantizing continuous numeric values (e.g., `age`) into discrete ranges (e.g., `age_range`)
2. Suppressing insignificant values that hold little analytical value (e.g., binary 0 or boolean false indicating absence of an attribute)
3. Selecting the minimum set of data columns (i.e., attributes) necessary to support downstream analysis and reporting tasks.

We can now work though the steps of senstive data preparation using the `customer_complaints_3k.csv` dataset above.

First, navigate to the `Prepare sensitive data` tab, select `Browse files`, and upload the `customer_complaints_3k.csv` dataset. A preview of the dataset should appear below.

#### Set subject identifier

Select `Set subject identifier` to expand the dropdown and indicate how individual data subjects (i.e., natural persons) are represented in the dataset. Since each record of the data file represents a different data subject, we can leave `Row number` as the selected option. Select the header again to collapse the dropdown and continue.

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

If an attribute value occurs only a small number of times, then attribute combinations containing that value will generally be even less frequent. Since these combinations will likely be eliminated during the data synthesis process anyway, removing low-frequency values from the sensitive dataset by specifying a `Minimum value count` of say `5` will reduce the number of combinations that need to be controlled. This typically raises the accuracy of the resulting synthetic dataset.

The checkbox `Suppress boolean False / binary 0` is also selected by default. This should be unchecked if `False` or `0` values are sensitive and/or counts of attribute combinations containing these values are important for analysis. In many cases, however, counts of cases that don't have certain attributes are less useful for analysis and lead to an explosion in the number of attribute combinations that need to be controlled. It is therefore recommended to leave this box checked and recode any meaningful alternatives using `Yes` and `No` values.

#### Rename attributes

Rename `age` to `age_range` and `quarter` to `period`.

#### Evaluating synthesizability

Pressing `Generate final dataset` applies all the specified transformations to the input dataset, with the result available for viewing and download under the `Final` tab of the data table panel.

The `Synthesizability summary` gives an initial indication of how easy it will be to generate high-accurary synthetic data given the number of attribute combinations in the final sensitive dataset. The smaller each of these numbers, the better:

- `Number of selected columns`: The number of columns after all data transformations
- `Number of distinct attribute values`: The number of distinct attribute values across selected columns
- `Theoretical attribute combinations`: The product of the number of distinct attribute values in each column
- `Theoretical combinations per record`: The number of theoretical combinations divided by the number of records
- `Typical values per record`: The mean number of attribute values per record
- `Typical combinations per record`: The number of possible combinations in the typical number of attribute values(2^typical_values)
- `Excess combinations ratio`: Theoretical combinations per record / Typical combinations per record

The last metric, `Excess combinations ratio`, is the main one to pay attention to in terms of synthesizability. As a rule of thumb, try to keep this ratio at or below `5`. The general idea here is that datasets should have sufficient records to support all possible combinations of attribute values, given the number of distinct attribute values in each column. Not all combinations will be present in most cases &ndash; data records tend to cluster around certain attribute patterns &ndash; so it is ok for this value to be greater than `1`. How far it can actually go and still yield high-accuracy synthetic data depends on how many of these possible attribute combinations are actually observed in the data.

### Generating anonymous data

Navigate to the `Generate anonymous data` tab to generate anonymous data from the sensitive dataset uploaded and prepared in previous steps.

The only user input required for synthesis is the `Epsilon` value, set to `12.00` by default. Lower values of epsilon give greater theoretical privacy protection at the expense of lower accuracy, while higher values of epsilon give higher accuracy at the expense of lower theoretical privacy protection. See [here](https://github.com/microsoft/synthetic-data-showcase/blob/main/docs/dp/README.md) for more details on how this "privacy budget" is allocated to different aspects of the synthesis process.

We recommend using the lowest `Epsilon` value that results in synthetic data with sufficient accurary to support downstream analysis. Start with the default value, and reduce it in small increments at a time. If the accuracy values with an `Epsilon` value of `12.00` are themselves too low, go back to the `Prepare sensitive data` tab and continue refining the sensitive dataset in ways that reduce the `Excess combinations ratio`.

After pressing `Anonymize data`, you will see two differential privacy parameters: the `Epsilon` value you set, and a `Delta` value that is generated based on the data (and indicates the very small thereoretical chance that the `Epsilon` privacy guarantee does not hold). It is important to publish both values alongside any DP dataset for correct interpretation of the privacy protection provided.

Once generated, you will see the `Aggregate data` and `Synthetic data` appear on the right hand side:

- The `Aggregate data` shows the protected counts of all combinations of up to four attribute values, plus the protected count of sensitive records. These aggregate counts are "protected" by the addition of noise calibrated to give the desired DP guarantees.
- The `Synthetic data` shows a dataset of records synthesized in a way that aims to replicate the protected counts of the aggregate dataset. In particular, new values are sampled and new records synthesized until the counts of individual attribute values match their protected counts, while ensuring that all combinations of up to four attributes in the synthesized record are present in the aggregate dataset.

Each of these datasets can be downloaded using the buttons provided.

### Evaluating anonymous data

The `Aggregate data` and `Synthetic data` generated by the workflow will always respect the privacy budget (i.e., `Epsilon` value) set by the user. However, the data may deviate from the actual sensitive data in three significant ways:

- `Error`: For a group of attribute combinations, the mean absolute difference between anonymous data counts and sensitive data counts
- `Suppressed %`: For a group of attribute combinations, the percentage of sensitive data counts not present in anonymous data counts
- `Fabrictaed %`: For a group of attribute combinations, the percentage of anonymous data counts not present in sensitive data counts

These metrics are summarized for each length of attribute combination up to four in the `Aggregate data quality` and `Synthetic data quality` tables to the left. For example, a value of `62 +/- 5` in the `Length 1` row should be read as "The mean count of attribute combinations with length 1 is 62, with a mean absolute error of 5".

As rules of thumb, under `Synthetic data quality`, the highest accuracy synthetic data will have:

- The mean `Overall Error` less than the mean `Overall Count`
- The `Suppressed %` less than `10%`
- The `Fabricated %` less than `1%`

### Querying anonymous data

Move to the `Query and visualize data` tab to start exploring the anonymous data.

Any record counts shown will use the corresponding protected counts from the aggregate data if these counts exist, since they will always be the most accurate, otherwise the synthetic data will be dynamically filtered to derive the desired count.

Before any filters are applied, the interface will use the `record_count` value from thr aggregate data as the estimate of sensitive records overall.

Try adding one or more attribute values to the query to observe the estimated count change. Notice that:

- selecting values from different attributes creates an "and" query, estimating the count of records with all of the selected attribute values
- selecting values from the same attribute creates an "or" query, estimating the count of records with any of the selected attribute values

### Visualizing anonymous data

The `Chart` panel to the right visualizes the current query using the selected `Chart type`, by default set to `Top attributes`. 

#### Top attributes chart

This chart groups values by attribute and shows these groups in descending count order.

The chart can be configured in multiple ways:

- Set `Subject label` to `Customer` to indicate what kind of individul is being counted
- Set `Types of  top attributes to show` to filter to a particular attribute, e.g., `product_code`
- Set `Number of top attribute values to show` to `5` to show only the top five attribute values

#### Time series chart

This chart plots the counts of multiple attribute values as time series. For example:

- Set `Time attribute` to `period`
- Set `Series attributes` to `price_issue` and `quality_issue` to compare these over time

#### Flow (alluvial) chart

This chart type is typically used to show flows from a source attribute to a target attribute, e.g., from origin to destination in a flight dataset. However, it can also be used to visualize the association between any pair of attributes. For example:

- Set `Source/origin attribute type` to `age_range`
- Set `Target/destination attribute type` to `product_code`
- Set `Highlight attribute` to `price_issue:True`

Mouse over the flow lines on the visualization to see how different combinations of `age_range` and `product_code` are associated with `price_issue:True` complaints.

### Exporting data and visuals

There are three different ways to export the current visual:

- Select `Data CSV` on the left to download a CSV file of the data displayed in the visual
- Select `Chart JSON` on the left to download a JSON file containing the specification of the [Plotly](https://plotly.com/python/) chart shown
- Press the camera icon above the chart to save it as a PNG image file, adjusting `Chart width` and `Chart height` as needed
