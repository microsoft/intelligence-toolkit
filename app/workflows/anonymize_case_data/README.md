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

First, navigate to the `Upload sensitive data` tab, select `Browse files`, and upload the `customer_complaints_3k.csv` dataset. A preview of the dataset should appear below.

#### Set subject identifier

Select `Set subject identifier` to expand the dropdown and indicate how individual data subjects (i.e., natural persons) are represented in the dataset. Since each record of the data file represents a different data subject, we can leave `Row number` as the selected option. Select the header again to collapse the dropdown and continue.

#### Select attribute columns to include

Under `Select attribute columns to include`, press `Select all` then deselect the identifying attributes `name`, `street`, and `email`.

#### Quantize datetime attributes

Under `Quantize datetime attributes`, select the `quarter` attribute and `Half` bin size to combine quarters into half years. After pressing `Quantize selected columns`, switching from the `Input data` to the `Processed data` view of the loaded data will show `quarter` now encoded into half years. The column name can be updated in a later step.

#### Quantize numeric attributes

Under `Quantize numeric attributes`, select `age`, set `Target bins` to `5`, and leave `Trim percent` at `0.00`. After pressing `Quantize selected columns`, the `Processed data` view of the loaded data will show `age` now encoded into five age ranges represented as (exclusive mininum value-inclusive maximum value]:

- `(0.0-20.0]`
- `(20.0-40.0]`
- `(40.0-60.0]`
- `(60.0-80.0]`
- `(80.0-100.0]`

Looking at the actual distribution of `age` values in the dataset, we see that there are very few data points in the `(0.0-20.0]` and `(80.0-100.0]` age ranges. We might therefore decide to trim some of these extreme values before determining appropriate bin sizes for the quantized data. Setting `Trim percent` to `0.01` and pressing `Quantize selected columns` again results in new age ranges as follows:

- `(20.0-30.0]`
- `(30.0-40.0]`
- `(40.0-50.0]`
- `(50.0-60.0]`

In general, the fewer the values of a data attribute and the more even the distribution of values across a dataset, the better.