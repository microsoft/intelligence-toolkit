# Anonymize Case Data

The `Anonymize Case Data` workflow generates differentially-private datasets and data summaries from sensitive case records.

Navigate to [example_outputs/anonymize_case_data](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/anonymize_case_data) (on GitHub) for examples.

## How it works

1. [**Input**] Sensitive case records representing attributes of individual data subjects, where there is risk that even deidentified subjects may be reidentified using distinctive combinations of attribute values.
2. [**Process**] [Private Accurate Combination Synthesizers (PAC-Synth)](https://github.com/microsoft/synthetic-data-showcase/blob/main/packages/lib-pacsynth/README.md) is used to generate synthetic datasets that are statistically similar to the original data, but with [differential privacy (DP)](https://en.wikipedia.org/wiki/Differential_privacy) ([proof](https://github.com/microsoft/synthetic-data-showcase/blob/main/docs/dp/dp_marginals.pdf)).
3. [**Process**] The synthesizer first creates a DP aggregate dataset in which the counts of all attribute combinations up to length 4 are precomputed and then adjusted with calibrated noise that provides DP.
4. [**Process**] The synthesizer then uses these counts to generate a synthetic dataset in which the synthetic records are consistent with these DP aggregates and thus inherit their DP properties.
5. [**Output**] A DP synthetic dataset CSV file with one row per synthetic record and a DP aggregate dataset CSV file with one row per count of attribute value combinations up to length 4. Can be created and used independently without any AI or embedding calls.
6. [**Process**] The user can query the DP datasets to explore summary datasets and charts.
7. [**Output**] Summary dataset CSVs, chart PNGs, and Plotly JSONs that can be readily shared with differential privacy for data subjects.

## Input requirements

- The input data file should be in CSV format with each row representing a different case (i.e., individual person or data subject).
- For data synthesis, each case must be represented as a collection of discrete (i.e., categorical or binary) attributes. Any continuous attributes must first be quantized via the user interface.
- Given the goal of creating an anonymous dataset, no direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) should be included in data outputs. Following the principle of [data minimization](https://en.wikipedia.org/wiki/Data_minimization), such direct identifiers should be removed from data inputs because they are not required for the processing purpose and create unnecessary risks for the data subject. Tools such as Microsoft Excel can be used to delete any direct identifier columns prior to use in Intelligence Toolkit.
- The nature of differential privacy means that indirect identifiers (e.g., age range, year of birth, gender, country, city) may be freely included in the data inputs. None of the combinations of these identifiers (or of any attributes) in the output data allow the presence of individuals to be inferred with any degree of certainty.

## Use with other workflows

`Anonymize Case Data` can be used to anonymize case data for privacy-preserving analysis in any other workflow accepting structured records as input:

- `Detect Case Patterns`
- `Compare Case Groups`

`Generate Record Data` can also be used to generate mock data for demonstration or evaluation of the `Anonymize Case Data` workflow.
