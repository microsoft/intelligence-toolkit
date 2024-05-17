# Data Synthesis

The **Data Synthesis** workflow generates differentially-private datasets and data summaries from sensitive case records.

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
