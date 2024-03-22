import plotly.express as px

outputs_dir = 'cache/data_synthesis/outputs'


att_separator = ';'
val_separator = ':'

color_schemes = {
    'Plotly': px.colors.qualitative.Plotly,
    'D3': px.colors.qualitative.D3,
    'G10': px.colors.qualitative.G10,
    'T10': px.colors.qualitative.T10,
    'Alphabet': px.colors.qualitative.Alphabet,
    'Dark24': px.colors.qualitative.Dark24,
    'Light24': px.colors.qualitative.Light24, 
    'Set1': px.colors.qualitative.Set1, 
    'Pastel1': px.colors.qualitative.Pastel1, 
    'Dark2': px.colors.qualitative.Dark2, 
    'Set2': px.colors.qualitative.Set2, 
    'Pastel2': px.colors.qualitative.Pastel2, 
    'Set3': px.colors.qualitative.Set3, 
    'Antique': px.colors.qualitative.Antique, 
    'Bold': px.colors.qualitative.Bold, 
    'Pastel': px.colors.qualitative.Pastel,
    'Prism': px.colors.qualitative.Prism,
    'Safe': px.colors.qualitative.Safe,
    'Vivid': px.colors.qualitative.Vivid
}

intro = """ \
# Data Synthesis

The **Data Synthesis** workflow generates differentially-private datasets and data summaries from sensitive case records.

## How it works

1. [**Input**] Personal case records representing sensitive attributes of data subjects, where there is risk that even de-identified subjects may be re-identified using distinctive combinations of attribute values.
2. [**Process**] [Private Accurate Combination Synthesizers (PAC-Synth)](https://github.com/microsoft/synthetic-data-showcase/blob/main/packages/lib-pacsynth/README.md) is used to generate synthetic datasets that are statistically similar to the original data, but with [differential privacy (DP)](https://en.wikipedia.org/wiki/Differential_privacy) ([proof](https://github.com/microsoft/synthetic-data-showcase/blob/main/docs/dp/dp_marginals.pdf)).
3. [**Process**] The synthesizer first creates a DP aggregate dataset in which the counts of all attribute combinations up to length 4 are precomputed and then adjusted with calibrated noise that provides DP.
4. [**Process**] The synthesizer then uses these counts to generate a synthetic dataset in which the synthetic records are consistent with these DP aggregates and thus inherit their DP properties.
5. [**Output**] A DP synthetic dataset CSV file with one row per synthetic record and a DP aggregate dataset CSV file with one row per count of attribute value combinations up to length 4. Can be created and used independently without any AI or embedding calls.
6. [**Process**] The user can query the DP datasets to explore summary datasets and charts.
7. [**Output**] Summary dataset CSVs, chart PNGs, and Plotly JSONs that can be readily shared with differential privacy for data subjects.
"""