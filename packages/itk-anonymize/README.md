# itk-anonymize

Differentially private anonymization of case data for Intelligence Toolkit, as
a pipeline of pure functions over polars frames:

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| 1 | `itk_anonymize.prepare` | Casts every column to text with `""` for missing (whole-number floats lose their `.0`), optional presence-only and rare-value suppression, and the header-plus-rows `RawTable` engines consume. |
| 2 | `itk_anonymize.synthesizability` | `SynthesizabilityStatistics`: attribute combination space against record count. |
| 3 | `itk_anonymize.engine` | The `Synthesizer` protocol. `itk_anonymize.pacsynth_engine` is the only module that imports pacsynth, lazily. |
| 4 | `itk_anonymize.aggregate` | Plain combination counts up to the reporting length, pacsynth key parsing, and the `selections`/`protected_count` frame with its `record_count` row. Keys are sorted tuples, so comparisons never depend on attribute order. |
| 5 | `itk_anonymize.evaluate` | `ErrorReport` per combination length: mean count, mean absolute error, suppressed and fabricated percentages. Numeric first; `to_text_frame()` renders the legacy strings. |
| 6 | `itk_anonymize.selection` | `AttributeValue` terms, their `attribute:value` keys, a readable description, and the data schema of a frame. |
| 7 | `itk_anonymize.queries` | `top_attributes`, `time_series`, `aggregate_links`, `synthetic_links`: exact DP counts where a combination was counted, synthetic row counts otherwise. Same columns as the legacy workflow. |

`itk_anonymize.Anonymizer` strings these together and keeps the parsed
aggregates on the `AnonymizeResult`, so `evaluate` and every query run without
calling the synthesizer again.

```python
import polars as pl
from itk_anonymize import AnonymizeConfig, Anonymizer, AttributeValue
from itk_anonymize.prepare import prepare_frame, suppress_boolean_binary
from itk_anonymize.queries import time_series, top_attributes

sensitive = suppress_boolean_binary(prepare_frame(pl.read_csv("complaints.csv")))
Anonymizer.analyze_synthesizability(sensitive)

anonymizer = Anonymizer.from_pacsynth()
result = anonymizer.anonymize(sensitive, AnonymizeConfig(epsilon=12.0))
result.synthetic                      # synthetic records
result.aggregates_frame               # DP aggregates, Synthetic Data Showcase layout
result.aggregate_error_report.frame   # numeric; .to_text_frame() for the legacy table
result.delta

selection = [AttributeValue(attribute="city", value="Lakeside")]
top_attributes(result.synthetic, result.dp_aggregates, selection, num_values=10)
time_series(result.synthetic, result.dp_aggregates, selection, "period", ["price_issue"])
Anonymizer.links(result, selection, "city", "product", AttributeValue(attribute="price_issue", value="True"))
```

Any object with `synthesize(table, config) -> SynthesisOutput` can replace
pacsynth; the tests use a deterministic fake, so differential privacy never runs
in the unit suite. Modes are plain names (`fabrication_mode="minimized"`), and
`sensitive_zeros` exposes pacsynth's rule that `"0"` is not an attribute unless
asked.

Plotly figures stay in the application layer: the query functions return the
frames those charts are drawn from. The package has no pandas, Streamlit or
Plotly dependency and, like `itk-core`, is bundled into the
`intelligence-toolkit` wheel rather than published on its own.

## Example notebook

Open [`examples/anonymize_case_data/anonymize_case_data.ipynb`](examples/anonymize_case_data/anonymize_case_data.ipynb)
for a walkthrough of the public API on the existing customer-complaints input,
[`customer_complaints_prepared.csv`](../../example_outputs/anonymize_case_data/customer_complaints/customer_complaints_prepared.csv)
(3,000 records, nine attributes). It prepares the records, reports their
synthesizability, releases DP aggregates and synthetic records through pacsynth,
shows both error reports, and answers top-attribute, time-series and flow
questions from the released data, with optional plotly charts drawn from those
frames. A second run at a smaller `epsilon` shows the privacy-utility trade-off.
Outputs are not committed: differential privacy adds fresh noise on every run.

Tests: `uv run pytest packages/itk-anonymize/tests`.
