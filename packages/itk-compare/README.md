# itk-compare

Case group comparison for Intelligence Toolkit, as a pipeline of pure functions
over polars frames. Given case records, it counts how often each attribute value
occurs in each group of records (and in each temporal window), ranks the groups,
and turns a selection of those rows into the prompt for an AI group report.

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| 1 | `itk_compare.prepare` | Casts every column to text once (`""` is missing), drops records lacking a group value, applies `attribute:value` filters, and lists the filter options a UI can offer. |
| 2 | `itk_compare.groups` | `group_counts`: records per group with `group_rank`. `attribute_counts`: every group crossed with every attribute value, zero-filled, with `group_attribute_rank` per value. |
| 3 | `itk_compare.temporal` | Counts per group, window and attribute value; missing windows filled with 0; `*_window_delta` against the previous window and dense `*_window_rank` per window. Vectorised, replacing the legacy three-level Python loop. |
| 4 | `itk_compare.summary` | Joins the stage frames into the summary frame with the legacy column names, writes the markdown description, lists the groups, and selects report rows by named groups or top ranks. |
| 5 | `itk_compare.report` | The `ChatFn` protocol (any `(messages, callbacks=None) -> str`), the prompt variables and messages, and `generate_report`. Prompts in `itk_compare.prompts`. |

`itk_compare.CaseGroupComparer` strings these together. `summarize` returns a
`CompareResult` that keeps the prepared, filtered, group, attribute and temporal
frames next to the summary, so report selections, descriptions and group lists
are answered from the result without re-running anything.

```python
import polars as pl
from itk_compare import AttributeFilter, CaseGroupComparer, CompareConfig, ReportSelection

records = pl.read_csv("complaints_prepared.csv", infer_schema_length=0)
config = CompareConfig(
    groups=("city",),
    aggregates=("price_issue", "quality_issue", "product_code"),
    temporal="period",
    filters=(AttributeFilter.parse("age_range:(30-40]"),),
)
result = CaseGroupComparer.summarize(records, config)
result.summary              # city, group_count, group_rank, attribute_value, ..., period_window_delta
result.groups_frame         # one row per group
result.description          # the markdown paragraph shown under the summary
result.dataset_proportion   # % of prepared records that passed the filters

rows, filter_text = CaseGroupComparer.report_rows(result, ReportSelection(top_group_ranks=3))

comparer = CaseGroupComparer.from_openai(OpenAIConfiguration())  # or CaseGroupComparer(any_chat_fn)
report_markdown = comparer.report(result, ReportSelection(groups=({"city": "Lakeside"},)))
```

Any callable with the `ChatFn` shape replaces the OpenAI client; the tests use a
recording fake, so the unit suite never touches the network. Read inputs as text
(`infer_schema_length=0`) to keep values such as `True` or `(30-40]` exactly as
uploaded; typed columns are cast with polars' own text forms (`true`, `1.0`).

## Differences from the legacy module

- Frames are sorted by group columns, attribute value and window, so equal
  inputs give identical frames. Legacy sorted by group columns only.
- Counts, ranks and deltas are `Int64`; legacy mixed `UInt32`, `Int32` and
  `Float64`.
- `dataset_proportion` is an `int` (`43%`), not a `float` (`43.0%`).
- Records whose group value is `""` are dropped with the null ones, so they no
  longer form a phantom group that takes part in `group_rank`.
- Records without a window value are skipped in the temporal stage instead of
  acting as the "previous window" of the first real one.
- Filter options skip missing values instead of listing `column:None`; filter
  values may contain `:`.
- The config rejects empty `groups`/`aggregates` and a temporal column that is
  also a group or aggregate, and `summarize` names any column the data lacks.

Charts stay in the application layer: the frames here are what they are drawn
from. The package has no pandas, Streamlit or Plotly dependency and, like
`itk-core`, is bundled into the `intelligence-toolkit` wheel rather than
published on its own.

## Example notebook

Open [`examples/compare_case_groups/compare_case_groups.ipynb`](examples/compare_case_groups/compare_case_groups.ipynb)
for a walkthrough on the existing customer-complaints input,
[`customer_complaints_prepared.csv`](../../example_outputs/compare_case_groups/customer_complaints/customer_complaints_prepared.csv)
(3,000 records, nine attributes). It builds the summary by `city` across
`period` windows, inspects the group, attribute and temporal frames, re-runs
with a filter, selects rows for a report and, when an OpenAI key is configured,
generates the AI group report; without a key that step is shown and skipped.

Tests: `uv run pytest packages/itk-compare/tests`.
