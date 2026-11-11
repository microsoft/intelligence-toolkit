# itk-patterns

Case pattern detection for Intelligence Toolkit, as a pipeline of pure functions
over polars frames, networkx graphs and numpy arrays:

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| 1 | `itk_patterns.prepare` | Casts every column to text with `""` for missing (whole-number floats lose their `.0`), numbers subjects in row order and unpivots the case table into `(subject, period, attribute type, value, full attribute)` links. Empty values and empty periods are dropped. |
| 2 | `itk_patterns.graph` | One weighted attribute co-occurrence graph per period over every attribute seen in any period: observed pairs scaled onto `[min_edge_weight, 1]`, unobserved pairs at `missing_edge_prop * min_edge_weight`. |
| 3 | `itk_patterns.embed` | The graph encoder embedding (`adjacency_matrix`, `add_diagonal`, `laplacian`, `encoder_embed`, `row_normalize`) and its fusion across periods: per-node positions per period plus an all-time centroid and prior centroids, with cosine shifts. `PeriodEmbedder` is the protocol; `EncoderEmbedder` the default. |
| 4 | `itk_patterns.detect` | `RecordCounter` (cached set intersections), converging pairs per period, growth of pairs into longer patterns, and the scored pattern frame with `detections` and a normalised `overall_score`. |
| 5 | `itk_patterns.series` | `time_series` for the detected patterns and `attribute_counts` for one pattern in one period: the frames the charts and the report are drawn from. |
| 6 | `itk_patterns.report` | `report_variables`, `build_report_messages` and `generate_report` behind a `ChatFn` protocol matching `OpenAIClient.generate_chat_async`. `itk_patterns.prompts` holds the legacy prompt text. |

`itk_patterns.PatternDetector` strings these together. Every stage takes a
`PatternResult` and returns a new one; the period graphs and the embedding
positions stay on the result, so `detect` re-runs with new thresholds without
re-embedding.

```python
import polars as pl
from itk_patterns import DetectionConfig, PatternConfig, PatternDetector

cases = pl.read_csv("customer_complaints_prepared.csv", infer_schema_length=0)
detector = PatternDetector()                       # deterministic, no network

result = PatternDetector.prepare(cases, PatternConfig(period_column="period"))
print(result.summary)                              # links, cases, attributes, periods
result = detector.embed(result)                    # graph encoder embedding per period
result = PatternDetector.detect(
    result, DetectionConfig(min_pattern_count=100, max_pattern_length=5)
)
result.patterns                                    # period, pattern, length, count, mean, z_score, detections, overall_score
result.patterns_display                            # legacy rounding
result.close_pairs, result.all_pairs
result.pattern_series("age_range=(30-40] & product_code=A")
PatternDetector.attribute_counts(result, "age_range=(30-40] & product_code=A", "2023-H1")

# cheaper thresholds, same embeddings
result = PatternDetector.detect(result, DetectionConfig(min_pattern_count=50))

# the LLM report, with any async (messages, stream, callbacks) -> str callable
detector = PatternDetector.from_openai(OpenAIConfiguration())
report = await detector.report(result, pattern, period)
PatternDetector.report_messages(result, pattern, period)   # the prompt, for inspection
```

The pipeline has no randomness, so equal inputs give equal frames. The unit
tests drive the embedding step through a tiny deterministic fake and never
import an embedding model or call the network.

## Differences from the legacy module

- Frames are numeric first (`Int64` counts, unrounded `Float64` scores) and
  sorted deterministically: patterns by `overall_score` descending, then
  period, then pattern; time series by pattern, then period; attribute counts
  by count descending, then value. `patterns_display` applies the legacy
  rounding (half to even, as Python's `round`).
- `time_series` holds one series per distinct pattern. The legacy frame repeated
  a pattern's series once per period it was detected in.
- Missing values are dropped at prepare time instead of surviving as `"nan"`
  attribute nodes; a `count_factor` with a zero mean is null rather than
  infinity.
- The adjacency matrix is symmetric. The legacy encoder filled one triangle
  only, in networkx edge order, which depended on pandas' unstable sort of the
  edge list. Pattern detection is unchanged on the example data because the
  convergence test only reads the sign of one embedding component.
- `PatternConfig.convergence` defaults to `"legacy"`, which compares the second
  component of each period position only, as the shipped workflow did;
  `"vector"` compares whole positions.
- Only the single-level fusion path is ported (`max_level=0`, the only value the
  workflow used), so the package needs no scikit-learn.
- Configuration is validated: period and attribute columns must exist, and
  thresholds have bounds.

Altair charts stay in the application layer: `time_series` and
`attribute_counts` return the frames those charts are drawn from. The package
has no pandas, Streamlit or Altair dependency and, like `itk-core`, is bundled
into the `intelligence-toolkit` wheel rather than published on its own.

## Example notebook

Open [`examples/detect_case_patterns/detect_case_patterns.ipynb`](examples/detect_case_patterns/detect_case_patterns.ipynb)
for a walkthrough of the public API on the existing customer-complaints input,
[`customer_complaints_prepared.csv`](../../example_outputs/detect_case_patterns/customer_complaints/customer_complaints_prepared.csv)
(3,000 records, eight attributes, twelve half-year periods). It builds the
attribute model, embeds the period graphs, detects and ranks patterns, re-runs
with different thresholds without re-embedding, shows the time series and
attribute counts behind one pattern, and generates the AI pattern report when
an OpenAI configuration is present. Outputs are not committed.

Tests: `uv run pytest packages/itk-patterns/tests`.
