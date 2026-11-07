# itk-match

Entity record matching for Intelligence Toolkit, as a pipeline of pure functions
over one unified polars frame:

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| 1 | `itk_match.unify` | Aligns several `Dataset`s onto canonical attribute labels and one `record_id` per row. |
| 2 | `itk_match.sentences` | Renders each record as a `NAME: …; LABEL: …` sentence for embedding. |
| 3 | `itk_match.embed` | Embeds sentences through any itk-core embedder into an aligned matrix. |
| 4 | `itk_match.blocking` | Finds candidate pairs by cosine distance over the k nearest neighbours (numpy only). |
| 5 | `itk_match.names` | Scores candidate pairs on entity name: trigram Jaccard or rapidfuzz. |
| 6 | `itk_match.cluster` | Union-find over pairs above the name threshold, producing `Group`s. |
| 7 | `itk_match.evaluate` | Asks the LLM for a structured JSON relatedness score per group. |

`itk_match.Matcher` strings these together and keeps the embeddings on the
`MatchResult`, so thresholds can be changed with `regroup` without re-embedding.

`MatchConfig.min_name_similarity` defaults to `0.75`, matching the legacy name
similarity threshold. Set it explicitly to `0.0` to disable name filtering.

```python
import polars as pl
from itk_core.openai_configuration import OpenAIConfiguration
from itk_match import Dataset, MatchConfig, Matcher

d1 = Dataset(
    name="D1",
    frame=pl.read_csv("input_1.csv"),
    name_column="employer_name",
    id_column="employer_id",
    attributes={"address": "address", "city": "city", "phone": "phone"},
)
d2 = Dataset(
    name="D2",
    frame=pl.read_csv("input_2.csv"),
    name_column="company_name",
    id_column="company_id",
    attributes={"street_address": "address", "city_address": "city", "phone_number": "phone"},
)

matcher = Matcher.from_openai(OpenAIConfiguration())
result = await matcher.match([d1, d2], MatchConfig(max_distance=0.05))
result.groups_frame            # one row per matched record, with group_id / group_size / name_similarity
tighter = matcher.regroup(result, MatchConfig(max_distance=0.01, min_name_similarity=0.75))
evaluations = await matcher.evaluate(tighter)   # group_id / relatedness / explanation
Matcher.integrate(tighter, evaluations)         # matched records joined with their evaluation
```

Only the embedding and evaluation stages call external APIs. It has no Streamlit,
scikit-learn or torch dependency and, like `itk-core`, is bundled into the
`intelligence-toolkit` wheel rather than published on its own.

Tests: `uv run pytest packages/itk-match/tests`.

## Offline notebook demo

Open [`examples/entity_matching/entity_matching.ipynb`](examples/entity_matching/entity_matching.ipynb)
for a runnable, executed walkthrough of the public matching API using the two
existing company-grievance inputs directly:
[`company_grievances_input_data_1.csv`](../../example_outputs/match_entity_records/company_grievances/company_grievances_input_data_1.csv)
and
[`company_grievances_input_data_2.csv`](../../example_outputs/match_entity_records/company_grievances/company_grievances_input_data_2.csv).
Each input has 2,000 records; no copied or newly generated datasets are needed.
It covers column alignment, an offline character-trigram embedder, cosine/name
thresholds, matched groups, unmatched records, and regrouping without re-embedding.
The notebook includes setup instructions for Python 3.11/3.12, input/result
integrity checks, and observed cross-source and source-only group counts. No API keys,
model downloads, or paid calls are needed after installing dependencies.

The custom embedder is a transparent lexical baseline, not a pretrained semantic
model. Its thresholds are illustrative, and no identity accuracy is claimed:
historical matching outputs are not used as ground truth.
