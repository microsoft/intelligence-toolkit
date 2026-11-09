# itk-networks

Entity network detection for Intelligence Toolkit, as a pipeline of pure
functions over polars frames and one networkx graph:

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| 0 | `itk_networks.text` | Node naming (`ENTITY==id`, `type==value`) and the legacy text cleaning. |
| 1 | `itk_networks.prepare` | `AttributeTable`/`FlagTable`/`GroupTable` to long link frames; the attribute graph with equality edges; the flags frame with max and mean flag counts. |
| 2 | `itk_networks.embed` | Embeds nodes of selected types through any itk-core embedder into an aligned matrix. |
| 3 | `itk_networks.neighbors` | `NeighborIndex` protocol (numpy brute-force cosine by default, sklearn adapter optional) and `infer_links`, the symmetric links within a distance threshold. |
| 4 | `itk_networks.project` | Trims attributes above the degree limit and projects the graph onto entities (three hops through attributes, equality edges and inferred links). |
| 5 | `itk_networks.communities` | Connected components, largest first; oversized ones split through a `CommunityDetector`. `itk_networks.leiden` is the graspologic adapter behind the `leiden` extra. |
| 6 | `itk_networks.records` | Entity records with each network's flag statistics, the entities frame and the grouped frame. |
| 7 | `itk_networks.explore` | The attribute subgraph around one network, its simplified form, and node/edge frames. Graph logic only. |
| 8 | `itk_networks.exposure` | Shortest paths from every flagged entity to a selected one, as an `Exposure` value and as the legacy markdown. |
| 9 | `itk_networks.report` | `ChatFn` protocol; builds the report prompt from the simplified network and exposure and asks the LLM. |

`itk_networks.NetworkDetector` strings these together. Every stage takes a
`NetworkResult` and returns a new one, keeping the graph, embeddings and
neighbour matrices, so `infer` can be re-run with a new threshold without
re-embedding and `identify` with a new config without re-indexing.

```python
import polars as pl
from itk_core.openai_configuration import OpenAIConfiguration
from itk_networks import AttributeTable, FlagTable, GroupTable, NetworkConfig, NetworkDetector

frame = pl.read_csv("company_grievances_input.csv")
attributes = AttributeTable(frame=frame, entity_column="name", columns=["address", "city", "email", "phone", "owner"])
flags = FlagTable(frame=frame, entity_column="name", columns=["safety_grievances", "pay_grievances"])
groups = GroupTable(frame=frame, entity_column="name", columns=["sector", "country"])

detector = NetworkDetector.from_openai(OpenAIConfiguration())
result = NetworkDetector.prepare([attributes], [flags], [groups])
print(result.summary)                                   # entities, attributes, flags, groups, links

result = await detector.index(result, ["ENTITY"])       # embed entity names, nearest neighbours
result = detector.infer(result, similarity_threshold=0.03)
result.inferred_links_frame                              # text / similar

result = detector.identify(result, NetworkConfig(max_attribute_degree=10, max_network_entities=20))
result.entities_frame                                    # entity_id, network_id, network_flags, ...
result.grouped_frame                                     # plus one column per group dimension

exposure = NetworkDetector.exposure(result, network_id=3, entity="Earths Bounty")
print(exposure.summary, exposure.markdown)
report = await detector.report(result, network_id=3, entity="Earths Bounty")
```

## Engines behind protocols

| Protocol | Default | Fake in tests |
| -------- | ------- | ------------- |
| `embed.Embedder` | `OpenAIEmbedder` or `LocalEmbedder` via `from_openai` | letter-bag vectors |
| `neighbors.NeighborIndex` | `BruteForceCosine` (numpy) | the default itself |
| `communities.CommunityDetector` | `leiden.GraspologicLeiden`, resolved lazily when a component needs splitting | `ChunkDetector` slicing sorted nodes |
| `report.ChatFn` | `OpenAIClient.generate_chat_async` | canned reply |

Unit tests never embed, fit sklearn or run Leiden. Install `itk-networks[leiden]`
for the graspologic adapter and `itk-networks[sklearn]` for the parity index.

## Differences from `intelligence_toolkit.detect_entity_networks`

Kept bit-for-bit: node names and the `==` encoding, text cleaning, equality
edges, the flags frame, the trimming rule and three-hop projection, component
ordering by size, network ids as list positions, entity record columns and
rounding, the exposure summary, path grouping and markdown, the prompt text.

Changed on purpose:

- `max_network_entities` is honoured. The legacy code passed it positionally
  into `random_seed`, so Leiden always split at 20 and the seed was the user's
  limit. The seed is now `NetworkConfig.leiden_seed` (default 42).
- Nearest neighbours uses `k = min(20, n)`, so fewer than 20 indexed nodes no
  longer raise.
- Ties in component size break on the smallest member and nodes are sorted
  before detection, so network ids are stable between runs.
- Exposure: attribute steps report their true degree (legacy reused the
  previous entity's flag count), no flags gives an empty `Exposure`, and a
  selected entity outside the subgraph gives zero paths instead of raising.
- Null entity ids and null flag/group values are dropped before linking, like
  attributes already were. `flag` is always text.
- `explore` ignores user-removed attributes as well as degree-trimmed ones.
- Inferred links are plain dicts of frozensets; report generation is async.

Graph colours and the streamlit-agraph node payloads (`get_entity_graph`,
`get_type_color`) stay with the Streamlit app; `explore.subgraph_frames`
provides everything they need.

Tests: `uv run pytest packages/itk-networks/tests`.
