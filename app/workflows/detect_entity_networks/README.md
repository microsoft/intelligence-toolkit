# Detect Entity Networks

The [`Detect Entity Networks`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/detect_entity_networks/README.md) workflow generates intelligence reports on risk exposure within detected networks of entities.

Select the `View example outputs` tab (in app) or navigate to [example_outputs/detect_entity_networks](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/detect_entity_networks) (on GitHub) for examples.

## How it works

1. [**Input**] Datasets describing entity attributes (used to infer relationships) and entity risks (used to infer risk exposure).
2. [**Process**] The user selects the data columns representing entity attributes, entity relationships, entity risk flags, and entity groups.
3. [**Embedding Calls**] Optional: the user selects node types (entity labels/attributes) to fuzzily match by embedding them into a multi-dimensional semantic space.
4. [**Process**] The system uses the entity relationships and shared attributes to infer networks of closely related entities.
5. [**Output**] The system shows (a) the structure of any selected network, and (b) risk exposure paths for any selected entity. Can be created and used independently without any AI or embedding calls.
6. [**AI Calls**] For entities or networks of interest selected by the user, generative AI is used to create AI network reports.
7. [**Output**] AI network report MD/PDF file(s) describing the structure of the network and the nature of risk propagation within it.

## Input requirements

- The input data files should be in CSV format and represent (a) attributes of individual entities, (b) relationships between pairs of entities, or (c) individual or aggregated flags linked to individual entities.
- Entities may be represented using a variety of attribute types, including unstructured text (e.g., street addresses) in different formats.
- Given the goal of linking closely-related entities based on their shared attributes, direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) of the respective entities should be included in data inputs as they increase the specificity of the links detected.
- Text representations of input data will be sent to external APIs for embedding and text generation. Using the entity data in such a way must comply with all applicable laws, regulations, and policies governing their source documents, including those pertaining to privacy and security.

## Tutorial

The task for this tutorial is detect networks of entities and their associated level of relationship-based risk using the `company_grievances` dataset available for download either:

- in app, via `View example outputs` tab &rarr; `Input data` tab
- on GitHub, at [example_outputs/detect_entity_networks/company_grievances/company_grievances_input.csv](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/detect_entity_networks/company_grievances/company_grievances_input.csv)

### Creating the data model

Navigate to the `Create data model` tab and upload the `company_grievances_input.csv` file.

Under `Map columns to model`, we will start with the `Link type` of `Entity-Attribute` to link entities to their attributes. These should be distinctive, i.e., linked only to that entity or closely related entities.

Set `name` as the `Entity ID column` and select `address`, `city`, `email`, `phone`, and `owner`  as `Attribute column(s) to link on`.

We would not select `sector` or `country` as attribute columns to link on since these are too broad, and would connect too many unrelated entities into the same networks. While `city` could be narrow or broad depending on the dataset (and city), the workflow has a way of showing shared attributes of relevance (like `city`) without using them to detect the entity networks.

Press `Add links to model` to see a summary of data model so far.

Next, set `Link type` to `Entity-Flag`, keep `name` as the `Entity ID column`, and set `safety_grievances`, `pay_grievances`, `conditions_grievances`, `treatment_grievances`, and `workload_grievances` as `Flag value column(s)`.

The format of these columns is as counts of the corresponding grievances, or "flags" more generally, so set `Flag format` to `Count`.

If flags were formatted as a column of flag labels representing instances of that flag type for the adjacent entity, then you would select `Instance` instead.

Press `Add links to model` to see the `Number of flags` update.

Finally, set `Link type` to `Entity-Group`, keep `name` as the `Entity ID column`, and set `Group value column(s) to group on` to `sector` and `country`. This will later allow detected entity networks to be analyzed by sector and city.

Press `Add links to model` to see the `Number of groups` update. The data model is now complete.

### Processing the data model

Navigate to the `Process data model` tab to continue.

From the "bipartite" graph of entity and attributes specified by the data model, it is possible to induce a "unipartite" graph of entity nodes whose edge weights represent counts of shared attributes. Within this unipartite graph, the workflow uses [Leiden](https://www.nature.com/articles/s41598-019-41695-z) community detection to identify networks of closely-related nodes. In many cases, however, these networks require further processing before they capture the groups of entities that a human expert would identify as "closely related" in the real world.

#### Connecting similar but not identical labels

The core modelling approach taken by `Detect Entity Networks` is to connect entities that share the same attribute values (e.g., email, phone number). However, not all shared attributes (or even entity names) will be recorded in the same format in all cases. Using the `Match similar nodes (optional)` panel on the left, you can fuzzily match similar node values such that they become connected in the resulting networks.

Under `Select node types to fuzzy match`, select `ENTITY` to match similar entity names. Pressing `Index nodes` creates a text embedding of the selected node values that can then be analyzed with different similarity thresholds to find the right matching levels. Pressing `Infer links` then infers pairs of similar nodes based on the current value of `Similarity threshold for fuzzy matching (max)`. The default value of `0.001` may not yield any matches, and if so, adjust the value upwards in small increments until matches are shown in a table. For example, a similarity threshold of `0.003` may infer a small number of links. The least similar matches are shown at the top of the table, so you can keep increasing the threshold while these values still represent a close match. For this example dataset, a value of `0.03` should give good-quality links.

#### Removing noisy attributes

There are multiple ways in which attributes can add noise to the resulting networks. For example, imagine several entities having an `address` of `USA`. These entities would likely end up in the same network, but it wouldn't represent a set of close real-world relationships. If such a situation arises (and it may not be detected until viewing the detected networks), then the `Remove attributes` controls under `Constrain networks` can be used to remove these attribute values and prevent such noisy networks from forming.

Another way in which certain attribute values can add noise to the detected networks is by connecting too many entities to represent a close real-world relationship. Setting a `Maximum attribute degree` prevents this problem by removing any attributes whose degree (number of connected entities) exceeds the specified limit.

Once networks exceed a certain size, it is similarly difficult to argue that they are all closely related. It is also difficult to make sense of the actual close relationships that do exist if the network visualization is too dense. The `Max network entities` field therefore sets a limit on the maximum number of entities that can be detected in a single network.

Finally, there are some kinds of `Supporting attribute types` that are helpful to see only when they link entities connected by other means. In this example, `city` is a good candidate for a supporting attribute: it is not a strong enough connection on its own, but in conjunction with other shared attributes it could help to communicate an even closer relationship.

Add `city` as a supporting attribute type before pressing `Identify networks`. The system will show the number of networks identified as well as a table of all attribute values removed because of their high degree.

### Exploring detected networks

Navigate to the `Explore networks` tab to begin exploring the detected networks.

In the `Select entity network` expander, the default table shows each detected entity network as a single row. Double clicking on column headers reranks networks by the associated metric:

- `Network Entities`: the number of entities in the network
- `Network Flags`: the total number of flags across all entities in the network
- `Flagged`: the total number of flagged entities in the network (i.e., with one or more flags)
- `Flags/Entity`: the average number of flags per entity in the network
- `Flagged/Unflagged`: the number of flagged entities for every unflagged entity

Selecting `Show entities` expands this table to show network statistics for each individual entity, including the number of `Entity Flags` directly linked to the entity. Entering a `quickFilter...` query above the table can then be used to search for specific entities and their network statistics.

Selecting `Show groups`, with or without `Show entities`, adds values from the specified grouping columns to the table. These are not used in the workflow, but they allow for group-based comparison of entity network using external analysis tools.

Deselect `Show entities` and select a row from the table to see a visualization of the associated entity network. This visualization can be panned/zoomed using the mouse/wheel.

In the visualization, entities are represented with larger nodes and attribute values with smaller nodes. Links (or edges) between an entity node and an attribute node indicate that the entity has that attribute value. Links between entity nodes or attribute nodes indicate that the names are similar based on the fuzzy matching performed.

Entity nodes are coloured blue unless the entity has associated flags, in which case they are coloured red. Otherwise, the colours of attribute nodes communicate the corresponding attribute type.

With `Graph type` set to `Full`, the network visualization shows all attribute values of all entity nodes, even if they are not shared by any other entity.

Changing `Graph type` to `Simplified` simplies the network visualization in two key ways:

1. Attribuute values are only shown if they are shared by two or more entity nodes
2. Nodes are merged if they were matched during the fuzzy matching process

`Simplified` networks are particularly helpful when evaluating the weight of evidence for a close relationship between entities, especially if one of them has a substantial number of flags.

Reverting back to `Full` as the `Graph type`, selecting `Show entities`, and selecting any entity with one or more network flags shows a slightly different kind of network view. Here, the node of the selected entity is represented by a larger node size, and a `Flag Exposure Paths` report on the right summarizes the various network paths by which flagged entities are related to the selected entity. This information is used by generative AI in the next step to reason about the overall level of flag (e.g., risk) exposure for the entity within its network of closely-related entities.

### Generating AI network reports

Navigate to the `Generate AI network reports` tab and press `Generate` to generate a report on the selected entity or network.

By default, the report will analyze entity relationships and flag exposure from a neutral perspective. To guide the generation of report text towards a "risk exposure" framing, modify the `Prompt text` accordingly. For example, try adding the following at the end of the prompt:

`Interpret flags as risks and add a section analyzing the overall level of risk exposure for the selected entity. Suggest actions that could help to clarify these potential risks.`
