# Match Entity Records

The [`Match Entity Records`](https://github.com/microsoft/intelligence-toolkit/blob/main/app/workflows/match_entity_records/README.md) workflow identifies fuzzy record matches across different entity datasets.

Select the `View example outputs` tab (in app) or navigate to [example_outputs/match_entity_records](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/match_entity_records) (on GitHub) for examples.

## How it works

1. [**Input**] Multiple datasets representing overlapping entities with inconsistent record formats/naming conventions.
2. [**Embedding Calls**] The records are harmonized and embedded into a multi-dimensional semantic space, with similar records close to one another.
3. [**Process**] Clusters of matching records are converted into groups, with user-controllable similarity thresholds.
4. [**Output**] Record groups CSV file containing all records matched to the same group. Can be created and used independently without any AI calls.
5. [**AI Calls**] On request from the user, the system uses generative AI to evaluate the likelihood of the records representing a real-world match.
6. [**Output**] AI record match CSV file evaluating the likelihood of a real-world match for each of the identified record groups.

## Input requirements

- The input data files should be in CSV format and represent individual entities to be matched against one another, with one entity per row.
- Entities may be represented using a variety of attribute types, including unstructured text (e.g., street addresses) in different formats.
- Given the goal of matching entity records, direct identifiers (e.g., names, aliases, ids, phone numbers, email addresses, street addresses) of the respective entities should be included in data inputs as they increase the specificity of the matches detected.
- Text representations of input records will be sent to external APIs for embedding and text generation. Using the entity records in such a way must comply with all applicable laws, regulations, and policies governing their source documents, including those pertaining to privacy and security.

## Tutorial

The task for this tutorial is detecting matching records across two related `company_grievances` datasets available for download either:

- in app, via `View example outputs` tab &rarr; `Input dataset 1`, `Input dataset 2` tabs
- on GitHub, at [example_outputs/match_entity_records/company_grievances/company_grievances_input_data_1.csv](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/match_entity_records/company_grievances/company_grievances_input_data_1.csv) and [example_outputs/match_entity_records/company_grievances/company_grievances_input_data_2.csv](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/match_entity_records/company_grievances/company_grievances_input_data_2.csv).

### How record embedding works

This workflow relies on a technique called *text embedding* to create vector-vased representations of data records. Since records with similar semantic content are "embedded" at similar points in vector space, we can use text embeddings to identify fuzzy matches between records that describe the same entity in ways that are semantically similar but lexically different (e.g., because of formatting, typos, etc.)

Using this approach, the records of the following dataset:

| company: *string*  | street: *string*    | city: *string* | country: *string* |
|--------------------|---------------------|----------------|-------------------|
| MediaWave          | 1111 Broadcast Blvd | Media City     | NewsLand          |
| UrbanTech          | 909 Innovation Dr   | Tech City      | Innovatia         |
| TechGurus          | 963 Innovation Dr   | Tech City      | Innovatia         |

would first be converted to the following "sentences":

```code
company: mediawave; street: 1111 broadcast blvd; city: media city; country: newsland
company: urbantech; street: 909 innovation dr; city: tech city; country: innovatia
company: techgurus; street: 963 innovation dr; city: tech city; country: innovatia
```

before being transformed into embedding vectors.

### Selecting the embedding approach

Intelligence Toolkit supports the use of [OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings/), which are suitable for up to `8191` input tokens (about 6000 words), as well as the [Hugging Face all-MiniLM-L6-v2 embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), which is suitable for up to `256` input tokens (about 200 words).

The advantage of OpenAI embedding models is that they provide much richer representations of much longer content. However, for those situations where the text units are short, the Hugging Face model can be run locally without calling any external APIs and can stil provide excellent results. This is done by setting `Use local embeddings` in the sidebar.

### Preparing the input datasets

Navigate to the `Upload record datasets` tab, press `Browse files`, then upload the `company_grievances_input_data_1.csv` and `company_grievances_input_data_2.csv` datasets. These datasets describe overlapping sets of entities, but in different formats and with different ID schemes. The goal of the workflow is to map different entity IDs into groups that represent the "same" actual real-world entity.

The interface will automatically select the first uploaded dataset for processing. If this is not `company_grievances_input_data_1.csv`, then select this filename under `Select a file to process` on the left.

Using the controls under `Map columns to data model` on the right:

- set `Dataset name` to `D1`
- set `Entity name column` to `employer_name`
- set `Entity ID column (optional)` to `employer_id`
- set `Entity attribute columns` to `sector`, `address`, `city`, `country`, `email`, `phone`, and `owner` (i.e., all remaining columns)
- press `Add records to model`

In general:

- if no ID column is selected, then the record index will be used as the record ID
- only select `Entity attribute columns` that you would expect to be shared across the different datasets

Next, we repeat this process for `company_grievances_input_data_2.csv`. First, select this filename under `Select a file to process` on the left. Then:

- set `Dataset name` to `D2`
- set `Entity name column` to `company_name`
- set `Entity ID column (optional)` to `company_id`
- set `Entity attribute columns` to `industry_sector`, `street_address`, `city_address`, `country_address`, `email_address`, `phone_number`, and `company_owner` (i.e., all remaining columns)
- press `Add records to model`

You can see that the two datasets represent the same kind of data using different column names. These discrepancies can be fixed in the next step.

Navigate to the `Detect record groups` tab to continue.

### Configuring the text embedding model

The interface on the left shows an empty selection box for `Attribute 1`. Within this field, the selectable values all have a suffix indicating their source dataset (here, `D1` or `D2`). Select `address::D1` and `street_address::D2` as the values for `Attribute 1`, and optionally enter either label (or a new label) for this attribute in the `Label (optional)` field. If no label is provided, the first value alphabetically will be used as the attribute label in the unified dataset.

Repeat this process to match the following pairs of attributes:

- `city::D1` and `city_address::D2`
- `company_owner::D1` and `city_address::D2`
- `country::D1` and `country_address::D2`
- `sector::D1` and `industry_sector::D2`
- `owner::D1` and `company_owner::D2`
- `city::D1` and `city_address::D2`
- `email::D1` and `email_address::D2`
- `phone::D1` and `phone_number::D2`

In general, if more than two datasets are being matched, then select all attribute labels across all datasets in each selection box.

### Configuring similarity thresholds

There are two key parameters for configuring the similarity-based matching process:

- `Matching record distance (max)`: the maximum distance between the embedding vectors of two records to consider an embedding match
- `Matching name similarity (min)`: the minimum Jaccard similarity between the names of embedding-matched records to consider an overall match

Starting out with a `Matching record distance (max)` of `0.001` may yield a small number of matches, while setting the threshold to `0.01` may return a greater number of matches. This *semantic* matching process can also be followed by a *lexical* matching process on entity names only, explained next.

#### Understanding lexical matching

With the default `Matching name similarity (min)` of `0.0`, there are no direct constraints on the match between the name fields of the record group. Setting this to a value of *p*, however, means that a fraction *p* of the 3-character combinations ("trigrams") in each name must overlap for there to be an overall match.

For example, setting it to `0.75` means that 0.75 or 75% of 3-character combinations ("trigrams") must overlap.

Consider `Sprout Ltd` (Entity A) and `Spout Ltd` (Entity B = Entity A with a typo) as entitiy names:

| Common trigrams | Entity A trigrams only | Entity B trigrams only |
|-----------------|------------------------|------------------------|
|                 | "spr"                  | "spo"                  |
|                 | "pro"                  | "pou"                  |
|                 | "rou"                  |                        |
| "out"           |                        |                        |
| "ut "           |                        |                        |
| "t l"           |                        |                        |
| " lt"           |                        |                        |
| "ltd"           |                        |                        |

The similarity measure used is the *Jaccard* similary, defined as the size of the intersection of comparable elements (the 5 common trigrams) divided by the size of the union of comparable elements (10 distinct trigrams overall). For this example, the Jaccard similarity resulting from a single omitted character is `0.5`, which would fail to reach the target level of `0.75`.

If the two entities were named `Sprout Technology Limited` (Entity A) and `Spout Tecnology Limited` (Entity B = Entity A with two typos), then the trigram sets change:

| Common trigrams | Entity A trigrams only | Entity B trigrams only |
|-----------------|------------------------|------------------------|
|                 | "spr"                  | "spo"                  |
|                 | "pro"                  | "pou"                  |
|                 | "rou"                  |                        |
| "out"           |                        |                        |
| "ut "           |                        |                        |
| "t t"           |                        |                        |
| " te"           |                        |                        |
| "tec"           |                        |                        |
|                 | "ech"                  | "ecn"                  |
|                 | "chn"                  | "cno"                  |
|                 | "hno"                  |                        |
| "nol"           |                        |                        |
| "olo"           |                        |                        |
| "log"           |                        |                        |
| "ogy"           |                        |                        |
| "gy "           |                        |                        |
| "y l"           |                        |                        |
| " li"           |                        |                        |
| "lim"           |                        |                        |
| "imi"           |                        |                        |
| "mit"           |                        |                        |
| "ite"           |                        |                        |
| "ted"           |                        |                        |

Now, the Jaccard simialrity is 17/27 = 0.63. Higher, but not quite at the `0.75` target.

Finally, if there was only a single typo distinguishing `Sprout Technology Limited` (Entity A) and `Spout Technology Limited` (Entity B = Entity A with a typo), then the trigram sets change again:

| Common trigrams | Entity A trigrams only | Entity B trigrams only |
|-----------------|------------------------|------------------------|
|                 | "spr"                  | "spo"                  |
|                 | "pro"                  | "pou"                  |
|                 | "rou"                  |                        |
| "out"           |                        |                        |
| "ut "           |                        |                        |
| "t t"           |                        |                        |
| " te"           |                        |                        |
| "tec"           |                        |                        |
| "ech"           |                        |                        |
| "chn"           |                        |                        |
| "hno"           |                        |                        |
| "nol"           |                        |                        |
| "olo"           |                        |                        |
| "log"           |                        |                        |
| "ogy"           |                        |                        |
| "gy "           |                        |                        |
| "y l"           |                        |                        |
| " li"           |                        |                        |
| "lim"           |                        |                        |
| "imi"           |                        |                        |
| "mit"           |                        |                        |
| "ite"           |                        |                        |
| "ted"           |                        |                        |

Now, the Jaccard simialrity is 20/25 = 0.75, or precisely at the `0.75` target.

As you can see, this *lexical* approach to determining text similarity can break down when the text representations of the same entity are substantially different (e.g., `Sprout Ltd` vs `Sprout & Partners Technology Ltd`). This is why the *semantic" approach of text embedding is so powerful. At the same time, sometimes embedding-based matching will return pairs of entity records with widely different names. In these cases, it can be helpful to require some dgeree of lexical overlap.

### Detecting and explaining record groups

Press `Detect record groups` to see the detected record groups on the right. The last two columns of this table show the size of the detected group and the average name similarity, the latter of which is used to create the group ranking. The results can be downloaded and used directly via the `Download record groups` button. Alternatively, you can proceed to the `Evaluate record groups` tab to have generative AI evaluate the likely real-world match represented by each of the groups. Here, press `Generate` to create a group-by-group evaluation and explanation of group relatedness.

A table of `Group ID`, `Relatedness`, and `Explantion` fields will be updated dynamically as generative AI processes the candidate record groups.

Press `Download AI match reports` to download the visible results, or `Download integrated results` to download a single result table combining matched records and AI group evaluations.
