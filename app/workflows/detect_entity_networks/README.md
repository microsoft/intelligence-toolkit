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
- on GitHub, at [example_outputs/detect_entity_networks/company_grievances](https://github.com/microsoft/intelligence-toolkit/tree/main/example_outputs/detect_entity_networks/company_grievances)

### Creating the data model

Navigate to the `Create data model` tab and upload the `company_grievances_data.csv` file.

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

The core modelling approach taken by `Detect Entity Networks` is to connect entities that share the same attribute values (e.g., email, phone number). However, not all shared attributes (or even entity names) will be recorded in the same format in all cases. Using the `Match similar nodes (optional)` panel on the left, you can  