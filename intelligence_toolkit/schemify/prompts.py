"""
Prompt templates for Schemify.

Optimized for web search grounding and structured extraction.
"""

# Record extraction from web search results
RECORD_EXTRACTION = """\
You are extracting structured entity records from web search results about {category}.

The following text contains information gathered from web sources. Extract entity records following these rules:

--Rules--

1. Each record label should be in ALL CAPITALS and represent a concrete instance of "{category}"
2. Only extract entities that are directly mentioned with factual information
3. Attribute names should use Initial Capitals with spaces (e.g., "Year Founded", "Primary Function")
4. Attribute values should start with a capital letter (e.g., "Machine learning", "North America")
5. Only use unicode characters if essential to the meaning (avoid decorative symbols, emojis, or special punctuation)
6. Prefer categorical attributes that could apply to multiple entities
7. Keep attribute values ATOMIC — each value should describe one concept/dimension.
   If a value spans two attributes (e.g., "open-source mobile app" = license + deployment model),
   split it: set License = "Open-source" and Deployment model = "Mobile app" separately.
8. Do not invent or hallucinate information not present in the text
9. If the text references the same entity with different spellings, use the most complete/formal name

--Guidance--

{guidance}

--Existing entities to avoid duplicating--

{existing_labels}

--Source text--

{text}
"""

# Record extraction WITH per-attribute citation tracking
RECORD_EXTRACTION_WITH_CITATIONS = """\
You are extracting structured entity records from web search results about {category}.

The following text contains information gathered from numbered web sources. Extract entity records and cite which sources support each attribute value.

--Rules--

1. Each record label should be in ALL CAPITALS and represent a concrete instance of "{category}"
2. Only extract entities that are directly mentioned with factual information
3. For each attribute value, include citation_indices listing which source(s) (by 0-based index) support that specific value
4. CRITICAL: Only cite sources that EXPLICITLY mention BOTH the entity AND the attribute value together
5. CRITICAL: Each attribute value MUST appear in the source text in the context of THAT SPECIFIC entity - do NOT mix attributes between entities
6. Attribute values should start with a capital letter (e.g., "Machine learning", "North America")
7. Only use unicode characters if essential to the meaning (avoid decorative symbols, emojis, or special punctuation)
8. Keep attribute values ATOMIC — each value should describe one concept/dimension.
   If a value spans two attributes (e.g., "open-source mobile app" = license + deployment model),
   assign each dimension to its own attribute rather than concatenating them.
9. If the same source mentions multiple different entities, extract each as a separate record with only the attributes mentioned for that specific entity
9. If a value comes from your general knowledge without source support, use an empty citation_indices array
10. Do not invent or hallucinate information not present in the text
11. If the text references the same entity with different spellings, use the most complete/formal name

--Guidance--

{guidance}

--Existing entities to avoid duplicating--

{existing_labels}

--Sources (numbered)--

{sources_list}

--Source text--

{text}
"""

# Web search query for discovering new entities
ENTITY_DISCOVERY_QUERY = """\
Search for specific, named examples of {category}.

{guidance}

{subcategory_focus}

{exclusion_list}

Provide factual information about each example including key attributes, characteristics, and notable details. Include specific names, dates, organizations, and verifiable facts where available.
"""

# Single-entity attribute verification — used when verifying/sourcing attributes for ONE known entity
SINGLE_ENTITY_VERIFICATION = """\
You are verifying attribute values for a SINGLE known entity from web search results.

The entity is: "{entity_label}"
(Also known as: {entity_aliases})
Category: {category}

--Rules--

1. Extract attributes ONLY for this entity — ignore other entities in the text.
2. You MUST use EXACTLY this label in your output: "{entity_label}"
3. If the entity appears under a different name (rebranding, acquisition, name change),
   still use the original label above. Note the new name in an additional_attribute
   called "Also known as" with the new/alternate name.
4. For each attribute value, include citation_indices listing which source(s) support it.
5. CRITICAL: Only cite sources that EXPLICITLY mention BOTH the entity AND the attribute value.
6. Attribute values should start with a capital letter.
7. Keep attribute values ATOMIC — one concept per value.
8. If a value comes from general knowledge without source support, use an empty citation_indices array.
9. Do not invent or hallucinate information.

--Guidance--

{guidance}

--Sources (numbered)--

{sources_list}

--Source text--

{text}
"""

# Web search query for expanding a specific entity
ENTITY_EXPANSION_QUERY = """\
Search for detailed information about "{label}" as an example of {category}.

Focus on the following attributes: {attributes}

{guidance}

Provide specific, factual information with concrete details. Include dates, organizations, statistics, and other verifiable information where available.
"""

# Subcategory generation for diverse exploration
SUBCATEGORY_GENERATION = """\
You are identifying underrepresented subcategories to ensure comprehensive coverage.

Given the following category and existing entities, identify a subcategory that is NOT yet well-represented but is important and broad.

Category: {category}

Guidance: {guidance}

--Existing entities--

{existing_labels}

--Instructions--

Generate a concise subcategory description (1-2 sentences) that should form the focus of the next search. The subcategory should:
1. Be significantly different from areas already covered
2. Be broad enough to contain multiple entities
3. Be important/notable within the overall category

Format: <Subcategory name>: <Brief description>
"""

# Multi-subcategory taxonomy generation (upfront)
TAXONOMY_GENERATION = """\
You are generating a comprehensive taxonomy of subcategories to ensure systematic exploration.

Category: {category}

Guidance: {guidance}

--Instructions--

Generate 8-12 distinct subcategories that together would provide comprehensive coverage of this category. 
Each subcategory should:
1. Be mutually exclusive (minimal overlap with other subcategories)
2. Be broad enough to contain 5+ entities each
3. Cover different dimensions: types, regions, time periods, use cases, stakeholders, etc.

Think about different ways to slice this category:
- By TYPE or FUNCTION (what kind of entity is it?)
- By GEOGRAPHY (where is it based or operates?)
- By TIME PERIOD (when was it created or active?)
- By TARGET/AUDIENCE (who does it serve?)
- By TECHNOLOGY/METHOD (how does it work?)
- By SCALE/SIZE (how big is it?)
- By SECTOR (which industry or domain?)

Provide subcategories as a structured list with name and description.
"""

# Diversified query strategies
QUERY_STRATEGY_ATTRIBUTE = """\
Search for {category} that are specifically known for their {attribute_focus}.

{guidance}

Find examples where {attribute_focus} is a notable or distinguishing characteristic. 
Provide specific names, organizations, and verifiable details.
"""

QUERY_STRATEGY_GEOGRAPHIC = """\
Search for {category} based in or primarily operating in {region}.

{guidance}

Focus on examples from this specific geographic region. Include local organizations, 
regional initiatives, and entities with strong presence in {region}.
Provide specific names and verifiable details.
"""

QUERY_STRATEGY_TEMPORAL = """\
Search for {category} that were founded, launched, or became prominent {time_period}.

{guidance}

Focus on examples from this specific time period. Include both well-known and lesser-known 
examples that emerged during this era.
Provide specific names, dates, and verifiable details.
"""

QUERY_STRATEGY_STAKEHOLDER = """\
Search for {category} that primarily serve or involve {stakeholder_type}.

{guidance}

Focus on examples designed for or operated by this stakeholder group.
Provide specific names and verifiable details.
"""

# Provisional attribute value generation
ATTRIBUTE_VALUES_GENERATION = """\
You are generating provisional values for attributes to enable systematic exploration.

Category: {category}
Guidance: {guidance}

For each attribute below, generate a comprehensive list of expected/possible values that would apply to entities in this category.

--Attributes--

{attributes}

--Instructions--

For each attribute, provide:
1. A list of 5-30 likely values (more for categorical attributes, fewer for less structured ones)
2. Whether this is a "closed" set (finite, enumerable like continents, countries) or "open" set (unbounded like names, descriptions)

Focus on values that:
- Are commonly used/referenced for this type of entity
- Would help differentiate between entities
- Are specific enough to be useful for targeted searches
- Start with a capital letter (e.g., "Machine learning", "North America")
- Use only essential unicode characters (avoid decorative symbols, emojis, or special punctuation)

Examples:
- "Geographic Region" → closed set: [Europe, North America, Asia, Africa, Latin America, Middle East, Oceania]
- "Technology Type" → closed set: [AI/ML, Blockchain, Mobile App, Web Platform, Database, API, Hardware]
- "Year Founded" → closed set: [2020s, 2010s, 2000s, 1990s, Before 1990]
- "Organization Name" → open set (too many to enumerate)
"""

# Query for single attribute value exploration
QUERY_SINGLE_ATTRIBUTE = """\
Search for {category} where {attribute_name} is specifically "{attribute_value}".

{guidance}

{exclusion_list}

Find examples that match this specific attribute value. Provide concrete entity names with verifiable details.
Include both well-known and lesser-known examples.
"""

# Query for attribute pair combination exploration  
QUERY_ATTRIBUTE_PAIR = """\
Search for {category} that combine these characteristics:
- {attribute1_name}: {attribute1_value}
- {attribute2_name}: {attribute2_value}

{guidance}

{exclusion_list}

Find examples that specifically match BOTH of these attribute values together.
Provide concrete entity names with verifiable details.
"""

# Query for attribute triple combination exploration
QUERY_ATTRIBUTE_TRIPLE = """\
Search for {category} that combine these characteristics:
- {attribute1_name}: {attribute1_value}
- {attribute2_name}: {attribute2_value}
- {attribute3_name}: {attribute3_value}

{guidance}

{exclusion_list}

Find examples that specifically match ALL THREE of these attribute values together.
Provide concrete entity names with verifiable details.
"""

# Value format cleanup for poorly formatted values
VALUE_FORMAT_CLEANUP = """\
You are cleaning up poorly formatted attribute values for the attribute "{attribute_name}".

Expected format: {expected_format}

These observed values need to be cleaned up or standardized:
{observed_values}

For each observed value:
1. Convert to the expected format if possible
2. If the value is ambiguous or cannot be reliably converted, mark as "UNKNOWN"
3. If the value is clearly invalid for this attribute, mark as "REMOVE"

Examples for Year/Date attributes:
- "April 2014" → "2014"
- "Apr-14" → "2014"  
- "2017 (update in 2020)" → "2017"
- "Founded in 2003; acquired by Nasdaq in 2020" → "2003"
- "2012, 2013, 2017" → take the earliest: "2012"
- "Unknown" → "UNKNOWN"
- "Reported significant activities around 2017" → "2017"

Output a mapping from each observed value to its cleaned form.
"""

# Attribute resolution for semantic deduplication
ATTRIBUTE_RESOLUTION = """\
You are resolving similar attribute names to canonical forms.

Given the following list of attribute names, identify any that are semantically equivalent and should be merged.

Only output mappings where names should change. Use these naming conventions:
- Initial capital letters
- Spaces between words (no underscores or camelCase)
- Fully capitalize acronyms
- No punctuation

--Attribute names--

{attribute_names}
"""

# Value normalization for closed-set attributes
VALUE_NORMALIZATION = """\
You are normalizing attribute values to canonical forms for the attribute "{attribute_name}".

These are the CANONICAL values (use these exactly):
{canonical_values}

These are the observed values that need normalization:
{observed_values}

NORMALIZATION RULES (apply in order):

1. EXACT/CASE MATCH: If observed value matches a canonical value (ignoring case), use the canonical form.
   Example: "united states" → "United States"

2. SYNONYMS & VARIANTS: Map common synonyms, abbreviations, and regional variants to canonical values.
   Examples:
   - "UK", "UK-wide", "Great Britain", "Britain" → "United Kingdom"
   - "US", "USA", "America" → "United States"
   - "APAC" → "Asia-Pacific"
   - "EMEA" → "Europe" (or split if needed)
   - "LATAM" → "Latin America"

3. COUNTRY-TO-REGION: If a specific country is observed but the canonical list only has regions, map to the appropriate region.
   Examples:
   - "Germany", "France", "Spain" → "Europe"
   - "Thailand", "Vietnam", "Cambodia" → "Southeast Asia"
   - "China", "Japan", "Korea" → "Asia-Pacific"
   - "Brazil", "Mexico", "Colombia" → "Latin America"
   - "Nigeria", "Kenya", "South Africa" → "Africa"
   - "UAE", "Saudi Arabia", "Qatar" → "Middle East"

4. COMPOUND VALUES: If observed value contains multiple items (comma-separated, "and", etc.), map to multiple canonical values separated by "|".
   Example: "North America and Europe" → "North America|Europe"

5. QUALIFIERS: Strip qualifiers like "primarily", "mainly", "with focus on", "-wide", "-based" and normalize the core value.
   Example: "UK-wide", "UK-based" → "United Kingdom"

6. NO MATCH: If an observed value genuinely doesn't match any canonical value and is too specific/unusual, output "REMOVE" to exclude it.

Be AGGRESSIVE about mapping to canonical values. The goal is consistency, not preserving original phrasing.
Every observed value should map to a canonical value whenever semantically appropriate.

FORMATTING RULES:
- Ensure all normalized values start with a capital letter
- Remove any non-essential unicode characters (decorative symbols, emojis, special punctuation)
- Keep only unicode characters that are essential to the meaning
"""

# Open-set value clustering - standardize equivalent values
OPEN_SET_VALUE_CLUSTERING = """\
You are standardizing values for the attribute "{attribute_name}" which has open-ended values (no predefined set).

Below are clusters of semantically equivalent values that should be standardized to a single canonical form.
For each cluster, choose the BEST canonical value that:
1. Is clear and descriptive
2. Uses proper capitalization and formatting
3. Represents the common meaning across all variants
4. Is concise but complete

CLUSTERS TO STANDARDIZE:
{clusters}

For each cluster, output:
- cluster_id: The cluster number
- canonical_value: The single best standardized form to use
- reasoning: Brief explanation of why this form was chosen

STANDARDIZATION GUIDELINES:
- Prefer noun phrases over verb phrases ("Data Analytics" not "Analyzing Data")
- Use title case for multi-word values ("Case Management System" not "case management system")
- Remove redundant words ("AI Platform" not "AI Software Platform Tool")
- Merge synonyms ("Mobile Application" and "Mobile App" → "Mobile App")
- Keep specificity when meaningful ("Machine Learning" not just "AI" if that's what variants mean)
- For technology terms, prefer the more commonly used industry term
"""

# Entity deduplication check
ENTITY_DEDUPLICATION = """\
You are checking if any of the new entity labels refer to the EXACT SAME entity as existing labels.

Existing labels:
{existing_labels}

New labels to check:
{new_labels}

CRITICAL: Only mark entities as duplicates if they are TRULY the same entity with different name variations.

Examples of TRUE duplicates (should merge):
- "MICROSOFT CORP" and "MICROSOFT CORPORATION" (same company)
- "POSTGRES" and "POSTGRESQL" (same product)
- "ACME HOTLINE" and "ACME CONSUMER HOTLINE" (same hotline)

Examples of NON-duplicates (should NOT merge):
- "ACME APP" and "BETA APP" (different apps by different organizations)
- "EXAMPLE FOUNDATION" and "EXAMPLE HOTLINE" (organization vs. their product)
- "BANQU PLATFORM" and "GENERIC BLOCKCHAIN PLATFORM" (different products)
- Apps/tools with different primary functions, even if they're in the same space

When in doubt, do NOT merge. It's better to have separate entries than to incorrectly combine different entities.

Only output pairs that are DEFINITELY the same entity with different name spellings.
"""

# Schema suggestion for proactive attribute discovery
SCHEMA_SUGGESTION = """\
You are suggesting core attributes for a category of entities.

Category: {category}

Guidance: {guidance}

Suggest 3-5 CORE attributes that would be most commonly available and useful for entities in this category.

IMPORTANT: Choose attributes that:
1. Are likely to be found in search results (commonly mentioned)
2. Apply broadly across different entities in the category  
3. Provide the most distinctive/identifying information
4. Can be answered concisely (not open-ended descriptions)

Avoid attributes that:
- Require specialized sources to find
- Are only applicable to a subset of entities
- Would frequently be empty or unknown
- Are overly technical or detailed

ATOMICITY RULE — each attribute must measure exactly ONE dimension:
- GOOD: "Deployment model" (how it's delivered), "Target sector" (who uses it)
- BAD:  "Deployment model and target sector" (two dimensions fused together)
- GOOD: "Geographic coverage", "Tool type"
- BAD:  "Tool type / primary function" (ambiguous — pick one name)
- If you're tempted to use "/" or "and" in an attribute name, split it into two attributes.

The same rule applies to attribute VALUES — each value should be a single concept:
- GOOD: "Mobile app" for Deployment model, "Open-source" for License
- BAD:  "Open-source mobile app" (mixes deployment model + license)
- If a value describes two dimensions, it belongs in two separate attributes.

Mark the most essential 2-3 attributes as "high" importance, others as "medium".

Format each attribute as: <Name>: <Brief description> [importance: high/medium]
"""

# Record completion for filling missing values
RECORD_COMPLETION_QUERY = """\
Search for specific information about "{label}" to fill in the following missing attributes:

{missing_attributes}

This is an instance of {category}.

{guidance}

Provide only factual, verifiable information. If information cannot be found, do not guess.
"""

# LLM Reflection for strategic query generation
QUERY_REFLECTION = """\
You are a research strategist analyzing an ongoing entity extraction project to identify gaps and suggest new search directions.

## Current Project

**Category:** {category}

**Guidance:** {guidance}

## Entities Discovered So Far ({entity_count} total)

{entity_summaries}

## Query History ({query_count} queries executed)

{query_history}

## Attribute Value Coverage

{coverage_summary}

## Your Task

Analyze this extraction project and suggest 3-5 NEW search queries that would:

1. **Fill coverage gaps** - Find entities in underrepresented attribute combinations
2. **Explore adjacent areas** - Look for related entity types not yet discovered
3. **Deepen thin areas** - Find more examples where we have few entities
4. **Discover outliers** - Search for unusual or edge-case entities

For each suggested query, provide:
- A clear, specific search focus (what to look for)
- Why this search would add value (what gap it fills)
- Expected entity types (what you might find)

Think creatively - consider:
- Geographic regions with low coverage
- Time periods (historical, emerging, etc.)
- Stakeholder perspectives not yet explored
- Technology approaches or methods underrepresented
- Scale variations (small/grassroots vs large/institutional)
- Unconventional or innovative approaches
"""

# Schema for reflection output
REFLECTION_QUERY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "reflection_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Brief analysis of current coverage and gaps"
                },
                "suggested_queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "focus": {
                                "type": "string",
                                "description": "Specific search focus - what to look for"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Why this query would add value"
                            },
                            "expected_entities": {
                                "type": "string",
                                "description": "Types of entities likely to be found"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Priority based on expected value"
                            }
                        },
                        "required": ["focus", "rationale", "expected_entities", "priority"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["analysis", "suggested_queries"],
            "additionalProperties": False
        }
    }
}


# Cardinality classification — decide open/closed and pick canonical values
CARDINALITY_CLASSIFICATION = """\
You are analyzing the observed values for the attribute "{attribute_name}" across a dataset of {total_records} entities in the category "{category}".

Below are CLUSTERS of similar observed values (grouped by fuzzy string matching).
Each value is annotated with how many entities have it.

{clusters}

SINGLETON VALUES (appeared only once or did not cluster):
{singletons}

Your tasks:

1. CLASSIFY this attribute as "closed" or "open":
   - "closed": The attribute has a FINITE, bounded set of meaningful categories.
     Most observed values are variants/synonyms/typos of a small set of concepts.
     Examples: license type, geographic region, technology category.
   - "open": The attribute is inherently unbounded — each entity may have a
     unique value. Examples: organization name, URL, description.
   Rule of thumb: if {num_clusters} clusters cover most of the {total_records}
   entities, it is likely closed.

2. For EACH cluster, pick or compose the single best CANONICAL value:
   - Clear, descriptive, properly capitalized
   - Prefer the most frequent variant if it's already well-formed
   - Consolidate synonyms ("Mobile app" and "Mobile application" → pick one)

3. For SINGLETON values, decide:
   - Map to an existing cluster's canonical value if semantically equivalent
   - Create a new canonical value if it represents a genuinely distinct concept
   - Mark as "REMOVE" if it is noise, too vague, or invalid

Output a list of canonical values and a mapping from every observed value to its
canonical form (or "REMOVE").
"""


# Dynamic enum expansion for closed-set attributes
ENUM_EXPANSION = """\
You are expanding the value set for the attribute "{attribute_name}" in a dataset about {category}.

Current canonical values:
{current_values}

These entities were assigned "Other" because none of the existing values fit:
{other_entities}

Based on these entities, suggest NEW values to add to the canonical set. These should be:
1. Broad enough to apply to multiple entities (not entity-specific)
2. Distinct from existing values (not synonyms)
3. Consistent in style/granularity with the existing values

Also re-classify each "Other" entity to either a new value you're proposing or an existing value if one actually fits.
"""


# Schema attribute merge for near-duplicate attributes
ATTRIBUTE_MERGE = """\
You are analyzing schema attributes for potential merges. Some attributes may be near-duplicates that should be combined.

Attribute pairs to evaluate:
{attribute_pairs}

For each pair, determine:
1. Are these semantically the SAME attribute with different names?
2. If yes, which name should be the canonical one?
3. How should values be reconciled?

Only merge attributes that are truly the same concept. Different aspects of an entity (e.g., "Technology Type" vs "Technology Description") should remain separate.
"""
