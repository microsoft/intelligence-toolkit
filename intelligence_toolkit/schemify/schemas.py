"""
JSON Schema definitions for structured LLM outputs.
"""
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SchemaAttribute


def get_record_extraction_schema(
    attributes: list | None = None, 
    with_citations: bool = False
) -> dict[str, Any]:
    """
    Get the JSON schema for record extraction.
    
    Args:
        attributes: List of attribute names OR SchemaAttribute objects.
                   If SchemaAttribute objects are passed and have canonical_values,
                   those are used as enum constraints (with "Other" option).
                   If None, uses flexible additional_attributes.
        with_citations: If True, each attribute includes citation_indices array and evidence text.
    """
    properties: dict[str, Any] = {
        "label": {
            "type": "string",
            "description": "Entity name in ALL CAPITALS"
        }
    }
    required = ["label"]
    
    # Add known attributes
    if attributes:
        for attr in attributes:
            # Handle both string names and SchemaAttribute objects
            if isinstance(attr, str):
                attr_name = attr
                canonical_values = None
                description = f"Value for {attr}, or empty string if unknown"
            else:
                # It's a SchemaAttribute object
                attr_name = attr.name
                canonical_values = attr.canonical_values if attr.canonical_values else None
                description = attr.description or f"Value for {attr.name}"
            
            if with_citations:
                # Attribute value with citation indices and evidence
                value_schema: dict[str, Any]
                if canonical_values:
                    # Use enum for canonical values (add "Other" for edge cases)
                    value_schema = {
                        "type": "string",
                        "enum": canonical_values + ["Other"],
                        "description": f"{description}. Choose from the listed options, or 'Other' if none fit."
                    }
                else:
                    value_schema = {
                        "type": "string",
                        "description": f"{description}, or empty string if unknown"
                    }
                
                properties[attr_name] = {
                    "type": "object",
                    "properties": {
                        "value": value_schema,
                        "citation_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Indices of sources that support this value (0-based)"
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Direct quote or paraphrase from the source text that supports this value for THIS SPECIFIC entity. Must mention the entity name."
                        }
                    },
                    "required": ["value", "citation_indices", "evidence"],
                    "additionalProperties": False
                }
            else:
                if canonical_values:
                    properties[attr_name] = {
                        "type": "string",
                        "enum": canonical_values + ["Other"],
                        "description": f"{description}. Choose from the listed options, or 'Other' if none fit."
                    }
                else:
                    properties[attr_name] = {
                        "type": "string",
                        "description": f"{description}, or empty string if unknown"
                    }
            required.append(attr_name)
    
    # Always include additional_attributes for overflow
    if with_citations:
        properties["additional_attributes"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "citation_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices of sources that support this value (0-based)"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Direct quote or paraphrase from the source text that supports this value for THIS SPECIFIC entity"
                    }
                },
                "required": ["name", "value", "citation_indices", "evidence"],
                "additionalProperties": False
            }
        }
    else:
        properties["additional_attributes"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["name", "value"],
                "additionalProperties": False
            }
        }
    required.append("additional_attributes")
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "record_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["records"],
                "additionalProperties": False
            }
        }
    }


def get_attribute_resolution_schema() -> dict:
    """Get the JSON schema for attribute resolution."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "attribute_resolution",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original": {"type": "string"},
                                "canonical": {"type": "string"}
                            },
                            "required": ["original", "canonical"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["mappings"],
                "additionalProperties": False
            }
        }
    }


def get_value_normalization_schema() -> dict[str, Any]:
    """Get the JSON schema for value normalization."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "value_normalization",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original": {"type": "string", "description": "The original observed value"},
                                "normalized": {"type": "string", "description": "The normalized canonical value, or REMOVE if invalid"}
                            },
                            "required": ["original", "normalized"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["mappings"],
                "additionalProperties": False
            }
        }
    }


def get_open_set_clustering_schema() -> dict[str, Any]:
    """Get the JSON schema for open-set value clustering/standardization."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "open_set_clustering",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "standardizations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cluster_id": {"type": "integer", "description": "The cluster number"},
                                "canonical_value": {"type": "string", "description": "The standardized canonical value for this cluster"},
                                "reasoning": {"type": "string", "description": "Brief explanation of why this form was chosen"}
                            },
                            "required": ["cluster_id", "canonical_value", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["standardizations"],
                "additionalProperties": False
            }
        }
    }


def get_entity_deduplication_schema() -> dict:
    """Get the JSON schema for entity deduplication."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_deduplication",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "duplicates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "new_label": {"type": "string"},
                                "existing_label": {"type": "string"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["new_label", "existing_label", "confidence"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["duplicates"],
                "additionalProperties": False
            }
        }
    }


def get_schema_suggestion_schema() -> dict:
    """Get the JSON schema for schema suggestions."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "schema_suggestion",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "importance": {"type": "string", "enum": ["high", "medium", "low"]}
                            },
                            "required": ["name", "description", "importance"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["attributes"],
                "additionalProperties": False
            }
        }
    }


def get_taxonomy_generation_schema() -> dict:
    """Get the JSON schema for taxonomy generation."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "taxonomy_generation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "subcategories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Short subcategory name"},
                                "description": {"type": "string", "description": "Brief description"},
                                "dimension": {"type": "string", "enum": ["type", "geography", "temporal", "stakeholder", "technology", "scale", "sector", "other"]},
                                "search_terms": {"type": "array", "items": {"type": "string"}, "description": "2-3 specific search terms for this subcategory"}
                            },
                            "required": ["name", "description", "dimension", "search_terms"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["subcategories"],
                "additionalProperties": False
            }
        }
    }


def get_attribute_values_schema() -> dict:
    """Get the JSON schema for provisional attribute value generation."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "attribute_values",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Attribute name"},
                                "is_closed_set": {"type": "boolean", "description": "True if finite/bounded values"},
                                "values": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of provisional values"
                                },
                                "reasoning": {"type": "string", "description": "Why closed/open classification"}
                            },
                            "required": ["name", "is_closed_set", "values", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["attributes"],
                "additionalProperties": False
            }
        }
    }


def get_cardinality_classification_schema() -> dict[str, Any]:
    """Get the JSON schema for cardinality classification and canonical value selection."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cardinality_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "enum": ["closed", "open"],
                        "description": "Whether the attribute has a finite (closed) or unbounded (open) value set"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the classification decision"
                    },
                    "canonical_values": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The final set of canonical values (for closed-set) or cluster representatives (for open-set)"
                    },
                    "mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original": {"type": "string", "description": "The observed value"},
                                "canonical": {"type": "string", "description": "The canonical value to map to, or REMOVE"}
                            },
                            "required": ["original", "canonical"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["classification", "reasoning", "canonical_values", "mappings"],
                "additionalProperties": False
            }
        }
    }


def get_enum_expansion_schema() -> dict[str, Any]:
    """Get the JSON schema for dynamic enum expansion."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "enum_expansion",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "new_values": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string", "description": "New canonical value to add"},
                                "reasoning": {"type": "string", "description": "Why this value is needed"}
                            },
                            "required": ["value", "reasoning"],
                            "additionalProperties": False
                        }
                    },
                    "reclassifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity": {"type": "string", "description": "Entity label"},
                                "new_value": {"type": "string", "description": "Correct value (new or existing)"},
                                "reasoning": {"type": "string", "description": "Why this classification"}
                            },
                            "required": ["entity", "new_value", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["new_values", "reclassifications"],
                "additionalProperties": False
            }
        }
    }


def get_attribute_merge_schema() -> dict[str, Any]:
    """Get the JSON schema for attribute merge decisions."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "attribute_merge",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "merges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "attr1": {"type": "string", "description": "First attribute name"},
                                "attr2": {"type": "string", "description": "Second attribute name"},
                                "should_merge": {"type": "boolean", "description": "Whether these should merge"},
                                "canonical_name": {"type": "string", "description": "Name to keep (if merging)"},
                                "reasoning": {"type": "string", "description": "Explanation"}
                            },
                            "required": ["attr1", "attr2", "should_merge", "canonical_name", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["merges"],
                "additionalProperties": False
            }
        }
    }
