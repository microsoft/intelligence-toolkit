answer_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_object",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string"
                },
                "title": {
                    "type": "string"
                },
                "introduction": {
                    "type": "string"
                },
                "content_id_sequence": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    }
                },
                "content_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "number"
                            },
                            "title": {
                                "type": "string"
                            },
                            "content": {
                                "type": "string"
                            }
                        },
                        "required": ["id", "title", "content"],
                        "additionalProperties": False,
                    }
                },
                "conclusion": {
                    "type": "string"
                },
            },
            "required": ["question", "title", "introduction", "content_id_sequence", "content_items", "conclusion"],
            "additionalProperties": False,
        }
    }
}


claim_extraction_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_claims",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "claim_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_context": {
                                "type": "string"
                            },
                            "claims": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "claim_statement": {
                                            "type": "string"
                                        },
                                        "claim_attribution": {
                                            "type": "string"
                                        },
                                        "supporting_sources": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "text_title": {
                                                        "type": "string"
                                                    },
                                                    "chunk_ids": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "number"
                                                        }
                                                    }
                                                },
                                                "required": ["text_title", "chunk_ids"],
                                                "additionalProperties": False,
                                            }
                                        },
                                        "contradicting_sources": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "text_title": {
                                                        "type": "string"
                                                    },
                                                    "chunk_ids": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "number"
                                                        }
                                                    }
                                                },
                                                "required": ["text_title", "chunk_ids"],
                                                "additionalProperties": False,
                                            }
                                        }
                                    },
                                    "required": ["claim_statement", "claim_attribution", "supporting_sources", "contradicting_sources"],
                                    "additionalProperties": False,
                                }                   
                            },
                        },
                        "required": ["claim_context", "claims"],
                        "additionalProperties": False,
                    }
                },
            },
            "required": ["claim_analysis"],
            "additionalProperties": False,
        }
    }
}

claim_summarization_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "content_items",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "content_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content_title": {
                                "type": "string"
                            },
                            "content_summary": {
                                "type": "string"
                            },
                            "content_commentary": {
                                "type": "string"
                            }
                        },
                        "required": ["content_title", "content_summary", "content_commentary"],
                        "additionalProperties": False,
                    }                   
                },
            },
            "required": ["content_items"],
            "additionalProperties": False,
        }
    }
}

content_integration_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "integrated_content",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                },
                "report_title": {
                    "type": "string"
                },
                "report_summary": {
                    "type": "string"
                },
                "theme_order": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "theme_title": {
                                "type": "string"
                            },
                            "theme_summary": {
                                "type": "string"
                            },
                            "content_id_order": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            },
                            "merge_content_ids": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "number"
                                    }
                                }
                            },
                            "theme_commentary": {
                                "type": "string"
                            }
                        },
                        "required": ["theme_title", "theme_summary", "content_id_order", "merge_content_ids", "theme_commentary"],
                        "additionalProperties": False,
                    }
                },
                "report_commentary": {
                    "type": "string"
                }
            },
            "required": ["question", "answer", "report_title", "report_summary", "theme_order", "report_commentary"],
            "additionalProperties": False,
        }
    }
}


claim_requery_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "requeried_claims",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {

                "supporting_source_indicies": {
                    "type": "array",
                    "items": {
                        "type": "number",
                    }
                },
                "contradicting_source_indicies": {
                    "type": "array",
                    "items": {
                        "type": "number",
                    }
                }
            },
            "required": ["supporting_source_indicies", "contradicting_source_indicies"],
            "additionalProperties": False,
        }
    }
}