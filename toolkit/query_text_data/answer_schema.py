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


intermediate_answer_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "intermediate_answer_object",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "chunk_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text_title": {
                                "type": "string"
                            },
                            "chunk_id": {
                                "type": "number"
                            },
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
                                    },
                                    "required": ["claim_statement", "claim_attribution"],
                                    "additionalProperties": False,
                                }
                            }
                        },
                        "required": ["text_title", "chunk_id", "claim_context", "claims"],
                        "additionalProperties": False,
                    }
                },
            },
            "required": ["chunk_analysis"],
            "additionalProperties": False,
        }
    }
}