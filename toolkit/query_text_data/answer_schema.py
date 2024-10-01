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
                "content_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "number"
                            },
                            "claim_summary": {
                                "type": "string"
                            },
                            "claim_text": {
                                "type": "string"
                            },
                            "claim_type": {
                                "type": "string",
                                "enum": ["fact", "value", "policy"]
                            },
                            "grounds": {
                                "type": "string"
                            },
                            "attribution": {
                                "type": "string"
                            },
                            "source": {
                                "type": "string"
                            }
                        },
                        "required": ["id", "claim_text", "claim_type", "grounds", "attribution", "source"],
                        "additionalProperties": False,
                    }
                },
            },
            "required": ["content_items"],
            "additionalProperties": False,
        }
    }
}
