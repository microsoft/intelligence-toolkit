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
                "claims": {
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
            "required": ["question", "title", "claims", "content_id_sequence", "content_items", "conclusion"],
            "additionalProperties": False,
        }
    }
}
