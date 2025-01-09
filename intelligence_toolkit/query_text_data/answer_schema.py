theme_integration_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "report_title": {
                    "type": "string"
                },
                "report_overview": {
                    "type": "string"
                },
                "report_implications": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                },
            },
            "required": ["report_title", "report_overview", "report_implications", "answer"],
            "additionalProperties": False,
        }
    }
}


theme_summarization_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "theme_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "theme_title": {
                    "type": "string"
                },
                "theme_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "point_title": {
                                "type": "string"
                            },
                            "point_evidence": {
                                "type": "string"
                            },
                            "point_commentary": {
                                "type": "string"
                            }
                        },
                        "required": ["point_title", "point_evidence", "point_commentary"],
                        "additionalProperties": False,
                    }                   
                }

            },
            "required": ["theme_title", "theme_points"],
            "additionalProperties": False,
        }
    }
}

thematic_update_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "thematic_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "point_id": {
                                "type": "number"
                            },
                            "point_title": {
                                "type": "string"
                            },
                            "source_ids": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            }
                        },
                        "required": ["point_id", "point_title", "source_ids"],
                        "additionalProperties": False,
                    }
                },
                "themes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "theme_title": {
                                "type": "string"
                            },
                            "point_ids": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            }
                        },
                        "required": ["theme_title", "point_ids"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["updates", "themes"],
            "additionalProperties": False,
        }
    }
}
