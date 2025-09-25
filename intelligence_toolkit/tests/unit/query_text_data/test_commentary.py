import json

import pytest

import intelligence_toolkit.query_text_data.commentary as commentary_module
from intelligence_toolkit.query_text_data.commentary import Commentary


def test_update_analysis_replaces_existing_themes(monkeypatch):
    outputs = [
        json.dumps(
            {
                "updates": [
                    {
                        "point_id": "point-1",
                        "point_title": "Initial insight",
                        "source_ids": [1],
                    }
                ],
                "themes": [
                    {
                        "theme_title": "Theme One",
                        "point_ids": ["point-1"],
                    }
                ],
            }
        ),
        json.dumps(
            {
                "updates": [
                    {
                        "point_id": "point-2",
                        "point_title": "Reframed insight",
                        "source_ids": [2],
                    }
                ],
                "themes": [
                    {
                        "theme_title": "Theme Two",
                        "point_ids": ["point-2"],
                    }
                ],
            }
        ),
    ]

    def fake_prepare_messages(prompt, variables, *args, **kwargs):
        return {"prompt": prompt, "variables": variables}

    def fake_generate_chat(self, messages, stream=False, response_format=None, callbacks=None):
        return outputs.pop(0)

    monkeypatch.setattr(commentary_module.utils, "prepare_messages", fake_prepare_messages)
    monkeypatch.setattr(commentary_module.OpenAIClient, "generate_chat", fake_generate_chat)

    cid_to_text = {
        1: json.dumps({"title": "Doc 1", "chunk_id": 1, "text_chunk": "Chunk 1"}),
        2: json.dumps({"title": "Doc 2", "chunk_id": 2, "text_chunk": "Chunk 2"}),
    }

    tracker = Commentary(
        ai_configuration=None,
        query="sample query",
        cid_to_text=cid_to_text,
        update_interval=1,
        analysis_callback=None,
        commentary_callback=None,
    )

    tracker.update_analysis({1: "Chunk 1"})
    assert tracker.structure["themes"] == {"Theme One": ["point-1"]}

    tracker.update_analysis({2: "Chunk 2"})
    assert tracker.structure["themes"] == {"Theme Two": ["point-2"]}

    assert outputs == []
    assert len(tracker.update_history) == 2
    first_update = tracker.update_history[0]
    second_update = tracker.update_history[1]

    assert first_update["chunks"] == {1: cid_to_text[1]}
    assert second_update["chunks"] == {2: cid_to_text[2]}
    assert first_update["response"]["themes"][0]["theme_title"] == "Theme One"
    assert tracker.last_update is second_update
