
import intelligence_toolkit.AI.utils as utils
import intelligence_toolkit.query_text_data.prompts as prompts
import intelligence_toolkit.query_text_data.answer_schema as answer_schema
from intelligence_toolkit.AI.client import OpenAIClient
from json import loads, dumps

class Commentary:

    def __init__(self, ai_configuration, query, callback):
        self.ai_configuration = ai_configuration
        self.query = query
        self.callback = callback
        self.structure = {
            "points": {},
            "point_sources": {},
            "themes": {},
        }

    def update_commentary(self, chunks: dict[int, str]):
        messages = utils.prepare_messages(
            prompts.thematic_update_prompt, {"sources": "\n\n".join([f"{k}:\n\n{v}" for k, v in chunks.items()]), "query": self.query, "structure": dumps(self.structure, indent=2)}
        )
        callbacks = [self.callback] if self.callback is not None else []
        updates = OpenAIClient(self.ai_configuration).generate_chat(
            messages,
            stream=False,
            response_format=answer_schema.thematic_update_format,
            callbacks=callbacks
        )
        update_obj = loads(updates)
        for u in update_obj["updates"]:
            point_id = u["point_id"]
            point_title = u["point_title"]
            source_ids = u["source_ids"]
            if point_id not in self.structure["points"]:
                self.structure["points"][point_id] = point_title
            if point_title != "":
                self.structure["points"][point_id] = point_title
            if point_id not in self.structure["point_sources"]:
                self.structure["point_sources"][point_id] = []
            for s in source_ids:
                if s not in self.structure["point_sources"][point_id]:
                    self.structure["point_sources"][point_id].append(s)
        for t in update_obj["themes"]:
            theme_title = t["theme_title"]
            point_ids = t["point_ids"]
            self.structure["themes"][theme_title] = point_ids
        print(dumps(self.structure, indent=2))
        for callback in callbacks:
            callback.on_llm_new_token(self.format_structure())

    def format_structure(self):
        output = ""
        for theme_title, point_ids in self.structure["themes"].items():
            output += f"- **{theme_title}**\n"
            for point_id in point_ids:
                source_list = ", ".join([str(x) for x in self.structure["point_sources"][point_id]])
                output += f"  - {self.structure['points'][point_id]} (sources: {source_list})\n"
        return output