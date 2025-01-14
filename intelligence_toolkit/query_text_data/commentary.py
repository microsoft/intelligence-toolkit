
import intelligence_toolkit.AI.utils as utils
import intelligence_toolkit.query_text_data.prompts as prompts
import intelligence_toolkit.query_text_data.answer_schema as answer_schema
from intelligence_toolkit.AI.client import OpenAIClient
from json import loads, dumps

class Commentary:

    def __init__(self, ai_configuration, query, cid_to_text, update_interval, analysis_callback, commentary_callback):
        self.ai_configuration = ai_configuration
        self.query = query
        self.analysis_callback = analysis_callback
        self.commentary_callback = commentary_callback
        self.cid_to_text = cid_to_text
        self.update_interval = update_interval
        self.unprocessed_chunks = {}
        self.structure = {
            "points": {},
            "point_sources": {},
            "themes": {},
        }

    def add_chunks(self, chunks: dict[int, str]):
        self.unprocessed_chunks.update(chunks)
        if self.update_interval > 0 and len(self.unprocessed_chunks) >= self.update_interval:
            self.update_analysis(self.unprocessed_chunks)
            self.unprocessed_chunks = {}

    def complete_analysis(self):
        if self.update_interval > 0 and len(self.unprocessed_chunks) > 0:
            self.update_analysis(self.unprocessed_chunks)
            self.unprocessed_chunks = {}

    def update_analysis(self, chunks: dict[int, str]):
        messages = utils.prepare_messages(
            prompts.thematic_update_prompt, {"sources": "\n\n".join([f"{k}:\n\n{v}" for k, v in chunks.items()]), "query": self.query, "structure": dumps(self.structure, indent=2)}
        )
        callbacks = [self.analysis_callback] if self.analysis_callback is not None else []
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
                if point_id in self.structure["point_sources"]:
                    source_list = ", ".join([str(x) for x in self.structure["point_sources"][point_id]])
                    output += f"  - {self.structure['points'][point_id]} (sources: {source_list})\n"
        return output
    
    def get_clustered_cids(self):
        if self.update_interval > 0:
            clustered_cids = {}
            current_cluster = []
            for theme_title, point_ids in self.structure["themes"].items():
                current_cluster = []
                for point_id in point_ids:
                    source_ids = self.structure["point_sources"][point_id]
                    for source_id in source_ids:
                        if source_id not in current_cluster:
                            current_cluster.append(source_id)
                clustered_cids[theme_title] = current_cluster
            return clustered_cids
        else:
            return {"All relevant chunks": list(self.unprocessed_chunks.keys())}
    
    async def generate_commentary(self):
        structure = self.format_structure()
        selected_cids = set()
        for theme, cid_list in self.get_clustered_cids().items():
            selected_cids.update(cid_list[:3])
        indexed_chunks = "\n\n".join([f"{cid}:\n\n{self.cid_to_text[cid]}" for cid in selected_cids])
        messages = utils.prepare_messages(
            prompts.commentary_prompt, {"query": self.query, "structure": structure, "chunks": indexed_chunks}
        )
        callbacks = [self.commentary_callback] if self.commentary_callback is not None else []
        commentary = await OpenAIClient(self.ai_configuration).generate_chat_async(
            messages,
            stream=True,
            callbacks=callbacks
        )
        return commentary