
from collections import defaultdict
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
import pandas as pd
from json import dumps, loads

def convert_files_to_chunks(
    input_filepaths,
    combine_text_under_n_chars,
    new_after_n_chars,
    max_characters,
    callbacks=[],
):
    text_to_chunks = defaultdict(list)
    def add_chunks(filename, text_chunks):
        for index, text in enumerate(text_chunks):
            chunk = {"title": filename, "text_chunk": text, "chunk_id": index + 1}
            text_to_chunks[filename].append(dumps(chunk, indent=2, ensure_ascii=False))
    
    def process_parts(rec_parts):
        rec_chunks = chunk_by_title(
            elements=rec_parts,
            combine_text_under_n_chars=combine_text_under_n_chars,
            new_after_n_chars=new_after_n_chars,
            max_characters=max_characters
        )
        rec_texts = [x.text for x in rec_chunks]
        add_chunks(f"{filename}_{ix+1}", rec_texts)

    for fx, filepath in enumerate(input_filepaths):
        filename = filepath.split("/")[-1]
        filename = filename.replace("(", "").replace(")", "").replace(" ", "_")
        for cb in callbacks:
            cb.on_batch_change(fx + 1, len(input_filepaths))

        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            cols = df.columns.values
            for ix, row in df.iterrows():
                rec_text = "; ".join([f"{col}: {str(row[col])}" for col in cols])
                rec_parts = partition_text(
                    text=rec_text
                )
                process_parts(rec_parts)
        elif filename.endswith(".json"):
            json_obj = loads(open(filepath).read())
            # check if json_obj is a list
            if isinstance(json_obj, list):
                for ix, js_rec in enumerate(json_obj):
                    rec_parts = partition_text(
                        text=dumps(js_rec)
                    )
                    process_parts(rec_parts)
            else:
                rec_parts = partition_text(
                    text=dumps(json_obj)
                )
                process_parts(rec_parts)
        else:
            rec_parts = partition(
                filepath,
                combine_text_under_n_chars=combine_text_under_n_chars,
                new_after_n_chars=new_after_n_chars,
                max_characters=max_characters
            )
            process_parts(rec_parts)
        
    return text_to_chunks
