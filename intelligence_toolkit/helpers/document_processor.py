
from collections import defaultdict
import pandas as pd
from json import dumps, loads
import pdfplumber
import io
from intelligence_toolkit.AI.text_splitter import TextSplitter

def convert_files_to_chunks(
    input_filepaths,
    chunk_size,
    callbacks=[],
):
    text_to_chunks = defaultdict(list)

    def add_chunks(filename, text, chunk_size):
        splitter = TextSplitter(chunk_size=chunk_size)
        text_chunks = splitter.split(text)
        for index, text in enumerate(text_chunks):
            chunk = {"title": filename, "text_chunk": text, "chunk_id": index + 1}
            text_to_chunks[filename].append(dumps(chunk, indent=2, ensure_ascii=False))

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
                add_chunks(f"{filename}_{ix+1}", rec_text, chunk_size)
        elif filename.endswith(".json"):
            json_obj = loads(open(filepath).read())
            # check if json_obj is a list
            if isinstance(json_obj, list):
                for ix, js_rec in enumerate(json_obj):
                    rec_text = dumps(js_rec)
                    add_chunks(f"{filename}_{ix+1}", rec_text, chunk_size)
            else:
                text = dumps(json_obj)
                add_chunks(filename, text, chunk_size)
        elif filename.endswith(".pdf"):
            page_texts = []
            bytes = open(filepath, "rb").read()
            pdf_reader = pdfplumber.open(io.BytesIO(bytes))
            for px in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[px].extract_text()
                page_texts.append(page_text)
            text = " ".join(page_texts)
            add_chunks(filename, text, chunk_size)
        else:
            text = open(filepath).read()
            add_chunks(filename, text, chunk_size)
        
    return text_to_chunks
