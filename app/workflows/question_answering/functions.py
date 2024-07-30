# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import io
import tempfile

import numpy as np
import pdfkit
import pdfplumber
import streamlit as st
import workflows.question_answering.classes as classes
from util import ui_components
from util.openai_wrapper import UIOpenAIConfiguration
from util.session_variables import SessionVariables
from util.wkhtmltopdf import config_pdfkit, pdfkit_options
from workflows.question_answering import config

from python.AI import utils
from python.AI.embedder import Embedder
from python.AI.text_splitter import TextSplitter

sv_home = SessionVariables("home")


def embedder():
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        return Embedder(
            ai_configuration, config.cache_dir, sv_home.local_embeddings.value
        )
    except Exception as e:
        st.error(f"Error creating connection: {e}")
        st.stop()


def chunk_files(sv, files):
    pb = st.progress(0, "Chunking files...")
    file_chunks = []
    text_splitter = TextSplitter()
    for fx, file_link in enumerate(files):
        pb.progress((fx + 1) / len(files), f"Chunking file {fx + 1} of {len(files)}...")
        file_names = [f.name for f in sv.answering_files.value.values()]
        doc_text = ""
        if file_link.name not in file_names:
            file_id = sv.answering_next_file_id.value
            sv.answering_next_file_id.value += 1
            file = classes.File(file_link.name, file_id)
            sv.answering_files.value[file_id] = file
            bytes = file_link.getvalue()

            txt_pdf_name = None
            if file_link.name.endswith(".txt"):
                config_pdf = config_pdfkit()
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    txt_pdf_name = temp_file.name
                    pdfkit.from_string(
                        bytes.decode("utf-8"),
                        txt_pdf_name,
                        options=pdfkit_options,
                        configuration=config_pdf,
                    )

            pdf_reader = pdfplumber.open(
                txt_pdf_name if txt_pdf_name else io.BytesIO(bytes)
            )
            for px in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[px].extract_text()
                doc_text += f"\n[PAGE {px + 1}]\n\n{page_text}\n\n"
                chunks = [
                    f"\n[PAGE {px + 1}]\n\n{x.strip()}\n\n"
                    for x in text_splitter.split(page_text)
                ]
                file_chunks.extend([(file, chunk) for chunk in chunks])
                file.set_text(doc_text)
    functions_embedder = embedder()
    for cx, (file, chunk) in enumerate(file_chunks):
        pb.progress(
            (cx + 1) / len(file_chunks),
            f"Embedding chunk {cx + 1} of {len(file_chunks)}...",
        )
        formatted_chunk = chunk.replace("\n", " ")
        chunk_vec = functions_embedder.embed_store_one(
            formatted_chunk, sv_home.save_cache.value
        )
        file.add_chunk(chunk, np.array(chunk_vec), cx + 1)
    pb.empty()


def update_question(question_history, new_questions, placeholder, prefix):
    response = question_history[-1]
    if len(new_questions) > 0:
        q_texts = [f"{q.id}: {q.text}\n\n{q.answer_texts[0]}" for q in new_questions]
        q_text = "\n\n".join(q_texts)

        system_message = """\
You are a helpful assistant augmenting a user question with any keywords (e.g., entities/concepts) found in a list of input questions, each of which is prefixed by the question ID.

Only keywords that clearly and directly help answer the question should be inserted as a list, enclosed by parentheses, at the appropriate point in the question, with each keywords item referencing the supporting question IDs using "(<keywords 1> [Q<ID>, Q<ID>...], <keywords 2> [Q<ID>, Q<ID>...], ...)".

Do not insert any text indicating lack of relevant keywords, and do not remove any text (including question references) already present in the previous augmented question unless it is clearly irrelevant.

Retain the structure of the original question, including any punctuation such as question marks. Do not add any new parts to the question other than the inserted keywords.
"""

        user_message = """\
Original question: {original_question}

Previous augmented question: {augmented_question}

Input questions:

{q_text}

New augmented question adding to the prevous augmented question:
"""
        variables = {
            "original_question": question_history[0],
            "augmented_question": question_history[-1],
            "q_text": q_text,
        }
        messages = utils.prepare_messages(
            system_message=system_message,
            user_message=user_message,
            variables=variables,
        )

        on_callback = ui_components.create_markdown_callback(placeholder, prefix)
        response = ui_components.generate_text(messages, [on_callback])
    else:
        print("Got no new questions!")
    return response
