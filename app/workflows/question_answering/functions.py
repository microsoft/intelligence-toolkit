import numpy as np
import streamlit as st
import os
import io
import tiktoken
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import util.Embedder
import workflows.question_answering.classes as classes
import workflows.question_answering.config as config
import tempfile
import pdfkit
from util.wkhtmltopdf import config_pdfkit, pdfkit_options

embedder = util.Embedder.create_embedder(cache=f'{config.cache_dir}/qa_mine')
encoder = tiktoken.get_encoding('cl100k_base')


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    length_function=len,
    is_separator_regex=False,
)

def chunk_files(sv, files):
    pb = st.progress(0, 'Chunking files...')
    file_chunks = []
    for fx, file_link in enumerate(files):
        pb.progress((fx+1) / len(files), f'Chunking file {fx+1} of {len(files)}...')
        file_names = [f.name for f in sv.answering_files.value.values()]
        doc_text = ''
        if file_link.name not in file_names:
            file_id = sv.answering_next_file_id.value
            sv.answering_next_file_id.value += 1
            file = classes.File(file_link.name, file_id)
            sv.answering_files.value[file_id] = file
            bytes = file_link.getvalue()
            
            txt_pdf_name = None
            if file_link.name.endswith('.txt'):
                config_pdf = config_pdfkit()
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    txt_pdf_name = temp_file.name
                    pdfkit.from_string(bytes.decode('utf-8'), txt_pdf_name, options=pdfkit_options, configuration=config_pdf)

            pdf_reader = pdfplumber.open(txt_pdf_name if txt_pdf_name else io.BytesIO(bytes))
            for px in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[px].extract_text()
                doc_text += f'\n[PAGE {px+1}]\n\n{page_text}\n\n'
                chunks = [f'\n[PAGE {px+1}]\n\n{x.strip()}\n\n' for x in text_splitter.split_text(page_text)]
                for chunk in chunks:
                    file_chunks.append((file, chunk))
                file.set_text(doc_text)

    for cx, (file, chunk) in enumerate(file_chunks):
        pb.progress((cx+1) / len(file_chunks), f'Embedding chunk {cx+1} of {len(file_chunks)}...')
        chunk_vec = embedder.encode(chunk)
        file.add_chunk(chunk, np.array(chunk_vec), cx+1)   

    pb.empty()

def update_question(sv, question_history, new_questions, placeholder, prefix):
    response = question_history[-1]
    if len(new_questions) > 0:
        q_texts = [f'{q.id}: {q.text}\n\n{q.answer_texts[0]}' for q in new_questions]
        q_text = '\n\n'.join(q_texts)    
        

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
            'original_question': question_history[0],
            'augmented_question': question_history[-1],
            'q_text': q_text
        }
        messages = util.AI_API.prepare_messages_from_message_pair(
            system_message=system_message,
            user_message=user_message,
            variables=variables,
        )
        response = util.AI_API.generate_text_from_message_list(
            messages=messages,
            placeholder=placeholder,
            prefix=prefix
        )
    else:
        print('Got no new questions!')
    return response

