import numpy as np
import streamlit as st
from collections import Counter
import re
import json
import scipy.spatial.distance

from util.download_pdf import add_download_pdf
import workflows.question_answering.functions as functions
import workflows.question_answering.classes as classes
import workflows.question_answering.config as config
import workflows.question_answering.prompts as prompts
import workflows.question_answering.variables as vars
import util.Embedder
import util.ui_components

embedder = util.Embedder.create_embedder(config.cache_dir)

def create():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Question Answering')
    sv = vars.SessionVariables('question_answering')

    intro_tab, uploader_tab, mining_tab, report_tab = st.tabs(['Question answering workflow:', 'Upload data', 'Mine & match questions', 'Generate AI answer reports'])
    
    df = None
    with intro_tab:
        st.markdown(config.intro)
    with uploader_tab:
        st.markdown('##### Upload data for processing')
        files = st.file_uploader("Upload PDF text files", type=['pdf', 'txt'], accept_multiple_files=True)
        if files != None:
            if st.button('Chunk and embed files'):
                functions.chunk_files(sv, files)

        num_files = len(sv.answering_files.value)
        num_chunks = sum([len(f.chunk_texts) for f in sv.answering_files.value.values()])
        if num_files > 0:
            st.markdown(f'Chunked **{num_files}** files into **{num_chunks}** chunks of up to **{config.chunk_size}** characters.')
    with mining_tab:
        c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
        with c1:
            question = st.text_input('Question', key='lazy_question')
        with c2:
            st.number_input('Target QA matches', min_value=1, step=1, value=sv.answering_target_matches.value, key=sv.answering_target_matches.key)
        with c3:
            st.number_input('Target source diversity', min_value=1, step=1, value=sv.answering_source_diversity.value, key=sv.answering_source_diversity.key)
        with c4:
            regenerate = st.button('Mine matching questions', key='lazy_regenerate', use_container_width=True)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown('#### Question mining')
            lazy_answering_placeholder = st.empty()
        with c2:
            st.markdown('#### Question matching')
            lazy_matches_placeholder = st.empty()
        lazy_answering_placeholder.markdown(sv.answering_status_history.value, unsafe_allow_html=True)
        lazy_matches_placeholder.markdown(sv.answering_matches.value, unsafe_allow_html=True)
        lazy_answer_placeholder = st.empty()  
        
        if question != '' and regenerate:
            sv.answering_question_history.value = []
            sv.answering_next_q_id.value = 1
            sv.answering_surface_questions.value = {}
            sv.answering_deeper_questions.value = {}
            sv.answering_last_lazy_question.value = question
            sv.answering_status_history.value = ''
            sv.answering_matches.value = f''
            sv.answering_question_history.value.append(question)
            lazy_answering_placeholder.markdown(sv.answering_status_history.value, unsafe_allow_html=True)
            lazy_matches_placeholder.markdown(sv.answering_matches.value, unsafe_allow_html=True)
            all_units = []
            for file in sv.answering_files.value.values():
                all_units += [('chunk', (file, cx), v) for cx, v in file.chunk_vectors.items() if v is not None]
            for qid, q in sv.answering_surface_questions.value.items():
                if q.vector is not None:
                    all_units.append(('question', q, q.vector))
                    at = q.answer_texts[0]
                    av = q.answer_vectors[0]
                    if av is not None:
                        all_units.append(('answer', q, av))
            matched_qs = []
            status_history = ''
            iteration = 0
            source_counts = Counter()
            used_chunks = set()
            while True:
                qe = embedder.encode(question)
                iteration += 1
                cosine_distances = sorted([(t, c, scipy.spatial.distance.cosine(qe, v)) for (t, c, v) in all_units], key=lambda x:x[2], reverse=False)
                chunk_index = sv.answering_target_matches.value
                for ix, (t, c, d) in enumerate(cosine_distances[:sv.answering_target_matches.value]):
                    if t == 'chunk':
                        chunk_index = ix
                        break
                        
                matched_qs = [c for t, c, d in cosine_distances[:chunk_index]]
                if len(matched_qs) == sv.answering_target_matches.value:
                    status_history += f'Iteration **{iteration}**...<br/><br/>Matched user question to **{len(matched_qs)}** mined questions:<br/>- ' + '<br/>- '.join([f'Q{matched_q.id}: {matched_q.text}' for matched_q in matched_qs]) + '<br/><br/>'
                    sv.answering_status_history.value = status_history
                    report_input = f'**User question**: {sv.answering_question_history.value[0]}\n\n**Augmented question**: {sv.answering_question_history.value[-1]}\n\n'
                    seen_qs = set()
                    for t, c, d in cosine_distances:
                        if t in ['question', 'answer']:
                            if c.id not in seen_qs:
                                delta = c.generate_outline(level=6)
                                qs = c.list_all_sub_questions()
                                candidate = report_input + delta
                                candidate_tokens = len(util.AI_API.encoder.encode(candidate))
                                if candidate_tokens <= sv.answering_outline_limit.value:
                                    report_input = candidate
                                    seen_qs.update([q.id for q in qs])
                                else:
                                    break
                    sv.answering_matches.value = report_input
                    # st.session_state['generate_lazy_answer'] = True
                    st.rerun()
                    break
                status_history += f'**Iteration {iteration}**...<br/><br/>Matched user question to **{len(matched_qs)}** of **{sv.answering_target_matches.value}** mined questions before reaching an unmined chunk'
                if len(matched_qs) > 0:
                    status_history += ':<br/>- ' + '<br/>- '.join([f'Q{matched_q.id}: {matched_q.text}' for matched_q in matched_qs])
                status_history += '<br/><br/>'

                new_questions = []
                target_diversity = min(sv.answering_source_diversity.value, len(sv.answering_files.value))
                for j in range(len(cosine_distances)):
                    t, c, d = cosine_distances[j]
                    
                    if t == 'chunk' and c not in used_chunks:
                        f = c[0]
                        cx = c[1]
                        source = c[0].id
                        max_source_counts = source_counts.most_common(target_diversity)
                        max_count = max_source_counts[0][1] if len(max_source_counts) > 0 else 0
                        num_max = len([x for x in max_source_counts if x[1] == max_count])
                        source_count = source_counts[source]
                        # print(f'Got a chunk with source count {source_count}. Max count is {max_count} and num max is {num_max}.')
                        if source_count > 0 and source_count == max_count and num_max < target_diversity:
                            pass
                            # print('Cannot use')
                        else:
                            # print('Can use')
                            used_chunks.add(c)
                            cosine_distances.pop(j)
                            source_counts.update([source])
                            break

                status_history +=f'Mining the next most similar chunk for question-answer pairs: chunk **{cx}** from file **{f.name}**...<br/>'
                variables = {
                    'text': f.chunk_texts[cx],
                    'file_id': f.id
                }
                messages = util.AI_API.prepare_messages_from_message_pair(
                    system_message=prompts.extraction_system_prompt,
                    user_message=prompts.extraction_user_prompt,
                    variables=variables
                )
                qas_raw = util.AI_API.generate_text_from_message_list(messages, placeholder=lazy_answering_placeholder, prefix=status_history)
                status_history += qas_raw + '<br/><br/>'

                try:
                    qas = json.loads(qas_raw)
                    for qa in qas:
                        q = qa['question']
                        a = qa['answer']
                        raw_refs = qa['source']
                        file_page_refs = [tuple([int(x[1:]) for x in r.split(';')]) for r in raw_refs]
                        
                        q_vec = embedder.encode(q)
                        a_vec = embedder.encode(a)

                        q_vec = np.array(q_vec)
                        a_vec = np.array(a_vec)
                        qid = sv.answering_next_q_id.value
                        sv.answering_next_q_id.value += 1
                        q = classes.Question(f, q, q_vec, 0, qid)
                        new_questions.append(q)
                        print(f'Created question {qid} from file {f.id}.')
                        f.add_question(q)
                        sv.answering_surface_questions.value[qid] = q
                        f.add_answer(qid, a, file_page_refs, a_vec)
                        all_units.append(('question', q, q.vector))
                    for (t, c, v) in list(all_units):
                        if t == 'chunk' and c[0].id == f.id and c[1] == cx:
                            all_units.remove((t, c, v))

                    status_history += f'Augmenting user question with partial answers:<br/>'
                    new_question = functions.update_question(sv, sv.answering_question_history.value, new_questions, lazy_answering_placeholder, status_history)
                    status_history += new_question + '<br/><br/>'
                    sv.answering_question_history.value.append(new_question)
                except Exception as e:
                    print('Error processing JSON')
                    print(e)
                    pass

    with report_tab:        
        # if 'generate_lazy_answer' in st.session_state:
        #     del st.session_state['generate_lazy_answer']
        if sv.answering_matches.value == '':
            st.markdown('Mine question matches to continue.')    
        else:
            c1, c2 = st.columns([2, 3])

            with c1:
                variables = {
                    'question': sv.answering_question_history.value[-1],
                    'outline': sv.answering_matches.value,
                    'source_diversity': sv.answering_source_diversity.value
                }
                generate, messages = util.ui_components.generative_ai_component(sv.answering_system_prompt, sv.answering_instructions, variables)
            with c2:
                report_placeholder = st.empty()
            
                if generate:
                    result = util.AI_API.generate_text_from_message_list(
                        placeholder=report_placeholder,
                        messages=messages,
                        prefix=''
                    )
                    sv.answering_lazy_answer_text.value = result
                report_placeholder.markdown(sv.answering_lazy_answer_text.value)
                report_data = sv.answering_lazy_answer_text.value
                is_download_disabled = report_data == ''
                name = sv.answering_lazy_answer_text.value.split('\n')[0].replace('#','').strip().replace(' ', '_')
                full_text = sv.answering_lazy_answer_text.value + '\n\n## Supporting FAQ\n\n' + re.sub(r' Q[\d]+: ', ' ', '\n\n'.join(sv.answering_matches.value.split('\n\n')[2:]), re.MULTILINE).replace('###### ', '### ')
                
                add_download_pdf(f'{name}.pdf', full_text, 'Download PDF report', disabled=is_download_disabled)
                st.download_button('Download markdown report', data=full_text, file_name=f'{name}.md', mime='text/markdown', disabled=sv.answering_lazy_answer_text.value == '', key='lazy_download_button')      
