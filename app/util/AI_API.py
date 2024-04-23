# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import tiktoken
import json
from util.session_variables import SessionVariables
from util.openai_instance import _OpenAI
from util.validation_prompt import GROUNDEDNESS_PROMPT

openai = _OpenAI()

text_encoder = 'cl100k_base'
max_gen_tokens = 4096
max_input_tokens = 128000
default_temperature = 0
max_embed_tokens = 8191

encoder = tiktoken.get_encoding(text_encoder)
session_var = SessionVariables('home')

def prepare_messages_from_message(system_message, variables):                
    messages = [
        {
            'role': 'system',
            'content': system_message.format(**variables)
        },
    ]
    return messages

def prepare_messages_from_message_pair(system_message, user_message, variables):                
    messages = [
        {
            'role': 'system',
            'content': system_message.format(**variables)
        },
        {
            'role': 'user',
            'content': user_message.format(**variables)
        }
    ]
    return messages

def count_tokens_in_message_list(messages):
    return len(encoder.encode(json.dumps(messages)))


def generate_embedding_from_text(text, model = None):
    if not model:
        model = session_var.embedding_model.value
    return openai.client().embeddings.create(input = text, model=model)

def generate_text_from_message_list(messages, placeholder=None, prefix='', model=None, temperature=default_temperature, max_tokens=max_gen_tokens):     
    if not model:
        model = session_var.generation_model.value
    
    response = ''
    try:
        responses = openai.client().chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            stream=True
        )
        response = ''             
        for chunk in responses:
            diff = chunk.choices[0].delta.content # type: ignore
            if diff is not None:
                response += diff
                if placeholder is not None:
                    show = prefix + response
                    if len(diff) > 0:
                        show += '▌'
                    placeholder.markdown(show, unsafe_allow_html=True)
        if placeholder is not None:
            placeholder.markdown(prefix + response, unsafe_allow_html=True)
    except Exception as e:
        print(f'Error generating from message list: {e}')
        raise Exception(f'Problem in OpenAI response. {e}')
    return response

def validate_report(messages, ai_response):
    model = session_var.generation_model.value

    message = [{
        'role': 'system',
        'content': GROUNDEDNESS_PROMPT.format(instructions=messages, report=ai_response)
    }]
    
    try:
        responses = openai.client().chat.completions.create(
            model=model,
            temperature=default_temperature,
            max_tokens=max_gen_tokens,
            messages=message,

        )
        return responses.choices[0].message.content, message
    except Exception as e:
        print(f'Error validating report: {e}')
        raise Exception(f'Problem in OpenAI response. {e}')
        
