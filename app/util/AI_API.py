from openai import OpenAI
import tiktoken
import json

gen_model = 'gpt-4-turbo-preview'
embed_model = 'text-embedding-3-small'
text_encoder = 'cl100k_base'
max_gen_tokens = 4096
max_input_tokens = 128000
default_temperature = 0
max_embed_tokens = 8191

client = OpenAI()
encoder = tiktoken.get_encoding(text_encoder)

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

def generate_text_from_message_list(messages, placeholder=None, prefix='', model=gen_model, temperature=default_temperature, max_tokens=max_gen_tokens):     
    response = ''
    try:
        responses = client.chat.completions.create(
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
    return response