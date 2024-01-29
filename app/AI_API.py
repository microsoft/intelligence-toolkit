import openai
import tiktoken

encoder = tiktoken.get_encoding('cl100k_base')

def generate(model, temperature, max_tokens, placeholder, system_message, user_message, variables, prefix):                
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
    return generate_from_messages(model, temperature, max_tokens, placeholder, messages, prefix)

def generate_from_messages(model, temperature, max_tokens, placeholder, messages, prefix):     
    # save messages
    with open('messages.txt', 'wb') as f:
        f.write(str(messages).encode('utf-8'))
    responses = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        stream=True
    )
    response = ''             
    for chunk in responses:
        response += chunk.choices[0].delta.get('content', '') # type: ignore
        show = prefix + response + '▌'
        show = show.replace('$', '\$')#.replace(':', '\:')
        placeholder.markdown(show, unsafe_allow_html=True)
    placeholder.markdown(show, unsafe_allow_html=True)
    return response