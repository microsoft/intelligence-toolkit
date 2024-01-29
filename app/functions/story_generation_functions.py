import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
import prompts.story_generation_prompts
import AI_API
import jsonschema
import traceback

def get_story_graph(graph_json, width, height):
    if 'characters' not in graph_json:
        return
    node_names = [n['name'] for n in graph_json['characters']]
    nodes = []
    edges = []
    for node_name in node_names:
        nodes.append(
            Node(
                title=node_name,
                id=node_name,
                label=node_name,
                size=10,
                color='blue'
            )
        )
    for relationship in graph_json['relationships']:
        edges.append(Edge(source=relationship['first_character'], target=relationship['second_character'], color="mediumgray", size=1))
    config = Config(
        width=width,
        height=height,
        directed=False,
        physics=True,
        hierarchical=False
    )
    return_value = agraph(nodes=nodes, edges=edges, config=config) # type: ignore
    return return_value

def generate_progressions(acts, states, target_scene_count):
    transition_tuples = []
    for i in range(len(states) - 1):
        transition_tuples.append((states[i], states[i + 1]))
    progressions = [(transition[0], transition[1], 'Major change') for transition in transition_tuples]
    # repeat the last transition until we have enough
    while len(progressions) < acts:
        progressions.append(progressions[-1])
    while len(progressions) < target_scene_count:
        new_progressions = []
        for progression in progressions:
            minor_change = (progression[0], progression[1], 'Minor change')
            new_progressions.append(minor_change)
            new_progressions.append(progression)
        progressions = new_progressions

    changes = [p[2] for p in progressions]
    major_indices = [-1] + [i for i, x in enumerate(changes) if x == 'Major change']

    major_spans = zip(major_indices[:-1], major_indices[1:])
    for last_major, next_major in major_spans:
        span = next_major - last_major
        if span >= 10:
            # two moderate changes in between
            moderate_index = last_major + (span // 3)
            moderate_index2 = last_major + 2 * (span // 3)
            progressions[moderate_index] = (progressions[moderate_index][0], progressions[moderate_index][1], 'Moderate change')
            progressions[moderate_index2] = (progressions[moderate_index2][0], progressions[moderate_index2][1], 'Moderate change')
        elif span >= 5:
            # one moderate change in between
            moderate_index = last_major + (span // 2)
            progressions[moderate_index] = (progressions[moderate_index][0], progressions[moderate_index][1], 'Moderate change')

    progressions[0] = (progressions[0][0], progressions[0][0], 'No change')

    progression_count = len(progressions)
    for ix, progression in enumerate(progressions):
        progressions[ix] = (progression[0], progression[1], progression[2], (ix+1)/progression_count)

    # if len(progressions) <= 5:
    #     # replace major changes with moderate changes
    #     for ix, progression in enumerate(progressions):
    #         if progression[2] == 'Major change':
    #             progressions[ix] = (progression[0], progression[1], 'Moderate change', progression[3])

    return progressions






# The story_schema is a JSON Schema in which values are described as follows:
# "value": {
#   "description": "Description of the value",
#   "type": "<type of the value>"
# }
# The type can be "string", "number", "boolean", "array", or "object".
# The generated template contains empty/null values for primitives, empty arrays for arrays of primitives, and empty objects as the sole elements of arrays of objects.
def generate_json_template_from_story_schema(story_schema):
    def generate_template(schema):
        if 'type' not in schema:
            return None
        if schema['type'] == 'object':
            if 'properties' not in schema:
                return None
            return {k: generate_template(v) for k, v in schema['properties'].items()}
        elif schema['type'] == 'array':
            if schema['items']['type'] == 'string':
                return []
            elif schema['items']['type'] == 'number':
                return []
            elif schema['items']['type'] == 'boolean':
                return []
            else:
                return [generate_template(schema['items'])]
        elif schema['type'] == 'string':
            return ''
        elif schema['type'] == 'number':
            return 0
        elif schema['type'] == 'boolean':
            return False
        else:
            return None

    ret = generate_template(story_schema)
    return ret

def add_json(template, key, value_type):
    def add():
        print('ADD JSON')
        template[key].append(generate_json_template_from_story_schema({'type': value_type}))
    return add

def edit_json(template, key, key_with_prefix):
    def edit():
        template[key] = st.session_state[key_with_prefix]
    return edit

def edit_json_list_item(template, key, i, key_with_prefix):
    def edit():
        template[key][i] = st.session_state[key_with_prefix]
    return edit

# The story_schema is a JSON Schema in which values are described as follows:
# "value": {
#   "description": "Description of the value",
#   "type": "<type of the value>"
# }
# The type can be "string", "number", "boolean", "array", or "object".
# If the type of the value is an object, then a properties key will contain the dictionary of keys and values of the object.
# The json_template contains null values for primitives, empty arrays for arrays of primitives, and empty objects as the sole elements of arrays of objects.
# The output is series of streamlit controls in which input controls are generated for each key in the story_schema according to the type of the key-value pair and labelled with the description from the story_schema.
# Both the json_template and the story_schema may have nested objects and arrays. The hierarchical index of a json value (e.g., 1.3.2) is used to identify the corresponding streamlit control in both its internal key and its label.
def generate_form_from_json_template(json_template, title, story_schema, prefix='', tier=3): # add print statements throughout
    if title != None:
        st.markdown('###' + ' ' + title)
    if type(json_template) != dict:
        return
    # print(f'Got json_template: {json_template}')
    # print(f'Got story_schema: {story_schema}')

    for key, value in json_template.items():
        key_with_prefix = f'{prefix}.{key}' if prefix else key
        schema_info = story_schema[key]
        description = schema_info['description']
        value_type = schema_info['type']
        if value_type == 'object':
            generate_form_from_json_template(value, None, schema_info['properties'], key_with_prefix, tier=tier+1)
        elif value_type == 'array':
            array_len = len(value)
            
            upper_key = key.title().replace('_', ' ')
            singular_key = upper_key if upper_key[-1] != 's' else upper_key[:-1]
            
            # st.markdown(tier * '#' + ' ' + upper_key)
            for i in range(array_len):
                label = f'{key_with_prefix} {i}'.replace('.', ' ').replace('_', ' ').replace('s ', ' ').title()
                st.markdown((tier+1) * '#' + ' ' + label)
                if 'properties' in schema_info['items']:
                    generate_form_from_json_template(value[i], None, schema_info['items']['properties'], f'{key_with_prefix}.{i}', tier=tier+1)
                else:
                    field_label = f'**{key.title().replace("_", " ")}**: {description}'
                    if schema_info['items']['type'] == 'string':
                        st.text_input(field_label, key=f'{key_with_prefix}.{i}', value=value[i], on_change=edit_json_list_item(json_template, key, i, f'{key_with_prefix}.{i}'))
                    elif schema_info['items']['type'] == 'number':
                        st.number_input(field_label, key=f'{key_with_prefix}.{i}', value=value[i], on_change=edit_json_list_item(json_template, key, i, f'{key_with_prefix}.{i}'))
                    elif schema_info['items']['type'] == 'boolean':
                        st.checkbox(field_label, key=f'{key_with_prefix}.{i}', value=value[i], on_change=edit_json_list_item(json_template, key, i, f'{key_with_prefix}.{i}'))
            st.button(f'Add {singular_key}', key=f'add_{key_with_prefix}', on_click=add_json(json_template, key_with_prefix, schema_info['items']['type']))
        else:
            field_label = f'**{key.title().replace("_", " ")}**: {description}'
            if value_type == 'string':
                st.text_input(field_label, key=key_with_prefix, value=value, on_change=edit_json(json_template, key, key_with_prefix))
            elif value_type == 'number':
                st.number_input(field_label, key=key_with_prefix, value=value, on_change=edit_json(json_template, key, key_with_prefix))
            elif value_type == 'boolean':
                st.checkbox(field_label, key=key_with_prefix, value=value, on_change=edit_json(json_template, key, key_with_prefix))

# Add all missing story_schema fields to the json_template, populated with default values (string: "", number: 0, boolean: False, array: [], object: {}).
# The story_schema is a JSON Schema in which values are described as follows:
# "value": {
#   "description": "Description of the value",
#   "type": "<type of the value>"
# }
# The type can be "string", "number", "boolean", "array", or "object".
# If the type of the value is an object, then a properties key will contain the dictionary of keys and values of the object.
# The json_template contains null values for primitives, empty arrays for arrays of primitives, and empty objects as the sole elements of arrays of objects.
# The output is a json_template with all missing fields added.
# The json_template may contain nested objects and arrays.
# The function should recurse through the json_template and story_schema, adding missing fields to the json_template as it goes. 
def add_missing_fields_to_json_template(json_template, story_schema):
    for key, value in story_schema.items():
        if key not in json_template:
            if type(value) == dict:
                if value['type'] == 'string':
                    json_template[key] = ""
                elif value['type'] == 'number':
                    json_template[key] = 0
                elif value['type'] == 'boolean':
                    json_template[key] = False
                elif value['type'] == 'array':
                    json_template[key] = []
                elif value['type'] == 'object':
                    json_template[key] = {}
            else:
                add_missing_fields_to_json_template(json_template[key], value['properties'])
        if value['type'] == 'object':
            add_missing_fields_to_json_template(json_template[key], value['properties'])
    return json_template

def generate_outline(sv, update_placeholder):
    perform_update(sv, prompts.story_generation_prompts.outline_prompt, update_placeholder)

def add_character(sv, update_placeholder):
    perform_update(sv, prompts.story_generation_prompts.character_prompt, update_placeholder)

def add_relationship(sv, update_placeholder):
    perform_update(sv, prompts.story_generation_prompts.relationship_prompt, update_placeholder)

def perform_update(sv, user_prompt, update_placeholder):
    output = sv.story_generation_output.value
    sv.story_generation_last_output.value = dict(output)
    story_schema = sv.story_schema.value

    system_message = prompts.story_generation_prompts.system_prompt
    variables = {
        'json_spec': output,
        'json_schema': story_schema,
    }

    commands = AI_API.generate(
        model=sv.model.value,
        temperature=0,
        max_tokens=4096,
        placeholder=update_placeholder,
        system_message=system_message,
        user_message=user_prompt,
        variables=variables,
        prefix=''
    )

    clean_commands = commands.replace('```python', '').replace('```', '').strip()
    sv.story_generation_history.value += f'\n\n```python\n\n{clean_commands}\n\n```\n\n'
    
    last_output = dict(output)
    try:
        exec(clean_commands)
        jsonschema.validate(output, story_schema)
        sv.story_generation_output.value = output
        sv.story_generation_history.value += f'Update validated and applied\n\n'
    except Exception:
        print('Exception on execution')
        traceback.print_exc() # type: ignore
        sv.story_generation_output.value = last_output
        output = sv.story_generation_output.value
        sv.story_generation_history.value += f'Update invalid, not applied\n\n'
    add_missing_fields_to_json_template(sv.story_generation_output.value, story_schema['properties'])    
    
    update_placeholder.markdown(sv.story_generation_history.value)