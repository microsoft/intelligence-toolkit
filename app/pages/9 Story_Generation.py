import traceback
import streamlit as st
import my_session_variables as msv
from session_variable import SessionVariable
import os
import AI_API
import json
from collections import defaultdict

from functions.question_answering_functions import *
import prompts.story_generation_prompts
from functions.story_generation_functions import *

class Story:
    def __init__(self):
        self.acts = []

    def add_act(self, act):
        self.acts.append(act)

    def list_progressions(self):
        progressions = []
        for act in self.acts:
            for chapter in act.chapters:
                for scene in chapter.scenes:
                    progressions += scene.progressions
        return progressions

class Progression:
    def __init__(self, percent, type, character_name_1, character_name_2, from_state, to_state, change, num_states):
        self.percent = percent
        self.type = type
        self.character_name_1 = character_name_1
        self.character_name_2 = character_name_2
        self.from_state = from_state
        self.to_state = to_state
        self.change = change
        self.num_states = num_states

    def set_index(self, index):
        self.index = index

class Chapter:
    def __init__(self, levels):
        self.scenes = []
        self.levels = levels

    def add_scene(self, scene):
        scene.progressions.sort(key=lambda x: self.levels.index(x.change))
        self.scenes.append(scene)
        

class Scene:
    def __init__(self, levels):
        self.progressions = []
        self.levels = levels

    def add_progression(self, progression):
        self.progressions.append(progression)
        self.progressions.sort(key=lambda x: self.levels.index(x.change))


class Act:
    def __init__(self, progressions, levels, G):
        self.progressions = progressions
        # self.level_runs, self.level_characters = self.break_by_level()
        self.chapters = []
        self.G = G
        self.levels = levels
        self.break_by_graph()

    def add_chapter(self, chapter):
        self.chapters.append(chapter)
    
    def break_by_graph(self):
        self.chapters = []
        current_chapter = Chapter(self.levels)
        current_scene = Scene(self.levels)
        # for level, progressions in self.level_runs.items():
        character_to_pxs = defaultdict(list)
        relationship_to_pxs = defaultdict(list)
        for px, progression in enumerate(self.progressions):
            if progression.type == 'character':
                character_to_pxs[progression.character_name_1].append(px)
            else:
                relationship_to_pxs[tuple(sorted([progression.character_name_1, progression.character_name_2]))].append(px)

        used_progressions = set()
        used_characters = set()
        last_change = None
        for px, progression in enumerate(self.progressions):
            if px in used_progressions:
                continue
            if progression.type == 'character':
                # if progression.character_name_1 in used_characters: # or (progression.change in ['No change', 'Minor change'] and last_change in ['Moderate change', 'Major change']):
                #     # New chapter whenever a character repeats or a change level drops

                # else: # we have a new character, pull in all related characters
                print(f'Adding character {progression.character_name_1} to scene')
                current_scene.add_progression(progression)
                used_progressions.add(px)
                used_characters.add(progression.character_name_1)
                neighbours = list(set(self.G.neighbors(progression.character_name_1)).intersection(character_to_pxs.keys()))
                neighbours.sort(key=lambda x : self.G.degree(x))
                added_rel = False
                for neighbor in neighbours:
                    if neighbor in used_characters:
                        continue
                    npxi = 0
                    next_px = character_to_pxs[neighbor][npxi]
                    while next_px in used_progressions:
                        npxi += 1
                        if npxi >= len(character_to_pxs[neighbor]):
                            npxi = -1
                            break
                        next_px = character_to_pxs[neighbor][npxi]
                    if npxi == -1:
                        continue
                    print(f'Adding related character {neighbor} to scene')
                    current_scene.add_progression(self.progressions[next_px])
                    last_change = self.progressions[next_px].change
                    used_progressions.add(next_px)

                    rel_tuple = tuple(sorted([progression.character_name_1, neighbor]))
                    if rel_tuple in relationship_to_pxs:
                        rpxi = 0
                        rpx = relationship_to_pxs[rel_tuple][rpxi]
                        while rpx in used_progressions:
                            rpxi += 1
                            if rpxi >= len(relationship_to_pxs[rel_tuple]):
                                rpxi = -1
                                break
                            rpx = relationship_to_pxs[rel_tuple][rpxi]
                        if rpxi == -1:
                            continue
                        current_scene.add_progression(self.progressions[rpx])
                        last_change = self.progressions[rpx].change
                        used_progressions.add(rpx)

                        current_chapter.add_scene(current_scene)
                        # self.add_chapter(current_chapter)
                        # current_chapter = Chapter(self.levels)
                        current_scene = Scene(self.levels)
                        # used_characters = set()
                print('Making new chapter')

                if len(current_chapter.scenes) > 0:
                    self.add_chapter(current_chapter)
                    used_characters = set()
                    current_chapter = Chapter(self.levels)
                    current_scene = Scene(self.levels)

            else:
                current_scene.add_progression(progression)
                

        if len(current_scene.progressions) > 0:
            current_chapter.add_scene(current_scene)
        if len(current_chapter.scenes) > 0:
            self.add_chapter(current_chapter)
            current_chapter = Chapter(self.levels)
            current_scene = Scene(self.levels)


def split_list(lst, n):
  # lst is the list to be split
  # n is the number of parts
  # returns a list of lists
  length = len(lst) # get the length of the list
  size = length // n # get the size of each part
  remainder = length % n # get the remainder
  result = [] # initialize an empty list to store the result
  index = 0 # initialize an index to track the position in the list
  for i in range(n): # loop through the number of parts
    part = lst[index:index+size] # get a part of the list
    if i < remainder: # if there is excess
      part.append(lst[index+size]) # add one more element to the part
      index += 1 # increment the index by one
    result.append(part) # append the part to the result
    index += size # increment the index by the size
  return result # return the result

def condense_prose_for_llm(input_text, character_limit=200000):
    # Always remove all but last scene text, relying on events and knowledge graph updates to provide context
    working_text = input_text
    while len(re.findall(r'\n### Scene text\n', working_text, flags=re.MULTILINE)) > 1:
        print('Got multiple scene text sections')
        st_match = re.search(r'\n### Scene text\n', working_text, flags=re.MULTILINE)
        kg_match = re.search(r'\n### Knowledge graph updates', working_text, flags=re.MULTILINE)
        working_text = working_text[:st_match.start()] + working_text[kg_match.end():]
        print(f'Working text is now {working_text}')

    # Remove all knowledge graph updates that are Minor
    if len(working_text) > character_limit:
        working_text = re.sub(r'\(.*, Minor change\)\n', '', working_text, flags=re.MULTILINE)
        
    # Remove all knowledge graph updates that are Moderate
    if len(working_text) > character_limit:
        working_text = re.sub(r'\(.*, Moderate change\)\n', '', working_text, flags=re.MULTILINE)
    
    working_text = re.sub(r'\n\#\#Scene \d+\.(\d+)\.\d+\n', '', working_text, flags=re.MULTILINE)

    return working_text

def clean_up_prose_for_user(prose):
    openings = set()
    output = f''
    lines = prose.split('\n')
    do_ouput = False
    last_chapter = 0
    overall_chapter = 1
    capture = False
    add_chapter = False
    for i, line in enumerate(lines):
        if i == 0 and line.startswith('# '):
            output += line + '\n\n'
            continue
        if line == '':
            continue
        if line.startswith('# '): #title
            continue
        
        if line == '### Scene event':
            do_ouput = False
        elif line == '### Knowledge graph updates':
            do_ouput = False
        elif line == '### Scene text':
            do_ouput = True
            add_chapter = True
            capture = True
        elif do_ouput:
            if capture:
                if line in openings:
                    print(f'Found duplicate opening: {line}')
                    do_ouput = False
                else:
                    if add_chapter:
                        output += f'## Chapter {overall_chapter}\n\n'
                        overall_chapter += 1
                        add_chapter = False
                    output += line + '\n\n'
                    openings.add(line)
                capture = False
            else:
                if add_chapter:
                    output += f'## Chapter {overall_chapter}\n\n'
                    overall_chapter += 1
                    add_chapter = False
                output += line + '\n\n'
    return output

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Story Generation')
    sv = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        sv = msv.SessionVariables()
    else:
        sv = st.session_state['my_session_variables']

    if not os.path.exists('extraction_schema'):
        os.mkdir('extraction_schema')

    sv.story_schema.value = json.loads(open('extraction_schema/story_outline_schema.json', 'r').read())
    
    
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(['Generate story outline', 'View story outline JSON', 'Create story graph JSON', 'View story graph network', 'View story scenes', 'Generate scene text', 'View LLM input', 'View final text'])
    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            story_file = st.selectbox('Load saved story file', [''] + [f for f in os.listdir('stories') if f.endswith('.json')])
            if story_file != '':
                sv.story_generation_output.value = json.loads(open(f'stories/{story_file}', 'r').read())
            else:
                if sv.story_generation_output.value == {}:
                    sv.story_generation_output.value = generate_json_template_from_story_schema(sv.story_schema.value)
            output = sv.story_generation_output.value
            if st.button('Save current story file'):
                open(f'stories/{sv.story_generation_output.value["title"].replace(":", "")}.json', 'w').write(json.dumps(sv.story_generation_output.value, indent=2))
            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                gen_outline = st.button('Generate outline', use_container_width=True)
            with b2:
                add_char = st.button('Add character', use_container_width=True)
            with b3:
                add_rel = st.button('Add relationship', use_container_width=True)

            with st.expander('AI updates in progress', expanded=True):
                update_placeholder = st.empty()

            if gen_outline:
                generate_outline(sv, update_placeholder)
            elif add_rel:
                add_relationship(sv, update_placeholder)
            elif add_char:
                add_character(sv, update_placeholder)


            # if undo:
            #     sv.case_extraction_output.value = dict(sv.case_extraction_last_output.value)
            #     update_placeholder.empty()
            #     sv.case_extraction_history.value = ''
            #     st.rerun()
            
        with c2:
            generate_form_from_json_template(sv.story_generation_output.value, sv.story_schema.value['description'], sv.story_schema.value['properties'])
        print('t1 done')
    with t2:
        st.download_button('Download story outline', json.dumps(sv.story_generation_output.value, indent=2), 'story_outline.json', 'text/json')
        st.markdown(f'```json\n\n{json.dumps(sv.story_generation_output.value, indent=2)}\n\n```\n\n')
        update_placeholder.markdown(sv.story_generation_history.value)
        print('t2 done')
    with t3:
        spec = sv.story_generation_output.value
        c1, c2, c3 = st.columns([1, 1, 1])
        sv.minor_scene_count = SessionVariable(2)
        sv.moderate_scene_count = SessionVariable(5)
        sv.major_scene_count = SessionVariable(9)
        with c1:
            major_scenes = st.number_input('Number of scenes per major character/relationship', min_value=1, key = sv.major_scene_count.key, value=sv.major_scene_count.value)
        with c2:
            moderate_scenes = st.number_input('Number of scenes per moderate character/relationship', min_value=1, key = sv.moderate_scene_count.key, value=sv.moderate_scene_count.value)
        with c3:
            minor_scenes = st.number_input('Number of scenes per minor character/relationship', min_value=1, key = sv.minor_scene_count.key, value=sv.minor_scene_count.value)
        # st.download_button('Download story graph', json.dumps(spec, indent=2), 'story_graph.json', 'text/json')
        if st.button('Generate story graph'):
            characters = []
            character_names = ['']
            for c in spec['characters']:
                if 'name' in c and c['name'] != '' and c['name'] not in character_names:
                    character_names.append(c['name'])
                    characters.append(c)
            relationships = []
            relationship_pairs = [('','')]
            for r in spec['relationships']:
                if 'first_character' in r and 'second_character' in r and r['first_character'] != '' and r['second_character'] != '':
                    pair = tuple(sorted([r['first_character'], r['second_character']]))
                    if pair not in relationship_pairs:
                        relationship_pairs.append(pair)
                        relationships.append(r)
            sv.story_graph.value = {
                'context': '\n'.join([f'{k}: {v}' for k, v in spec.items() if k not in ['characters', 'relationships']]),
                'characters': [{
                    'name': c['name'],
                    'context': '\n'.join([f'{k}: {v}' for k, v in c.items() if k not in ['name', 'state_progression']]),
                    'state_progression': generate_progressions(spec['act_count'], c['state_progression'], major_scenes if c['prominance'] == 'Major' else moderate_scenes if c['prominance'] == 'Moderate' else minor_scenes)
                } for c in characters if 'state_progression' in c],
                'relationships': [{
                    'first_character': r['first_character'],
                    'second_character': r['second_character'],
                    'context': '\n'.join([f'{k}: {v}' for k, v in r.items() if k not in ['state_progression', 'first_character', 'second_character', 'scene_count']]),
                    'state_progression': generate_progressions(spec['act_count'], r['state_progression'], major_scenes if r['prominance'] == 'Major' else moderate_scenes if r['prominance'] == 'Moderate' else minor_scenes)
                } for r in relationships if 'state_progression' in r]
            }
            
        st.markdown(f'```json\n\n{json.dumps(sv.story_graph.value, indent=2)}\n\n```\n\n')
        print('t3 done')
    with t4:
        get_story_graph(sv.story_graph.value, 1000, 1000)
        print('t4 done')
    with t5:
        spec = sv.story_generation_output.value
        graph = sv.story_graph.value
        sv.story_scenes.value = []
        if len(graph) > 0:
            levels = ['No change', 'Minor change', 'Moderate change', 'Major change']
            # state_progression_count = 0
            num_char_state_progressions = sum([len(c['state_progression']) for c in graph['characters']])
            num_rel_state_progressions = sum([len(r['state_progression']) for r in graph['relationships']])
            num_state_progressions = num_char_state_progressions + num_rel_state_progressions

            # level_index = 0
            # context = graph['context'] if 'context' in graph else ''
            characters = graph['characters'] if 'characters' in graph else []
            relationships = graph['relationships'] if 'relationships' in graph else []
            
            G = nx.Graph()
            G.add_nodes_from([c['name'] for c in characters])
            G.add_edges_from([(r['first_character'], r['second_character']) for r in relationships])

            progressions = []

            for character in characters:
                states = character['state_progression']
                num_states = len(states)
                for state in states:
                    from_state, to_state, change, percent = state
                    progressions.append(Progression(percent, 'character', character['name'], None, from_state, to_state, change, num_states))
            for relationship in relationships:
                states = relationship['state_progression']
                num_states = len(states)
                for state in states:
                    from_state, to_state, change, percent = state
                    progressions.append(Progression(percent, 'relationship', relationship["first_character"], relationship["second_character"], from_state, to_state, change, num_states))
            progressions.sort(key=lambda x: x.percent)
            for px, progression in enumerate(progressions):
                progression.set_index(px)
            act_count = int(spec['act_count'])
            if act_count == 0:
                act_count = 3

            # divide the progressions into acts
            act_progressions = split_list(progressions, act_count)


            story = Story()

            for act_progression in act_progressions:
                act = Act(act_progression, levels, G)
                story.add_act(act)



            # st.markdown(f'```json\n\n{json.dumps(act_chapter_progressions, indent=2)}\n\n```\n\n')

            sv.story_structure_text.value = f'# {spec["title"]}\n\n'
            for ax, act in enumerate(story.acts):
                sv.story_structure_text.value += f'## Act {ax+1}\n\n'
                for cx, chapter in enumerate(act.chapters):
                    sv.story_structure_text.value += f'### Chapter {ax+1}.{cx+1}\n\n'
                    for sx, scene in enumerate(chapter.scenes):
                        scene_title = f'Scene {ax+1}.{cx+1}.{sx+1}'
                        sv.story_structure_text.value += f'#### {scene_title}\n\n'
                        scene_changes = []
                        scene_placement = ''
                        if ax == 0:
                            scene_placement += 'Opening act; '
                        elif ax == len(story.acts) - 1:
                            scene_placement += 'Closing act; '
                        else:
                            scene_placement += 'Middle act; '
                        if cx == 0:
                            scene_placement += 'Opening chapter; '
                        elif cx == len(act.chapters) - 1:
                            scene_placement += 'Closing chapter; '
                        else:
                            scene_placement += 'Middle chapter; '
                        if sx == len(chapter.scenes) - 1:
                            scene_placement += 'Closing scene'
                        elif sx == 0:
                            scene_placement += 'Opening scene'
                        else:
                            scene_placement += 'Middle scene'
                        for progression in scene.progressions:
                            if progression.type == 'character':
                                if change == 'No change':
                                    character_change = f'Character **{progression.character_name_1}** is established as being *"{progression.from_state}"*'
                                else:
                                    character_change = f'Character **{progression.character_name_1}** undergoes **{progression.change}** from *"{progression.from_state}"* towards *"{progression.to_state}"*' # at {percent} through {num_states} states'
                                sv.story_structure_text.value += f'{character_change}\n\n'
                                scene_changes.append(character_change)

                            else:
                                if change == 'No change':
                                    relationship_change = f'Relationship between **{progression.character_name_1}** and **{progression.character_name_2}** is established as being *"{progression.from_state}"*'
                                else:
                                    relationship_change = f'Relationship between **{progression.character_name_1}** and **{progression.character_name_2}** undergoes **{progression.change}** from *"{progression.from_state}"* towards *"{progression.to_state}"*'# at {percent}*'
                                sv.story_structure_text.value += f'{relationship_change}\n\n'
                                scene_changes.append(relationship_change)
                            
                        sv.story_scenes.value.append((scene_title, scene_placement, scene_changes))

            st.markdown(sv.story_structure_text.value)
            actual_progressions = story.list_progressions()
            st.markdown(f'*{num_state_progressions}/{len(actual_progressions)} progressions generated*')
        

        print('t5 done')
    with t6:
        scenes = sv.story_scenes.value
        graph = sv.story_graph.value
        sv.scenes_completed = SessionVariable([])
        for scene_title, scene_placement, scene_changes in scenes:
            if scene_title == "Scene 3.7.1":
                break
            sv.scenes_completed.value.append(scene_title)
        print(f'Scenes completed: {sv.scenes_completed.value}')
        if st.button('Generate scenes'):
            if sv.story_prose.value == '':
                sv.story_prose.value = f'# {sv.story_generation_output.value["title"]}\n\n'
            scene_placeholder = st.empty()
            story_context = sv.story_graph.value['context'] if 'context' in sv.story_graph.value else ''
            sv.story_memory.value = ''  

            context = '# Story context:\n\n' + story_context + '\n\n' 
            for character in graph['characters']:
                context += f'# {character["name"]} context:\n'
                context += character['context'] + '\n\n'
            for relationship in graph['relationships']:
                context += f'# {relationship["first_character"]} and {relationship["second_character"]} context:\n'
                context += relationship['context'] + '\n\n'

            for scene_title, scene_placement, scene_changes in scenes:
                if scene_title in sv.scenes_completed.value:
                    continue
                print(f'Generating scene {scene_title}')
                print(f'Scene placement: {scene_placement}')
                print(f'Scene changes: {scene_changes}')
   
                system_prompt = prompts.story_generation_prompts.scene_system_prompt
                user_prompt = prompts.story_generation_prompts.scene_user_prompt
                variables = {
                    'context': context,
                    'changes': '\n'.join(scene_changes),
                    'progress': sv.story_memory.value,
                    'placement': scene_placement
                }
                # sv.story_prose.value += f'## {scene_title}\n\n'

                try:
                    new_prose = AI_API.generate(
                        model=sv.model.value,
                        temperature=1,
                        max_tokens=4096,
                        placeholder=scene_placeholder,
                        system_message=system_prompt,
                        user_message=user_prompt,
                        variables=variables,
                        prefix=sv.story_prose.value
                    )
                except Exception as e:
                    print('Error generating prose')
                    traceback.print_exc()
                sv.story_prose.value += new_prose + '\n\n'
                sv.story_memory.value = condense_prose_for_llm(sv.story_prose.value)
                open('stories/story_prose.md', 'w', encoding='utf-8').write(sv.story_prose.value)
                open('stories/story_memory.md', 'w', encoding='utf-8').write(sv.story_memory.value)
                sv.scenes_completed.value.append(scene_title)
            scene_placeholder.markdown(sv.story_prose.value)
        print('t6 done')
    with t7:
        # show story memory
        st.markdown(sv.story_memory.value)
    with t8:

        cleaned = clean_up_prose_for_user(open('stories/story_prose.md', 'r', encoding='utf-8').read())
        st.download_button('Download story text', cleaned, 'story_text.md', 'text/markdown')
        st.markdown(cleaned)
        print('t7 done')

if __name__ == '__main__':
    main()