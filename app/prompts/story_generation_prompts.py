system_prompt = """\
You are a helpful assistant generating a Python object conforming to a JSON schema representing a story.

Output Python code that makes the requested update to the output object, which is stored in a Python dictionary variable named `output`.

Escape any " characters in the values with a \ character.

Do not assign any variables other than those in the schema.
Do not assign empty values.
Do not modify values once assigned.

Preface the code with ```python\n and end with ```\n.
"""
                        
outline_prompt = """\
Your goal is to populate all top-level fields in the JSON specification, except for characters and relationships.

Do not add any characters or relationships.

Output in progress:

{json_spec}

JSON schema guiding the output:

{json_schema}

Python code to update the JSON specification towards completion of the JSON schema:
"""

character_prompt = """\
Your goal is to identify the need for a story to have a new character that contributes to the story.

Explain your reasoning for the need for the new character in code comments.

Do not used cliched or stereotypical character names, traits, objects, locations, or other entities.

Output in progress:

{json_spec}

JSON schema guiding the output:

{json_schema}

Python code to update the JSON specification towards completion of the JSON schema:
"""


relationship_prompt = """\
Your goal is to identify the need for a story to have a relationship between two characters who do not already appear as relations in the output in progress.

Explain your reasoning for the need for the new character in code comments.

If there are no characters in the output yet, create two new characters and add them to the characters field before adding the relationship.

Otherwise, create a relationship between existing characters if appropriate and provided they don't already have a relationship.

Otherwise, create a new character and add it to the characters field before adding a relationship to an existing character.

Not all characters need to have a relationship with one another - only create relationships that are fundamental to the story.

Do not used cliched or stereotypical character names, traits, objects, locations, or other entities.

Output in progress:

{json_spec}

JSON schema guiding the output:

{json_schema}

Python code to update the JSON specification towards completion of the JSON schema:
"""

scene_system_prompt = """\
Your goal is to generate a scene of a story that is consistent with the context provided and which develops certain characters and relationships as specified.

Write the output in the style of genre fiction, but do not add any structure beyond paragraphs (so no chapters, sections, etc.). Paragraphs should extend to several sentences where possible, avoiding the pattern of many short paragraphs. Integrate world-building throughout the scene, using relevant knowledge from prior scenes for reinforcement. Each scene should reference at least 3 pieces of knowledge from prior scenes or story context, and more as appropriate for the targeted development. Favour dialogue and inner monologue over narration for character and relationship development. Show these developments with character's words and actions, don't simply describe them with narration.

Opening scenes of new acts require a high ratio of narrative world building to dialogue and inner monologue, followed by an balanced ratio for the opening scenes of new chapters, followed by a low ratio for all other scenes.

Similarly, scenes should end with words or actions, sequences should end with additional narration, and acts should end with even more narration, especially for the denouement (for the final act).

Steps to follow when creating the scene:
1. Create world-building narrative that is consistent with the context of the scene and the story so far.
2. Establish an event that is consistent with the placement of the scene in the story and the level of development that characters and relationships need to undergo in the scene.
3. Determine whether new characters, objects, locations, and other entities need to be introduced to support the event.
4. Determine how the event affects characters' goals, and what actions they would take in response to the event.
5. Have the character's make minimal and conservative actions in response to the event, but have the world react in a way that is different to the reaction expected by the character.
6. Determine a renewed action that the character is forced to take in response to the world's reaction and which transforms the character and/or their relationships in the way specified.
7. Write fluent prose that captures all of the above in the style of a genre fiction novel. Focus on the characters' inner monologue and dialogue, and use narration sparingly. Never use narration to describe a character's emotions or thoughts - always use inner monologue for this purpose. Use narration to describe the world, the actions of characters, and the reactions of the world to the actions of characters.

The premise and/or inciting incident of the story should NOT occur in the first chapter. This chapter should instead establish the characters and their relationships. They may hint at the premise and/or inciting incident, but not reveal them until the approproate time in the story.

The closing chapter should bring the story to its conclusion, resolving the premise and/or inciting incident, while the closing scene should end with a satisfying last word or act.

Format the response as a Markdown document with the following structure:
1. Add a "### Scene event" subheading containing a description of the event. The significance of events and their stakes for the characters should increase over the course of an act and story as a whole, reaching a climax in the closing chapter.
2. Add a "### Scene text" subheading containing the output prose in the style of genre fiction. The prose in a scene should not replicate the prose or prose structure in any prior scene - variety is crucical.
3. Add a "### Knowledge graph updates" subheading describing updates to knowledge about the story world that are implied by the scene. Updates should be written as a bulleted list of (subject, predicate, object, significance) tuples, where:
- subject and object are entities
- predicate is a relationship between the entities
- significance is the significance of the update to the plot (Minor, Moderate, or Major)
- For example, if the scene describes a character learning that another character is a spy, the update would be written as (Character1, Knows is spy, Character2, Major).

Add no headings other than those described above.

Aim to generate for 12-15 paragraphs of prose overall, with 3-7 sentences per paragraph. Complexity of language should be appropriate for a young adult audience.
"""

scene_user_prompt = """\
Context of the story:

{context}

Placement of the scene in the story:

{placement}

Desired development of characters and relationships over the course of the scene:

{changes}

Output in progress, to be continued:

{progress}
"""