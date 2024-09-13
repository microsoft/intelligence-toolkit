data_extraction_prompt = """
You are a helpful assistant tasked with extracting a structured JSON object from unstructured text following the JSON schema provided.

You should generate a new object that adheres to the schema and contains all relevant information from the text.

--Generation guidance--

{generation_guidance}

--Input text--

{input_text}

"""