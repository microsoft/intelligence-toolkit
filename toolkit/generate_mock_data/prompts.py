unseeded_data_generation_prompt = """
You are a helpful assistant tasked with generating a JSON object following the JSON schema provided.

You should generate a new object that adheres to the schema and contains mock data that is plausible but not linked to any real-world entities (e.g., person, organization). All output data records should be unrelated and contain substantial variety (e.g., covering all enum/binary values). The content of the generated records and fields should follow the guidance below, if provided.

--Generation guidance--

{generation_guidance}

--Primary Record Array--

{primary_record_array}

--Record Targets--

Output record count: {total_records}
"""

seeded_data_generation_prompt = """
You are a helpful assistant tasked with generating a JSON object following the JSON schema provided. You should generate mock data that is plausible but not linked to any real-world entities (e.g., person, organization). 

The JSON object may contain multiple arrays representing collections of data records. For the purposes of this task, only consider the primary record array specified when counting records and generate any other auxiliary records as needed to complete and/or connect these primary records.

The seed record provided should be used to generate certain numbers of records in the output object that are either near duplicates or close relations of the seed record, as follows:

- Near duplicate: A record that is very similar to a record in the example object but not identical, with minor variations in data fields but recognisable as the same real-world entity.
- Close relation: A record that is related to a record in the example object, but not a direct duplicate, with some shared data fields or common attributes indicating a close real-world relationship between distinct entities.

Once the target numbers of near duplicates and close relaitions have been generated, generate the remaining records. These records should be unrelated to the seed record and contain substantial variety (e.g., covering all enum/binary values in the data schema).

Do not include the seed record in the output object.

The content of the generated records and fields should follow the guidance below, if provided.

--Generation guidance--

{generation_guidance}

--Primary Record Array--

{primary_record_array}

--Seed Record--

{seed_record}

--Record Targets--

{record_targets}
"""