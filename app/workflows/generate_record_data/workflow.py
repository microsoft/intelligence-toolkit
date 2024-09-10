import os
from json import dumps, loads

import pandas as pd
import streamlit as st

import app.workflows.generate_record_data.functions as bds_functions
import app.workflows.generate_record_data.variables as bds_variables
import toolkit.generate_record_data.data_generator as data_generator
import toolkit.generate_record_data.schema_builder as schema_builder
from app.util.openai_wrapper import UIOpenAIConfiguration
from toolkit.generate_record_data import get_readme as get_intro

ai_configuration = UIOpenAIConfiguration().get_configuration()

def create(sv: bds_variables.SessionVariables, workflow: None):
    intro_tab, schema_tab, generator_tab, mock_tab = st.tabs(['Generate Record Data workflow:', 'Prepare data schema', 'Generate mock data', 'View example outputs'])
    with intro_tab:
        st.markdown(get_intro())
    with schema_tab:
        form, preview = st.columns([1, 1])
        with form:
            file = st.file_uploader('Upload schema', type=['json'], key='schema_uploader')
            if file is not None and sv.loaded_filename.value != file.name:
                sv.loaded_filename.value = file.name
                sv.schema.value.clear()
                jsn = loads(file.read())
                for k, v in jsn.items():
                    sv.schema.value[k] = v
                print(f'Loaded schema: {sv.schema.value}')
            st.markdown('### Edit data schema')
            bds_functions.generate_form_from_json_schema(
                global_schema=sv.schema.value,
                default_schema=schema_builder.create_boilerplate_schema(),
            )
        with preview:
            st.markdown('### Preview')
            schema_tab, object_tab = st.tabs(['JSON schema', 'Sample object'])
            obj = schema_builder.generate_object_from_schema(sv.schema.value)
            with schema_tab:
                st.write(sv.schema.value)
            with object_tab:
                st.write(obj)
            validation = schema_builder.evaluate_object_and_schema(obj, sv.schema.value)
            if validation == schema_builder.ValidationResult.VALID:
                st.success('Schema is valid')
            elif validation == schema_builder.ValidationResult.OBJECT_INVALID:
                st.error('Object is invalid')
            elif validation == schema_builder.ValidationResult.SCHEMA_INVALID:
                st.error('Schema is invalid')
            name = sv.schema.value["title"].replace(" ", "_").lower() + "_[schema].json"
            st.download_button(
                label=f'Download {name}',
                data=dumps(sv.schema.value, indent=2),
                file_name=name,
                mime='application/json'
            )
    with generator_tab:
        if len(sv.schema.value['properties']) == 0:
            st.warning("Prepare data schema to continue.")
        else:
            st.markdown("##### Data generation controls")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                array_field_arrays = data_generator.extract_array_fields(sv.schema.value)
                sv.record_arrays.value = ['.'.join(a) for a in array_field_arrays]
                st.selectbox("Primary record array", sv.record_arrays.value, key=sv.primary_record_array.key,
                             help="In the presence of multiple arrays, select the one that represents the primary record type whose records should be counted towards the `Total records to generate` target")
            with c2:
                st.number_input("Records per batch", min_value=1, value=sv.records_per_batch.value, key=sv.records_per_batch.key,
                                help="How many records to generate in a single LLM call")
            with c3:
                st.number_input("Parallel batches", min_value=0, step=100, value=sv.parallel_batches.value, key=sv.parallel_batches.key,
                                help="In a single iteration, how many batches to generate via parallel LLM calls")
            with c4:
                st.number_input("Total records to generate", min_value=0, step=sv.records_per_batch.value*sv.parallel_batches.value, value=sv.num_records_overall.value, key=sv.num_records_overall.key, help="How many records to generate. Must be a multiple of `Records per batch` x `Parallel batches`")    
            with c5:
                st.number_input("Duplicate records per batch", min_value=0, value=sv.duplicate_records_per_batch.value, key=sv.duplicate_records_per_batch.key,
                                help="Within each batch, how many records should be near-duplicates of a seed record randomly selected from existing records")
            with c6:
                st.number_input("Related records per batch", min_value=0, value=sv.related_records_per_batch.value, key=sv.related_records_per_batch.key,
                                help="Within each batch, how many records should appear closely related to (but not the same as) a seed record randomly selected from existing records")
            st.text_area("AI data generation guidance", key=sv.generation_guidance.key, value=sv.generation_guidance.value,
                         help="Guidance to the generative AI model about how mock data should be generated (e.g., targeted a specific region, time period, industry, etc.)")
            
            generate = st.button('Generate mock data')
            df_placeholders = []
            dl_placeholders = []
            if len(sv.record_arrays.value) == 0:
                st.warning("No record arrays found in schema.")
            else:
                tabs = st.tabs(sv.record_arrays.value)
                for ix, record_array in enumerate(sv.record_arrays.value):
                    with tabs[ix]:
                        df_placeholder = st.empty()
                        df_placeholders.append(df_placeholder)
                        dl_placeholder = st.empty()
                        dl_placeholders.append(dl_placeholder)                

                def on_dfs_update(path_to_df):
                    for ix, record_array in enumerate(sv.record_arrays.value):
                        with df_placeholders[ix]:
                            df = path_to_df[record_array]
                            st.dataframe(df, height=250)
                    sv.generated_dfs.value = path_to_df
                                
                if generate:
                    sv.generated_dfs.value = {k: pd.DataFrame() for k in sv.record_arrays.value}
                    for placeholder in df_placeholders:
                        placeholder.empty()

                    sv.final_object.value, sv.generated_objects.value, sv.generated_dfs.value = data_generator.generate_data(
                        ai_configuration=ai_configuration,
                        generation_guidance=sv.generation_guidance.value,
                        primary_record_array=sv.primary_record_array.value,
                        record_arrays=sv.record_arrays.value,
                        num_records_overall=sv.num_records_overall.value,
                        records_per_batch=sv.records_per_batch.value,
                        parallel_batches=sv.parallel_batches.value,
                        duplicate_records_per_batch=sv.duplicate_records_per_batch.value,
                        related_records_per_batch=sv.related_records_per_batch.value,
                        data_schema=sv.schema.value,
                        df_update_callback=on_dfs_update
                    )

                for ix, record_array in enumerate(sv.record_arrays.value):
                        with df_placeholders[ix]:
                            if record_array in sv.generated_dfs.value:
                                df = sv.generated_dfs.value[record_array]
                                st.dataframe(df, height=250)

                for ix, record_array in enumerate(sv.record_arrays.value):
                    with dl_placeholders[ix]:
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            name = sv.schema.value["title"].replace(" ", "_").lower() + "_[data].json"
                            st.download_button(
                                label=f'Download {name}',
                                data=dumps(sv.final_object.value, indent=2),
                                file_name=f'{name}',
                                mime='application/json',
                                key=f'{name}_{ix}_json_download'
                            )
                        with c2:
                            if record_array in sv.generated_dfs.value:
                                st.download_button(
                                    label=f'Download {record_array}.csv',
                                    data=sv.generated_dfs.value[record_array].to_csv(index=False, encoding='utf-8'),
                                    file_name=f'{record_array}.csv',
                                    mime='text/csv',
                                    key=f'{record_array}_csv_download'
                                )
    with mock_tab:
        workflow_home = 'example_outputs/generate_record_data'

        mock_data_files = [x for x in os.listdir(f'{workflow_home}') if x.endswith('.csv')]
        mock_dfs = {}
        for file in mock_data_files:
            mock_dfs[file] = pd.read_csv(f'{workflow_home}/{file}')
        selected_data = st.selectbox('Select mock data', mock_data_files)
        if selected_data != None:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('### Schema')
                schema_file = '_'.join(selected_data.split('_')[:-1]) + '_schema.json'
                schema_text = loads(open(f'{workflow_home}/{schema_file}', 'r').read())
                st.write(schema_text)
                st.download_button(
                    label=f'Download {schema_file}',
                    data=dumps(schema_text, indent=2),
                    file_name=schema_file,
                    mime='application/json',
                )
            with c2:
                st.markdown('### Mock data')
                st.dataframe(mock_dfs[selected_data], height=500)
                st.download_button(
                    label=f'Download {selected_data}',
                    data=mock_dfs[selected_data].to_csv(index=False, encoding='utf-8'),
                    file_name=f'{selected_data}',
                    mime='text/csv',
                )