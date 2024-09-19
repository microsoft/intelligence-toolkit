import os
from json import dumps, loads

import pandas as pd
import streamlit as st

import app.workflows.generate_mock_data.variables as bds_variables
import toolkit.generate_mock_data.data_generator as data_generator
import toolkit.generate_mock_data.text_generator as text_generator
import app.util.schema_ui as schema_ui
from app.util.openai_wrapper import UIOpenAIConfiguration
import app.util.ui_components as ui_components
import app.util.example_outputs_ui as example_outputs_ui

ai_configuration = UIOpenAIConfiguration().get_configuration()

def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


async def create(sv: bds_variables.SessionVariables, workflow: None):
    intro_tab, schema_tab, record_generator_tab, text_generator_tab, mock_tab = st.tabs(['Generate Mock Data workflow:', 'Prepare data schema', 'Generate mock records', 'Generate mock text', 'View example outputs'])
    with intro_tab:
        st.markdown(get_intro())
    with schema_tab:
        sv.loaded_filename.value = schema_ui.build_schema_ui(sv.schema.value, sv.loaded_filename.value)
    with record_generator_tab:
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
                         help="Guidance to the generative AI model about how mock data should be generated (e.g., targeting a specific region, time period, industry, etc.)")
            
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

                    sv.final_object.value, sv.generated_objects.value, sv.generated_dfs.value = await data_generator.generate_data(
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
                        df_update_callback=on_dfs_update,
                        callback_batch=None
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
                            name = sv.schema.value["title"].replace(" ", "_").lower() + "_[mock_data].json"
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

    with text_generator_tab:
        d1, d2 = st.columns([1, 1])
        with d1:
            st.markdown("##### Text generation controls")
            selected_file, selected_df = ui_components.multi_csv_uploader(
                "Upload CSV file(s)",
                sv.uploaded_synthesis_files,
                "synthesis_uploader",
                "synthesis_uploader",
                sv.synthesis_max_rows_to_process,
            )
            if selected_df is not None:
                input_texts = []
                for ix, row in selected_df.iterrows():
                    input_texts.append(row.to_json())
            st.text_area("AI text generation guidance", key=sv.text_generation_guidance.key, value=sv.text_generation_guidance.value,
                        help="Guidance to the generative AI model about how text should be generated from records")
            
            generate = st.button('Generate text data')

        with d2:
            df_placeholders = []
            dl_placeholders = []
            
            df_placeholder = st.empty()
            df_placeholders.append(df_placeholder)
            dl_placeholder = st.empty()
            dl_placeholders.append(dl_placeholder)                

            def on_dfs_update(df):
                with df_placeholder:
                    st.dataframe(df, height=250)
                sv.generated_text_df.value = df
                            
            if generate:
                sv.generated_texts.value = pd.DataFrame()
                df_placeholder.empty()

                (
                    sv.generated_texts.value,
                    sv.generated_text_df.value
                ) = await text_generator.generate_text_data(
                    ai_configuration=ai_configuration,
                    input_texts=input_texts,
                    generation_guidance=sv.text_generation_guidance.value,
                    df_update_callback=on_dfs_update,
                    callback_batch=None
                )

            if sv.generated_text_df.value is not None:
                with df_placeholder:
                    st.dataframe(sv.generated_text_df.value, height=600, use_container_width=True)

                with dl_placeholder:
                    name = '.'.join(selected_file.split(".")[:-1]) + "_[mock_text].csv"
                    st.download_button(
                        label=f'Download {name}',
                        data=sv.generated_text_df.value.to_csv(index=False, encoding='utf-8'),
                        file_name=f'{name}',
                        mime='text/csv',
                        key=f'{name}_csv_download'
                    )
               
    with mock_tab:
        example_outputs_ui.create_example_outputs_ui(mock_tab, workflow)