import os
from json import dumps, loads

import pandas as pd
import streamlit as st

import app.workflows.extract_record_data.variables as variables
import toolkit.extract_record_data.prompts as prompts
import toolkit.extract_record_data.data_extractor as data_extractor
import app.util.schema_ui as schema_ui
from app.util.openai_wrapper import UIOpenAIConfiguration

ai_configuration = UIOpenAIConfiguration().get_configuration()

def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


async def create(sv: variables.SessionVariables, workflow: None):
    intro_tab, schema_tab, generator_tab, mock_tab = st.tabs(['Extract Record Data workflow:', 'Prepare data schema', 'Extract structured records', 'View example outputs'])
    with intro_tab:
        st.markdown(get_intro())
    with schema_tab:
        sv.loaded_filename.value = schema_ui.build_schema_ui(sv.schema.value, sv.loaded_filename.value)
        array_field_arrays = data_extractor.extract_array_fields(sv.schema.value)
        sv.record_arrays.value = ['.'.join(a) for a in array_field_arrays]
    with generator_tab:
        if len(sv.schema.value['properties']) == 0:
            st.warning("Prepare data schema to continue.")
        else:
            # ABC Ltd. is headquartered at 10 Downing Street, London, UK. The company has 3 employees: John Doe, Jane Doe, and Alice Smith. John Doe is the CEO, Jane Doe is the CFO, and Alice Smith is the CTO. The company has 2 departments: Sales and Marketing. The Sales department has 2 employees: John Doe and Jane Doe. The Marketing department has 1 employee: Alice Smith. I called them on 07876545432 and they have an email address of abc@example.com. Grievances include safety (10) and poor treatment (5) of employees.

            st.markdown("##### Record extraction controls")
            st.selectbox("Primary record array", sv.record_arrays.value, key=sv.primary_record_array.key,
                             help="In the presence of multiple arrays, select the one that represents the primary record type.")
            st.text_area("Unstructured text input", key=sv.text_input.key, value=sv.text_input.value)
            st.text_area("AI record extraction guidance", key=sv.generation_guidance.key, value=sv.generation_guidance.value,
                         help="Guidance to the generative AI model about how data records should be extracted")
            
            generate = st.button('Extract record data')
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

                    (
                        sv.final_object.value,
                        sv.generated_objects.value,
                        sv.generated_dfs.value
                    ) = await data_extractor.extract_record_data(
                        ai_configuration=ai_configuration,
                        input_texts=[sv.text_input.value],
                        generation_guidance=sv.generation_guidance.value,
                        record_arrays=sv.record_arrays.value,
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
    # with mock_tab:
    #     workflow_home = 'example_outputs/generate_mock_data'

    #     mock_data_folders = [x for x in os.listdir(f'{workflow_home}')]
    #     print(mock_data_folders)
    #     mock_dfs = {}
    #     for folder in mock_data_folders:
    #         mock_data_file = [x for x in os.listdir(f'{workflow_home}/{folder}') if x.endswith('.csv')][0]
    #         mock_dfs[folder] = pd.read_csv(f'{workflow_home}/{folder}/{mock_data_file}')
    #     selected_data = st.selectbox('Select example', mock_data_folders)
    #     if selected_data != None:
    #         t1, t2 = st.tabs(['JSON schema', 'Mock data'])
    #         with t1:
    #             schema_file = f'{workflow_home}/{selected_data}/{selected_data}_schema.json'
    #             schema_text = loads(open(schema_file, 'r').read())
    #             st.write(schema_text)
    #             st.download_button(
    #                 label=f'Download {schema_file}',
    #                 data=dumps(schema_text, indent=2),
    #                 file_name=schema_file,
    #                 mime='application/json',
    #             )
    #         with t2:
    #             st.dataframe(mock_dfs[selected_data], height=500)
    #             st.download_button(
    #                 label=f'Download {selected_data}',
    #                 data=mock_dfs[selected_data].to_csv(index=False, encoding='utf-8'),
    #                 file_name=f'{selected_data}',
    #                 mime='text/csv',
    #             )