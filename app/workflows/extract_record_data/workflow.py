import os
from json import dumps, loads

import pandas as pd
import streamlit as st

import app.util.example_outputs_ui as example_outputs_ui
import app.util.schema_ui as schema_ui
import app.util.ui_components as ui_components
import app.workflows.extract_record_data.variables as variables
import toolkit.extract_record_data.data_extractor as data_extractor
from app.util.download_pdf import add_download_pdf
from app.util.openai_wrapper import UIOpenAIConfiguration

ai_configuration = UIOpenAIConfiguration().get_configuration()

def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


async def create(sv: variables.SessionVariables, workflow: None):
    ui_components.check_ai_configuration()

    intro_tab, schema_tab, generator_tab, mock_tab = st.tabs(['Extract Record Data workflow:', 'Prepare data schema', 'Extract structured records', 'View example outputs'])
    with intro_tab:
        file_content = get_intro()
        st.markdown(file_content)
        add_download_pdf(
            f"{workflow}_introduction_tutorial.pdf",
            file_content,
            ":floppy_disk: Download as PDF",
        )
    with schema_tab:
        sv.loaded_schema_filename.value = schema_ui.build_schema_ui(
            sv.schema.value, sv.loaded_schema_filename.value)
        array_field_arrays = data_extractor.extract_array_fields(sv.schema.value)
        sv.record_arrays.value = ['.'.join(a) for a in array_field_arrays]
    with generator_tab:
        d1, d2 = st.columns([1, 1])
        with d1:
            if len(sv.schema.value['properties']) == 0:
                st.warning("Prepare data schema to continue.")
            else:
                st.markdown("##### Record extraction controls")
                mode = st.radio("Mode", ["Extract from single text", "Extract from rows of CSV file"], horizontal=True)
                input_texts = []
                if mode == "Extract from single text":
                    st.text_area("Unstructured text input", key=sv.text_input.key, value=sv.text_input.value, height=400)
                    input_texts.append(sv.text_input.value)
                else:
                    _, selected_df, changed = ui_components.multi_csv_uploader(
                        "Upload CSV file(s)",
                        sv.uploaded_synthesis_files,
                        workflow + "uploader",
                        sv.synthesis_max_rows_to_process,
                    )
                    if selected_df is not None:
                        input_texts = []
                        for ix, row in selected_df.iterrows():
                            input_texts.append(row.to_json())
                st.text_area("AI record extraction guidance", key=sv.generation_guidance.key, value=sv.generation_guidance.value,
                            help="Guidance to the generative AI model about how data records should be extracted")
                
                generate = st.button('Extract record data')

            with d2:
                st.markdown("##### Extracted records")
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
                                st.dataframe(df, height=250, hide_index=True, use_container_width=True)
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
                            input_texts=input_texts,
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
                                    st.dataframe(df, height=600, use_container_width=True, hide_index=True)

                    for ix, record_array in enumerate(sv.record_arrays.value):
                        with dl_placeholders[ix]:
                            c1, c2 = st.columns([1, 1])
                            with c1:
                                name = sv.schema.value["title"].replace(" ", "_").lower() + "_[schema].json"
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
                                        label=f'Download {record_array}_[extracted].csv',
                                        data=sv.generated_dfs.value[record_array].to_csv(index=False, encoding='utf-8'),
                                        file_name=f'{record_array}_[extracted].csv',
                                        mime='text/csv',
                                        key=f'{record_array}_extracted_csv_download'
                                    )
    with mock_tab:
        example_outputs_ui.create_example_outputs_ui(mock_tab, workflow)