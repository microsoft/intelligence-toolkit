import os
import streamlit as st
from json import dumps, loads
import app.workflows.generate_record_data.variables as bds_variables
import app.workflows.generate_record_data.functions as bds_functions
import toolkit.generate_record_data.schema_builder as schema_builder
from toolkit.generate_record_data import get_readme as get_intro

def create(sv: bds_variables.SessionVariables, workflow: None):
    intro_tab, schema_tab, generator_tab = st.tabs(['Generate Record Data workflow:', 'Prepare data schema', 'Generate sample data'])
    with intro_tab:
        st.markdown(get_intro())
    with schema_tab:
        form, preview = st.columns([1, 1])
        with form:
            file = st.file_uploader('Upload schema', type=['json'], key='schema_uploader')
            if file is not None:
                jsn = loads(file.read())
                sv.schema.value = jsn
            st.markdown('### Edit Data Schema')
            bds_functions.generate_form_from_json_schema(
                global_schema=sv.schema.value
            )
        with preview:
            st.markdown('### Preview')
            schema_tab, object_tab = st.tabs(['JSON Schema', 'Sample Object'])
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
            st.download_button(
                label='Download schema',
                data=dumps(sv.schema.value, indent=2),
                file_name='schema.json',
                mime='application/json'
            )
