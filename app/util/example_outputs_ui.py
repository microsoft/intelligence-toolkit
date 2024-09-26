import os
import streamlit as st
import pandas as pd
from json import loads, dumps
import PIL
import io

def create_example_outputs_ui(container, workflow):
    with container:
        workflow_home = f'example_outputs/{workflow}'
        mock_data_folders = [x for x in os.listdir(f'{workflow_home}') if x != 'example_format.json']
        example_json = loads(open(f'{workflow_home}/example_format.json', 'r').read())
        order = example_json['example_order']
        metadata = example_json['example_metadata']
        selected_data = st.selectbox('Select example', mock_data_folders)
        if selected_data != None:
            headings = [metadata[x]['heading'] for x in order]
            tabs = st.tabs(headings)
            for i, tab in enumerate(tabs):
                with tab:
                    this_key = order[i]
                    this_metadata = metadata[this_key]
                    st.markdown(this_metadata['description'])
                    this_type = this_metadata['type']
                    if type(this_key) == str:
                        filename = f'{selected_data}_{this_key}.{this_type}'
                        data_file = f'{workflow_home}/{selected_data}/{filename}'
                        if this_type == 'list':
                            seq = this_metadata['item_order']
                            index = 1
                            terminate = False
                            while not terminate:
                                
                                st.markdown('')
                                for sx, item in enumerate(seq):
                                    item_type = this_metadata['item_types'][item]
                                    filename = f'{selected_data}_{this_key}_{item}_{index}.{item_type}'
                                    if os.path.exists(f'{workflow_home}/{selected_data}/{filename}'):
                                        if sx == 0:
                                            st.divider()
                                            st.info(f'**Example {index}**')
                                        if item_type == 'csv':
                                            df = pd.read_csv(f'{workflow_home}/{selected_data}/{filename}')
                                            st.dataframe(df, height=450, hide_index=True, use_container_width=True)
                                        elif item_type == 'json':
                                            js = loads(open(f'{workflow_home}/{selected_data}/{filename}', 'r').read())
                                            st.write(js)
                                        elif item_type == 'md':
                                            md = open(f'{workflow_home}/{selected_data}/{filename}', 'r').read()
                                            st.markdown(md)
                                        elif item_type == 'png':
                                            with open(f'{workflow_home}/{selected_data}/{filename}', 'rb') as f:
                                                im = PIL.Image.open(io.BytesIO(f.read()))    
                                                st.image(im)
                                    else:
                                        terminate = True
                                        break
                                index += 1
                        else:
                            mime = ''
                            data = ''
                            if this_type == 'csv':
                                mime = 'text/csv'
                                df = pd.read_csv(data_file)
                                data = df.to_csv(index=False, encoding='utf-8')
                                st.dataframe(df, height=450, hide_index=True, use_container_width=True)
                            elif this_type == 'json':
                                mime = 'application/json'
                                js = loads(open(data_file, 'r').read())
                                data = dumps(js, indent=2)
                                st.write(js)
                            elif this_type == 'md':
                                mime = 'text/markdown'
                                md = open(data_file, 'r').read()
                                st.markdown(md)
                            
                            st.download_button(
                                label=f'Download {filename}',
                                data=data,
                                file_name=filename,
                                mime=mime,
                            )