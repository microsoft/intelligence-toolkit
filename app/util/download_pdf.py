import tempfile
import markdown2
import pdfkit
from util.wkhtmltopdf import config_pdfkit, pdfkit_options
import streamlit as st

css = '''body {
    font-family: 'helvetica';
}
'''
#itk-label
text_label = 'Report generated using Intelligence Toolkit (https://aka.ms/itk)'

def add_download_pdf(name, text, button_text='Download PDF', is_markdown=True, disabled=False):
    # Ensure the file name ends with '.pdf'
    if not name.endswith('.pdf'):
        name += '.pdf'
    text = f'{text}<hr>{text_label}'
    # Convert text to HTML if it's in Markdown format
    text = markdown2.markdown(text) if is_markdown else text

    # Generate PDF from HTML string
    config_pdf = config_pdfkit()
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        text = f'<style>{css}</style>' + text

        pdfkit.from_string(text, temp_file.name, options=pdfkit_options, configuration=config_pdf)
        
        # Provide download button for the generated PDF
        with open(temp_file.name, 'rb') as f:
            st.download_button(button_text, f, file_name=name, mime='application/pdf', disabled=disabled)
