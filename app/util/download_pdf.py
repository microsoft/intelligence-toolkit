import tempfile

import markdown2
import pdfkit
import streamlit as st
from javascript.styles import style_pdf
from util.wkhtmltopdf import config_pdfkit, pdfkit_options

# itk-label
text_label = "Report generated using Intelligence Toolkit (https://aka.ms/itk)"


def add_download_pdf(
    name, text, button_text="Download PDF", is_markdown=True, disabled=False
):
    if not name.endswith(".pdf"):
        name += ".pdf"
    # Convert text to HTML if it's in Markdown format
    text = markdown2.markdown(text) if is_markdown else text
    text = f"<style>{style_pdf}</style> \n\n {text} <hr>{text_label}"

    # Generate PDF from HTML string
    config_pdf = config_pdfkit()

    with st.spinner("Preparing PDF download..."):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            pdfkit.from_string(
                text, temp_file.name, options=pdfkit_options, configuration=config_pdf
            )

            # Provide download button for the generated PDF
            with open(temp_file.name, "rb") as f:
                st.download_button(
                    button_text,
                    f,
                    file_name=name,
                    mime="application/pdf",
                    disabled=disabled,
                )
