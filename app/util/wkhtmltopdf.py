# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import os
import pdfkit
import util.constants as constants

# Specify the name of the executable
# Check if the executable is in the system PATH
def is_in_path(executable):
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(os.path.join(path.strip('"'), executable)):
            return True
    return False

def config_pdfkit():
    path_wkhtmltopdf = constants.PDF_WKHTMLTOPDF_PATH

    # Verify if wkhtmltopdf is in PATH
    if is_in_path('wkhtmltopdf'):
        path_wkhtmltopdf=''

    return pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

pdfkit_options = {
    'margin-top': f'{constants.PDF_MARGIN_INCHES}in',
    'margin-right': f'{constants.PDF_MARGIN_INCHES}in',
    'margin-bottom': f'{constants.PDF_MARGIN_INCHES}in',
    'margin-left': f'{constants.PDF_MARGIN_INCHES}in',
    'encoding': constants.PDF_ENCODING,
    'enable-local-file-access': True,
}