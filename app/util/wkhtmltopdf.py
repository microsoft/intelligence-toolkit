import os
import pdfkit

# Specify the name of the executable
executable = 'wkhtmltopdf'

# Check if the executable is in the system PATH
def is_in_path(executable):
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(os.path.join(path.strip('"'), executable)):
            return True
    return False

def config_pdfkit():
    path_wkhtmltopdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
    # Verify if wkhtmltopdf is in PATH
    if is_in_path(executable):
        path_wkhtmltopdf=''
    else:
        path_wkhtmltopdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'

    return pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

pdfkit_options = {
    'encoding': 'UTF-8',
    'enable-local-file-access': True,
}