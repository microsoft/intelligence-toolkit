# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

# PDF Generation
import os

PDF_ENCODING = "UTF-8"
PDF_WKHTMLTOPDF_PATH = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
EMBEDDINGS_FILE_NAME = "embeddings.pickle"

EMBEDDINGS_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.getcwd()), "intelligence-toolkit", "cache"
)
# create a new directory if it does not exist
if not os.path.exists(EMBEDDINGS_PATH):
    os.makedirs(EMBEDDINGS_PATH)
