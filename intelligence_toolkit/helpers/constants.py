# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

# PDF Generation
import os

PDF_ENCODING = "UTF-8"
PDF_WKHTMLTOPDF_PATH = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
ATTRIBUTE_VALUE_SEPARATOR = "=="

CACHE_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.getcwd()), "intelligence-toolkit-data", "cache"
)
# create a new directory if it does not exist
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

VECTOR_STORE_MAX_RETRIES = 5
VECTOR_STORE_MAX_RETRIES_WAIT_TIME = 1
