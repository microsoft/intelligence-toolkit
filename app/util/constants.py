# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

# PDF Generation
PDF_MARGIN_INCHES = 0.75
PDF_ENCODING = "UTF-8"
PDF_WKHTMLTOPDF_PATH = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
MAX_SIZE_EMBEDDINGS_KEY = "max_embedding"
LOCAL_EMBEDDING_MODEL_KEY = "local_embedding_model"


FILE_ENCODING_OPTIONS = [
    "unicode-escape",
    "utf-8",
    "utf-8-sig",
    "Windows-1252",
    "ISO-8859-1",
    "ASCII",
    "Big5",
    "Shift JIS",
]
FILE_ENCODING_DEFAULT = "unicode-escape"