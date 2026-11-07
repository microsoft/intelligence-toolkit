# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

# PDF settings live in the backend so the library and app share one definition
from intelligence_toolkit.helpers.constants import (  # noqa: F401
    PDF_ENCODING,
    PDF_MARGIN_INCHES,
    PDF_WKHTMLTOPDF_PATH,
)

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