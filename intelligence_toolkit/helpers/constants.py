# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

# PDF Generation
import os

PDF_ENCODING = "UTF-8"
# Fallback location when wkhtmltopdf is not on PATH; override with WKHTMLTOPDF_PATH
PDF_WKHTMLTOPDF_PATH = os.environ.get(
    "WKHTMLTOPDF_PATH",
    "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
    if os.name == "nt"
    else "/usr/bin/wkhtmltopdf",
)
PDF_MARGIN_INCHES = 0.75
ATTRIBUTE_VALUE_SEPARATOR = "=="

from itk_core.paths import CACHE_PATH

# The app stores per-workflow data under CACHE_PATH and expects it to exist.
os.makedirs(CACHE_PATH, exist_ok=True)
