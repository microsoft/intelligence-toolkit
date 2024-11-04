# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

import plotly.express as px

from intelligence_toolkit.helpers.constants import CACHE_PATH

cache_dir = os.path.join(CACHE_PATH, "anonymize_case_data")
outputs_dir = os.path.join(cache_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

att_separator = ";"
val_separator = ":"
