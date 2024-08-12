# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

from toolkit.helpers.constants import CACHE_PATH

list_sep = "; "
max_rows_to_show = 1000
entity_label = "Entity"
cache_dir = os.path.join(CACHE_PATH, "record_matching")
outputs_dir = os.path.join(cache_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)
