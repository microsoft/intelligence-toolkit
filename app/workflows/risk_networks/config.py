# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

from python.helpers.constants import EMBEDDINGS_PATH

att_val_sep = "=="
list_sep = "; "
max_rows_to_show = 1000
entity_label = "ENTITY"
cache_dir = os.path.join(EMBEDDINGS_PATH, "risk_networks")
outputs_dir = os.path.join(cache_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)
