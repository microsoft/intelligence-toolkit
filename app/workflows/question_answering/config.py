# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

from toolkit.helpers.constants import CACHE_PATH

cache_dir = os.path.join(CACHE_PATH, "question_answering")
os.makedirs(cache_dir, exist_ok=True)
