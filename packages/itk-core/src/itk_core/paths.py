# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

# Default location for embedding stores. Not created at import time; consumers
# (e.g. VectorStore) create it on first use.
CACHE_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.getcwd()), "intelligence-toolkit-data", "cache"
)
