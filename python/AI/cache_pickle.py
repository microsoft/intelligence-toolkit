# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
import pickle

from .defaults import PICKLE_FILE_NAME


class CachePickle:
    
    def __init__(self, file_name:str = PICKLE_FILE_NAME, path = None):
        if not path:
            if os.environ.get("CACHE_DIR"):
                path = os.environ.get("CACHE_DIR")
            else:
                path = os.getcwd()
        self.file_path = os.path.join(path, file_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            with open(self.file_path, 'wb') as f:
                f.write(b'')

    def get_all(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                if os.path.getsize(self.file_path) > 0:
                    return pickle.load(f)
                else:
                    return {}
        return {}

    def save(self, items: dict, max_size=0):
        count = len(items)
        if max_size > 0 and count > max_size:
            items = dict(list(items.items())[-max_size:])
        with open(self.file_path, 'wb') as f:
            pickle.dump(items, f)

    def reset(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def get(self, hsh, items):
        if hsh in items:
            return items[hsh]
        return {}
