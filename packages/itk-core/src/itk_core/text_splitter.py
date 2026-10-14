# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import semchunk
import tiktoken

from .defaults import CHUNK_SIZE, DEFAULT_ENCODING, DEFAULT_LLM_MODEL


class TextSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, model: str = DEFAULT_LLM_MODEL):
        self.chunk_size = chunk_size
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING)

        self._chunk = semchunk.chunkerify(
            encoding, chunk_size
        )

    def split(self, text: str):  # -> Any:
        return self._chunk(text)
