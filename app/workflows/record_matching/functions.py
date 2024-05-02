# Copyright (c) 2024 Microsoft Corporation. All rights reserved.

from AI.embedder import Embedder
from workflows.record_matching import config

embedder = Embedder(pickle_path=config.cache_dir)

def convert_to_sentences(df, skip):
    sentences = []
    cols = df.columns 
    for row in df.iter_rows(named=True):
        sentence = ''
        for field in cols:
            if field not in skip:
                val = str(row[field]).upper()
                if val == 'NAN':
                    val = ''
                sentence += field.upper() + ': ' + val + '; '
        sentences.append(sentence.strip())
    return sentences