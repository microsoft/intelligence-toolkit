# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from util.openai_wrapper import UIOpenAIConfiguration
from workflows.record_matching import config

from python.AI.embedder import Embedder

ai_configuration = UIOpenAIConfiguration().get_configuration()
embedder = Embedder(ai_configuration, config.cache_dir)

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