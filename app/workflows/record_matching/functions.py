# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st
from util.openai_wrapper import UIOpenAIConfiguration
from util.session_variables import SessionVariables
from workflows.record_matching import config

from python.AI.embedder import Embedder

sv_home = SessionVariables("home")


def embedder() -> Embedder:
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        return Embedder(
            configuration=ai_configuration,
            db_name=config.cache_name,
            local=sv_home.local_embeddings.value,
        )
    except Exception as e:
        st.error(f"Error creating connection: {e}")
        st.stop()


def convert_to_sentences(df, skip):
    sentences = []
    cols = df.columns
    for row in df.iter_rows(named=True):
        sentence = ""
        for field in cols:
            if field not in skip:
                val = str(row[field]).upper()
                if val == "NAN":
                    val = ""
                sentence += field.upper() + ": " + val + "; "
        sentences.append(sentence.strip())
    return sentences
