import os

import numpy as np
import streamlit as st
from gensim import corpora, models


def lda_basic(list_of_list_of_tokens, num_topics):
    dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
    dictionary_LDA.filter_extremes(no_below=2)
    corpus = [
        dictionary_LDA.doc2bow(list_of_tokens)
        for list_of_tokens in list_of_list_of_tokens
    ]
    lda_model = models.LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary_LDA,
        passes=20,
        alpha="auto",
        eta="auto",
        random_state=42,
    )

    cm = models.CoherenceModel(model=lda_model, corpus=corpus, coherence="u_mass")
    coherence = cm.get_coherence()  # get coherence value

    return corpus, lda_model, coherence


@st.cache(persist=eval(os.getenv("PERSISTENT")))
def extract_topics_lda(text_file_paths, num_topics=0, num_words=10):
    list_of_list_of_tokens = []
    for text_file_path in text_file_paths:
        with open(text_file_path, "r") as f:
            text = f.read()
            doc_words = text.replace("\n", " ").split(" ")
            list_of_list_of_tokens.append(doc_words)

    if num_topics == 0:
        coherence_scores = []
        range_of_topics = list(range(5, 31))
        for num_topics_in_lst in range_of_topics:
            _, lda_model, coherence = lda_basic(
                list_of_list_of_tokens, num_topics_in_lst
            )
            coherence_scores.append(abs(coherence))
        num_topics = 3 + np.argmin(coherence_scores)

    corpus, lda_model, _ = lda_basic(list_of_list_of_tokens, num_topics)

    year_wise_topics = []
    for i, list_of_tokens in enumerate(list_of_list_of_tokens):
        year_wise_topics.append(lda_model[corpus[i]])

    return year_wise_topics, lda_model.show_topics(
        formatted=True, num_topics=num_topics, num_words=num_words
    )
