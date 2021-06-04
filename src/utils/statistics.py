from itertools import islice

import numpy as np
import streamlit as st
from nltk import FreqDist, ngrams


def find_ngrams_for_sentences(sentences, n=1):
    sentence_list = sentences.split("\n")
    ngrams_list = []
    for sentence in sentence_list:
        ngrams_list += list(ngrams(sentence.split(), n))
    return ngrams_list


def find_freq(text, n=1, sort=False):
    ngrams_lst = find_ngrams_for_sentences(text, n)
    gram_count_mapping = dict(FreqDist(ngrams_lst))
    gram_count_mapping = {" ".join(k): v for k, v in gram_count_mapping.items()}
    if sort:
        sorted_gram_count_tuple = sorted(
            gram_count_mapping.items(), key=lambda x: x[1], reverse=True
        )
        gram_count_mapping = {k: v for k, v in sorted_gram_count_tuple}
    return gram_count_mapping


def find_norm_freq(text, n=1, sort=False):
    gram_count_mapping = find_freq(text=text, n=n, sort=sort)
    norm_factor = sum(list(gram_count_mapping.values()))
    gram_count_mapping = {k: v / norm_factor for k, v in gram_count_mapping.items()}
    return gram_count_mapping


def find_productivity(words, text, n=1):
    fdist = find_freq(text=text, n=n, sort=True)
    ngrams_lst = list(fdist.keys())
    prods = {}
    for word in words:
        ngrams_lst_having_word = []
        for ngram in ngrams_lst:
            if word in ngram:
                ngrams_lst_having_word.append(ngram)

        ngrams_lst_having_word = list(set(ngrams_lst_having_word))

        # find f_m_i for every ngram
        f_m_is = []
        for i in range(len(ngrams_lst_having_word)):
            f_m_i = fdist[ngrams_lst_having_word[i]]
            f_m_is.append(f_m_i)

        total_f_m_i = sum(f_m_is)
        p_m_is = [i / total_f_m_i for i in f_m_is]
        prod = -np.sum(np.multiply(np.log2(p_m_is), p_m_is))
        prods[word] = prod

    return prods


@st.cache(persist=True)
def freq_top_k(text, top_k=20, n=1, normalize=False):
    if normalize:
        sorted_gram_count_mapping = find_norm_freq(text, n=n, sort=True)
    else:
        sorted_gram_count_mapping = find_freq(text, n=n, sort=True)

    if top_k < len(sorted_gram_count_mapping):
        sorted_gram_count_mapping = dict(
            islice(sorted_gram_count_mapping.items(), top_k)
        )

    return sorted_gram_count_mapping
