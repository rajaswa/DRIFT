import json
import operator
import os
from itertools import islice

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import FreqDist, ngrams


def find_freq(text, n=1):
    ngrams_lst = list(ngrams(text.split(), n))
    gram_count_mapping = dict(FreqDist(ngrams_lst))
    gram_count_mapping = {" ".join(k): v for k, v in gram_count_mapping.items()}
    return gram_count_mapping


def find_norm_freq(text, n=1):
    gram_count_mapping = find_freq(text=text, n=n)
    norm_factor = sum(list(gram_count_mapping.values()))
    gram_count_mapping = {k: v / norm_factor for k, v in gram_count_mapping.items()}
    return gram_count_mapping


def find_productivity(words, text, n=2, save_load_path=None, overwrite=False):
    if (
        save_load_path is not None
        and os.path.isfile(save_load_path)
        and not (overwrite)
    ):
        print(f"Loading Productivity Dictionary from {save_load_path}")
        with open(save_load_path, "r") as prod_f:
            word_prod_mapping = json.load(prod_f)
        return word_prod_mapping

    fdist = find_freq(text=text, n=n)
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

        p_m_is = [i / sum(f_m_is) for i in f_m_is]
        prod = -np.sum(np.multiply(np.log2(p_m_is), p_m_is))
        prods[word] = prod

    print(f"Saving Productivity Dictionary at {save_load_path}")
    with open(save_load_path, "w") as prod_f:
        json.dump(prods, prod_f, indent=4)
    return prods


def freq_top_k(text, save_load_path=None, top_k=200, n=1, overwrite=False):
    if (
        save_load_path is not None
        and os.path.isfile(save_load_path)
        and not (overwrite)
    ):
        print("Loading Frequency Dictionary from {save_load_path}")
        with open(save_load_path, "r") as freq_dict:
            sorted_gram_count_mapping = json.load(freq_dict)
    else:
        print("Computing Frequency Dictionary and Saving at {save_load_path}")
        gram_count_mapping = find_freq(text=text, n=n)
        sorted_gram_count_tuple = sorted(
            gram_count_mapping.items(), key=operator.itemgetter(1), reverse=True
        )
        sorted_gram_count_mapping = {k: v for k, v in sorted_gram_count_tuple}
        if save_load_path is not None:
            with open(save_load_path, "w") as freq_json:
                json.dump(sorted_gram_count_mapping, freq_json, indent=4)

        if top_k < len(sorted_gram_count_mapping):
            sorted_gram_count_mapping = dict(
                islice(sorted_gram_count_mapping.items(), top_k)
            )

    return sorted_gram_count_mapping


def norm_freq_top_k(text, save_load_path=None, top_k=200, n=1, overwrite=False):
    if (
        save_load_path is not None
        and os.path.isfile(save_load_path)
        and not (overwrite)
    ):
        print(f"Loading Normalised Frequency Dictionary from {save_load_path}")

        with open(save_load_path, "r") as norm_freq_dict:
            sorted_gram_count_mapping = json.load(norm_freq_dict)
    else:
        print(
            f"Computing Normalised Frequency Dictionary and Saving at {save_load_path}"
        )
        gram_count_mapping = find_norm_freq(text=text, n=n)
        sorted_gram_count_tuple = sorted(
            gram_count_mapping.items(), key=operator.itemgetter(1), reverse=True
        )
        sorted_gram_count_mapping = {k: v for k, v in sorted_gram_count_tuple}
        if save_load_path is not None:
            with open(save_load_path, "w") as norm_freq_json:
                json.dump(sorted_gram_count_mapping, norm_freq_json, indent=4)

        if top_k < len(sorted_gram_count_mapping):
            sorted_gram_count_mapping = dict(
                islice(sorted_gram_count_mapping.items(), top_k)
            )

    return sorted_gram_count_mapping
