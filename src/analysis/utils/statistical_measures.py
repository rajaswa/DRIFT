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


def find_productivity(words, text, n=2):

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

    return prods
