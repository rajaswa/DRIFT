import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import ngrams


def find_freq(text_file):
    with open(text_file, "r") as f:
        text = f.read()
    unique_words, counts = np.unique(text.split(), return_counts=True)

    word_count_mapping = {word: ct for word, ct in zip(unique_words, counts)}

    return word_count_mapping


def plot_freq(word, year_wise_word_count_mappings):
    years = []
    freqs = []
    for year in year_wise_word_count_mappings:
        word_count_mapping = year_wise_word_count_mappings[year]
        if word in word_count_mapping:
            freqs.append(word_count_mapping[word])
        else:
            freqs.append(0)
        years.append(year)

    plt.plot(years, freqs)
    plt.title("Frequency Plot")
    plt.xlabel("Years")
    plt.ylabel("Frequencies")
    plt.show()


def find_productivity(word, text_file, n=2):
    with open(text_file, "r") as f:
        text = f.read()

    ngrams_lst = list(ngrams(text.split(), n))
    fdist = dict(nltk.FreqDist(ngrams_lst))

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

    return prod


def plot_prod(word, year_wise_prods_mappings):
    years = []
    prods = []
    for year in year_wise_prods_mappings:
        word_prod_mapping = year_wise_prods_mappings[year]
        if word in word_prod_mapping:
            prods.append(word_prod_mapping[word])
        else:
            prods.append(0)
        years.append(year)

    plt.plot(years, freqs)
    plt.title("Productivity Plot")
    plt.xlabel("Years")
    plt.ylabel("Productivities")
    plt.show()
