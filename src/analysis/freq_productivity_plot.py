import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import ngrams


def plot_freq(word, year_wise_word_count_mappings, save_path, keyword_type):
    years = []
    freqs = []
    for year in year_wise_word_count_mappings:
        word_count_mapping = year_wise_word_count_mappings[year]
        if word in word_count_mapping:
            freqs.append(word_count_mapping[word])
        else:
            freqs.append(0)
        years.append(year)

    plt.clf()
    plt.plot(years, freqs)
    if keyword_type == "freq":
        plt.title(f"{word} Frequency Plot")
        plt.ylabel("Frequencies")
    elif keyword_type == "norm_freq":
        plt.title(f"{word} Normalised Frequency Plot")
        plt.ylabel("Normalised Frequencies")
    plt.xlabel("Years")

    plt.savefig(save_path)


def plot_prod(word, year_wise_prods_mappings, save_path):
    years = []
    prods = []
    for year in year_wise_prods_mappings:
        word_prod_mapping = year_wise_prods_mappings[year]
        if word in word_prod_mapping:
            prods.append(word_prod_mapping[word])
        else:
            prods.append(0)
        years.append(year)
    plt.clf()
    plt.plot(years, prods)
    plt.title("Productivity Plot")
    plt.xlabel("Years")
    plt.ylabel("Productivities")
    plt.savefig(save_path)
