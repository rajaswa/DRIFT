import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import ngrams

from src.analysis.utils.statistical_measures import plot_freq, plot_prod


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
