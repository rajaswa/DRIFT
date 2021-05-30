import json
import operator
import os
from itertools import islice

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import ngrams

from src.analysis.utils import *


def save_freq_dict(text, save_path, n=1):
    print("Computing Frequency Dictionary and Saving at {save_path}")
    gram_count_mapping = find_freq(text=text, n=n)
    sorted_gram_count_tuple = sorted(
        gram_count_mapping.items(), key=operator.itemgetter(1), reverse=True
    )
    sorted_gram_count_mapping = {k: v for k, v in sorted_gram_count_tuple}
    if save_path is not None:
        with open(save_path, "w") as freq_json:
            json.dump(sorted_gram_count_mapping, freq_json, indent=4)

    return sorted_gram_count_mapping


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
        sorted_gram_count_mapping = save_freq_dict(text, save_load_path, n)

    if top_k < len(sorted_gram_count_mapping):
        sorted_gram_count_mapping = dict(
            islice(sorted_gram_count_mapping.items(), top_k)
        )

    return sorted_gram_count_mapping


def save_norm_freq_dict(text, save_path=None, n=1):
    print(f"Computing Normalised Frequency Dictionary and Saving at {save_path}")
    gram_count_mapping = find_norm_freq(text=text, n=n)
    sorted_gram_count_tuple = sorted(
        gram_count_mapping.items(), key=operator.itemgetter(1), reverse=True
    )
    sorted_gram_count_mapping = {k: v for k, v in sorted_gram_count_tuple}
    if save_path is not None:
        with open(save_path, "w") as norm_freq_json:
            json.dump(sorted_gram_count_mapping, norm_freq_json, indent=4)

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
        sorted_gram_count_mapping = save_norm_freq_dict(text, save_load_path, n)

    if top_k < len(sorted_gram_count_mapping):
        sorted_gram_count_mapping = dict(
            islice(sorted_gram_count_mapping.items(), top_k)
        )

    return sorted_gram_count_mapping
