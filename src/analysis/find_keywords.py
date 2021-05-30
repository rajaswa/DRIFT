import json
import operator
import os
from itertools import islice

from ..utils import find_freq, find_norm_freq, save_json


def freq_top_k(text, save_load_path=None, top_k=200, n=1, overwrite=False):
    if (
        save_load_path is not None
        and os.path.isfile(save_load_path)
        and not (overwrite)
    ):
        print(f"Loading Frequency Dictionary from {save_load_path}")
        with open(save_load_path, "r") as freq_dict:
            sorted_gram_count_mapping = json.load(freq_dict)
    else:
        sorted_gram_count_mapping = find_freq(text, n=n, sort=True)

        if top_k < len(sorted_gram_count_mapping):
            sorted_gram_count_mapping = dict(
                islice(sorted_gram_count_mapping.items(), top_k)
            )
        print(f"Saving Frequency Dictionary at {save_load_path}")
        save_json(sorted_gram_count_mapping, save_load_path)

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
        sorted_gram_count_mapping = find_norm_freq(text, n=n, sort=True)

        if top_k < len(sorted_gram_count_mapping):
            sorted_gram_count_mapping = dict(
                islice(sorted_gram_count_mapping.items(), top_k)
            )
        print(f"Saving Norm Frequency Dictionary at {save_load_path}")
        save_json(sorted_gram_count_mapping, save_load_path)

    return sorted_gram_count_mapping
