import inspect
import os

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from src.analysis.similarity_acc_matrix import compute_acc_between_years
from src.analysis.track_trends_sim import compute_similarity_matrix_years
from src.utils.misc import get_sub, get_tail_from_data_path
from src.utils.statistics import find_freq, find_norm_freq, find_productivity


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_values_from_indices(lst, idx_list):
    return [lst[idx] for idx in idx_list], [
        lst[idx] for idx in range(len(lst)) if idx not in idx_list
    ]


def get_years_from_data_path(data_path):
    years = sorted(
        [fil.split(".")[0] for fil in os.listdir(data_path) if fil != "compass.txt"]
    )
    return years


def get_productivity_for_range(
    start_year, end_year, selected_ngrams, years, data_path, n, normalize=False
):
    yearss = []
    words = []
    prodss = []

    start_year_idx = years.index(start_year)
    end_year_idx = years.index(end_year)
    for year_idx in range(start_year_idx, end_year_idx + 1):
        year = years[year_idx]
        year_text = read_text_file(data_path, year)
        prods = find_productivity(selected_ngrams, year_text, n, normalize)

        for word, productivity in prods.items():
            yearss.append(year)
            words.append(word)
            prodss.append(productivity)
    productivity_df = pd.DataFrame.from_dict(
        {"Year": yearss, "Word": words, "Productivity": prodss}
    )
    return productivity_df


def get_frequency_for_range(
    start_year, end_year, selected_ngrams, years, data_path, n, normalize=False
):
    yearss = []
    words = []
    freqss = []
    start_year_idx = years.index(start_year)
    end_year_idx = years.index(end_year)
    for year_idx in range(start_year_idx, end_year_idx + 1):
        year = years[year_idx]
        year_text = read_text_file(data_path, year)
        if normalize:
            freqs = find_norm_freq(year_text, n=n, sort=False)
        else:
            freqs = find_freq(year_text, n=n, sort=False)
        for word in selected_ngrams:
            yearss.append(year)
            words.append(word)
            freqss.append(freqs[word] if word in freqs else 0)
    frequency_df = pd.DataFrame.from_dict(
        {"Year": yearss, "Word": words, "Frequency": freqss}
    )
    return frequency_df


def get_acceleration_bw_models(
    year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
):
    model_path1 = os.path.join(model_path, year1 + ".model")
    model_path2 = os.path.join(model_path, year2 + ".model")

    word_pairs, em1, em2 = compute_acc_between_years(
        selected_ngrams,
        model_path1,
        model_path2,
        all_model_vectors=all_model_vectors,
        top_k_acc=top_k_acc,
        skip_same_word_pairs=True,
        skip_duplicates=True,
    )
    return word_pairs, em1, em2


def get_word_pair_sim_bw_models(
    year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
):
    word_pairs, em1, em2 = get_acceleration_bw_models(
        year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
    )
    word_pair_sim_df = pd.DataFrame(
        list(word_pairs.items()), columns=["Word Pair", "Acceleration"]
    )
    word_pair_sim_df = word_pair_sim_df.sort_values(by="Acceleration", ascending=False)

    word_pair_sim_df_words = []
    for word1, word2 in word_pair_sim_df["Word Pair"].values:
        if word1 not in word_pair_sim_df_words:
            word_pair_sim_df_words.append(word1)
        if word2 not in word_pair_sim_df_words:
            word_pair_sim_df_words.append(word2)
    return word_pair_sim_df, word_pair_sim_df_words


def read_text_file(data_path, name):
    with open(os.path.join(data_path, name + ".txt"), encoding="utf-8") as f:
        words = f.read()
    return words


def get_curve_hull_objects(embeds, labels):
    label_to_point_map = {}
    for idx, label in enumerate(labels):
        if label not in label_to_point_map:
            label_to_point_map[label] = [embeds[idx]]
        else:
            label_to_point_map[label] += [embeds[idx]]

    label_to_vertices_map = {}

    for label, label_points in label_to_point_map.items():
        label_points = np.array(label_points)
        hull = ConvexHull(label_points)
        vertices = label_points[hull.vertices]
        label_to_vertices_map[label] = vertices

    return label_to_vertices_map


def create_word_to_entry_dict(word, model_path, sim_dict):
    return {
        "{}{}".format(word, get_sub(get_tail_from_data_path(model_path))): [
            "{}{} ({})".format(
                k.split("_")[0],
                get_sub(get_tail_from_data_path(k.split("_")[1])),
                round(float(sim_dict[k]), 2),
            )
            for k in sim_dict
        ]
    }


def get_dict_with_new_words(model_paths, selected_ngram, top_k_sim):
    sim_dict = compute_similarity_matrix_years(
        model_paths, selected_ngram, top_k_sim=top_k_sim
    )
    return create_word_to_entry_dict(selected_ngram, model_paths[0], sim_dict)


def word_to_entry_dict(word_year, year1, year2, years, stride, top_k_sim, model_path):
    # TO-DO: Need better logic here. `eighy-eighty2008` becomes `eightyeight2008`
    # TO-DO: `2016a2020` appears as `a`, `20162020`
    word_pure = "".join([i for i in word_year if not i.isdigit()]).strip()
    year = int(
        get_sub("".join([i for i in word_year if i.isdigit()]).strip(), rev=True)
    ) - int(year1)

    if str(year + int(year1)) == year2:
        return {}

    else:
        model_paths = [
            os.path.join(model_path, str(yr) + ".model")
            for yr in years[year : min(stride + year + 1, len(years))]
        ]
        return get_dict_with_new_words(model_paths, word_pure, top_k_sim)
