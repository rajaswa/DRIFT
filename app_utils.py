import inspect
import os

import pandas as pd

from src.utils.statistics import find_productivity


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
    start_year, end_year, selected_ngrams, years, data_path, n
):
    yearss = []
    words = []
    prodss = []
    start_year_idx = years.index(start_year)
    end_year_idx = years.index(end_year)
    for year_idx in range(start_year_idx, end_year_idx + 1):
        year = years[year_idx]
        year_text = read_text_file(data_path, year)
        prods = find_productivity(selected_ngrams, year_text, n)
        for word, productivity in prods.items():
            yearss.append(year)
            words.append(word)
            prodss.append(productivity)
    productivity_df = pd.DataFrame.from_dict(
        {"Year": yearss, "Word": words, "Productivity": prodss}
    )
    return productivity_df


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
