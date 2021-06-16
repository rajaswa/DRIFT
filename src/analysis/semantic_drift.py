import os

import numpy as np
import streamlit as st
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import (
    cosine_distances,
    cosine_similarity,
    euclidean_distances,
)

from src.utils.words import get_word_embeddings

from ..utils import intersection


# @st.cache(persist=eval(os.getenv("PERSISTENT")))
# def find_most_similar_words(
#     words,
#     year_model_path,
#     compass_model_path,
#     top_k_sim=10,
# ):
#     compass_model = Word2Vec.load(compass_model_path)
#     compass_words = list(compass_model.wv.vocab.keys())
#     compass_vectors = compass_model.wv.vectors

#     word_vectors = get_word_embeddings(year_model_path, words)
#     compass_word_vectors_for_year = get_word_embeddings(year_model_path, compass_words)

#     sim_matrix = cosine_similarity(word_vectors, compass_vectors)
#     top_sims = np.argsort(sim_matrix, axis=1)
#     top_sims = top_sims[:, -top_k_sim:]

#     # Find most similar compass words to each of these words
#     most_sim_words = [
#         [compass_words[top_sims[i][j]] for j in range(top_sims[i].shape[0])]
#         for i in range(top_sims.shape[0])
#     ]

#     most_sim_scores = [
#         [sim_matrix[i][top_sims[i][j]] for j in range(top_sims[i].shape[0])]
#         for i in range(top_sims.shape[0])
#     ]

#     sim_word_embs = [
#         [
#             compass_word_vectors_for_year[compass_words.index(sim_word)]
#             for sim_word in most_sim_words_per_word
#         ]
#         for most_sim_words_per_word in most_sim_words
#     ]

#     sim_dict = {}
#     for idx, word in enumerate(words):
#         sim_dict[word] = {}
#         sim_dict[word]["emb"] = word_vectors[idx]
#         sim_dict[word]["sim_words"] = most_sim_words[idx]
#         sim_dict[word]["sim_embs"] = sim_word_embs[idx]
#         sim_dict[word]["sim_scores"] = most_sim_scores[idx]
#     return sim_dict


# @st.cache(persist=eval(os.getenv("PERSISTENT")))
# def find_most_drifted_words(
#     words, year_model_paths, compass_model_path, top_k_sim=10, top_k_drift=5
# ):

#     sim_dicts = [
#         find_most_similar_words(words, year_model_path, compass_model_path, top_k_sim)
#         for year_model_path in year_model_paths
#     ]
#     scores = {}
#     for word in sim_dicts[0].keys():
#         scores[word] = 0
#         year_1_sim_words = sim_dicts[0][word]["sim_words"]
#         year_last_sim_words = sim_dicts[-1][word]["sim_words"]
#         year_1_sim_scores = sim_dicts[0][word]["sim_scores"]
#         year_last_sim_scores = sim_dicts[-1][word]["sim_scores"]
#         common_sim_words = intersection(year_1_sim_words, year_last_sim_words)

#         year_1_score_map = dict(zip(year_1_sim_words, year_1_sim_scores))
#         year_last_score_map = dict(zip(year_last_sim_words, year_last_sim_scores))

#         for sim_word in common_sim_words:
#             scores[word] += abs(
#                 year_last_score_map[sim_word] - year_1_score_map[sim_word]
#             )

#         scores[word] += len(year_1_sim_words + year_last_sim_words) - 2 * len(
#             common_sim_words
#         )

#     sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
#         :top_k_drift
#     ]

#     words_for_plotting = []
#     embs_for_plotting = []
#     years = [year_path.split("/")[-1].split(".")[0] for year_path in year_model_paths]
#     for idx, year_sim_dict in enumerate(sim_dicts):
#         for word, score in sorted_scores:
#             words_for_plotting.append(word + "_" + years[idx])
#             embs_for_plotting.append(year_sim_dict[word]["emb"])

#         sim_word_to_sim_embds = dict(
#             zip(year_sim_dict[word]["sim_words"], year_sim_dict[word]["sim_embs"])
#         )
#         for sim_word in year_sim_dict[word]["sim_words"]:
#             new_sim_word = sim_word + "_" + years[idx]
#             if new_sim_word not in words_for_plotting:
#                 words_for_plotting.append(new_sim_word)
#                 embs_for_plotting.append(sim_word_to_sim_embds[sim_word])

#     return words_for_plotting, embs_for_plotting


@st.cache(persist=eval(os.getenv("PERSISTENT")))
def find_most_drifted_words(
    year_1_path, year_2_path, words=[], top_k_drift=10, distance_measure="euclidean"
):
    assert distance_measure in [
        "euclidean",
        "cosine",
    ], 'distance_measure should be one of ["euclidean", "cosine"]'
    year_1_model = Word2Vec.load(year_1_path)
    year_2_model = Word2Vec.load(year_2_path)

    if words == []:
        year_1_vocab = list(year_1_model.wv.vocab.keys())
        year_2_vocab = list(year_2_model.wv.vocab.keys())
    else:
        year_1_vocab = [word for word in words if word in year_1_model.wv.vocab]
        year_2_vocab = [word for word in words if word in year_2_model.wv.vocab]
    common_vocab = list(set(year_1_vocab).intersection(set(year_2_vocab)))

    distance_dict = {}
    for word in common_vocab:
        if distance_measure == "euclidean":
            distance_dict[word] = euclidean_distances(
                year_1_model.wv[word].reshape(1, -1),
                year_2_model.wv[word].reshape(1, -1),
            )[0][0]
        else:
            distance_dict[word] = cosine_distances(
                year_1_model.wv[word].reshape(1, -1),
                year_2_model.wv[word].reshape(1, -1),
            )[0][0]
    distance_dict = dict(
        sorted(distance_dict.items(), key=lambda item: item[1])[:top_k_drift]
    )

    return distance_dict
