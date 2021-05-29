import operator
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from src.analysis.utils import *


def find_most_similar_words(
    words,
    year_model_path,
    compass_model_path,
    top_k=10,
):
    year_model = Word2Vec.load(year_model_path)
    compass_model = Word2Vec.load(compass_model_path)

    word_vectors = []
    for word in words:
        if word in year_model.wv.vocab:
            word_vectors.append(year_model.wv[word])
        else:
            word_vectors.append(np.zeros(year_model.wv["paper"].shape))
    word_vectors = np.array(word_vectors)
    # word_vectors = word_vectors/np.linalg.norm(word_vectors,axis=1)[:,None]

    compass_vectors = compass_model.wv.vectors
    # compass_vectors = compass_vectors/np.linalg.norm(compass_vectors,axis=1)[:,None]

    top_sims = np.argsort(cosine_similarity(word_vectors, compass_vectors), axis=1)
    top_sims = top_sims[:, -top_k:]

    compass_words = list(compass_model.wv.vocab.keys())

    most_sim_words = []

    for i in range(top_sims.shape[0]):
        sim_words = []
        for j in range(top_sims[i].shape[0]):
            sim_words.append(compass_words[top_sims[i][j]])
        most_sim_words.append(sim_words)

    return word_vectors, most_sim_words, top_sims


def find_most_drifted_words(
    words, year_model_paths, compass_model_path, top_k=10, top_most_drifted_k=5
):
    year_wise_stats = []
    for year_model_path in year_model_paths:
        year_wise_stats.append(
            find_most_similar_words(words, year_model_path, compass_model_path, top_k)
        )

    scores = [0 for i in range(len(words))]
    for year1_stats, year2_stats in zip(year_wise_stats[:-1], year_wise_stats[1:]):

        common_words = [
            intersection(year1_stats[1][k], year2_stats[1][k])
            for k in range(len(year1_stats[1]))
        ]

        scores = [scores[k] - len(common_words[k]) + top_k for k in range(len(words))]
        for l, word_wise_common_words in enumerate(common_words):
            # print(year1_stats[1])
            # print(year2_stats[1])
            year1_common_indices = [
                year1_stats[1][l].index(ele) for ele in word_wise_common_words
            ]
            year2_common_indices = [
                year2_stats[1][l].index(ele) for ele in word_wise_common_words
            ]
            year1_common_sims = [
                year1_stats[2][l][year1_common_indices[k]]
                for k in range(len(year1_common_indices))
            ]
            year2_common_sims = [
                year2_stats[2][l][year2_common_indices[k]]
                for k in range(len(year2_common_indices))
            ]
            scores[l] += abs(
                sum([x1 - x2 for (x1, x2) in zip(year1_common_sims, year2_common_sims)])
            )

    top_most_drifted_indices = np.argsort(scores)[-top_most_drifted_k:]

    return [words[k] for k in top_most_drifted_indices]


def plot_semantic_drift(word, year_model_path, compass_model_path, save_path, top_k=10):
    compass_model = Word2Vec.load(compass_model_path)

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(
            words,
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    plt.savefig(save_path)
