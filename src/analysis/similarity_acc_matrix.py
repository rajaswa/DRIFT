import json
import os

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from ..utils import save_json, save_npy


def compute_similarity_matrix_keywords(keywords, model_path, all_model_vectors=False):
    model = Word2Vec.load(model_path)

    if all_model_vectors:
        keywords = list(model.wv.vocab.keys())

    unk_emb = np.mean([model.wv[word] for word in model.wv.vocab])
    word_embs = np.array(
        [model.wv if keyword in model.wv.vocab else unk_emb for keyword in keywords]
    )
    sim_matrix = cosine_similarity(word_embs, word_embs)

    return keywords, sim_matrix


def compute_acceleration_matrix(sim_matrix_1, sim_matrix_2, top_k_acc=None):
    acceleration_matrix = sim_matrix_2 - sim_matrix_1

    acc_matrix_sorted = (-acceleration_matrix).argsort(axis=None)
    acc_matrix_indices = np.unravel_index(acc_matrix_sorted, acceleration_matrix.shape)
    sorted_indices = np.vstack(acc_matrix_indices).T

    if top_k_acc is not None:
        sorted_indices = sorted_indices[:top_k_acc]

    word_pairs = []
    for sorted_idx in sorted_indices:
        word_1_index, word_2_index = sorted_idx

        word_1 = keywords[word_1_index]
        word_2 = keywords[word_2_index]

        word_pairs.append(
            (word_1, word_2, acceleration_matrix[word_1_index][word_2_index])
        )

    return word_pairs
