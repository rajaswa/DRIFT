import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import get_word_embeddings


def compute_similarity_matrix_keywords(model_path, keywords=[], all_model_vectors=False):
    word_embs = np.array(get_word_embeddings(model_path, keywords, all_model_vectors=all_model_vectors))
    sim_matrix = cosine_similarity(word_embs, word_embs)

    return keywords, word_embs, sim_matrix


def compute_acceleration_matrix(keywords, sim_matrix_1, sim_matrix_2, top_k_acc=200, skip_same_word_pairs=True, skip_duplicates=True):
    acceleration_matrix = sim_matrix_2 - sim_matrix_1

    acc_matrix_sorted = (-acceleration_matrix).argsort(axis=None)
    acc_matrix_indices = np.unravel_index(acc_matrix_sorted, acceleration_matrix.shape)
    sorted_indices = np.vstack(acc_matrix_indices).T

    word_pairs = {}
    top_k_count = 0
    idx = 0

    while top_k_count<top_k_acc and idx<len(sorted_indices):
        word_1_index, word_2_index = sorted_indices[idx]
        idx+=1

        word_1 = keywords[word_1_index]
        word_2 = keywords[word_2_index]
        if skip_same_word_pairs and word_1==word_2:
            continue
        if skip_duplicates and (word_2, word_1) in word_pairs:
            continue

        word_pairs[(word_1,word_2)] = acceleration_matrix[word_1_index][
            word_2_index
        ]
        top_k_count+=1

    return word_pairs

def compute_acc_between_years(keywords, model_path_1, model_path_2,all_model_vectors=False,top_k_acc=200, skip_same_word_pairs=True, skip_duplicates=True):
    kw1,em1,sim_matrix_1 = compute_similarity_matrix_keywords(model_path_1, all_model_vectors=all_model_vectors, keywords=keywords)
    kw2,em2,sim_matrix_2 = compute_similarity_matrix_keywords(model_path_2, all_model_vectors=all_model_vectors, keywords=keywords)
    word_pairs = compute_acceleration_matrix(keywords, sim_matrix_1=sim_matrix_1, sim_matrix_2=sim_matrix_2, top_k_acc=top_k_acc, skip_same_word_pairs=skip_same_word_pairs, skip_duplicates=skip_duplicates)

    return word_pairs, em1, em2