import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(model_path, keywords):
    keywords_size = len(keywords)

    model = Word2Vec.load(model_path)
    sim_matrix = np.ones((keywords_size, keywords_size))
    for i in range(keywords_size):
        for j in range(i + 1, keywords_size - 1):
            row_word_embedding = model.wv[keywords[i]]
            column_word_embedding = model.wv[keywords[j]]

            sim_matrix[i][j] = cosine_similarity(
                row_word_embedding.reshape(1, -1), column_word_embedding.reshape(1, -1)
            )[0]

    return sim_matrix


def compute_acceleration_matrix(sim_matrix_1, sim_matrix_2):
    return sim_matrix_2 - sim_matrix_1


def top_k_acceleration(keywords, acceleration_matrix, k=10):
    i = (-acceleration_matrix).argsort(axis=None)
    j = np.unravel_index(i, acceleration_matrix.shape)
    sorted_indices = np.vstack(j).T[:k]

    word_pairs = []
    for sorted_idx in sorted_indices:
        word_1_index, word_2_index = sorted_idx

        word_1 = keywords[word_1_index]
        word_2 = keywords[word_2_index]

        word_pairs.append(
            (word_1, word_2, acceleration_matrix[word_1_index][word_2_index])
        )

    return word_pairs
