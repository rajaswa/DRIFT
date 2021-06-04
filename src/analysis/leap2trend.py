from .similarity_acc_matrix import compute_similarity_matrix_keywords
import numpy as np
import tqdm


def predict_trends(keywords, year_model_paths, delta=20):
    ranked_matrices = []
    keywords = list(keywords.keys())
    for model_path in year_model_paths:
        sim_matrix = compute_similarity_matrix_keywords(model_path, keywords, False)[-1]
        ranked_matrix = rank_matrices(sim_matrix)
        ranked_matrices.append(ranked_matrix)
    leap_indices = rank_ascent_identification(ranked_matrices, delta)
    word_pairs = [
        (keywords[leap_indices[0][i]], keywords[leap_indices[1][i]])
        for i in range(leap_indices[0].shape[0])
    ]
    return word_pairs


def rank_ascent_identification(ranked_matrices, delta=20):
    rank_identification_matrix = np.zeros(ranked_matrices[0].shape)
    for matrix_t, matrix_t_1 in zip(ranked_matrices[:-1], ranked_matrices[1:]):
        rank_diff_matrix = np.subtract(matrix_t, matrix_t_1)
        leap_indices = np.where(rank_diff_matrix >= delta)
    return leap_indices


def rank_matrices(matrix):
    sorted_matrix = np.sort(matrix.flatten())[::-1].reshape(matrix.shape)
    rank = 1
    ranked_matrix = np.zeros(matrix.shape)
    pbar = tqdm.tqdm(total=int((matrix.shape[0] ** 2) / 2))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            temp = sorted_matrix[i][j]
            flag = 0
            if i > j:
                continue
            else:
                pbar.update(1)
            for k in range(matrix.shape[0]):
                for p in range(matrix.shape[0]):
                    if matrix[k][p] == temp:
                        rank += 1
                        ranked_matrix[k][p] = rank
                        flag = 1
                        break
                if flag == 1:
                    break

    return ranked_matrix
