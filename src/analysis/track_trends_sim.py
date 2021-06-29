import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import get_word_embeddings


def compute_similarity_matrix_years(model_paths, keyword, top_k_sim=5):
    first_year_embs = get_word_embeddings(model_paths[0], [keyword], return_words=False)
    sim_dict = {}
    for model_path in model_paths[1:]:
        second_year_vocab, second_year_embs = get_word_embeddings(
            model_path, [], all_model_vectors=True, return_words=True
        )
        if keyword in second_year_vocab:
            word_remove_idx = second_year_vocab.index(keyword)
            del second_year_vocab[word_remove_idx]
            second_year_embs = np.delete(second_year_embs, word_remove_idx, axis=0)
        sim_vec = cosine_similarity(first_year_embs, second_year_embs)[0]
        sim_dict.update(
            {
                word + "_" + model_path: sim_vec[i]
                for i, word in enumerate(second_year_vocab)
            }
        )
    sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
    sim_dict_wo_dup = {}
    sim_dups = []
    ctr = 0
    for key in sim_dict:
        word = key.split("_")[0]
        if word not in sim_dups:
            sim_dict_wo_dup[key] = sim_dict[key]
            sim_dups.append(word)
            ctr += 1
            if ctr == top_k_sim:
                break
    return sim_dict_wo_dup
