import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import intersection


def find_most_similar_words(
    words,
    year_model_path,
    compass_model_path,
    top_k_for_sim=10,
):
    year_model = Word2Vec.load(year_model_path)
    compass_model = Word2Vec.load(compass_model_path)

    compass_words = list(compass_model.wv.vocab.keys())
    unk_emb = np.mean([year_model.wv[word] for word in year_model.wv.vocab], axis=0)
    word_vectors = np.array(
        [
            year_model.wv[word] if word in year_model.wv.vocab else unk_emb
            for word in words
        ]
    )
    compass_vectors = compass_model.wv.vectors
    sim_matrix = cosine_similarity(word_vectors, compass_vectors)
    top_sims = np.argsort(sim_matrix, axis=1)
    top_sims = top_sims[:, -top_k_for_sim:]

    most_sim_words = [
        [compass_words[top_sims[i][j]] for j in range(top_sims[i].shape[0])]
        for i in range(top_sims.shape[0])
    ]
    print(most_sim_words[0])

    sim_dict = {}
    for idx, word in enumerate(words):
        sim_dict[word] = {}
        sim_dict[word]["emb"] = word_vectors[idx, :]
        sim_dict[word]["sim_info"] = {}
        for sim_word in most_sim_words[idx]:
            sim_dict[word]["sim_info"][sim_word] = {}
            sim_dict[word]["sim_info"][sim_word]["emb"] = compass_model.wv[sim_word]
            sim_dict[word]["sim_info"][sim_word]["sim_score"] = sim_matrix[
                words.index(word)
            ][compass_words.index(sim_word)]
    return sim_dict


def find_most_drifted_words(
    words, year_model_paths, compass_model_path, top_k_for_sim=10, top_most_drifted_k=5
):

    sim_dicts = [
        find_most_similar_words(
            words, year_model_path, compass_model_path, top_k_for_sim
        )
        for year_model_path in year_model_paths
    ]
    scores = {}
    for word in sim_dicts[0]:
        scores[word] = 0
        year_1_info = sim_dicts[0][word]["sim_info"]
        year_last_info = sim_dicts[-1][word]["sim_info"]
        year_1_sim_words = list(year_1_info.keys())
        year_last_sim_words = list(year_last_info.keys())
        common_sim_words = intersection(year_1_sim_words, year_last_sim_words)

        for sim_word in common_sim_words:
            scores[word] += abs(
                year_last_info[sim_word]["sim_score"]
                - year_1_info[sim_word]["sim_score"]
            )

        scores[word] += len(year_1_sim_words + year_last_sim_words) - 2 * len(
            common_sim_words
        )

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
        :top_most_drifted_k
    ]

    words_for_plotting = []
    embs_for_plotting = []
    for idx, year_sim_dict in enumerate(sim_dicts):
        for word in sorted_scores:
            words_for_plotting.append(word + "_" +str(idx))
            embs_for_plotting.append(year_sim_dict[word]["emb"])

            for sim_word in year_sim_dict[word]["sim_info"]:
                if sim_word not in words_for_plotting:
                    words_for_plotting.append(sim_word)
                    embs_for_plotting.append(
                        year_sim_dict[word]["sim_info"][sim_word]["emb"]
                    )

    return words_for_plotting, embs_for_plotting
