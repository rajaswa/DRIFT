import operator

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


def find_most_similar_words(
    word, year_model_path, compass_model, top_k=10, compass_model_path=None
):
    year_model = Word2Vec.load(year_model_path)

    assert (compass_model is None) ^ (
        compass_model_path is None
    ), "one of compass_model and compass_model_path should not be None"

    if compass_model is None:
        compass_model = Word2Vec.load(compass_model_path)

    assert word in year_model.wv.vocab, str(word) + " out of vocabulary"

    word_embedding = year_model.wv[word]

    sim_dict = {}
    for word_compare in compass_model.wv.vocab:
        word_compare_embedding = compass_model.wv[word_compare]
        sim = cosine_similarity(
            word_embedding.reshape(1, -1), word_compare_embedding.reshape(1, -1)
        )[0]
        sim_dict[word_compare] = (sim, word_compare_embedding)

    sorted_sim_tuple = sorted(
        sim_dict.items(), key=operator.itemgetter(1), reverse=True
    )
    sim_dict = {k: v for k, v in sorted_sim_tuple}

    if top_k < len(sorted_sim_tuple):
        sim_dict = dict(islice(sim_dict.items(), top_k))

    return sim_dict, word_embedding


def plot_semantic_drift(
    word, year_model_paths, compass_model_path, save_path, top_k=10
):
    compass_model = Word2Vec.load(compass_model_path)

    sim_dicts = []
    word_embeddings = []
    for year_model_path in year_model_paths:
        sim_dict, word_embedding = find_most_similar_words(
            word, year_model_path, compass_model, top_k
        )
        sim_dicts.append(sim_dict)
        word_embeddings.append(word_embedding)

    # fit t-SNE on compass word embeddings
    train_embs = []
    for word in compass_model.wv.vocab:
        train_embs.append(compass_model.wv[word])

    tsne = TSNE(n_components=2, init="pca", random_state=42)
    tsne.fit(train_embs)

    words = []
    x = []
    y = []

    for i, sim_dict, word_embedding in enumerate(zip(sim_dicts, word_embeddings)):
        words.append(word + "_" + str(i))
        red_emb = tsne.transform([word_embedding])
        x.append(red_emb[0][0])
        y.append(red_emb[0][1])

        for sim_word in sim_dict:
            words.append(sim_word)
            red_emb = tsne.transform([sim_dict[sim_word][1]])
            x.append(red_emb[0][0])
            y.append(red_emb[0][1])

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
