import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import json

def compute_similarity_matrix_keywords(keywords, model_path=None,save_load_path=None):
    assert model_path is not None or save_load_path is not None, "One of model_path and save_load_path should not be None"

    keywords_size = len(keywords)
    if model_path is not None:
        model = Word2Vec.load(model_path)
        sim_matrix = np.ones((keywords_size, keywords_size))
        
        for i in range(keywords_size):
            for j in range(i + 1, keywords_size - 1):
                row_word_embedding = model.wv[keywords[i]]
                column_word_embedding = model.wv[keywords[j]]

                sim_matrix[i][j] = cosine_similarity(
                    row_word_embedding.reshape(1, -1), column_word_embedding.reshape(1, -1)
                )[0][0]
        
        if save_load_path is not None:
            print(f"COMPUTING SIMILARITY MATRIX FOR {model_path} AND SAVING AT {save_load_path}")
            if not os.path.exists(save_load_path):
                os.makedirs(save_load_path)
            
            vocab_path = os.path.join(save_load_path, "vocab.json")
            sim_path = os.path.join(save_load_path, "sim.npy")
            
            with open(vocab_path, "w") as vocab_json:
                json.dump(keywords, vocab_json)

            with open(sim_path, 'wb') as sim_npy:
                np.save(sim_npy, sim_matrix)                  

    elif save_load_path is not None:
        
        print(f"LOADING SIMILARITY MATRIX FROM {save_load_path}")
        if not os.path.exists(save_load_path):
            os.makedirs(save_load_path)
        vocab_path = os.path.join(save_load_path, "vocab.json")
        sim_path = os.path.join(save_load_path, "sim.npy")

        with open(vocab_path, "r") as vocab_json:
            keywords = json.load(vocab_json)

        with open(sim_path, 'rb') as sim_npy:
            sim_matrix = np.load(sim_npy)


    return keywords, sim_matrix


def compute_similarity_matrix(keywords,model_path=None, save_load_path=None):
    
    assert model_path is not None or save_load_path is not None, "One of model_path and save_load_path should not be None"
    
    if model_path is not None:
        model = Word2Vec.load(model_path)
        model_vocab = list(model.wv.vocab)
        vocab_size = len(model_vocab)
        sim_matrix = np.ones((vocab_size, vocab_size))

        for i in tqdm(range(vocab_size)):
            for j in range(i+1, vocab_size-1):
                row_word_embedding = model.wv[model_vocab[i]]
                column_word_embedding = model.wv[model_vocab[j]]

                sim_matrix[i][j] = cosine_similarity(row_word_embedding.reshape(1,-1), column_word_embedding.reshape(1,-1))[0][0]

        if save_load_path is not None:
            print(f"COMPUTING SIMILARITY MATRIX FOR {model_path} AND SAVING AT {save_load_path}")
            if not os.path.exists(save_load_path):
                os.makedirs(save_load_path)
            vocab_path = os.path.join(save_load_path, "vocab.json")
            sim_path = os.path.join(save_load_path, "sim.npy")

            with open(vocab_path, "w") as vocab_json:
                json.dump(model_vocab, vocab_json)

            with open(sim_path, 'wb') as sim_npy:
                np.save(sim_npy, sim_matrix)

    elif save_load_path is not None:
        print(f"LOADING SIMILARITY MATRIX FROM {save_load_path}")
        if not os.path.exists(save_load_path):
            os.makedirs(save_load_path)
        vocab_path = os.path.join(save_load_path, "vocab.json")
        sim_path = os.path.join(save_load_path, "sim.npy")

        with open(vocab_path, "r") as vocab_json:
            model_vocab = json.load(vocab_json)

        with open(sim_path, 'rb') as sim_npy:
            sim_matrix = np.load(sim_npy)

    return model_vocab,sim_matrix


def compute_acceleration_matrix(sim_matrix_1, sim_matrix_2):
    # print(sim_matrix_1)
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

def plot_tsne(model_path, word_pair, top_unigrams, save_path):
    year_model = Word2Vec.load(model_path)

    # fit t-SNE on compass word embeddings
    train_words = top_unigrams
    train_words.extend(word_pair)
    train_embs = []
    for word in train_words:
        train_embs.append(year_model.wv[word])

    tsne = TSNE(n_components=2, init="pca", random_state=42)
    red_embs = tsne.fit_transform(train_embs)

    x = []
    y = []

    for ele in red_embs:
        x.append(ele[0])
        y.append(ele[1])
    plt.clf()
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)-2):
        plt.scatter(x[i],y[i], s=0)
        plt.annotate(train_words[i],
                    xy=(x[i], y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    for i in range(len(x)-2, len(x)):
        plt.scatter(x[i],y[i], s=0)
        plt.annotate(train_words[i],
                    xy=(x[i], y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom',
                    # fontstyle='oblique',
                    fontweight='bold',
                    # fontsize='large',
                    color='red')

    plt.savefig(save_path)