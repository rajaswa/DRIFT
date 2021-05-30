import time

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from gensim.models.word2vec import Word2Vec
from joblib import Parallel, delayed
from nltk.cluster import KMeansClusterer
from numba import jit, prange
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from tqdm.auto import tqdm


def kmeans_parallel(X, n_clusters, repeats=25):
    kmeans = KMeansClusterer(
        n_clusters, distance=nltk.cluster.util.cosine_distance, repeats=repeats
    )
    labels = kmeans.cluster(X, assign_clusters=True)
    return silhouette_score(X, labels, metric="cosine")


def k_means_clustering(keywords, model_path, save_path, k_opt=None, k_max=10):

    model = Word2Vec.load(model_path)
    keywords = [keyword for keyword in keywords if keyword in model.wv.vocab]
    X = [model.wv[keyword] for keyword in keywords]

    if k_opt is None:
        silhouette_scores = []
        t = time.time()
        silhouette_scores = Parallel(n_jobs=-1)(
            delayed(kmeans_parallel)(X, n_clusters, 25)
            for n_clusters in tqdm(range(2, k_max + 1))
        )
        k_opt = 2 + silhouette_scores.index(max(silhouette_scores))

    kmeans = KMeansClusterer(
        k_opt, distance=nltk.cluster.util.cosine_distance, repeats=25
    )
    labels = kmeans.cluster(X, assign_clusters=True)

    # PLOT THE DIAGRAM
    tsne = TSNE(n_components=2, init="pca", random_state=42)
    X_red = tsne.fit_transform(X)

    df = pd.DataFrame(columns=["x", "y", "label"])
    for i in range(len(keywords)):
        df = df.append(
            {"x": X_red[i][0], "y": X_red[i][1], "label": labels[i]}, ignore_index=True
        )

    facet = sns.lmplot(
        data=df, x="x", y="y", hue="label", fit_reg=False, legend=True, legend_out=True
    )
    facet.savefig(save_path)

    # for i, text in enumerate(leg.get_texts()):
    # 	plt.setp(text)

    return keywords, labels
