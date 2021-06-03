import faiss
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm


def kmeans_train(X, n_cluster, method="faiss"):
    if method == "faiss":
        kmeans = faiss.Kmeans(
            d=X.shape[1],
            k=n_cluster,
        )
        kmeans.train(X.astype(np.float32))
        labels = kmeans.index.search(X.astype(np.float32), 1)[1].reshape(-1)
    elif method == "sklearn":
        kmeans = KMeans(
            n_clusters=n_cluster,
        ).fit(X)
        labels = np.array(kmeans.labels_)

    return labels


def kmeans_clustering(
    keywords, model_path, k_opt=None, k_max=10, method="faiss"
):
    assert method in ["faiss", "sklearn"], "method should be one of " + str(
        ["faiss", "sklearn"]
    )

    model = Word2Vec.load(model_path)
    keywords = [keyword for keyword in keywords if keyword in model.wv.vocab]
    embs = np.array([model.wv[keyword] for keyword in keywords])

    if k_opt is None:

        silhouette_scores = []
        for n_cluster in range(2, k_max + 1):
            labels = kmeans_train(embs, n_cluster, method)
            silhouette_scores.append(silhouette_score(embs, labels, metric="euclidean"))

        k_opt = 2 + silhouette_scores.index(max(silhouette_scores))

    labels = kmeans_train(embs, k_opt, method)

    return keywords, embs, labels, k_opt
