import os

import faiss
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.words import get_word_embeddings


# TO-DO: Switch to FaissKMeans class, based on this: https://towardsdatascience.com/k-means-8x-faster-27x-lower-error-than-scikit-learns-in-25-lines-eaedc7a3a0c8


def kmeans_train(X, n_cluster, method="faiss", return_fitted_model=False):
    assert method in ["faiss", "sklearn"], "method should be one of " + str(
        ["faiss", "sklearn"]
    )
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
    if return_fitted_model:
        return labels, kmeans
    else:
        return labels


def kmeans_embeddings(
    embs, k_opt=None, k_max=10, method="faiss", return_fitted_model=False
):

    if k_opt is None:

        silhouette_scores = []
        for n_cluster in range(2, min(k_max + 1, len(embs))):
            labels = kmeans_train(embs, n_cluster, method)
            silhouette_scores.append(silhouette_score(embs, labels, metric="euclidean"))

        k_opt = 2 + silhouette_scores.index(max(silhouette_scores))

    if return_fitted_model:
        labels, kmeans = kmeans_train(embs, k_opt, method, return_fitted_model)
        return labels, k_opt, kmeans
    else:
        labels = kmeans_train(embs, k_opt, method, return_fitted_model)
        return labels, k_opt


def kmeans_clustering(
    keywords,
    model_path,
    k_opt=None,
    k_max=10,
    method="faiss",
    return_fitted_model=False,
):
    embs = get_word_embeddings(
        model_path,
        keywords,
        all_model_vectors=False,
        return_words=False,
        filter_missing_words=True,
    )
    if return_fitted_model:
        labels, kmeans = kmeans_embeddings(
            embs,
            k_opt=k_opt,
            k_max=k_max,
            method=method,
            return_fitted_model=return_fitted_model,
        )
        return keywords, embs, labels, k_opt, kmeans
    else:
        labels = kmeans_embeddings(
            embs,
            k_opt=k_opt,
            k_max=k_max,
            method=method,
            return_fitted_model=return_fitted_model,
        )
        return keywords, embs, labels, k_opt
