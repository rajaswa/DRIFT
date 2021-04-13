import nltk
from gensim.models.word2vec import Word2Vec
from sklearn import cluster


def kmeans_clustering(model_path, k=10):
    model = Word2Vec.load(model_path)
    X = model[model.vocab]

    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return model.vocab, X, labels, centroids


# def track_clusters()
