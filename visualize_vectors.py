import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from viz import plotly_scatter_embeddings


cache_dir = ".cache"

with open(f"{cache_dir}/vectors", "rb") as f:
    vectors = pkl.load(f)
with open(f"{cache_dir}/all_words", "rb") as f:
    all_words = pkl.load(f)
with open(f"{cache_dir}/years", "rb") as f:
    years = pkl.load(f)
with open(f"{cache_dir}/compass_vectors", "rb") as f:
    compass_vectors = pkl.load(f)

type = "tsne"
if type == "pca":
    pca = PCA(2, random_state=42)
    compass_embeddings = pca.fit_transform(compass_vectors)
    embeddings = pca.transform(vectors)
elif type == "tsne":
    tsne = TSNE(n_components=2, init="pca", random_state=42)
    # compass_embeddings = tsne.fit_transform(compass_vectors)
    embeddings = tsne.fit_transform(vectors)
elif type == "umap":
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(vectors)

sorted_unique_years = np.array(sorted(np.unique(list(map(int, years)))))

sorted_unique_cs = (sorted_unique_years - np.min(sorted_unique_years)) / (
    np.max(sorted_unique_years) - np.min(sorted_unique_years)
)

year_to_color = dict(zip(sorted_unique_years, sorted_unique_cs))

fig = plotly_scatter_embeddings(years, all_words, embeddings, year_to_color)

fig.show()
