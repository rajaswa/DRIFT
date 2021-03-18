from pickle import UnpicklingError
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np
from gensim.models.word2vec import Word2Vec
import os
import pickle as pkl
from tqdm.auto import tqdm
np.random.seed(42)
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
if not os.path.exists('.cache'):
    os.makedirs('.cache')

load_from_cache = False
stopwords = stopwords.words('english')

if not load_from_cache:
    with open('./data/compass.txt', 'r') as f:
        d = f.read()
    unique_words, counts = np.unique(d.split(), return_counts=True)
    sorted_words_counts = sorted(list(zip(unique_words, counts)), key=lambda x: x[1], reverse=True)
    print(sorted_words_counts[:100])
    new_words = []
    for tups in sorted_words_counts:
        if tups[0] not in stopwords and tups[0] not in ['neural','language','models','data', 'model','using','results','show']:
            new_words.append(tups[0])
            if len(new_words) == 4:
                break
    print(new_words)
    
    years = []
    all_words = []
    vectors = []
    for model_name in tqdm(sorted(os.listdir('./model'))):
        if model_name != 'compass.model' and model_name != 'log.txt':
            try:
                model = Word2Vec.load(f'./model/{model_name}')
                for word in new_words:
                    if word in model.wv:
                        all_words.append(word)
                        vectors.append(model.wv[word])
                        years.append(model_name.split('.')[0])
            except UnpicklingError as e:
                print(e)
                print("Unable to load model for ", model_name)
    with open('.cache/vectors', 'wb') as f:
        pkl.dump(vectors, f)
    with open('.cache/all_words', 'wb') as f:
        pkl.dump(all_words, f)
    with open('.cache/years', 'wb') as f:
        pkl.dump(years, f)
    compass_vectors = []
    try:
        model = Word2Vec.load('./model/compass.model')
        for word in new_words:
            if word in model.wv:
                compass_vectors.append(model.wv[word])
    except UnpicklingError as e:
        print(e)
        print("Unable to load model for ", model_name)
    with open('.cache/compass_vectors', 'wb') as f:
        pkl.dump(compass_vectors, f)

    
else:
    with open('.cache/vectors', 'rb') as f:
        vectors = pkl.load(f)
    with open('.cache/all_words', 'rb') as f:
        all_words = pkl.load(f)
    with open('.cache/years', 'rb') as f:
        years = pkl.load(f)
    with open('.cache/compass_vectors', 'rb') as f:
        compass_vectors = pkl.load(f)
           


type='umap'
if type=='pca':
    pca = PCA(2, random_state=42)
    compass_embeddings = pca.fit_transform(compass_vectors)
    embeddings = pca.transform(vectors)
elif type=='tsne':
    tsne = TSNE(n_components=2, init='pca',random_state=42)
    # compass_embeddings = tsne.fit_transform(compass_vectors)
    embeddings = tsne.fit_transform(vectors)
elif type=='umap':
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(vectors)


sorted_unique_years = np.array(sorted(np.unique(list(map(int, years)))))

sorted_unique_cs = (sorted_unique_years - np.min(sorted_unique_years))/(np.max(sorted_unique_years) - np.min(sorted_unique_years))

year_to_color = dict(zip(sorted_unique_years, sorted_unique_cs))
fig = plt.figure(figsize=(10, 10)) 

sc = plt.scatter(embeddings[:, 0], embeddings[:, 1], c = list(map(year_to_color.get, list(map(int, years)))), alpha=0.5)
ax = plt.gca()
for i, txt in enumerate(all_words):
    ax.annotate(txt, (embeddings[i,0], embeddings[i,1]))
cbar = plt.colorbar(sc, ticks=sorted_unique_cs + 1/12, label='Years')
cbar.ax.set_yticklabels(sorted_unique_years)
cbar.ax.axes.tick_params(length=0)
plt.savefig(f'{type} Plot.png', bbox_inches='tight')
plt.close()



