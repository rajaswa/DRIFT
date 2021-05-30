import os
import pickle as pkl
from pickle import UnpicklingError

import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from tqdm.auto import tqdm


np.random.seed(42)

if not os.path.exists(".cache"):
    os.makedirs(".cache")

stopwords = stopwords.words("english")

with open("./data/compass.txt", "r") as f:
    d = f.read()
unique_words, counts = np.unique(d.split(), return_counts=True)
sorted_words_counts = sorted(
    list(zip(unique_words, counts)), key=lambda x: x[1], reverse=True
)


# WORD SELECTION TOP-10
new_words = []
for tups in sorted_words_counts:
    if tups[0] not in stopwords:
        new_words.append(tups[0])
        if len(new_words) == 10:
            break

years = []
all_words = []
vectors = []
for model_name in tqdm(sorted(os.listdir("./model"))):
    if model_name != "compass.model" and model_name != "log.txt":
        try:
            model = Word2Vec.load(f"./model/{model_name}")
            for word in new_words:
                if word in model.wv:
                    all_words.append(word)
                    vectors.append(model.wv[word])
                    years.append(model_name.split(".")[0])
        except UnpicklingError as e:
            print(e)
            print("Unable to load model for ", model_name)
with open(".cache/vectors", "wb") as f:
    pkl.dump(vectors, f)
with open(".cache/all_words", "wb") as f:
    pkl.dump(all_words, f)
with open(".cache/years", "wb") as f:
    pkl.dump(years, f)
compass_vectors = []
try:
    model = Word2Vec.load("./model/compass.model")
    for word in new_words:
        if word in model.wv:
            compass_vectors.append(model.wv[word])
except UnpicklingError as e:
    print(e)
    print("Unable to load model for ", model_name)
with open(".cache/compass_vectors", "wb") as f:
    pkl.dump(compass_vectors, f)
