import os
from pickle import UnpicklingError

import numpy as np
import streamlit as st
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from tqdm.auto import tqdm

def load_word_embedding(model_path, word):
    model = Word2Vec.load(model_path)
    return model.wv[word]


def get_all_word_counts(file_path="./data/compass", sort=True, remove_stopwords=True):
    with open(file_path, "r") as f:
        data = f.read()
    unique_words, counts = np.unique(data.split(), return_counts=True)
    word_counts = dict(zip(unique_words, counts))
    new_word_counts = {}
    if remove_stopwords:
        english_stopwords = stopwords.words("english")
        for word in word_counts.keys():
            if word not in english_stopwords:
                new_word_counts[word] = word_counts[word]
        word_counts = new_word_counts
    if sort:
        word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return word_counts

def get_vectors(word_list, models_dir, skip_models_list=[]):
    years = []
    all_words = []
    vectors = []
    for model_name in tqdm(sorted(os.listdir(models_dir))):
        if (
            model_name != "compass.model"
            and model_name != "log.txt"
            and model_name not in skip_models_list
        ):
            try:
                model = Word2Vec.load(os.path.join(models_dir, model_name))
                for word in word_list:
                    if word in model.wv:
                        all_words.append(word)
                        vectors.append()
                        years.append(model_name.split(".")[0])
            except UnpicklingError as e:
                print(e)
                print("Unable to load model for ", model_name)
    compass_vectors = []
    try:
        model = Word2Vec.load(os.path.join(models_dir, "compass.model"))
        for word in word_list:
            if word in model.wv:
                compass_vectors.append(model.wv[word])
    except UnpicklingError as e:
        print(e)
        print("Unable to load model for ", model_name)
    return (
        years,
        all_words,
        vectors,
        compass_vectors,
    )  # compass vectors have same shape as word list
