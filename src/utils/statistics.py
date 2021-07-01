import os
from itertools import islice

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import yake
from nltk import FreqDist, StanfordTagger, ngrams, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache(persist=eval(os.getenv("PERSISTENT")))
def get_tfidf_features(sentences, sort=True, filter_pos_tags=[]):
    new_text = sentences
    if filter_pos_tags != []:
        new_text = ""
        pos_tags = get_stanford_tags(sentences)
        for sentence in pos_tags:
            new_text += (
                " ".join(
                    [
                        word
                        for word, word_class in sentence
                        if word_class in filter_pos_tags
                    ]
                )
                + "\n"
            )

    lines = new_text.split("\n")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(lines).toarray()
    X = np.mean(X, axis=0)
    X = X.reshape(-1)
    result_dict = dict(zip(list(vectorizer.get_feature_names()), list(X)))
    if sort:
        result_dict_tuple = sorted(
            result_dict.items(), key=lambda x: x[1], reverse=True
        )
        result_dict = {k: v for k, v in result_dict_tuple}
    return result_dict


from .misc import length_removed_keywords, remove_keywords_util


@st.cache(persist=eval(os.getenv("PERSISTENT")))
def get_stanford_tags(sentences):
    pos_tags = []
    for line in sentences.split("\n"):
        pos_tagged = nltk.pos_tag(word_tokenize(line))
        pos_tags.append(pos_tagged)
    return pos_tags


def find_ngrams_for_sentences(sentences, n=1):
    sentence_list = sentences.split("\n")
    ngrams_list = []
    for sentence in sentence_list:
        ngrams_list += list(ngrams(sentence.split(), n))
    return ngrams_list


@st.cache(persist=eval(os.getenv("PERSISTENT")))
def find_freq(text, n=1, sort=False, filter_pos_tags=[]):
    new_text = text
    if filter_pos_tags != []:
        new_text = ""
        pos_tags = get_stanford_tags(text)
        for sentence in pos_tags:
            new_text += (
                " ".join(
                    [
                        word
                        for word, word_class in sentence
                        if word_class in filter_pos_tags
                    ]
                )
                + "\n"
            )

    ngrams_lst = find_ngrams_for_sentences(new_text, n)
    gram_count_mapping = dict(FreqDist(ngrams_lst))
    gram_count_mapping = {" ".join(k): v for k, v in gram_count_mapping.items()}
    if sort:
        sorted_gram_count_tuple = sorted(
            gram_count_mapping.items(), key=lambda x: x[1], reverse=True
        )
        gram_count_mapping = {k: v for k, v in sorted_gram_count_tuple}
    return gram_count_mapping


@st.cache(persist=eval(os.getenv("PERSISTENT")))
def find_norm_freq(text, n=1, sort=False, filter_pos_tags=[]):
    gram_count_mapping = find_freq(
        text=text, n=n, sort=sort, filter_pos_tags=filter_pos_tags
    )
    norm_factor = sum(list(gram_count_mapping.values()))
    gram_count_mapping = {k: v / norm_factor for k, v in gram_count_mapping.items()}
    return gram_count_mapping


@st.cache(persist=eval(os.getenv("PERSISTENT")))
def find_productivity(words, text, n=2, normalize=False, filter_pos_tags=[]):
    if normalize:
        fdist = find_norm_freq(
            text=text, n=n, sort=True, filter_pos_tags=filter_pos_tags
        )
    else:
        fdist = find_freq(text=text, n=n, sort=True, filter_pos_tags=filter_pos_tags)

    ngrams_lst = list(fdist.keys())
    prods = {}
    for word in words:
        ngrams_lst_having_word = []
        for ngram in ngrams_lst:
            if word in ngram:
                ngrams_lst_having_word.append(ngram)

        ngrams_lst_having_word = list(set(ngrams_lst_having_word))

        # find f_m_i for every ngram
        f_m_is = []
        for i in range(len(ngrams_lst_having_word)):
            f_m_i = fdist[ngrams_lst_having_word[i]]
            f_m_is.append(f_m_i)

        total_f_m_i = sum(f_m_is)
        p_m_is = [i / total_f_m_i for i in f_m_is]
        prod = -np.sum(np.multiply(np.log2(p_m_is), p_m_is))
        prods[word] = prod

    return prods


@st.cache(persist=eval(os.getenv("PERSISTENT")))
def freq_top_k(
    text,
    top_k=20,
    n=1,
    normalize=True,
    filter_pos_tags=[],
    tfidf=False,
    remove_keywords_path="./removed_keywords/removedphrases.txt",
):
    if tfidf:
        sorted_gram_count_mapping = get_tfidf_features(
            text, sort=True, filter_pos_tags=filter_pos_tags
        )
    else:
        if normalize:
            sorted_gram_count_mapping = find_norm_freq(
                text, n=n, sort=True, filter_pos_tags=filter_pos_tags
            )
        else:
            sorted_gram_count_mapping = find_freq(
                text, n=n, sort=True, filter_pos_tags=filter_pos_tags
            )

    if remove_keywords_path is not None and os.path.isfile(remove_keywords_path):
        sorted_gram_count_mapping = remove_keywords_util(
            remove_keywords_path, sorted_gram_count_mapping
        )

    if top_k < len(sorted_gram_count_mapping):
        sorted_gram_count_mapping = dict(
            islice(sorted_gram_count_mapping.items(), top_k)
        )

    return sorted_gram_count_mapping


def yake_keyword_extraction(
    text_file,
    top_k=20,
    language="en",
    max_ngram_size=2,
    window_size=2,
    deduplication_threshold=0.9,
    deduplication_algo="seqm",
    remove_keywords_path="./removed_keywords/removedphrases.txt",
):
    with open(text_file, "r") as f:
        text = f.read()
    if remove_keywords_path is not None and os.path.isfile(remove_keywords_path):
        len_rem_keywords = length_removed_keywords(remove_keywords_path)
    else:
        len_rem_keywords = 0
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        dedupFunc=deduplication_algo,
        windowsSize=window_size,
        top=top_k + len_rem_keywords,
        features=None,
    )
    keywords = custom_kw_extractor.extract_keywords(text)
    if remove_keywords_path is not None and os.path.isfile(remove_keywords_path):
        keywords = remove_keywords_util(remove_keywords_path, dict(keywords))
    x = list(keywords.keys())
    y = [1 / (1e5 * keywords[key]) for key in keywords.keys()]

    df = pd.DataFrame()
    df["ngram"] = x
    df["score"] = y
    return df
