import unicodedata

import contractions
import inflect
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


# noise removal
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def to_lower_case(text):
    return text.lower()


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def remove_punctuation(text):

    for punct in "/-'":
        text = text.replace(punct, " ")

    for punct in "&":
        text = text.replace(punct, "and")

    for punct in "?!.,\"#$%'()*+-/:;<=>@[\\]^_`{|}~" + "“”’":
        text = text.replace(punct, "")
    return text


# tokenisation
def tokenise(text):
    words = word_tokenize(text)
    return words


# normalisation
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = (
            unicodedata.normalize("NFKD", word)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    inflect_engine = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = inflect_engine.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words("english"):
            new_words.append(word)
    return new_words


def lemmatise_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatiser = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatiser.lemmatize(word, pos="v")
        lemmas.append(lemma)
    return lemmas


def preprocess_text(text):
    text = to_lower_case(text)

    # noise removal
    text = strip_html(text)
    text = replace_contractions(text)
    text = remove_punctuation(text)

    # tokenisation
    words = tokenise(text)

    # normalisation
    words = remove_non_ascii(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatise_verbs(words)

    return " ".join(words)
