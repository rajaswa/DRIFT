import argparse
import glob
import json
import os
import re
import string
import unicodedata
import xml.etree.ElementTree as ET

import contractions
import inflect
import nltk
from bs4 import BeautifulSoup
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm


translator = Translator()

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

lemmatiser = WordNetLemmatizer()
inflect_engine = inflect.engine()


def to_lower_case(text):
    return text.lower()


# noise removal
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


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
    lemmas = []
    for word in words:
        lemma = lemmatiser.lemmatize(word, pos="v")
        lemmas.append(lemma)
    return lemmas


def preprocess(text):
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


def innertext(elt):
    return (elt.text or "") + (
        "".join(innertext(e) + (e.tail or "") for e in elt) or ""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_save_path",
        type=str,
        default="data/acl_anthology.json",
        help=".json path where the file is to saved",
    )
    args = parser.parse_args()
    json_save_path = args.json_save_path
    split_path = os.path.split(json_save_path)
    if not os.path.exists(split_path[0]):
        os.makedirs(split_path[0])

    cmd = "git clone {}".format("https://github.com/acl-org/acl-anthology.git")
    print("Running " + cmd)
    os.system(cmd)

    # find the paths all XML files
    lst_of_xml_files = glob.glob("acl-anthology/data/xml/*.xml")

    all_conf = {}

    for path in tqdm(lst_of_xml_files):
        with open(path) as xml_file:
            xml = xml_file.read()
            if xml.find("<abstract>") == -1:
                continue
            tree = ET.fromstring(xml)

            for vol_node in tree.iter("volume"):

                # year
                year = vol_node.find(".//year")
                if year is None:
                    continue
                else:
                    year = innertext(year)

                if year not in all_conf:
                    all_conf[year] = {}

                # publisher
                publisher = vol_node.find(".//publisher")
                if publisher is None:
                    publisher_keys = list(all_conf[year].keys())
                    max_anon = -1
                    for publisher_key in publisher_keys:
                        if publisher_key.startswith("anonymous"):
                            max_anon = max(int(publisher_key.split("_")[1]), max_anon)
                    if max_anon == -1:
                        publisher = "anonymous_1"
                    else:
                        publisher = "anonymous_" + str(max_anon + 1)
                else:
                    publisher = innertext(publisher)

                if publisher not in all_conf[year]:
                    all_conf[year][publisher] = {}

                # booktitle
                booktitle = vol_node.find(".//booktitle")
                if booktitle is None:
                    booktitle_keys = list(all_conf[year][publisher].keys())
                    max_anon = -1
                    for booktitle_key in booktitle_keys:
                        if booktitle_key.startswith("anonymous"):
                            max_anon = max(int(booktitle_key.split("_")[1]), max_anon)
                    if max_anon == -1:
                        booktitle = "anonymous_1"
                    else:
                        booktitle = "anonymous_" + str(max_anon + 1)
                else:
                    booktitle = innertext(booktitle)

                if booktitle not in all_conf[year][publisher]:
                    all_conf[year][publisher][booktitle] = {}

                month = vol_node.find(".//month")
                if month is not None:
                    all_conf[year][publisher][booktitle]["month"] = innertext(month)
                else:
                    all_conf[year][publisher][booktitle]["month"] = None

                url = vol_node.find(".//url")
                if url is not None:
                    all_conf[year][publisher][booktitle]["url"] = innertext(url)
                else:
                    all_conf[year][publisher][booktitle]["url"] = None

                all_conf[year][publisher][booktitle]["papers"] = []

                for node in vol_node.iter("paper"):
                    paper_dict = {}
                    lang_flag = 1
                    paper_dict["authors"] = []
                    author_name = ""
                    for elem in node.iter():
                        if not elem.tag == node.tag:

                            if elem.tag == "language":
                                if elem.text != "eng":
                                    lang_flag = 0
                                    break

                            if elem.tag == "author":
                                continue
                            elif elem.tag == "first":
                                if elem.text is not None:
                                    author_name = elem.text
                            elif elem.tag == "last":
                                if elem.text is not None:
                                    author_name += " " + elem.text
                                paper_dict["authors"].append(author_name)
                            else:
                                paper_dict[elem.tag] = innertext(elem)
                                if elem.tag == "abstract":
                                    if (
                                        translator.detect(paper_dict["abstract"]).lang
                                        != "en"
                                    ):
                                        lang_flag = 0
                                        break
                                    paper_dict["preprocessed_abstract"] = preprocess(
                                        paper_dict["abstract"]
                                    )

                    if lang_flag == 1:
                        all_conf[year][publisher][booktitle]["papers"].append(
                            paper_dict
                        )

    with open(json_save_path, "w") as f:
        json.dump(all_conf, f)


if __name__ == "__main__":
    main()
