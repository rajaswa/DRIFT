from io import StringIO
from bs4 import BeautifulSoup
from tika import parser

import os
import argparse
import glob
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import json

import re
import string
import unicodedata

from googletrans import Translator

import contractions

from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import inflect

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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--json_save_path",
        type=str,
        default="data/acl_anthology.json",
        help=".json path where the file is to saved",
    )
    args = arg_parser.parse_args()
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
                                if elem.tag == "abstract":
                                    continue
                                paper_dict[elem.tag] = innertext(elem)

                                if elem.tag == "url":

                                    paper_url = innertext(elem)
                                    if not (
                                        paper_url.startswith("http")
                                        or paper_url.startswith("www")
                                    ):
                                        paper_url = (
                                            "https://www.aclweb.org/anthology/"
                                            + paper_url
                                        )
                                    if not (paper_url.endswith(".pdf")):
                                        paper_url = paper_url + ".pdf"

                                    try:
                                        _buffer = StringIO()
                                        data = parser.from_file(
                                            paper_url, xmlContent=True
                                        )
                                        xhtml_data = BeautifulSoup(data["content"])

                                        page_1_content = xhtml_data.find_all(
                                            "div", attrs={"class": "page"}
                                        )[0]
                                        _buffer.write(str(page_1_content))
                                        parsed_content = parser.from_buffer(
                                            _buffer.getvalue(), xmlContent=True
                                        )
                                        _buffer.truncate()
                                        paragraphs_content = BeautifulSoup(
                                            parsed_content["content"]
                                        ).find_all("p")
                                        flag_for_abs_format = 0
                                        for idx, paragraph_content in enumerate(
                                            paragraphs_content
                                        ):
                                            if (
                                                paragraph_content.text.strip()
                                                .strip(".")
                                                .replace(" ", "")
                                                .lower()
                                                == "abstract"
                                            ):
                                                break
                                            if (
                                                paragraph_content.text.strip()
                                                .strip(".")
                                                .replace(" ", "")
                                                .lower()
                                                == "résumé-abstract"
                                            ):
                                                flag_for_abs_format = 1
                                                break

                                        if flag_for_abs_format == 0:
                                            abstract_text = paragraphs_content[
                                                idx + 1
                                            ].text.strip()
                                        else:
                                            abstract_text = paragraphs_content[
                                                idx + 2
                                            ].text.strip()

                                        # basic preprocessing
                                        abstract_text_lines = abstract_text.split("\n")
                                        abstract_text_lines = [
                                            abstract_text_line.strip()
                                            for abstract_text_line in abstract_text_lines
                                        ]

                                        abstract_text = ""
                                        for line_no in range(len(abstract_text_lines)):
                                            abstract_text_lines[
                                                line_no
                                            ] = abstract_text_lines[line_no].strip()
                                            if line_no == 0:
                                                abstract_text = abstract_text_lines[
                                                    line_no
                                                ]
                                            else:
                                                if abstract_text[-1] == "-":
                                                    abstract_text = (
                                                        abstract_text[:-1]
                                                        + abstract_text_lines[line_no]
                                                    )
                                                else:
                                                    abstract_text = (
                                                        abstract_text
                                                        + " "
                                                        + abstract_text_lines[line_no]
                                                    )

                                        paper_dict["abstract"] = abstract_text
                                        paper_dict[
                                            "preprocessed_abstract"
                                        ] = preprocess(abstract_text)
                                    except Exception as e:
                                        # print(e)
                                        continue

                    if lang_flag == 1:
                        all_conf[year][publisher][booktitle]["papers"].append(
                            paper_dict
                        )

        with open(json_save_path, "w") as f:
            json.dump(all_conf, f)


if __name__ == "__main__":
    main()
