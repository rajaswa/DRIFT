import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from io import StringIO

from bs4 import BeautifulSoup
from googletrans import Translator
from tika import parser
from tqdm.auto import tqdm

from src.utils.preprocess import preprocess_text


translator = Translator()


def innertext(elt):
    return (elt.text or "") + (
        "".join(innertext(e) + (e.tail or "") for e in elt) or ""
    )


def get_abstract_from_pdf(paper_url):
    print(f"Getting PDF from URL: {paper_url}")
    _buffer = StringIO()
    data = parser.from_file(paper_url, xmlContent=True)
    xhtml_data = BeautifulSoup(data["content"])

    page_1_content = xhtml_data.find_all("div", attrs={"class": "page"})[0]
    _buffer.write(str(page_1_content))
    parsed_content = parser.from_buffer(_buffer.getvalue(), xmlContent=True)
    _buffer.truncate()
    paragraphs_content = BeautifulSoup(parsed_content["content"]).find_all("p")
    flag_for_abs_format = 0

    for idx, paragraph_content in enumerate(paragraphs_content):
        paragraph_content_text = paragraph_content.text.strip().replace(" ", "").lower()

        if paragraph_content_text == "abstract":
            break
        if paragraph_content_text == "résumé-abstract":
            flag_for_abs_format = 1
            break
        if (
            (
                paragraph_content_text[:9] == "abstract:"
                and len(paragraph_content_text) > 9
            )
            or (
                paragraph_content_text[:8] == "abstract"
                and len(paragraph_content_text) > 8
            )
            or (
                paragraph_content_text[:9] == "abstract."
                and len(paragraph_content_text) > 9
            )
        ):
            flag_for_abs_format = 2
            break

    if flag_for_abs_format == 0:
        abstract_text = paragraphs_content[idx + 1].text.strip()
    elif flag_for_abs_format == 1:
        abstract_text = paragraphs_content[idx + 2].text.strip()
    else:
        abstract_text = paragraphs_content[idx].text.strip()

    # basic preprocessing
    abstract_text_lines = abstract_text.split("\n")
    abstract_text_lines = [
        abstract_text_line.strip() for abstract_text_line in abstract_text_lines
    ]

    abstract_text = ""
    for line_no in range(len(abstract_text_lines)):
        abstract_text_lines[line_no] = (
            abstract_text_lines[line_no]
            .strip()
            .replace("\xad", "-")
            .replace("\u00ad", "-")
            .replace("\N{SOFT HYPHEN}", "-")
            .replace("\xa0", " ")
        )
        if line_no == 0:
            abstract_text = abstract_text_lines[line_no]
        else:
            if abstract_text[-1] == "-":
                abstract_text = abstract_text[:-1] + abstract_text_lines[line_no]
            else:
                abstract_text = abstract_text + " " + abstract_text_lines[line_no]
    return abstract_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_save_path",
        type=str,
        default="data/acl_anthology.json",
        help=".json path where the file is to saved",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Whether to use preprocess abstract during JSON creation.",
    )
    parser.add_argument(
        "--use_pdf",
        action="store_true",
        default=False,
        help="Whether to use PDF to get the abstract instead of the XML.",
    )
    args = parser.parse_args()
    json_save_path = args.json_save_path
    split_path = os.path.split(json_save_path)
    if not os.path.exists(split_path[0]):
        os.makedirs(split_path[0])

    cmd = "git clone {}".format("https://github.com/acl-org/acl-anthology.git")
    print("Running " + cmd)
    os.system(cmd)

    lst_of_xml_files = glob.glob("acl-anthology/data/xml/*.xml")
    lst_of_xml_files.sort()

    all_conf = {}

    for path in tqdm(lst_of_xml_files):
        with open(path) as xml_file:
            xml = xml_file.read()
            if not args.use_pdf and xml.find("<abstract>") == -1:
                continue
            tree = ET.fromstring(xml)

            for vol_node in tree.iter("volume"):

                year = vol_node.find(".//year")
                if year is None:
                    continue
                else:
                    year = innertext(year)

                if year not in all_conf:
                    all_conf[year] = []

                publisher_keys = []
                booktitle_keys = []
                common_info_dict = {}

                publisher = vol_node.find(".//publisher")

                if publisher is None:
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

                publisher_keys.append(publisher)
                common_info_dict["publisher"] = publisher

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

                booktitle_keys.append(booktitle)
                common_info_dict["book_title"] = booktitle

                month = vol_node.find(".//month")
                if month is not None:
                    common_info_dict["month"] = innertext(month)
                else:
                    common_info_dict["month"] = None

                url = vol_node.find(".//url")
                if url is not None:
                    common_info_dict["url"] = innertext(url)
                else:
                    common_info_dict["url"] = None

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
                                if not args.use_pdf:
                                    paper_dict[elem.tag] = innertext(elem)
                                    if elem.tag == "abstract":
                                        if (
                                            translator.detect(
                                                paper_dict["abstract"]
                                            ).lang
                                            != "en"
                                        ):
                                            lang_flag = 0
                                            break
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
                                                abstract_text = get_abstract_from_pdf(
                                                    paper_url
                                                )
                                                paper_dict["abstract"] = abstract_text
                                            except Exception as e:
                                                continue
                                if args.preprocess:
                                    paper_dict[
                                        "preprocessed_abstract"
                                    ] = preprocess_text(paper_dict["abstract"])

                    if lang_flag == 1:
                        paper_dict.update(common_info_dict)
                        all_conf[year].append(paper_dict)

    with open(json_save_path, "w") as f:
        json.dump(all_conf, f)


if __name__ == "__main__":
    main()
