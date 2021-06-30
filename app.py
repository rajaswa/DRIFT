import glob
import os


# Need this here to prevent errors
os.environ["PERSISTENT"] = "True"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import streamlit as st
from nltk.corpus import stopwords
from streamlit import caching

from app_utils import (
    get_curve_hull_objects,
    get_default_args,
    get_dict_with_new_words,
    get_frequency_for_range,
    get_productivity_for_range,
    get_word_pair_sim_bw_models,
    get_years_from_data_path,
    read_text_file,
    word_to_entry_dict,
)
from preprocess_and_save_txt import preprocess_and_save
from session import _get_state
from src.analysis.productivity_inference import cluster_productivity
from src.analysis.semantic_drift import find_most_drifted_words
from src.analysis.similarity_acc_matrix import (
    compute_acc_between_years,
    compute_acc_heatmap_between_years,
)
from src.analysis.topic_extraction_lda import extract_topics_lda
from src.analysis.track_trends_sim import compute_similarity_matrix_years
from src.analysis.tracking_clusters import kmeans_clustering, kmeans_embeddings
from src.utils import get_word_embeddings, plotly_line_dataframe, word_cloud
from src.utils.misc import get_sub, reduce_dimensions
from src.utils.statistics import freq_top_k, yake_keyword_extraction
from src.utils.viz import (
    embs_for_plotting,
    plotly_heatmap,
    plotly_histogram,
    plotly_scatter,
    plotly_scatter_df,
)
from train_twec import train


template = "simple_white"
pio.templates.default = template
plotly_template = pio.templates[template]
colorscale = plotly_template.layout["colorscale"]["diverging"]


# Folder selection not directly support in Streamlit as of now
# https://github.com/streamlit/streamlit/issues/1019
# import tkinter as tk
# from tkinter import filedialog

# root = tk.Tk()
# root.withdraw()
# # Make folder picker dialog appear on top of other windows
# root.wm_attributes('-topmost', 1)

np.random.seed(42)

pos_tag_dict = {}
with open('nltk_pos_tag_list.txt') as f:
    for line in f:
        line_split = line.strip().split('\t')
        tag = line_split[0]
        desc = line_split[1]
        pos_tag_dict[tag]=desc

# @st.cache(allow_output_mutation=True)
# def get_sim_dict():
#     return OrderedDict()


def plot(obj, col1, col2, typ="plotly", key="key"):
    formats = [".eps", ".pdf", ".svg", ".png", ".jpeg", ".webp"]
    col2.header("Save your graph")
    if typ == "plotly":
        formats.append(".html")
        col1.plotly_chart(obj, use_container_width=True)
        col2.write(
            "**Note:** You can also save a particular portion of the graph using the camera icon on the chart."
        )
    elif typ == "pyplot":
        col1.pyplot(obj)
    elif typ == "array":
        col1.image(obj)
    elif typ == "PIL":
        formats.remove(".svg")
        col1.image(obj)

    name = col2.text_input("Name", value="MyFigure", key=key)
    format = col2.selectbox(
        "Format", formats, help="The format to be used to save the file.", key=key
    )
    # Caveat: This only works on local host. See comment https://github.com/streamlit/streamlit/issues/1019#issuecomment-813001320
    # Caveat 2: The folder selection can only be done once and not repetitively
    # dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))

    if col2.button("Save", key=key):
        with st.spinner(text="Saving"):
            save_name = f"{name}{format}"
            if typ == "plotly":
                if format == ".html":
                    obj.write_html(save_name)
                else:
                    obj.write_image(save_name, width=1920, height=1080)
            elif typ == "array":
                plt.imsave(save_name, obj)
            elif typ == "pyplot":
                plt.savefig(save_name)
            elif typ == "PIL":
                obj.save(save_name)


def display_caching_option():
    col1, col2 = st.beta_columns(2)
    if col1.checkbox("Persistent Caching", value=False):
        caching._clear_mem_cache()
        os.environ["PERSISTENT"] = "True"
    else:
        caching._clear_disk_cache()
        os.environ["PERSISTENT"] = "False"

    if col2.button("Clear All Cache"):
        caching.clear_cache()


def get_component(component_var, typ, params):
    return component_var.__getattribute__(typ)(**params)


def generate_components_from_dict(comp_dict, variable_params):
    vars_ = {}

    for component_key, component_dict in comp_dict.items():
        component_var = component_dict["component_var"]
        typ = component_dict["typ"]
        params = component_dict["params"]
        component_variable_params = component_dict["variable_params"]
        params.update(
            {
                key: variable_params[value]
                for key, value in component_variable_params.items()
            }
        )
        vars_[component_key] = get_component(component_var, typ, params)

    return vars_


def generate_analysis_components(analysis_type, variable_params):
    sidebar_summary_text.write(ANALYSIS_METHODS[analysis_type]["SUMMARY"])
    figure1_title.header(analysis_type)

    vars_ = generate_components_from_dict(
        ANALYSIS_METHODS[analysis_type]["COMPONENTS"], variable_params
    )
    return vars_


st.set_page_config(
    page_title="Diachronic Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

state = _get_state()


# LAYOUT
main = st.beta_container()

title = main.empty()
settings = main.empty()
about = main.empty()

figure1_block = main.beta_container()
figure2_block = main.beta_container()

figure1_title = figure1_block.empty()
figure1_params = figure1_block.beta_container()
figure1_plot = figure1_block.beta_container()

figure2_title = figure2_block.empty()
figure2_params = figure2_block.beta_container()
figure2_plot = figure2_block.beta_container()

sidebar = st.sidebar.beta_container()
sidebar_title = sidebar.empty()
sidebar_mode = sidebar.empty()
sidebar_analysis_type = sidebar.empty()
sidebar_summary_header = sidebar.empty()
sidebar_summary_text = sidebar.empty()
sidebar_parameters = sidebar.beta_container()


# Analysis Methods Resource Bundle
# COMMON RESOURCES

COMMON = dict(
    TITLE="Diachronic",
    SIDEBAR_TITLE="Settings",
    SIDEBAR_SUMMARY_HEADER="Summary",
    STOPWORDS=list(set(stopwords.words("english"))),
)

ANALYSIS_METHODS = {
    "WordCloud": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
        COMPONENTS=dict(
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            max_words=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "max_words"},
                params=dict(
                    label="#words (max)",
                    min_value=10,
                    format="%d",
                    help="Maximum number of words to be displayed",
                ),
            ),
            min_font_size=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "min_font_size"},
                params=dict(
                    label="Min. font size", min_value=10, max_value=80, format="%d"
                ),
            ),
            max_font_size=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "max_font_size"},
                params=dict(
                    label="Max. font size", min_value=25, max_value=100, format="%d"
                ),
            ),
            background_color=dict(
                component_var=sidebar_parameters,
                typ="color_picker",
                variable_params={"value": "background_color"},
                params=dict(label="Background Color"),
            ),
            width=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "width"},
                params=dict(
                    label="Width", min_value=100, max_value=10000, format="%d", step=50
                ),
            ),
            height=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "height"},
                params=dict(
                    label="Height",
                    min_value=100,
                    max_value=10000,
                    format="%d",
                    step=50,
                ),
            ),
            collocations=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={"value": "collocations"},
                params=dict(
                    label="Collocations",
                    help="Whether to include collocations (bigrams) of two words.",
                ),
            ),
        ),
    ),
    "Productivity/Frequency Plot": dict(
        ABOUT="",
        SUMMARY="Term productivity, that is, a measure for the ability of a concept (lexicalised as a singleword term) to produce new, subordinated concepts (lexicalised as multi-word terms).",
        COMPONENTS=dict(
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            n=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "n"},
                params=dict(
                    label="N",
                    min_value=1,
                    format="%d",
                    help="N in N-gram for productivity calculation.",
                ),
            ),
            top_k=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k"},
                params=dict(
                    label="K",
                    min_value=0,
                    format="%d",
                    help="Top-K words to be chosen from.",
                ),
            ),
            filter_pos_tags=dict(
                component_var=sidebar_parameters,
                typ="multiselect",
                variable_params={"default": "filter_pos_tags"},
                params=dict(
                    label="POS-Tags Filter",
                    options=list(pos_tag_dict.keys()),
                    format_func=lambda x:x+' : '+pos_tag_dict[x],
                    help="The POS Tags that should be selected. If empty, no filtering is done.",
                ),
            ),

            tfidf=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={"value": "tfidf"},
                params=dict(
                    label="Use TF-IDF",
                    help="Whether to use TF-IDF for selection instead of frequency.",
                ),
            ),
            normalize=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={"value": "normalize"},
                params=dict(
                    label="Normalize",
                    help="Whether to use normalize frequency.",
                ),
            ),
        ),
    ),
    "Acceleration Plot": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
        COMPONENTS=dict(
            note=dict(
                component_var=figure1_block,
                typ="info",
                variable_params={},
                params=dict(
                    body="Note that all words may not be present in all years. In that case mean of all word vectors is taken."
                ),
            ),
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            model_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Model Dir",
                    value="./model/",
                    help="Directory path to the folder containing year-wise model files.",
                ),
            ),
            top_k=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k"},
                params=dict(
                    label="K",
                    min_value=1,
                    format="%d",
                    help="Top-K words to be chosen from.",
                ),
            ),
            filter_pos_tags=dict(
                component_var=sidebar_parameters,
                typ="multiselect",
                variable_params={"default": "filter_pos_tags"},
                params=dict(
                    label="POS-Tags Filter",
                    options=list(pos_tag_dict.keys()),
                    format_func=lambda x:x+' : '+pos_tag_dict[x],
                    help="The POS Tags that should be selected. If empty, no filtering is done.",
                ),
            ),

            tfidf=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={"value": "tfidf"},
                params=dict(
                    label="Use TF-IDF",
                    help="Whether to use TF-IDF for selection instead of frequency.",
                ),
            ),
            top_k_sim=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k_sim"},
                params=dict(
                    label="K (sim.)",
                    min_value=2,
                    format="%d",
                    help="Top-K words for similarity.",
                ),
            ),
            top_k_acc=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k_acc"},
                params=dict(
                    label="K (acc.)",
                    min_value=1,
                    format="%d",
                    help="Top-K words for accuracy.",
                ),
            ),
            p2f=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={},
                params=dict(
                    label="Past→Future",
                    value=True,
                    help="Whether to calculate past to future acceleration",
                ),
            ),
        ),
    ),
    "Semantic Drift": dict(
        ABOUT="",
        SUMMARY="Computing the semantic drift of words, i.e.,  change in meaning, by finding the Euclidean/Cosine Distance between two representations of the word from different years and showing the drift on a t-SNE plot",
        COMPONENTS=dict(
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            model_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Model Dir",
                    value="./model/",
                    help="Directory path to the folder containing year-wise model files.",
                ),
            ),
            top_k=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k"},
                params=dict(
                    label="K",
                    value=200,
                    min_value=1,
                    format="%d",
                    help="Top-K words on which we will calculate drift.",
                ),
            ),
            filter_pos_tags=dict(
                component_var=sidebar_parameters,
                typ="multiselect",
                variable_params={"default": "filter_pos_tags"},
                params=dict(
                    label="POS-Tags Filter",
                    options=list(pos_tag_dict.keys()),
                    format_func=lambda x:x+' : '+pos_tag_dict[x],
                    help="The POS Tags that should be selected. If empty, no filtering is done.",
                ),
            ),

            tfidf=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={"value": "tfidf"},
                params=dict(
                    label="Use TF-IDF",
                    help="Whether to use TF-IDF for selection instead of frequency.",
                ),
            ),
            top_k_sim=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k_sim"},
                params=dict(
                    label="K (sim.)",
                    min_value=1,
                    format="%d",
                    help="Top-K words for similarity.",
                ),
            ),
            top_k_drift=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k_drift"},
                params=dict(
                    label="K (drift.)",
                    min_value=1,
                    format="%d",
                    help="Top-K words for drift.",
                ),
            ),
            distance_measure=dict(
                component_var=sidebar_parameters,
                typ="selectbox",
                variable_params={},
                params=dict(
                    label="Distance Method",
                    options=["euclidean", "cosine"],
                    help="Distance Method to compute the drift.",
                ),
            ),
        ),
    ),
    "Tracking Clusters": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
        COMPONENTS=dict(
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            model_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Model Dir",
                    value="./model/",
                    help="Directory path to the folder containing year-wise model files.",
                ),
            ),
            top_k=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k"},
                params=dict(
                    label="K",
                    min_value=1,
                    format="%d",
                    help="Top-K words to be chosen from.",
                ),
            ),
            filter_pos_tags=dict(
                component_var=sidebar_parameters,
                typ="multiselect",
                variable_params={"default": "filter_pos_tags"},
                params=dict(
                    label="POS-Tags Filter",
                    options=list(pos_tag_dict.keys()),
                    format_func=lambda x:x+' : '+pos_tag_dict[x],
                    help="The POS Tags that should be selected. If empty, no filtering is done.",
                ),
            ),

            tfidf=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={"value": "tfidf"},
                params=dict(
                    label="Use TF-IDF",
                    help="Whether to use TF-IDF for selection instead of frequency.",
                ),
            ),
            n_clusters=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={},
                params=dict(
                    label="Number of clusters",
                    min_value=0,
                    value=0,
                    format="%d",
                    help="Number of clusters. If set to 0, optimal number of clusters are found using the silhouette score.",
                ),
            ),
            max_clusters=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "k_max"},
                params=dict(
                    label="Max number of clusters",
                    min_value=1,
                    value=0,
                    format="%d",
                    help="Maximum number of clusters. Used when number of clusters is 0.",
                ),
            ),
            method=dict(
                component_var=sidebar_parameters,
                typ="selectbox",
                variable_params={},
                params=dict(
                    label="Method",
                    options=["faiss", "sklearn"],
                    help="Method to use for K-means. `faiss` is recommended when calculating optimal number of clusters.",
                ),
            ),
        ),
    ),
    "Acceleration Heatmap": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
        COMPONENTS=dict(
            note=dict(
                component_var=figure1_params,
                typ="info",
                variable_params={},
                params=dict(
                    body="Note that all words may not be present in all years. In that case mean of all word vectors is taken."
                ),
            ),
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            model_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Model Dir",
                    value="./model/",
                    help="Directory path to the folder containing year-wise model files.",
                ),
            ),
            top_k=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k"},
                params=dict(
                    label="K",
                    min_value=1,
                    format="%d",
                    help="Top-K words to be chosen from.",
                ),
            ),
            filter_pos_tags=dict(
                component_var=sidebar_parameters,
                typ="multiselect",
                variable_params={"default": "filter_pos_tags"},
                params=dict(
                    label="POS-Tags Filter",
                    options=list(pos_tag_dict.keys()),
                    format_func=lambda x:x+' : '+pos_tag_dict[x],
                    help="The POS Tags that should be selected. If empty, no filtering is done.",
                ),
            ),

            tfidf=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={"value": "tfidf"},
                params=dict(
                    label="Use TF-IDF",
                    help="Whether to use TF-IDF for selection instead of frequency.",
                ),
            ),
            p2f=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={},
                params=dict(
                    label="Year 1 → Year 2",
                    value=True,
                    help="Whether to calculate Year 1 to Year 2 acceleration",
                ),
            ),
        ),
    ),
    "Track Trends with Similarity": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
        COMPONENTS=dict(
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            model_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Model Dir",
                    value="./model/",
                    help="Directory path to the folder containing year-wise model files.",
                ),
            ),
            top_k=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k"},
                params=dict(
                    label="K",
                    min_value=1,
                    format="%d",
                    help="Top-K words to be chosen from.",
                ),
            ),
            filter_pos_tags=dict(
                component_var=sidebar_parameters,
                typ="multiselect",
                variable_params={"default": "filter_pos_tags"},
                params=dict(
                    label="POS-Tags Filter",
                    options=list(pos_tag_dict.keys()),
                    format_func=lambda x:x+' : '+pos_tag_dict[x],
                    help="The POS Tags that should be selected. If empty, no filtering is done.",
                ),
            ),

            tfidf=dict(
                component_var=sidebar_parameters,
                typ="checkbox",
                variable_params={"value": "tfidf"},
                params=dict(
                    label="Use TF-IDF",
                    help="Whether to use TF-IDF for selection instead of frequency.",
                ),
            ),
            top_k_sim=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k_sim"},
                params=dict(
                    label="K (sim.)",
                    min_value=1,
                    format="%d",
                    help="Top-K words for similarity.",
                ),
            ),
            stride=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "stride"},
                params=dict(
                    label="Stride",
                    min_value=1,
                    format="%d",
                    help="Stride",
                ),
            ),
        ),
    ),
    "Keyword Visualisation": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
        COMPONENTS=dict(
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            top_k=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "top_k"},
                params=dict(
                    label="K",
                    min_value=1,
                    format="%d",
                    help="Top-K words to be chosen from.",
                ),
            ),
            max_ngram_size=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "max_ngram_size"},
                params=dict(
                    label="Max Ngram Size",
                    min_value=1,
                    value=2,
                    format="%d",
                    help="N-gram size.",
                ),
            ),
        ),
    ),
    "LDA Topic Modelling": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
        COMPONENTS=dict(
            data_path=dict(
                component_var=sidebar_parameters,
                typ="text_input",
                variable_params={},
                params=dict(
                    label="Data Path",
                    value="./data/",
                    help="Directory path to the folder containing year-wise text files.",
                ),
            ),
            num_topics=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "num_topics"},
                params=dict(
                    label="Number of Topics",
                    min_value=0,
                    value=0,
                    format="%d",
                    help="Number of LDA Topics",
                ),
            ),
            num_words=dict(
                component_var=sidebar_parameters,
                typ="number_input",
                variable_params={"value": "num_words"},
                params=dict(
                    label="Number of Words",
                    min_value=10,
                    value=10,
                    format="%d",
                    help="Number of Words Per Topic.",
                ),
            ),
        ),
    ),
}

PREPROCESS = dict(
    ABOUT="",
    SUMMARY="",
    COMPONENTS=dict(
        json_path=dict(
            component_var=st,
            typ="text_input",
            variable_params={"value": "json_path"},
            params=dict(
                label="JSON Path", help="Path to the JSON file containing raw data"
            ),
        ),
        text_key=dict(
            component_var=st,
            typ="text_input",
            variable_params={"value": "text_key"},
            params=dict(label="Text Key", help="Key in JSON containing the text"),
        ),
        save_dir=dict(
            component_var=st,
            typ="text_input",
            variable_params={"value": "save_dir"},
            params=dict(
                label="Data Path",
                help="Directory path to the folder where you want to store year-wise processed text files.",
            ),
        ),
    ),
)

TRAIN = dict(
    ABOUT="",
    SUMMARY="",
    COMPONENTS=dict(
        data_dir=dict(
            component_var=st,
            typ="text_input",
            variable_params={"value": "data_dir"},
            params=dict(
                label="Data Path",
                help="Directory path to the folder where year-wise processed text files are stored.",
            ),
        ),
        embedding_size=dict(
            component_var=st,
            typ="number_input",
            variable_params={"value": "embedding_size"},
            params=dict(label="Embedding Size", help="Embedding size to be used"),
        ),
        skipgram=dict(
            component_var=st,
            typ="checkbox",
            variable_params={"value": "skipgram"},
            params=dict(label="Skipgram", help="Whether to use skipgram or CBOW"),
        ),
        siter=dict(
            component_var=st,
            typ="number_input",
            variable_params={"value": "siter"},
            params=dict(
                label="Static Iterations",
                help="Iterations when training the compass",  # TO-DO: VERIFY
            ),
        ),
        diter=dict(
            component_var=st,
            typ="number_input",
            variable_params={"value": "diter"},
            params=dict(
                label="Dynamic Iterations",
                help="Iterations when training the slices",  # TO-DO: VERIFY
            ),
        ),
        negative_samples=dict(
            component_var=st,
            typ="number_input",
            variable_params={"value": "negative_samples"},
            params=dict(
                label="Negative Samples",
                help="Number of negative samples to use during training",  # TO-DO: VERIFY
            ),
        ),
        window_size=dict(
            component_var=st,
            typ="number_input",
            variable_params={"value": "window_size"},
            params=dict(
                label="Window Size",
                help="Window size used to create input-output pairs",  # TO-DO: VERIFY
            ),
        ),
        output_path=dict(
            component_var=st,
            typ="text_input",
            variable_params={"value": "output_path"},
            params=dict(
                label="Model Path",
                help="Directory path to the folder where you want to store year-wise model files.",
            ),
        ),
        overwrite_compass=dict(
            component_var=st,
            typ="checkbox",
            variable_params={"value": "overwrite_compass"},
            params=dict(
                label="Overwrite Compass",
                help="Whether to overwrite the compass if it already exists",  # TO-DO: VERIFY
            ),
        ),
    ),
)

# SIDEBAR COMMON SETUP
title.title(COMMON["TITLE"])
sidebar_title.title(COMMON["SIDEBAR_TITLE"])

with settings.beta_expander("App Settings"):
    display_caching_option()

mode = sidebar_mode.radio(label="Mode", options=["Train", "Analysis"], index=1)

if mode == "Train":
    variable_params = get_default_args(train)
    variable_params.update(get_default_args(preprocess_and_save))
    with sidebar.beta_expander("Preprocessing"):
        preprocess_vars_ = generate_components_from_dict(
            PREPROCESS["COMPONENTS"], variable_params
        )
        if st.button("Preprocess"):
            preprocess_and_save(**preprocess_vars_, streamlit=True, component=main)

    with sidebar.beta_expander("Training"):
        train_vars_ = generate_components_from_dict(
            TRAIN["COMPONENTS"], variable_params
        )
        if st.button("Train"):
            train(**train_vars_, streamlit=True, component=main)

elif mode == "Analysis":
    analysis_type = sidebar_analysis_type.selectbox(
        label="Analysis Type",
        options=list(ANALYSIS_METHODS.keys()),
        help="The type of analysis you want to perform.",
    )

    sidebar_summary_header.header(COMMON["SIDEBAR_SUMMARY_HEADER"])
    if analysis_type == "WordCloud":
        # setup
        variable_params = get_default_args(word_cloud)
        vars_ = generate_analysis_components(analysis_type, variable_params)

        # get words
        years = get_years_from_data_path(vars_["data_path"])
        with figure1_params.beta_expander("Plot Parameters"):
            selected_year = st.select_slider(
                label="Year",
                options=years,
                help="Year for which world cloud is to be generated",
            )

        words = read_text_file(vars_["data_path"], selected_year)

        # plot
        col1, col2 = figure1_plot.beta_columns([8, 2])
        with st.spinner("Plotting"):
            word_cloud_image = word_cloud(
                words=words,
                stop_words=COMMON["STOPWORDS"],
                max_words=vars_["max_words"],
                min_font_size=vars_["min_font_size"],
                max_font_size=vars_["max_font_size"],
                background_color=vars_["background_color"],
                width=vars_["width"],
                height=vars_["height"],
            )
            plot(word_cloud_image, col1, col2, typ="PIL")

    elif analysis_type == "Productivity/Frequency Plot":
        variable_params = get_default_args(freq_top_k)
        vars_ = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars_["data_path"])
        compass_text = read_text_file(vars_["data_path"], "compass")

        choose_list_freq = freq_top_k(
            compass_text,
            top_k=vars_["top_k"],
            n=vars_["n"],
            normalize=vars_["normalize"],
            filter_pos_tags = vars_["filter_pos_tags"],
            tfidf = vars_["tfidf"]
        )
        choose_list = list(choose_list_freq.keys())

        with figure1_params.beta_expander("Plot Parameters"):
            selected_ngrams = st.multiselect(
                "Selected N-grams", default=choose_list, options=choose_list
            )
            custom_ngrams = st.text_area(
                "Custom N-grams",
                value="",
                help="A comma-separated list of n-grams. Ensure that you only use the `n` you chose.",
            )

            start_year, end_year = st.select_slider(
                "Range in years", options=years, value=(years[0], years[-1])
            )
            plot_title_prod = st.text_input(
                label="Productivity Plot Title",
                value=f"Productivity Plot for range {start_year}-{end_year}",
            )
            plot_title_freq = st.text_input(
                label="Frequency Plot Title",
                value=f"Frequency Plot for range {start_year}-{end_year}",
            )

        if custom_ngrams.strip() != "":
            custom_ngrams_list = [
                word.strip() for word in custom_ngrams.split(",") if word.strip() != ""
            ]
            for ngram in custom_ngrams_list:
                if len(ngram.split(" ")) != vars_["n"]:
                    raise ValueError(
                        f"Found n-gram: `{ngram}` which does not have the specified value of n: {vars_['n']}."
                    )
            selected_ngrams = selected_ngrams + custom_ngrams_list

        if selected_ngrams == []:
            raise ValueError(
                "Found an empty list of n-grams. Please select some value of K > 0 or enter custom n-grams."
            )
        productivity_df = get_productivity_for_range(
            start_year,
            end_year,
            selected_ngrams,
            years,
            vars_["data_path"],
            2,
            vars_["normalize"],
        )
        frequency_df = get_frequency_for_range(
            start_year,
            end_year,
            selected_ngrams,
            years,
            vars_["data_path"],
            vars_["n"],
            vars_["normalize"],
        )

        final_clusters = cluster_productivity(productivity_df, frequency_df)

        n_gram_freq_df = pd.DataFrame(
            list(choose_list_freq.items()), columns=["N-gram", "Frequency"]
        )

        # plot
        col11, col12 = figure1_block.beta_columns([6, 4])

        with st.spinner("Plotting"):
            fig = plotly_line_dataframe(
                productivity_df,
                x_col="Year",
                y_col="Productivity",
                word_col="Word",
                title=plot_title_prod,
            )
            plot(fig, col11, col12, key="prod")

        col21, col22 = figure2_block.beta_columns([6, 4])

        with st.spinner("Plotting"):
            fig = plotly_line_dataframe(
                frequency_df,
                x_col="Year",
                y_col="Frequency",
                word_col="Word",
                title=plot_title_freq,
            )
            plot(fig, col21, col22, key="freq")
        st.write("Clusters")
        st.write(final_clusters)

    elif analysis_type == "Acceleration Plot":
        variable_params = get_default_args(compute_acc_between_years)
        variable_params.update(get_default_args(freq_top_k))
        variable_params["top_k_sim"] = 10
        variable_params["top_k"] = 200

        vars_ = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars_["data_path"])
        compass_text = read_text_file(vars_["data_path"], "compass")

        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")

        with figure1_params_expander:
            year1, year2 = st.select_slider(
                "Range in years",
                options=years if vars_["p2f"] else years[::-1],
                value=(years[0], years[-1]) if vars_["p2f"] else (years[-1], years[0]),
            )

        # TO-DO: Check if n is needed here
        # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

        choose_list_freq = freq_top_k(
            compass_text,
            top_k=vars_["top_k"],
            n=1,
            normalize=False,
            filter_pos_tags = vars_["filter_pos_tags"],
            tfidf = vars_["tfidf"]
        )
        choose_list = list(choose_list_freq.keys())

        # with figure1_params_expander:
        #     selected_ngrams = st.multiselect(
        #         "Selected N-grams", default=choose_list, options=choose_list
        #     )
        selected_ngrams = choose_list
        word_pair_sim_df, word_pair_sim_df_words = get_word_pair_sim_bw_models(
            year1,
            year2,
            vars_["model_path"],
            selected_ngrams,
            False,
            vars_["top_k_acc"],
        )

        with figure1_params_expander:
            st.dataframe(word_pair_sim_df.T)
            plot_years = st.multiselect(
                "Select Years",
                options=years,
                default=[year1, year2],
                help="Year for which plot is to be made.",
            )
            plot_word_1 = st.selectbox(
                "Select the first word to be plotted", index=0, options=choose_list
            )
            plot_word_2 = st.selectbox("Select word pair", index=1, options=choose_list)
            target_calc_words = [plot_word_1, plot_word_2]
            typ = st.selectbox(
                "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
            )
            plot_title = st.text_input(
                label="Plot Title",
                value=f"{analysis_type} for year given acceleration range {year1}-{year2}",
            )
        if plot_word_1 == plot_word_2:
            st.error("Please select two different words to calculate acceleration!")

        elif len(plot_years) < 2:
            st.error("Please select at least two years to calculate acceleration!")
        else:
            word_embeddings = []
            plot_words = []
            colors = []
            for plot_year in plot_years:
                year_model_path = os.path.join(
                    vars_["model_path"], plot_year + ".model"
                )
                word_embeddings += get_word_embeddings(
                    year_model_path, target_calc_words
                )
                plot_words += [word + get_sub(plot_year) for word in target_calc_words]
            model_path_1 = os.path.join(vars_["model_path"], year1 + ".model")
            model_path_2 = os.path.join(vars_["model_path"], year2 + ".model")

            similar_words = []
            similarity_embeddings = []
            for plot_word in target_calc_words:
                for model_path_iter in [model_path_1, model_path_2]:
                    similar_words_temp, similarity_embeddings_temp = embs_for_plotting(
                        plot_word,
                        model_path_iter,
                        top_k_sim=vars_["top_k_sim"],
                        skip_words=plot_words,
                    )
                    similar_words += similar_words_temp[1:]
                    similarity_embeddings += similarity_embeddings_temp[1:]

            embeddings = word_embeddings + similarity_embeddings
            plot_words += similar_words
            two_dim_embs = reduce_dimensions(embeddings, typ=typ, fit_on_compass=False)
            col1, col2 = figure1_block.beta_columns([8, 2])
            colors += [
                word if "_" not in word else word.split("_")[-1] for word in plot_words
            ]
            plot_words = [
                word.split("_")[0] if "_" in word else word for word in plot_words
            ]

            with st.spinner("Plotting"):
                fig = plotly_scatter(
                    two_dim_embs[:, 0],
                    two_dim_embs[:, 1],
                    text_annot=plot_words,
                    title=plot_title,
                    color_by_values=colors,
                )
                plot(fig, col1, col2)

    elif analysis_type == "Semantic Drift":
        variable_params = get_default_args(find_most_drifted_words)
        variable_params.update(get_default_args(freq_top_k))
        del variable_params["words"]
        variable_params["top_k_sim"] = 10
        variable_params["top_k"] = 100
        # print(variable_params)

        vars_ = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars_["data_path"])
        compass_text = read_text_file(vars_["data_path"], "compass")

        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")
        with figure1_params_expander:
            year1, year2 = st.select_slider(
                "Range in years",
                options=years,
                value=(years[0], years[-1]),
            )

            typ = st.selectbox(
                "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
            )
            plot_title = st.text_input(
                label="Plot Title", value=f"{analysis_type} for range {year1}-{year2}"
            )

        list_top_k_freq = freq_top_k(
            compass_text,
            top_k=vars_["top_k"],
            n=1,
            normalize=False,
            filter_pos_tags = vars_["filter_pos_tags"],
            tfidf = vars_["tfidf"]
        )
        list_top_k_freq = list(list_top_k_freq.keys())

        model_path_1 = os.path.join(vars_["model_path"], year1 + ".model")
        model_path_2 = os.path.join(vars_["model_path"], year2 + ".model")
        distance_dict = find_most_drifted_words(
            year_1_path=model_path_1,
            year_2_path=model_path_2,
            words=list_top_k_freq,
            top_k_drift=vars_["top_k_drift"],
            distance_measure=vars_["distance_measure"],
        )
        selected_ngrams = st.selectbox(
            "Select N-grams from list", index=0, options=list(distance_dict.keys())
        )
        selected_ngrams_text = st.text_input(
            "Write your own N-gram to analyze", value=""
        )
        if selected_ngrams_text != "":
            selected_ngrams = selected_ngrams_text

        words1, embs1 = embs_for_plotting(
            selected_ngrams, model_path_1, top_k_sim=vars_["top_k_sim"]
        )
        words2, embs2 = embs_for_plotting(
            selected_ngrams, model_path_2, top_k_sim=vars_["top_k_sim"]
        )
        words = words1 + words2
        embs = embs1 + embs2

        two_dim_embs = reduce_dimensions(embs, typ=typ, fit_on_compass=False)

        plot_years = [word.split("_")[1] for word in words]
        plot_words = [word.split("_")[0] for word in words]
        plot_text = [
            word + get_sub(year) if word in selected_ngrams else word
            for word, year in zip(plot_words, plot_years)
        ]
        col1, col2 = figure1_block.beta_columns([8, 2])
        with st.spinner("Plotting"):
            fig = plotly_scatter(
                x=two_dim_embs[:, 0],
                y=two_dim_embs[:, 1],
                color_by_values=[
                    word if ngram in selected_ngrams else f"Similar Words from {year}"
                    for ngram, word, year in zip(plot_words, plot_text, plot_years)
                ],
                text_annot=plot_text,
                title=plot_title,
            )
            plot(fig, col1, col2)

    elif analysis_type == "Tracking Clusters":
        variable_params = get_default_args(kmeans_clustering)
        variable_params.update(get_default_args(freq_top_k))
        vars_ = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars_["data_path"])
        compass_text = read_text_file(vars_["data_path"], "compass")

        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")
        with figure1_params_expander:
            year1, year2 = st.select_slider(
                "Range in years", options=years, value=(years[-4], years[-1])
            )
        selected_years = [str(i) for i in range(int(year1), int(year2) + 1)]

        choose_list_freq = freq_top_k(compass_text, top_k=vars_["top_k"], n=1, filter_pos_tags = vars_["filter_pos_tags"],
            tfidf = vars_["tfidf"])

        keywords_list = list(choose_list_freq.keys())

        with figure1_params_expander:
            selected_ngrams = st.multiselect(
                "Selected N-grams", default=keywords_list, options=keywords_list
            )
            typ = st.selectbox(
                "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
            )

        for selected_year in selected_years:
            year_model_path = os.path.join(
                vars_["model_path"], selected_year + ".model"
            )
            keywords, embs = get_word_embeddings(
                year_model_path,
                selected_ngrams,
                all_model_vectors=False,
                return_words=True,
                filter_missing_words=True,
            )
            two_dim_embs = reduce_dimensions(embs, typ=typ, fit_on_compass=False)
            labels, k_opt, kmeans = kmeans_embeddings(
                two_dim_embs,
                k_opt=None if vars_["n_clusters"] == 0 else vars_["n_clusters"],
                k_max=vars_["max_clusters"],
                method=vars_["method"],
                return_fitted_model=True,
            )
            figure1_block.write(f"Optimal Number of Clusters: {k_opt}")

            label_to_vertices_map = get_curve_hull_objects(two_dim_embs, labels)

            clusters_df = pd.DataFrame(
                {
                    "X": two_dim_embs[:, 0],
                    "Y": two_dim_embs[:, 1],
                    "Label": list(map(str, labels)),
                    "Word": keywords,
                }
            )
            col1, col2 = figure1_block.beta_columns([8, 2])
            with st.spinner("Plotting"):
                fig = plotly_scatter_df(
                    clusters_df,
                    x_col="X",
                    y_col="Y",
                    color_col="Label",
                    text_annot="Word",
                    title=f"{analysis_type} for year {selected_year}",
                    labels={"Label": "Cluster Label"},
                    colorscale=colorscale,
                    label_to_vertices_map=label_to_vertices_map,
                )
                plot(fig, col1, col2, key="tracking_clusters_" + selected_year)

    elif analysis_type == "Acceleration Heatmap":
        variable_params = get_default_args(compute_acc_heatmap_between_years)
        variable_params.update(get_default_args(freq_top_k))
        vars_ = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars_["data_path"])
        compass_text = read_text_file(vars_["data_path"], "compass")

        # TO-DO: Check if n is needed here
        # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

        choose_list_freq = freq_top_k(
            compass_text,
            top_k=vars_["top_k"],
            n=1,
            normalize=False,
            filter_pos_tags = vars_["filter_pos_tags"],
            tfidf = vars_["tfidf"]
        )
        choose_list = list(choose_list_freq.keys())
        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")
        with figure1_params_expander:
            selected_ngrams = st.multiselect(
                "Selected N-grams", default=choose_list, options=choose_list
            )

        col_1_1, col_1_2 = figure1_params.beta_columns([5, 5])
        year_1 = col_1_1.selectbox("Year 1", options=years, index=0)

        year_2 = col_1_2.selectbox("Year 2", options=years, index=len(years) - 1)

        if not vars_["p2f"]:
            year_2, year_1 = year_1, year_2

        with figure1_params_expander:
            plot_title = st.text_input(
                label="Plot Title", value=f"{analysis_type} for range {year_1}-{year_2}"
            )

        model_path_1 = os.path.join(vars_["model_path"], year_1 + ".model")
        model_path_2 = os.path.join(vars_["model_path"], year_2 + ".model")

        # print(len(selected_ngrams))
        acc_matrix = compute_acc_heatmap_between_years(
            selected_ngrams,
            model_path_1,
            model_path_2,
            False,
        )

        col1, col2 = figure1_plot.beta_columns([9, 1])
        with st.spinner("Plotting"):
            fig = plotly_heatmap(
                acc_matrix,
                x=selected_ngrams,
                y=selected_ngrams,
                title=plot_title,
            )
            plot(fig, col1, col2)

    elif analysis_type == "Track Trends with Similarity":

        variable_params = get_default_args(compute_similarity_matrix_years)
        variable_params["stride"] = 3
        variable_params.update(get_default_args(freq_top_k))
        vars_ = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars_["data_path"])
        compass_text = read_text_file(vars_["data_path"], "compass")
        stride = vars_["stride"]

        # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

        # TO-DO: SHOULD WE SWITCH TO YEAR WISE TEXT INSTEAD?
        choose_list_freq = freq_top_k(
            compass_text, top_k=vars_["top_k"], n=1, normalize=True
        )
        keywords_list = list(choose_list_freq.keys())
        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")
        compass_model_path = os.path.join(vars_["model_path"], "compass.model")
        with figure1_params_expander:
            year1, year2 = st.select_slider(
                "Range in years",
                options=years,
                value=(years[0], years[-1]),
            )
            st.text(body="Top K words: " + ", ".join(keywords_list))

        years = years[years.index(year1) : years.index(year2) + 1]

        model_paths = [
            os.path.join(vars_["model_path"], str(year) + ".model")
            for year in years[: min(stride + 1, len(years))]
        ]
        selected_ngram = figure1_plot.text_input(
            label="Type a Word", value=keywords_list[0]
        )

        if figure1_plot.button("Generate Dataframe"):
            state.sim_dict = get_dict_with_new_words(
                model_paths, selected_ngram, top_k_sim=vars_["top_k_sim"]
            )

        select_list = figure1_plot.empty()

        if state.sim_dict != {} and state.sim_dict is not None:
            state.new_word = select_list.selectbox(
                "Select Next Word",
                [ele.split("(")[0] for ele in list(state.sim_dict.values())[-1]],
            )
            if figure2_plot.button("Generate Next Column"):
                state.sim_dict = {
                    **state.sim_dict,
                    **word_to_entry_dict(
                        state.new_word,
                        year1,
                        year2,
                        years,
                        stride,
                        vars_["top_k_sim"],
                        vars_["model_path"],
                    ),
                }

        if figure2_plot.button(label="Clear Data"):
            state.clear()
            
        if state.sim_dict != {} and state.sim_dict is not None:
            df = pd.DataFrame(state.sim_dict)
            figure2_plot.write(df)
        state.sync()

    elif analysis_type == "Keyword Visualisation":
        variable_params = get_default_args(yake_keyword_extraction)
        variable_params.update(get_default_args(freq_top_k))
        vars_ = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars_["data_path"])

        with figure1_params.beta_expander("Plot Parameters"):
            selected_year = st.select_slider(
                label="Year",
                options=years,
                help="Year for which world cloud is to be generated",
            )
        text_file = os.path.join(vars_["data_path"], selected_year + ".txt")

        keywords_df = yake_keyword_extraction(
            text_file,
            top_k=vars_["top_k"],
            language="en",
            max_ngram_size=vars_["max_ngram_size"],
            window_size=2,
            deduplication_threshold=0.9,
            deduplication_algo="seqm",
        )
        col1, col2 = figure1_block.beta_columns([8, 2])

        with st.spinner("Plotting"):
            # fig = go.Figure(data=[go.Histogram(x=x,y=y)])
            fig = plotly_histogram(
                keywords_df,
                y_label="ngram",
                x_label="score",
                orientation="h",
                title="X",
            )
            plot(fig, col1, col2)

        keywords_df = keywords_df.round(2)
        col1.dataframe(keywords_df.T)

    elif analysis_type == "LDA Topic Modelling":
        variable_params = get_default_args(extract_topics_lda)
        variable_params["num_topics"] = 20
        vars_ = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars_["data_path"])

        with figure1_params.beta_expander("Plot Parameters"):
            selected_year = st.select_slider(
                label="Year",
                options=years,
                help="Year for which the topical analysis is to be shown.",
            )
        year_paths = glob.glob(os.path.join(vars_["data_path"], "*.txt"))
        year_paths.remove(os.path.join(vars_["data_path"], "compass.txt"))
        year_wise_topics, topic_wise_info = extract_topics_lda(
            year_paths, num_topics=vars_["num_topics"], num_words=vars_["num_words"]
        )
        vars_["num_topics"] = len(topic_wise_info)
        topic_wise_info_for_graph = []
        for topic_info in topic_wise_info:
            topic_info_for_graph_wt = []
            topic_info_for_graph_word = []
            info_str = topic_info[1]
            weight_ngram_pairs = info_str.split("+")
            weight_ngram_pairs = [
                (
                    weight_ngram_pair.strip().split("*")[0],
                    weight_ngram_pair.strip().split("*")[1].replace('"', ""),
                )
                for weight_ngram_pair in weight_ngram_pairs
            ]
            for ele in weight_ngram_pairs:
                topic_info_for_graph_wt.append(ele[0])
                topic_info_for_graph_word.append(ele[1])
            df_topic_info = pd.DataFrame()
            df_topic_info["Word"] = topic_info_for_graph_word
            df_topic_info["WT"] = topic_info_for_graph_wt
            topic_wise_info_for_graph.append(df_topic_info)

        selected_year_idx = year_paths.index(
            os.path.join(vars_["data_path"], f"{selected_year}.txt")
        )
        selected_year_topics = year_wise_topics[selected_year_idx]

        dict_for_graph = {}
        for selected_year_topic in selected_year_topics:
            dict_for_graph[int(selected_year_topic[0])] = selected_year_topic[1]

        st.write("Number of Topics: ", vars_["num_topics"])

        topics_not_present = list(
            set([i for i in range(vars_["num_topics"])])
            - set(list(dict_for_graph.keys()))
        )
        for topic_not_present in topics_not_present:
            dict_for_graph[topic_not_present] = np.array(0.0).astype(np.float32)

        df_for_graph = pd.DataFrame.from_dict(
            {
                "Topic": list(dict_for_graph.keys()),
                "Probability": list(dict_for_graph.values()),
            }
        )

        col1, col2 = figure1_block.beta_columns([8, 2])

        with st.spinner("Plotting"):
            fig = plotly_histogram(
                df_for_graph,
                y_label="Topic",
                x_label="Probability",
                orientation="h",
                title="X",
            )
            plot(fig, col1, col2)

        topics_of_interest = [
            key for key in list(set(dict_for_graph.keys()) - set(topics_not_present))
        ]
        topic_wise_info_list = [
            topic_wise_info_for_graph[index] for index in topics_of_interest
        ]

        for topic_wise_info in topic_wise_info_list:
            fig = plotly_histogram(
                topic_wise_info,
                y_label="Word",
                x_label="WT",
                orientation="h",
                title="X",
            )
            st.write(fig)
