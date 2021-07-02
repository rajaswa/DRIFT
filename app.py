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
with open("nltk_pos_tag_list.txt") as f:
    for line in f:
        line_split = line.strip().split("\t")
        tag = line_split[0]
        desc = line_split[1]
        pos_tag_dict[tag] = desc

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


def display_caching_option(element):
    col1, col2 = element.beta_columns(2)
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

    vars_ = generate_components_from_dict(
        ANALYSIS_METHODS[analysis_type]["COMPONENTS"], variable_params
    )
    
    figure1_title.header(analysis_type)
    with about.beta_expander("About"):
        st.markdown(ANALYSIS_METHODS[analysis_type]["ABOUT"], unsafe_allow_html=True)
    return vars_


st.set_page_config(
    page_title="DRIFT",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="./misc/images/logo_letter_png.png",
)

state = _get_state()


# LAYOUT
main = st.beta_container()

title = main.empty()
settings = main.empty()

figure1_block = main.beta_container()
figure2_block = main.beta_container()

figure1_title = figure1_block.empty()
about = figure1_block.empty()
figure1_params = figure1_block.beta_container()
figure1_plot = figure1_block.beta_container()

figure2_title = figure2_block.empty()
figure2_params = figure2_block.beta_container()
figure2_plot = figure2_block.beta_container()

sidebar = st.sidebar.beta_container()
sidebar_image = sidebar.empty()
sidebar_title = sidebar.empty()
sidebar_mode = sidebar.empty()
sidebar_settings = sidebar.beta_container()
sidebar_analysis_type = sidebar.empty()
sidebar_summary_header = sidebar.empty()
sidebar_summary_text = sidebar.empty()
sidebar_parameters = sidebar.beta_container()


# Analysis Methods Resource Bundle
# COMMON RESOURCES

COMMON = dict(
    TITLE="DRIFT: Diachronic Analysis of Scientific Literature",
    SIDEBAR_TITLE="Settings",
    SIDEBAR_SUMMARY_HEADER="Summary",
    STOPWORDS=list(set(stopwords.words("english"))),
)

ANALYSIS_METHODS = {
    "WordCloud": dict(
        ABOUT='''A word cloud, or tag cloud, is a textual data visualization which allows anyone to see in a single glance the words which have the highest frequency within a given body of text. Word clouds are typically used as a tool for processing, analyzing and disseminating qualitative sentiment data.

References:
- [Word Cloud Explorer: Text Analytics based on Word Clouds](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6758829)
- [wordcloud Package](https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html)
- [Free Word Cloud Generator](https://www.freewordcloudgenerator.com)
''',
        SUMMARY="WordClouds are visual representation of text data, used to depict the most frequent words in the corpus. The depictions are usually single words, and the frequency is shown with font size.",
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
        ABOUT=r'''Our main reference for this method is [this paper](https://www.aclweb.org/anthology/W16-2101.pdf).
In short, this paper uses normalized term frequency and term producitvity as their measures.

- **Term Frequency**: This is the normalized frequency of a given term in a given year.
- **Term Productivity**: This is a measure of the ability of the concept to produce new multi-word terms. In our case we use bigrams. For each year *y* and single-word term *t*, and associated *n* multi-word terms *m*, the productivity is given by the entropy:
    
    $$
        e(t,y) = - \sum_{i=1}^{n} \log_{2}(p_{m_{i},y}).p_{m_{i},y}
        \\
        p_{m,y} = \frac{f(m)}{\sum_{i=1}^{n}f(m_{i})}
    $$

Based on these two measures, they hypothesize three kinds of terms:
- **Growing Terms**: Those which have increasing frequency and productivity in the recent years.
- **Consolidated Terms**: Those that are growing in frequency, but not in productivity.
- **Terms in Decline**: Those which have reached an upper bound of productivity and are being used less in terms of frequency.

Then, they perform clustering of the terms based on their frequency and productivity curves over the years to test their hypothesis.
They find that the clusters formed show similar trends as expected.

**Note**: They also evaluate quality of their clusters using pseudo-labels, but we do not use any automated labels here. They also try with and without double-counting multi-word terms, but we stick to double-counting. They suggest it is more explanable.
''',
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
                    format_func=lambda x: x + " : " + pos_tag_dict[x],
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
        ABOUT=r'''This plot is based on the word-pair acceleration over time. Our inspiration for this method is [this paper](https://sci-hub.se/10.1109/ijcnn.2019.8852140).
Acceleration is a metric which calculates how quickly the word embeddings for a pair of word get close together or farther apart. If they are getting closer together, it means these two terms have started appearing more frequently in similar contexts, which leads to similar embeddings.
In the paper, it is described as:

$$
    acceleration(w_{i}, w_{j}) = sim(w_{i}, w_{j})^{t+1} - sim(w_{i}, w_{j})^{t}\\
    sim(w_{i}, w_{j}) = cosine (u_{w_{i}}, u_{w_{j}}) = \frac{u_{w_{i}}.u_{w_{j}}}{\left\lVert u_{w_{i}}\right\rVert . \left\lVert u_{w_{j}}\right\rVert}
$$

Below, we display the top few pairs between the given start and end year in  dataframe, then one can select years and then select word-pairs in the plot parameters expander. A reduced dimension plot is displayed.

**Note**: They suggest using skip-gram method over CBOW for the model. They use t-SNE representation to view the embeddings. But their way of aligning the embeddings is different. They also use some stability measure to find the best Word2Vec model. The also use *Word2Phrase* which we are planning to add soon.

''',
        SUMMARY="A pair of words converges/diverges over time. Acceleration is a measure of convergence of a pair of keywords. This module identifies fast converging keywords and depicts this convergence on a 2-D plot.",
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
                    format_func=lambda x: x + " : " + pos_tag_dict[x],
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
                    help="Top-K words to be reported for acceleration.",
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
        ABOUT=r'''This plot represents the change in meaning of a word over time. This shift is represented on a 2-dimensional representation of the embedding space.
To find the drift of a word, we calculate the distance between the embeddings of the word in the final year and in the initial year. We find the drift for all words and sort them in descending order to find the most drifted words.
We give an option to use one of two distance metrics: Euclidean Distance and Cosine Distance.

$$
    euclidean\_distance = \sqrt{\vec{u}.\vec{u} - 2 \times \vec{u}.\vec{v} + \vec{v}.\vec{v}} \\
    cosine\_distance = 1 - \frac{\vec{u}.\vec{v}}{||\vec{u}||||\vec{v}||}    
$$

We plot top-K (sim.) most similar words around the two representations of the selected word.

In the ```Plot Parameters``` expander, the user can select the range of years over which the drift will be computed. He/She can also select the dimensionality reduction method for plotting the embeddings.

Below the graph, we provide a list of most drifted words (from the top-K keywords). The user can also choose a custom word.

''',
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
                    format_func=lambda x: x + " : " + pos_tag_dict[x],
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
        ABOUT=r'''Word meanings change over time. They come closer or drift apart. In a certain year, words are clumped together, i.e., they belong to one cluster. But over time, clusters can break into two/coalesce together to form one. Unlike the previous module which tracks movement of one word at a time, here, we track the movement of clusters.
We plot the formed clusters for all the years lying in the selected range of years.
**Note:** We give an option to use one of two libraries for clustering: sklearn or faiss. faiss' KMeans implementation is around 10 times faster than sklearn's.

''',
        SUMMARY="Cluster word embeddings for different years and track how these clusters change over a time period.",
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
                    format_func=lambda x: x + " : " + pos_tag_dict[x],
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
        ABOUT=r'''This plot is based on the word-pair acceleration over time. Our inspiration for this method is [this paper](https://sci-hub.se/10.1109/ijcnn.2019.8852140).
Acceleration is a metric which calculates how quickly the word embeddings for a pair of word get close together or farther apart. If they are getting closer together, it means these two terms have started appearing more frequently in similar contexts, which leads to similar embeddings.
In the paper, it is described as:

$$
    acceleration(w_{i}, w_{j}) = sim(w_{i}, w_{j})^{t+1} - sim(w_{i}, w_{j})^{t}\\
    sim(w_{i}, w_{j}) = cosine (u_{w_{i}}, u_{w_{j}}) = \frac{u_{w_{i}}.u_{w_{j}}}{\left\lVert u_{w_{i}}\right\rVert . \left\lVert u_{w_{j}}\right\rVert}
$$

For all the selected keywords, we display a heatmap, where the brightness of the colour determines the value of the acceleration between that pair, i.e., the brightness is directly proportional to the acceleration value.

**Note**: They suggest using skip-gram method over CBOW for the model.

''',
        SUMMARY="A pair of words converges/diverges over time. Acceleration is a measure of convergence of a pair of keywords. This module identifies fast converging keywords and depicts the convergence using a heatmap.",
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
                    format_func=lambda x: x + " : " + pos_tag_dict[x],
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
        ABOUT=r'''In this method, we wish to chart the trajectory of a word/topic from year 1 to year 2. 

To accomplish this, we allow the user to pick a word from year 1. At the same time, we ask the user to provide the desired stride. We search for the most similar word in the next stride years. We keep doing this iteratively till we reach year 2, updating the word at each step.

The user has to select a word and click on ```Generate Dataframe```. This gives a list of most similar words in the next stride years. The user can now iteratively select the next word from the drop-down till the final year is reached.
''',
        SUMMARY="We identify trends by recursively finding most similar words over years. In this way, we are able to chart the trajectory of a word from one year to another.",
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
                    format_func=lambda x: x + " : " + pos_tag_dict[x],
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
        ABOUT=r'''Here, we use the [YAKE Keyword Extraction](https://www.sciencedirect.com/science/article/abs/pii/S0020025519308588) method to extract keywords.

In our code, we use an [open source implementation](https://github.com/LIAAD/yake) of YAKE.

**NOTE:** Yake returns scores which are indirectly proportional to the keyword importance. Hence, we do the following to report the final scores:

$$
new\_score = \frac{1}{10^{5} \times yake\_score}
$$

''',
        SUMMARY="Bar Graph visualisations for keywords (words vs score).",
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
        ABOUT=r'''[Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) is a generative probabilistic model for an assortment of documents, generally used for topic modelling and extraction. LDA clusters the text data into imaginary topics. 

Every topic can be represented as a probability distribution over ngrams and every document can be represented as a probability distribution over these generated topics. 

We train LDA on a corpus where each document contains the abstracts of a particular year. We express every year as a probability distribution of topics.

In the first bar graph, we show how a year can be decomposed into topics. The graphs below the first one show a decomposition of the relevant topics.
''',
        SUMMARY="LDA clusters the text data into imaginary topics. Every topic can be represented as a probability distribution over ngrams and every document can be represented as a probability distribution over these generated topics.",
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
sidebar_image.markdown(
    '<img src="https://i.ibb.co/FV8rwYd/Driftlogo1.png" style="display: block;margin-left: auto;margin-right: auto;width:65px;height:120px;">',
    unsafe_allow_html=True,
)

# with settings.beta_expander("App Settings"):
#     display_caching_option()

mode = sidebar_mode.radio(label="Mode", options=["Train", "Analysis"], index=1)
display_caching_option(sidebar_settings)
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
            filter_pos_tags=vars_["filter_pos_tags"],
            tfidf=vars_["tfidf"],
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
            filter_pos_tags=vars_["filter_pos_tags"],
            tfidf=vars_["tfidf"],
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
                "Dimensionality Reduction Method", options=["umap", "pca", "tsne"]
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
                "Dimensionality Reduction Method", options=["umap", "pca", "tsne"]
            )
            plot_title = st.text_input(
                label="Plot Title", value=f"{analysis_type} for range {year1}-{year2}"
            )

        list_top_k_freq = freq_top_k(
            compass_text,
            top_k=vars_["top_k"],
            n=1,
            normalize=False,
            filter_pos_tags=vars_["filter_pos_tags"],
            tfidf=vars_["tfidf"],
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

        choose_list_freq = freq_top_k(
            compass_text,
            top_k=vars_["top_k"],
            n=1,
            filter_pos_tags=vars_["filter_pos_tags"],
            tfidf=vars_["tfidf"],
        )

        keywords_list = list(choose_list_freq.keys())

        with figure1_params_expander:
            selected_ngrams = st.multiselect(
                "Selected N-grams", default=keywords_list, options=keywords_list
            )
            typ = st.selectbox(
                "Dimensionality Reduction Method", options=["umap", "pca", "tsne"]
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
            filter_pos_tags=vars_["filter_pos_tags"],
            tfidf=vars_["tfidf"],
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
            topic_wise_info = topic_wise_info.sort_values(by=["WT"])
            fig = plotly_histogram(
                topic_wise_info,
                y_label="Word",
                x_label="WT",
                orientation="h",
                title="X",
            )
            st.write(fig)
