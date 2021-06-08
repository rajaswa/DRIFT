import inspect
import os


# Need this here to prevent errors
os.environ["PERSISTENT"] = "True"
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from streamlit import caching
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

from preprocess_and_save_txt import preprocess_and_save
from src.analysis.semantic_drift import find_most_drifted_words
from src.analysis.track_trends_sim import compute_similarity_matrix_years
from src.analysis.similarity_acc_matrix import (
    compute_acc_between_years,
    compute_acc_heatmap_between_years,
    compute_acceleration_matrix,
)
from src.analysis.tracking_clusters import kmeans_clustering
from src.utils import get_word_embeddings, plotly_line_dataframe, word_cloud
from src.utils.misc import get_sub, get_super, reduce_dimensions
from src.utils.statistics import find_productivity, freq_top_k
from src.utils.viz import plotly_heatmap, plotly_scatter
from train_twec import train


# Folder selection not directly support in Streamlit as of now
# https://github.com/streamlit/streamlit/issues/1019
# import tkinter as tk
# from tkinter import filedialog

# root = tk.Tk()
# root.withdraw()
# # Make folder picker dialog appear on top of other windows
# root.wm_attributes('-topmost', 1)

np.random.seed(42)


@st.cache(allow_output_mutation=True)
def get_df():
    return {}


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_values_from_indices(lst, idx_list):
    return [lst[idx] for idx in idx_list], [
        lst[idx] for idx in range(len(lst)) if idx not in idx_list
    ]


def get_component(component_var, typ, params):
    return component_var.__getattribute__(typ)(**params)


def plot(obj, col1, col2, typ="plotly"):
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

    name = col2.text_input("Name", value="MyFigure")
    format = col2.selectbox(
        "Format", formats, help="The format to be used to save the file."
    )
    # Caveat: This only works on local host. See comment https://github.com/streamlit/streamlit/issues/1019#issuecomment-813001320
    # Caveat 2: The folder selection can only be done once and not repetitively
    # dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))

    if col2.button("Save"):
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


def generate_components_from_dict(comp_dict, variable_params):
    vars = {}

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
        vars[component_key] = get_component(component_var, typ, params)

    return vars


def generate_analysis_components(analysis_type, variable_params):
    # print(ANALYSIS_METHODS[analysis_type])
    # print(variable_params)
    sidebar_summary_text.write(ANALYSIS_METHODS[analysis_type]["SUMMARY"])
    figure1_title.header(analysis_type)

    vars = generate_components_from_dict(
        ANALYSIS_METHODS[analysis_type]["COMPONENTS"], variable_params
    )
    return vars

def get_tail_from_data_path(data_path):
    return os.path.split(data_path)[-1].split(".")[0]

def get_years_from_data_path(data_path):
    years = sorted(
        [fil.split(".")[0] for fil in os.listdir(data_path) if fil != "compass.txt"]
    )
    return years


def read_text_file(data_path, name):
    with open(os.path.join(data_path, name + ".txt"), encoding="utf-8") as f:
        words = f.read()
    return words


def get_productivity_for_range(
    start_year, end_year, selected_ngrams, years, data_path, n
):
    yearss = []
    words = []
    prodss = []
    start_year_idx = years.index(start_year)
    end_year_idx = years.index(end_year)
    for year_idx in range(start_year_idx, end_year_idx + 1):
        year = years[year_idx]
        year_text = read_text_file(data_path, year)
        prods = find_productivity(selected_ngrams, year_text, n)
        for word, productivity in prods.items():
            yearss.append(year)
            words.append(word)
            prodss.append(productivity)
    productivity_df = pd.DataFrame.from_dict(
        {"Year": yearss, "Word": words, "Productivity": prodss}
    )
    return productivity_df


def get_acceleration_bw_models(
    year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
):
    model_path1 = os.path.join(model_path, year1 + ".model")
    model_path2 = os.path.join(model_path, year2 + ".model")

    word_pairs, em1, em2 = compute_acc_between_years(
        selected_ngrams,
        model_path1,
        model_path2,
        all_model_vectors=all_model_vectors,
        top_k_acc=top_k_acc,
        skip_same_word_pairs=True,
        skip_duplicates=True,
    )
    return word_pairs, em1, em2


def get_word_pair_sim_bw_models(
    year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
):
    word_pairs, em1, em2 = get_acceleration_bw_models(
        year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
    )
    word_pair_sim_df = pd.DataFrame(
        list(word_pairs.items()), columns=["Word Pair", "Acceleration"]
    )
    word_pair_sim_df = word_pair_sim_df.sort_values(by="Acceleration", ascending=False)

    word_pair_sim_df_words = []
    for word1, word2 in word_pair_sim_df["Word Pair"].values:
        if word1 not in word_pair_sim_df_words:
            word_pair_sim_df_words.append(word1)
        if word2 not in word_pair_sim_df_words:
            word_pair_sim_df_words.append(word2)
    return word_pair_sim_df, word_pair_sim_df_words


st.set_page_config(
    page_title="Diachronic Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    "Productivity Plot": dict(
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
                    min_value=1,
                    format="%d",
                    help="Top-K words to be chosen from.",
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
        preprocess_vars = generate_components_from_dict(
            PREPROCESS["COMPONENTS"], variable_params
        )
        if st.button("Preprocess"):
            preprocess_and_save(**preprocess_vars, streamlit=True, component=main)

    with sidebar.beta_expander("Training"):
        train_vars = generate_components_from_dict(TRAIN["COMPONENTS"], variable_params)
        if st.button("Train"):
            train(**train_vars, streamlit=True, component=main)

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
        vars = generate_analysis_components(analysis_type, variable_params)

        # get words
        years = get_years_from_data_path(vars["data_path"])
        with figure1_params.beta_expander("Plot Parameters"):
            selected_year = st.select_slider(
                label="Year",
                options=years,
                help="Year for which world cloud is to be generated",
            )

        words = read_text_file(vars["data_path"], selected_year)

        # plot
        col1, col2 = figure1_plot.beta_columns([8, 2])
        with st.spinner("Plotting"):
            word_cloud_image = word_cloud(
                words=words,
                stop_words=COMMON["STOPWORDS"],
                max_words=vars["max_words"],
                min_font_size=vars["min_font_size"],
                max_font_size=vars["max_font_size"],
                background_color=vars["background_color"],
                width=vars["width"],
                height=vars["height"],
            )
            plot(word_cloud_image, col1, col2, typ="PIL")

    elif analysis_type == "Productivity Plot":
        variable_params = get_default_args(freq_top_k)
        vars = generate_analysis_components(analysis_type, variable_params)

        years = get_years_from_data_path(vars["data_path"])
        compass_text = read_text_file(vars["data_path"], "compass")

        choose_list_freq = freq_top_k(
            compass_text, top_k=vars["top_k"], n=vars["n"], normalize=vars["normalize"]
        )
        choose_list = list(choose_list_freq.keys())

        with figure1_params.beta_expander("Plot Parameters"):
            selected_ngrams = st.multiselect(
                "Selected N-grams", default=choose_list, options=choose_list
            )

            start_year, end_year = st.select_slider(
                "Range in years", options=years, value=(years[0], years[-1])
            )
            plot_title = st.text_input(
                label="Plot Title",
                value=f"{analysis_type} for range {start_year}-{end_year}",
            )

        productivity_df = get_productivity_for_range(
            start_year, end_year, selected_ngrams, years, vars["data_path"], vars["n"]
        )
        n_gram_freq_df = pd.DataFrame(
            list(choose_list_freq.items()), columns=["N-gram", "Frequency"]
        )

        # plot
        col1, col2 = figure1_block.beta_columns([8, 2])
        with st.spinner("Plotting"):
            fig = plotly_line_dataframe(
                productivity_df,
                x_col="Year",
                y_col="Productivity",
                word_col="Word",
                title=plot_title,
            )
            plot(fig, col1, col2)
        col1.dataframe(n_gram_freq_df.T)

    elif analysis_type == "Acceleration Plot":
        variable_params = get_default_args(compute_acc_between_years)
        variable_params.update(get_default_args(freq_top_k))
        vars = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars["data_path"])
        compass_text = read_text_file(vars["data_path"], "compass")

        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")

        with figure1_params_expander:
            year1, year2 = st.select_slider(
                "Range in years",
                options=years if vars["p2f"] else years[::-1],
                value=(years[0], years[-1]) if vars["p2f"] else (years[-1], years[0]),
            )

        # TO-DO: Check if n is needed here
        # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

        choose_list_freq = freq_top_k(
            compass_text,
            top_k=vars["top_k"],
            n=1,
            normalize=False,
        )
        choose_list = list(choose_list_freq.keys())

        with figure1_params_expander:
            selected_ngrams = st.multiselect(
                "Selected N-grams", default=choose_list, options=choose_list
            )

        word_pair_sim_df, word_pair_sim_df_words = get_word_pair_sim_bw_models(
            year1,
            year2,
            vars["model_path"],
            selected_ngrams,
            False,
            vars["top_k_acc"],
        )

        with figure1_params_expander:
            st.dataframe(word_pair_sim_df.T)
            plot_year = st.select_slider(
                "Year",
                options=years,
                value=years[-1],
                help="Year for which plot is to be made.",
            )

            plot_words_string = st.text_area(
                label="Words to be plotted", value=",".join(word_pair_sim_df_words)
            )

            typ = st.selectbox(
                "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
            )
            plot_title = st.text_input(
                label="Plot Title",
                value=f"{analysis_type} for year {plot_year} given acc range {year1}-{year2}",
            )
        plot_words = plot_words_string.split(",")

        year_model_path = os.path.join(vars["model_path"], plot_year + ".model")

        word_embeddings = get_word_embeddings(year_model_path, plot_words)

        two_dim_embs = reduce_dimensions(word_embeddings, typ=typ, fit_on_compass=False)

        col1, col2 = figure1_block.beta_columns([8, 2])
        with st.spinner("Plotting"):
            fig = plotly_scatter(
                two_dim_embs[:, 0],
                two_dim_embs[:, 1],
                text_annot=plot_words,
                title=plot_title,
            )
            plot(fig, col1, col2)

    elif analysis_type == "Semantic Drift":
        variable_params = get_default_args(find_most_drifted_words)
        variable_params.update(get_default_args(freq_top_k))
        vars = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars["data_path"])
        compass_text = read_text_file(vars["data_path"], "compass")

        # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

        choose_list_freq = freq_top_k(
            compass_text, top_k=vars["top_k"], n=1, normalize=True
        )
        keywords_list = list(choose_list_freq.keys())

        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")
        with figure1_params_expander:
            year1, year2 = st.select_slider(
                "Range in years",
                options=years,
                value=(years[0], years[-1]),
            )

            selected_ngrams = st.multiselect(
                "Selected N-grams", default=keywords_list, options=keywords_list
            )
            typ = st.selectbox(
                "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
            )
            plot_title = st.text_input(
                label="Plot Title", value=f"{analysis_type} for range {year1}-{year2}"
            )

        model_path_1 = os.path.join(vars["model_path"], year1 + ".model")
        model_path_2 = os.path.join(vars["model_path"], year2 + ".model")
        compass_model_path = os.path.join(vars["model_path"], "compass.model")

        words, embs = find_most_drifted_words(
            selected_ngrams,
            [model_path_1, model_path_2],
            compass_model_path,
            top_k_sim=vars["top_k_sim"],
            top_k_drift=vars["top_k_drift"],
        )

        two_dim_embs = reduce_dimensions(embs, typ=typ, fit_on_compass=False)
        plot_words = [word.split("_")[0] for word in words]
        plot_years = [word.split("_")[1] for word in words]

        col1, col2 = figure1_block.beta_columns([8, 2])
        with st.spinner("Plotting"):
            fig = plotly_scatter(
                x=two_dim_embs[:, 0],
                y=two_dim_embs[:, 1],
                color_by_values=plot_years,
                text_annot=plot_words,
                title=plot_title,
            )
            plot(fig, col1, col2)

    elif analysis_type == "Tracking Clusters":
        variable_params = get_default_args(kmeans_clustering)
        variable_params.update(get_default_args(freq_top_k))
        vars = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars["data_path"])
        compass_text = read_text_file(vars["data_path"], "compass")

        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")
        with figure1_params_expander:
            selected_year = st.select_slider(
                "Range in years", options=years, value=years[0]
            )

        choose_list_freq = freq_top_k(compass_text, top_k=vars["top_k"], n=1)

        keywords_list = list(choose_list_freq.keys())

        with figure1_params_expander:
            selected_ngrams = st.multiselect(
                "Selected N-grams", default=keywords_list, options=keywords_list
            )
            typ = st.selectbox(
                "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
            )
            plot_title = st.text_input(
                label="Plot Title", value=f"{analysis_type} for year {selected_year}"
            )

        year_model_path = os.path.join(vars["model_path"], selected_year + ".model")

        keywords, embs, labels, k_opt = kmeans_clustering(
            selected_ngrams,
            year_model_path,
            k_opt=None if vars["n_clusters"] == 0 else vars["n_clusters"],
            k_max=vars["max_clusters"],
            method=vars["method"],
        )

        figure1_block.write(f"Optimal Number of Clusters: {k_opt}")

        two_dim_embs = reduce_dimensions(embs, typ=typ, fit_on_compass=False)

        col1, col2 = figure1_block.beta_columns([8, 2])
        with st.spinner("Plotting"):
            fig = plotly_scatter(
                x=two_dim_embs[:, 0],
                y=two_dim_embs[:, 1],
                color_by_values=labels,
                text_annot=keywords,
                title=plot_title,
            )
            plot(fig, col1, col2)

    elif analysis_type == "Acceleration Heatmap":
        variable_params = get_default_args(compute_acc_heatmap_between_years)
        variable_params.update(get_default_args(freq_top_k))
        vars = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars["data_path"])
        compass_text = read_text_file(vars["data_path"], "compass")

        # TO-DO: Check if n is needed here
        # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

        choose_list_freq = freq_top_k(
            compass_text,
            top_k=vars["top_k"],
            n=1,
            normalize=False,
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

        if not vars["p2f"]:
            year_2, year_1 = year_1, year_2

        with figure1_params_expander:
            plot_title = st.text_input(
                label="Plot Title", value=f"{analysis_type} for range {year_1}-{year_2}"
            )

        model_path_1 = os.path.join(vars["model_path"], year_1 + ".model")
        model_path_2 = os.path.join(vars["model_path"], year_2 + ".model")

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
        vars = generate_analysis_components(analysis_type, variable_params)
        years = get_years_from_data_path(vars["data_path"])
        compass_text = read_text_file(vars["data_path"], "compass")

        # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

        choose_list_freq = freq_top_k(
            compass_text, top_k=vars["top_k"], n=1, normalize=True
        )
        keywords_list = list(choose_list_freq.keys())
        figure1_params_expander = figure1_params.beta_expander("Plot Parameters")
        with figure1_params_expander:
            year1, year2 = st.select_slider(
                "Range in years",
                options=years,
                value=(years[0], years[-1]),
            )
            st.text(body=", ".join(keywords_list))
            selected_ngram = st.text_input(label='Type a Word', value="language")
            model_paths = [os.path.join(vars["model_path"], str(year) + ".model") for year in range(int(year1), int(year2)+1)]
            compass_model_path = os.path.join(vars["model_path"], "compass.model")
            if st.button("Generate Dataframe"):
               
                sim_dict = compute_similarity_matrix_years(model_paths, selected_ngram, top_k_sim=vars["top_k_sim"])

                get_df()[
                    "{}{}".format(
                        selected_ngram, get_sub(get_tail_from_data_path(model_paths[0]))
                    )
                ] = [
                    "{}{} ({})".format(
                        k.split("_")[0],
                        get_sub(get_tail_from_data_path(k.split("_")[1])),
                        round(float(sim_dict[k]), 2),
                    )
                    for k in sim_dict
                ]
                # get_df()["add"] = ["fpfe", "onfpo;wnf"] 

            # selected_ngram = st.selectbox(label="Choose a Word", freq)
            # selected_ngram = st.text_input("Type Word")

   


        if get_df() != {}:

            next_word_form = st.form(key='next_word_form')
            next_word = next_word_form.selectbox("Select a Word", [ele.split("(")[0] for ele in list(pd.DataFrame.from_dict(get_df()).iloc[:, -1])])
            gen_next_word_button = next_word_form.form_submit_button(label='Generate Next Word')
            if gen_next_word_button:
                next_word_pure = "".join([i for i in next_word if not i.isdigit()]).strip()
                sim_dict = compute_similarity_matrix_years(model_paths, next_word_pure, top_k_sim=vars["top_k_sim"])

                get_df()[
                    "{}{}".format(
                        next_word_pure, get_sub(get_tail_from_data_path(model_paths[0]))
                    )
                ] = [
                    "{}{} ({})".format(
                        k.split("_")[0],
                        get_sub(get_tail_from_data_path(k.split("_")[1])),
                        round(float(sim_dict[k]), 2),
                    )
                    for k in sim_dict
                ]

            # if st.button("ADD2"):
            #     get_df()["add2"] = ["fpfe", "onfpo;wnf"] 

            st.write(pd.DataFrame.from_dict(get_df()))
        clear_data = st.button(label="Clear Data")
        if clear_data:
            get_df().clear()
