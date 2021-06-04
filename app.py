import inspect
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
from streamlit import caching

from src.analysis.semantic_drift import find_most_drifted_words
from src.analysis.similarity_acc_matrix import compute_acc_between_years
from src.analysis.tracking_clusters import kmeans_clustering
from src.utils import get_word_embeddings, plotly_line_dataframe, word_cloud
from src.utils.misc import reduce_dimensions
from src.utils.statistics import find_productivity, freq_top_k
from src.utils.viz import plotly_scatter


# Folder selection not directly support in Streamlit as of now
# https://github.com/streamlit/streamlit/issues/1019
# import tkinter as tk
# from tkinter import filedialog

# root = tk.Tk()
# root.withdraw()
# # Make folder picker dialog appear on top of other windows
# root.wm_attributes('-topmost', 1)

np.random.seed(42)


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
sidebar_summary_header = sidebar.empty()
sidebar_summary_text = sidebar.empty()
sidebar_mode = sidebar.empty()
sidebar_analysis_type = sidebar.empty()
sidebar_parameters = sidebar.beta_container()  # this will be a beta container



# Analysis Methods Resource Bundle
# COMMON RESOURCES
COMMON = dict(
    TITLE="Diachronic", SIDEBAR_TITLE="Settings", SIDEBAR_SUMMARY_HEADER="Summary"
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
            keyword_method=dict(
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
        COMPONENTS=



            data_path = st.sidebar.text_input(
                "Data Dir",
                value="./data/",
                help="Directory path to the folder containing year-wise text files.",
            )
            model_path = st.sidebar.text_input(
                "Model Dir",
                value="./model/",
                help="Directory path to the folder containing year-wise model files.",
            )

            keyword_method = st.sidebar.radio(
                "Keyword Method", options=["Frequency", "Norm Frequency"]
            )  # Needs format_func
            top_k = st.sidebar.number_input(
                "K", value=variable_params["top_k_acc"], min_value=1, format="%d"
            )

            top_k_keywords = st.sidebar.number_input(
                "K (for keywords selection)",
                value=variable_params["top_k"],
                min_value=1,
                format="%d",
            )
    ),
    "Semantic Drift": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
    ),
    "Tracking Clusters": dict(
        ABOUT="",
        SUMMARY="A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.",
    ),
}
# SIDEBAR COMMON SETUP
title.title(COMMON["TITLE"])
sidebar_title.title(COMMON["SIDEBAR_TITLE"])

analysis_type = sidebar_analysis_type.selectbox(
    label="Analysis Type",
    options=list(ANALYSIS_METHODS.keys()),
    help="The type of analysis you want to perform.",
)

sidebar_summary_header.header(COMMON["SIDEBAR_SUMMARY_HEADER"])


if analysis_type == "WordCloud":
    variable_params = get_default_args(word_cloud)
    sidebar_summary_text.write(ANALYSIS_METHODS[analysis_type]["SUMMARY"])
    figure1_title.header(analysis_type)
    vars = {}

    for component_key, component_dict in ANALYSIS_METHODS[analysis_type][
        "COMPONENTS"
    ].items():
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

    years = sorted(
        [
            fil.split(".")[0]
            for fil in os.listdir(vars["data_path"])
            if fil != "compass.txt"
        ]
    )
    with figure1_params.beta_expander("Plot Parameters"):
        selected_year = st.select_slider(
            label="Year",
            options=years,
            help="Year for which world cloud is to be generated",
        )
    # TO-DO: Check if more stopwords are needed.
    stop_words = list(set(stopwords.words("english")))

    with open(
        os.path.join(vars["data_path"], selected_year + ".txt"), encoding="utf-8"
    ) as f:
        words = f.read()
    col1, col2 = figure1_plot.beta_columns([8, 2])
    with st.spinner("Plotting"):
        word_cloud_image = word_cloud(
            words=words,
            stop_words=stop_words,
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
    sidebar_summary_text.write(ANALYSIS_METHODS[analysis_type]["SUMMARY"])
    vars = {}

    for component_key, component_dict in ANALYSIS_METHODS[analysis_type][
        "COMPONENTS"
    ].items():
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

    years = sorted(
        [
            fil.split(".")[0]
            for fil in os.listdir(vars["data_path"])
            if fil != "compass.txt"
        ]
    )
    stop_words = list(set(stopwords.words("english")))

    with open(os.path.join(vars["data_path"], "compass.txt"), encoding="utf-8") as f:
        compass_text = f.read()

    choose_list_freq = freq_top_k(
        compass_text,
        top_k=vars["top_k"],
        n=vars["n"],
        normalize=vars["keyword_method"]
    )
    choose_list = list(choose_list_freq.keys())

    with figure1_params.beta_expander("Plot Parameters"):
        selected_ngrams = st.multiselect(
            "Selected N-grams", default=choose_list, options=choose_list
        )

        start_year, end_year = st.select_slider(
            "Range in years", options=years, value=(years[0], years[-1])
        )

    yearss = []
    words = []
    prodss = []

    start_year_idx = years.index(start_year)
    end_year_idx = years.index(end_year)
    for year_idx in range(start_year_idx, end_year_idx + 1):
        year = years[year_idx]
        with open(
            os.path.join(vars["data_path"], year + ".txt"), encoding="utf-8"
        ) as f:
            year_text = f.read()
            prods = find_productivity(selected_ngrams, year_text, vars["n"])
            for word, productivity in prods.items():
                yearss.append(year)
                words.append(word)
                prodss.append(productivity)
    productivity_df = pd.DataFrame.from_dict(
        {"Year": yearss, "Word": words, "Productivity": prodss}
    )
    n_gram_freq_df = pd.DataFrame(
        list(choose_list_freq.items()), columns=["N-gram", "Frequency"]
    )
    col1, col2 = figure1_block.beta_columns([8, 2])

    with st.spinner("Plotting"):
        fig = plotly_line_dataframe(
            productivity_df, x_col="Year", y_col="Productivity", word_col="Word"
        )
        plot(fig, col1, col2)
    col1.dataframe(n_gram_freq_df.T)

elif analysis_type == "Acceleration Plot":
    variable_params = get_default_args(compute_acc_between_years)
    variable_params.update(get_default_args(freq_top_k))
    sidebar_summary_text.write(
        ANALYSIS_METHODS[analysis_type]["SUMMARY"]
    )

    st.info(
        "Note that all words may not be present in all years. In that case mean of all word vectors is taken."
    )

    vars = {}

    for component_key, component_dict in ANALYSIS_METHODS[analysis_type][
        "COMPONENTS"
    ].items():
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

    years = sorted(
        [fil.split(".")[0] for fil in os.listdir(data_path) if fil != "compass.txt"]
    )

    with open(os.path.join(data_path, "compass.txt"), encoding="utf-8") as f:
        compass_text = f.read()
    p2f = st.sidebar.checkbox("""Past→Future""", value=True)
    year1, year2 = st.sidebar.select_slider(
        "Range in years",
        options=years if p2f else years[::-1],
        value=(years[0], years[-1]) if p2f else (years[-1], years[0]),
    )
    
    # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

    choose_list_freq = freq_top_k(
        compass_text,
        top_k=top_k_keywords,
        n=1,
        normalize=keyword_method == "Norm Frequency",
    )
    keywords_list = list(choose_list_freq.keys())

    compass_words = compass_text.replace("\n", " ").split()

    selected_ngrams = st.sidebar.multiselect(
        "Selected N-grams", default=keywords_list, options=compass_words
    )

    model_path1 = os.path.join(model_path, year1 + ".model")
    model_path2 = os.path.join(model_path, year2 + ".model")

    word_pairs, em1, em2 = compute_acc_between_years(
        selected_ngrams,
        model_path1,
        model_path2,
        all_model_vectors=acc_default_values_dict["all_model_vectors"],
        top_k_acc=top_k,
        skip_same_word_pairs=True,
        skip_duplicates=True,
    )
    word_pair_sim_df = pd.DataFrame(
        list(word_pairs.items()), columns=["Word Pair", "Acceleration"]
    )
    word_pair_sim_df = word_pair_sim_df.sort_values(by="Acceleration", ascending=False)
    st.dataframe(word_pair_sim_df.T)

    plot_year = st.select_slider(
        "Year",
        options=years,
        value=years[-1],
        help="Year for which plot is to be made.",
    )
    word_pair_sim_df_words = []
    for word1, word2 in word_pair_sim_df["Word Pair"].values:
        if word1 not in word_pair_sim_df_words:
            word_pair_sim_df_words.append(word1)
        if word2 not in word_pair_sim_df_words:
            word_pair_sim_df_words.append(word2)

    plot_words_string = st.text_area(
        label="Words to be plotted", value=",".join(word_pair_sim_df_words)
    )
    plot_words = plot_words_string.split(",")

    year_model_path = os.path.join(model_path, plot_year + ".model")

    word_embeddings = get_word_embeddings(year_model_path, plot_words)

    # compass_embeddings = None
    # if st.checkbox("Fit on Compass"):
    #     compass_embeddings = None
    #     compass_words = compass_text.replace("\n"," ").split() # TO-DO: CHECK THIS APPROACH
    #     compass_model_path = os.path.join(model_path, "compass.model")
    #     compass_embeddings = get_word_embeddings(compass_model_path, compass_words)

    typ = st.selectbox(
        "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
    )
    two_dim_embs = reduce_dimensions(word_embeddings, typ=typ, fit_on_compass=False)

    fig = plotly_scatter(two_dim_embs[:, 0], two_dim_embs[:, 1], text_annot=plot_words)

    col1, col2 = st.beta_columns([8, 2])
    plot(fig, col1, col2)


elif analysis_type == "Semantic Drift":
    text_place_holder.write(
        "We find the top k words which have drifted the most WRT some fixed words."
    )

    drif_default_values_dict = get_default_args(find_most_drifted_words)
    freq_default_values_dict = get_default_args(freq_top_k)

    data_path = st.sidebar.text_input(
        "Data Dir",
        value="./data/",
        help="Directory path to the folder containing year-wise text files.",
    )
    model_path = st.sidebar.text_input(
        "Model Dir",
        value="./model/",
        help="Directory path to the folder containing year-wise model files.",
    )

    years = sorted(
        [fil.split(".")[0] for fil in os.listdir(data_path) if fil != "compass.txt"]
    )

    with open(os.path.join(data_path, "compass.txt"), encoding="utf-8") as f:
        compass_text = f.read()
    p2f = st.sidebar.checkbox("""Past→Future""", value=True)
    year1, year2 = st.sidebar.select_slider(
        "Range in years",
        options=years if p2f else years[::-1],
        value=(years[0], years[-1]) if p2f else (years[-1], years[0]),
    )
    keyword_method = st.sidebar.radio(
        "Keyword Method", options=["Frequency", "Norm Frequency"]
    )  # Needs format_func
    top_k_for_sim = st.sidebar.number_input(
        "K for sim",
        value=drif_default_values_dict["top_k_for_sim"],
        min_value=1,
        format="%d",
    )
    top_k_for_drift = st.sidebar.number_input(
        "K for drift",
        value=drif_default_values_dict["top_most_drifted_k"],
        min_value=1,
        format="%d",
    )
    top_k_keywords = st.sidebar.number_input(
        "K (for keywords selection)",
        value=freq_default_values_dict["top_k"],
        min_value=1,
        format="%d",
    )
    # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

    choose_list_freq = freq_top_k(
        compass_text,
        top_k=top_k_keywords,
        n=1,
        normalize=keyword_method == "Norm Frequency",
    )
    keywords_list = list(choose_list_freq.keys())

    compass_words = compass_text.replace("\n", " ").split()
    selected_ngrams = st.sidebar.multiselect(
        "Selected N-grams", default=keywords_list, options=compass_words
    )

    model_path_1 = os.path.join(model_path, year1 + ".model")
    model_path_2 = os.path.join(model_path, year2 + ".model")
    compass_model_path = os.path.join(model_path, "compass.model")

    words, embs = find_most_drifted_words(
        selected_ngrams,
        [model_path_1, model_path_2],
        compass_model_path,
        top_k_for_sim=top_k_for_sim,
        top_most_drifted_k=top_k_for_drift,
    )
    typ = st.selectbox(
        "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
    )
    two_dim_embs = reduce_dimensions(embs, typ=typ, fit_on_compass=False)
    plot_words = [word.split("_")[0] for word in words]
    plot_years = [word.split("_")[1] for word in words]

    # plot_unique_words,plot_word_counts = np.unique(plot_words, return_counts=True)
    # same_words = []
    # for idx, plot_word_count in enumerate(plot_word_counts):
    #     if plot_word_count>1:
    #         same_words.append(plot_unique_words[idx])

    # plot_same_words_indices = [idx for idx,word in enumerate(plot_words) if word in same_words]
    # plot_dstnct_words_indices = [idx for idx,word in enumerate(plot_words) if word not in same_words]

    # plot_same_words = get_values_from_indices(plot_words,  plot_same_words_indices)
    # plot_dstnct_words = get_values_from_indices(plot_words,  plot_dstnct_words_indices)
    # plot_same_years = get_values_from_indices(plot_years,  plot_same_words_indices)
    # plot_dstnct_years = get_values_from_indices(plot_years,  plot_dstnct_words_indices)
    # plot_same_embs = np.array(get_values_from_indices(two_dim_embs,  plot_same_words_indices))
    # plot_dstnct_embs = np.array(get_values_from_indices(two_dim_embs,  plot_dstnct_words_indices))

    # print(plot_dstnct_words)
    # fig = px.line(x = plot_same_embs[:,0], y = plot_same_embs[:,1], color = plot_same_years,text=plot_same_words)
    fig = plotly_scatter(
        x=two_dim_embs[:, 0],
        y=two_dim_embs[:, 1],
        color_by_values=plot_years,
        text_annot=plot_words,
    )
    # fig = plotly_add_lines_to_scatter(fig, plot_same_embs[:,0], plot_same_embs[:,1], color_by_values=plot_same_years, text_annot=plot_same_words)
    col1, col2 = st.beta_columns([8, 2])
    plot(fig, col1, col2)


elif analysis_type == "Tracking Clusters":
    text_place_holder.write(
        "We find the top k words which have drifted the most WRT some fixed words."
    )

    kmeans_default_values_dict = get_default_args(kmeans_clustering)
    freq_default_values_dict = get_default_args(freq_top_k)

    data_path = st.sidebar.text_input(
        "Data Dir",
        value="./data/",
        help="Directory path to the folder containing year-wise text files.",
    )
    model_path = st.sidebar.text_input(
        "Model Dir",
        value="./model/",
        help="Directory path to the folder containing year-wise model files.",
    )

    years = sorted(
        [fil.split(".")[0] for fil in os.listdir(data_path) if fil != "compass.txt"]
    )

    selected_year = st.sidebar.select_slider(
        "Range in years", options=years, value=years[0]
    )

    with open(os.path.join(data_path, "compass.txt"), encoding="utf-8") as f:
        compass_text = f.read()

    top_k_keywords = st.sidebar.number_input(
        "K (for keywords selection)",
        value=freq_default_values_dict["top_k"],
        min_value=1,
        format="%d",
    )
    # n = st.sidebar.number_input("N", value=freq_default_values_dict['n'], min_value=1, format="%d", help="N in N-gram for productivity calculation.")

    choose_list_freq = freq_top_k(compass_text, top_k=top_k_keywords, n=1)

    keywords_list = list(choose_list_freq.keys())

    compass_words = compass_text.replace("\n", " ").split()
    selected_ngrams = st.sidebar.multiselect(
        "Selected N-grams", default=keywords_list, options=compass_words
    )

    n_clusters = st.sidebar.number_input(
        "Number of clusters",
        value=0,
        min_value=0,
        format="%d",
    )
    max_clusters = st.sidebar.number_input(
        "Max number of clusters",
        value=kmeans_default_values_dict["k_max"],
        min_value=1,
        format="%d",
    )

    method = st.sidebar.selectbox("Method", options=["faiss", "sklearn"])
    year_model_path = os.path.join(model_path, selected_year + ".model")

    keywords, embs, labels, k_opt = kmeans_clustering(
        selected_ngrams,
        year_model_path,
        k_opt=None if n_clusters == 0 else n_clusters,
        k_max=max_clusters,
        method=method,
    )

    st.write(f"Optimal Number of Clusters: {k_opt}")
    typ = st.selectbox(
        "Dimensionality Reduction Method", options=["tsne", "pca", "umap"]
    )
    two_dim_embs = reduce_dimensions(embs, typ=typ, fit_on_compass=False)

    fig = plotly_scatter(
        x=two_dim_embs[:, 0],
        y=two_dim_embs[:, 1],
        color_by_values=labels,
        text_annot=keywords,
    )
    col1, col2 = st.beta_columns([8, 2])
    plot(fig, col1, col2)
