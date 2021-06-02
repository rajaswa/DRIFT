import matplotlib.pyplot as plt
from nltk import collocations
import numpy as np
import plotly.express as px
import streamlit as st
import os
from streamlit import caching
from nltk.corpus import stopwords
from src.utils.viz import word_cloud
import inspect

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

# Folder selection not directly support in Streamlit as of now
# https://github.com/streamlit/streamlit/issues/1019
# import tkinter as tk
# from tkinter import filedialog

# root = tk.Tk()
# root.withdraw()
# # Make folder picker dialog appear on top of other windows
# root.wm_attributes('-topmost', 1)

np.random.seed(42)


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


st.title("Diachronic Analysis")
df = px.data.gapminder().query("country=='Canada'")
fig = px.line(df, x="year", y="lifeExp", title="Life expectancy in Canada")

st.sidebar.title("Dummy Title")
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    [
        "WordCloud",
        "Productivity Plot",
        "Acceleration Plot",
        "Semantic Drift",
        "Tracking Clusters",
    ],
    help="The type of analysis you want to perform.",
)
st.sidebar.header("Description")
text_place_holder = st.sidebar.empty()

if analysis_type == "WordCloud":
    text_place_holder.write(
        "A tag cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color."
    )

    default_values_dict = get_default_args(word_cloud)
    data_path = st.sidebar.text_input("Data Path",value="./data/", help="Directory path to the folder containing year-wise text files.")
    years = sorted([fil.split('.')[0] for fil in os.listdir(data_path) if fil!='compass.txt'])
    selected_year = st.sidebar.select_slider("Year", options=years, help="Year for which world cloud is to be generated.")
    max_words = st.sidebar.number_input(
        "Max number of words",value=default_values_dict['max_words'], min_value=10, format="%d"
    )
    min_font_size = st.sidebar.number_input(
        "Min font size", value=default_values_dict['min_font_size'], min_value = 10, max_value=80, format="%d"
    )
    max_font_size = st.sidebar.number_input(
        "Max font size", value = default_values_dict['max_font_size'], min_value = 25, max_value=100, format="%d"
    )
    stop_words = list(set(stopwords.words("english")))
    background_color = st.sidebar.color_picker("Background Color", value=default_values_dict['background_color'])
    width = st.sidebar.number_input(
        "Width", value=default_values_dict['width'], min_value = 100, max_value=10000, format="%d", step=50
    )
    height = st.sidebar.number_input(
        "Height", value=default_values_dict['height'], min_value = 100, max_value=10000, format="%d", step=50
    )
    collocations =  st.checkbox("Collocations", value=default_values_dict['collocations'], help="Whether to include collocations (bigrams) of two words.")

    # TO-DO: Check if more stopwords are needed.

    with open(os.path.join(data_path,selected_year+".txt")) as f:
        words = f.read()
    col1, col2 = st.beta_columns([8, 2])
    with st.spinner("Plotting"):
        word_cloud_image= word_cloud(
                words = words,
                max_words = max_words,
                stop_words = stop_words,
                min_font_size = min_font_size,
                max_font_size = max_font_size,
                background_color=background_color,
                width=width,
                height=height,
            )

    plot(word_cloud_image, col1, col2, typ="PIL")

elif analysis_type == "Productivity Plot":
    text_place_holder.write(
        "Term productivity, that is, a measure for the ability of a concept (lexicalised as a singleword term) to produce new, subordinated concepts (lexicalised as multi-word terms)."
    )
    unigrams = ["this", "is", "a", "dummy", "list"]
    years = ["1990", "2016", "2017", "2020"]
    selected_unigram = st.sidebar.selectbox("Unigram", options=unigrams)
    keyword_method = st.sidebar.radio(
        "Keyword Method", options=["Frequency", "Norm Frequency"]
    )
    start_year, end_year = st.sidebar.select_slider(
        "Range in years", options=years, value=(years[0], years[-1])
    )
    top_k = st.sidebar.number_input("K in Top-K", min_value=1, format="%d")

elif analysis_type == "Acceleration Plot":
    temp = ["the", "blue", "ball"] * 100
    text_place_holder.write(
        "Identification of fast converging keywords over X Number of years."
    )
    years = ["1990", "2016", "2017", "2020"]
    start_year, end_year = st.sidebar.select_slider(
        "Range in years", options=years, value=(years[0], years[-1])
    )
    keyword_method = st.sidebar.radio(
        "Keyword Method", options=["Frequency", "Norm Frequency"]
    )  # Needs format_func
    top_k = st.sidebar.number_input("K in Top-K", min_value=1, format="%d")

    list_of_words = temp[:top_k]
    selected_values = st.selectbox(label="Top-K words", options=list_of_words)

elif analysis_type == "Semantic Drift":
    text_place_holder.write(
        "We find the top k words which have drifted the most WRT some fixed words."
    )
    years = ["1990", "2016", "2017", "2020"]
    start_year, end_year = st.sidebar.select_slider(
        "Range in years", options=years, value=(years[0], years[-1])
    )
    keyword_method = st.sidebar.radio(
        "Keyword Method", options=["Frequency", "Norm Frequency"]
    )  # Needs format_func
    top_k = st.sidebar.number_input("K in Top-K", min_value=1, format="%d")

elif analysis_type == "Tracking Clusters":
    text_place_holder.write(
        "See how clusters of word vectors change in various years.."
    )
    years = ["1990", "2016", "2017", "2020"]
    start_year, end_year = st.sidebar.select_slider(
        "Range in years", options=years, value=(years[0], years[-1])
    )
    keyword_method = st.sidebar.radio(
        "Keyword Method", options=["Frequency", "Norm Frequency"]
    )  # Needs format_func
    top_k = st.sidebar.number_input("K in Top-K", min_value=1, format="%d")
