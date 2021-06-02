import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
import wordcloud
from wordcloud import WordCloud
from wordcloud.wordcloud import STOPWORDS


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
    # # Caveat: This only works on local host. See comment https://github.com/streamlit/streamlit/issues/1019#issuecomment-813001320
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
    years = ["1990", "2016", "2017", "2020"]
    selected_year = st.sidebar.select_slider("Year", options=years)
    max_words = st.sidebar.number_input(
        "Max number of words", min_value=10, format="%d"
    )
    col1, col2 = st.beta_columns([8, 2])
    word_cloud = (
        WordCloud(
            background_color="white",
            width=int(selected_year) // 2,
            height=int(selected_year) // 2,
            random_state=42,
        )
        .generate(
            "Temporary wordcloud. Just testing this utility by adding random text here."
        )
        .to_array()
    )
    plot(word_cloud, col1, col2, typ="array")

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
