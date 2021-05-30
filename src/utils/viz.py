import matplotlib.pyplot as plt
import nltk
import numpy as np
import plotly.express as px
from nltk.corpus import stopwords
from wordcloud import WordCloud


nltk.download("stopwords")
nltk.download("wordnet")


def pyplot_scatter_embeddings(
    years, words, embeddings, year_to_color_map=None, save_fig=False
):
    if year_to_color_map is None:
        sorted_unique_years = np.array(sorted(np.unique(list(map(int, years)))))

        sorted_unique_colors = (sorted_unique_years - np.min(sorted_unique_years)) / (
            np.max(sorted_unique_years) - np.min(sorted_unique_years)
        )

        year_to_color_map = dict(zip(sorted_unique_years, sorted_unique_colors))

    fig = plt.figure(figsize=(10, 10))

    sc = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=list(map(year_to_color_map.get, list(map(int, years)))),
        alpha=0.5,
        cmap="Spectral",
    )
    ax = plt.gca()
    for i, txt in enumerate(words):
        ax.annotate(txt, (embeddings[i, 0], embeddings[i, 1]))
    cbar = plt.colorbar(sc, ticks=sorted_unique_colors + 1 / 12, label="Years")
    cbar.ax.set_yticklabels(sorted_unique_years)
    cbar.ax.axes.tick_params(length=0)
    if save_fig:
        plt.savefig("Pyplot Scatter Plot.png", bbox_inches="tight")
        plt.close()
    return fig


def plotly_scatter_embeddings(years, words, embeddings, save_fig=False):

    fig = px.scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        color=years,
        text=words,
        color_continuous_scale="Spectral",
    )

    fig.update_traces(textposition="top center")
    if save_fig:
        fig.write_html("Plotly Scatter Embeddings Plot.html")
    return fig


def word_cloud(words, max_words=100, stop_words=None, save_fig=False):
    """ Returns PIL WordCloud Image """
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    wordcloud = WordCloud(
        background_color="black",
        stopwords=stop_words,
        max_words=max_words,
        max_font_size=25,
        random_state=42,
    ).generate(str(words))

    if save_fig:
        wordcloud_svg = wordcloud.to_svg(embed_font=True)
        with open("WordCloud", "w+") as wordcloud_f:
            wordcloud_f.write(wordcloud_svg)
    return word_cloud.to_image()


def plotly_line_dataframe(df, x_col, y_col, word_col, save_fig=False):

    fig = px.line(
        df, x=x_col, y=y_col, color=word_col, color_continuous_scale="Spectral"
    )
    if save_fig:
        fig.write_html("Plotly  Line Plot.html")
    return fig
