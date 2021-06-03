import matplotlib.pyplot as plt
import nltk
import numpy as np
import plotly.express as px
from nltk.corpus import stopwords
from wordcloud import WordCloud
import plotly.graph_objects as go



nltk.download("stopwords")
nltk.download("wordnet")


def pyplot_scatter_embeddings(
    x,
    y,
    color_by_values,
    text_annot=None,
    color_map=None,
    save_path=None,
    color_bar_label="Years",
):
    if color_map is None:
        sorted_unique_color_values = np.array(
            sorted(np.unique(list(map(int, color_by_values))))
        )

        sorted_unique_color_values_rescaled = (
            sorted_unique_color_values - np.min(sorted_unique_color_values)
        ) / (np.max(sorted_unique_color_values) - np.min(sorted_unique_color_values))

        value_to_color_map = dict(
            zip(sorted_unique_color_values, sorted_unique_color_values_rescaled)
        )

    fig = plt.figure(figsize=(10, 10))

    sc = plt.scatter(
        x,
        y,
        c=list(map(value_to_color_map.get, list(map(int, color_by_values)))),
        alpha=0.5,
        cmap="Spectral",
    )

    if text_annot is not None:
        ax = plt.gca()
        for i, txt in enumerate(text_annot):
            ax.annotate(txt, (x[i], y[i]))
    cbar = plt.colorbar(
        sc, ticks=sorted_unique_color_values_rescaled + 1 / 12, label=color_bar_label
    )
    cbar.ax.set_yticklabels(sorted_unique_color_values)
    cbar.ax.axes.tick_params(length=0)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plotly_scatter(x, y, color_by_values=None, text_annot=None, save_path=None):

    fig = px.scatter(
        x=x,
        y=y,
        color=color_by_values,
        text=text_annot,
        color_continuous_scale="Spectral",
    )

    fig.update_traces(textposition="top center")
    if save_path:
        fig.write_html(save_path)
    return fig

def word_cloud(
    words,
    max_words=100,
    stop_words=None,
    save_path=None,
    min_font_size=10,
    max_font_size=25,
    background_color="#FFFFFF",
    width=500,
    height=500,
    collocations=True,
):
    """ Returns PIL WordCloud Image """
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    word_cloud_obj = WordCloud(
        background_color=background_color,
        stopwords=stop_words,
        max_words=max_words,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        width=width,
        height=height,
        random_state=42,
        collocations=collocations,
    ).generate(words)

    if save_path:
        word_cloud_svg = word_cloud_obj.to_svg(embed_font=True)
        with open(save_path, "w+") as word_cloud_f:
            word_cloud_f.write(word_cloud_svg)
    return word_cloud_obj.to_image()


def plotly_line_dataframe(df, x_col, y_col, word_col, save_path=None):

    fig = px.line(df, x=x_col, y=y_col, color=word_col)
    if save_path:
        fig.write_html(save_path)
    return fig
