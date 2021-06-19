import matplotlib.pyplot as plt
import nltk
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from wordcloud import WordCloud

from src.utils.misc import get_tail_from_data_path

from ..analysis import similarity_acc_matrix


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


def plotly_scatter(
    x, y, color_by_values=None, text_annot=None, title=None, save_path=None
):

    fig = px.scatter(
        x=x,
        y=y,
        color=color_by_values,
        text=text_annot,
        color_continuous_scale="Spectral",
        title=title,
    )

    fig.update_traces(textposition="top center")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    if save_path:
        fig.write_html(save_path)
    return fig


def plotly_scatter_df(
    df, x_col, y_col, color_col=None,size_col=None, facet_col=None, labels=None, text_annot=None, title=None, save_path=None
):

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        facet_col=facet_col,
        text=text_annot,
        title=title,
        labels=labels,
    )

    fig.update_traces(textposition="top center")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    # fig.update_layout(legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.02,
    #     xanchor="right",
    #     x=1
    # ))
    if save_path:
        fig.write_html(save_path)
    return fig

def plotly_histogram(
    df, x_label=None, y_label=None, orientation="h", title=None, save_path=None
):

    fig = px.bar(df, x=x_label, y=y_label, orientation=orientation, title=title)

    fig.update_xaxes(side="top")

    if save_path:
        fig.write_html(save_path)
    return fig


def plotly_heatmap(
    arr,
    x=None,
    y=None,
    x_label=None,
    y_label=None,
    color_label=None,
    title=None,
    save_path=None,
):

    fig = px.imshow(
        arr,
        labels=dict(x=x_label, y=y_label, color=color_label),
        x=x,
        y=y,
        color_continuous_scale="Spectral",
        title=title,
    )

    fig.update_xaxes(side="top")

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


def plotly_line_dataframe(df, x_col, y_col, word_col, title=None, save_path=None):

    fig = px.line(df, x=x_col, y=y_col, color=word_col, title=title)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    if save_path:
        fig.write_html(save_path)
    return fig


def embs_for_plotting(word, year_path, top_k_sim=10, skip_words=[]):
    year = get_tail_from_data_path(year_path)
    (
        keywords,
        embs,
        sim_matrix,
    ) = similarity_acc_matrix.compute_similarity_matrix_keywords(
        model_path=year_path, keywords=[], all_model_vectors=True
    )
    word_idx = keywords.index(word)
    # print(skip_words)
    sim_vector = sim_matrix[word_idx]
    top_sims = np.argsort(sim_vector)
    top_sims = top_sims[-top_k_sim:]

    words = []
    word_embs = []
    words.append(word + "_" + year)
    word_embs.append(embs[word_idx])
    skip_words_modified = [skip_word + "_" + year for skip_word in skip_words]
    for top_sim in top_sims:
        if keywords[top_sim] == word or keywords[top_sim] in skip_words_modified:
            continue
        words.append(keywords[top_sim] + "_" + year)
        word_embs.append(embs[top_sim])
    return words, word_embs
