import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud


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


def plotly_scatter_embeddings(
    years, words, embeddings, year_to_color_map=None, save_fig=False
):
    if year_to_color_map is None:
        sorted_unique_years = np.array(sorted(np.unique(list(map(int, years)))))

        sorted_unique_colors = (sorted_unique_years - np.min(sorted_unique_years)) / (
            np.max(sorted_unique_years) - np.min(sorted_unique_years)
        )

        year_to_color_map = dict(zip(sorted_unique_years, sorted_unique_colors))

    fig = go.Figure(
        data=go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode="markers",
            marker_color=list(map(year_to_color_map.get, list(map(int, years)))),
            alpha=0.5,
            cmap="Spectral",
        )
    )

    for i, txt in enumerate(words):
        fig.add_annotation(
            dict(
                font=dict(color="rgba(0,0,200,0.8)", size=12),
                x=embeddings[i, 0],
                y=embeddings[i, 1],
                showarrow=False,
                text=txt,
                textangle=0,
                xanchor="left",
                xref="x",
                yref="y",
            )
        )
    if save_fig:
        fig.write_html("Plotly Plot.html")
    return fig
