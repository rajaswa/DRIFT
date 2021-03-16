"""Visualization utils."""

import numpy as np
from IPython.core.display import HTML, display


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 220
        sat = 100
        lig = 100 - int(127 * attr)
    else:
        hue = 220
        sat = 100
        lig = 100 - int(-125 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def format_special_tokens(token):
    """Convert <> to # if there are any HTML syntax tags.

    Example: '<Hello>' will be converted to '#Hello' to avoid confusion
    with HTML tags.

    Args:
        token (str): The token to be formatted.
    Returns:
        (str): The formatted token.
    """
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_word_importances(words, importances):
    if np.isnan(importances[0]):
        importances = np.zeros_like(importances)

    assert len(words) <= len(importances)
    tags = ["<div>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_special_tokens(word)
        for character in word:  ## Printing Weird Words
            if ord(character) >= 128:
                print(word)
                break
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</div>")
    return HTML("".join(tags))


def format_word_colors(words, colors):
    assert len(words) == len(colors)
    tags = ["<div style='width:50%;'>"]
    for word, color in zip(words, colors):
        word = format_special_tokens(word)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</div>")
    return HTML("".join(tags))


def display_html(html):
    display(html)


def save_to_file(html, path):
    with open(path, "w") as f:
        f.write(html.data)
