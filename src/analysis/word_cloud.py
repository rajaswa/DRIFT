import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud


nltk.download("stopwords")
nltk.download("wordnet")


def make_word_cloud(corpus, save_path, truncate_popular=200):
    with open("src/analysis/utils/popular_words.txt", "r") as popular_f:
        popular_words = popular_f.read().split("\n")

    popular_words = popular_words[:truncate_popular]
    stop_words = set(stopwords.words("english") + popular_words)
    wordcloud = WordCloud(
        background_color="black",
        stopwords=stop_words,
        max_words=100,
        max_font_size=25,
        random_state=42,
    ).generate(str(corpus))
    # wordcloud.to_file(save_path)
    wordcloud_svg = wordcloud.to_svg(embed_font=True)
    with open(save_path, "w+") as wordcloud_f:
        wordcloud_f.write(wordcloud_svg)
