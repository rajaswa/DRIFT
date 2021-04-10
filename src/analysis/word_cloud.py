import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud


nltk.download("stopwords")
nltk.download("wordnet")


def make_word_cloud(corpus, save_path):
    stop_words = set(stopwords.words("english"))
    wordcloud = WordCloud(
        background_color="white",
        stopwords=stop_words,
        max_words=100,
        max_font_size=25,
        random_state=42,
    ).generate(str(corpus))
    wordcloud.to_file(save_path)
