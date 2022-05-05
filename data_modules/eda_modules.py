from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns

def word_cloud_from_text(text, nlp, custom_stopwords, titre=""):
    stopwords = list(nlp.Defaults.stop_words)
    if custom_stopwords:
        stopwords += custom_stopwords

    wordcloud = WordCloud(background_color="white", stopwords=stopwords, max_words=50, scale=10).generate_from_text(text)

    return wordcloud

def barplot_from_data(data, x, y):
    fig, ax = plt.subplots()
    sns.barplot(x=x, y=y, data=data, ax=ax)
    return fig

def get_tophashtags(tweets):
    return tweets.apply(lambda x: pd.value_counts(re.findall('(#\w+)', x.lower() ))).sum(axis=0).to_frame().reset_index().sort_values(by=0,ascending=False)
