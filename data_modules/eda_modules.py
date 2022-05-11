from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
from streamlit_folium import folium_static
import folium
import random
import streamlit as st


def word_cloud_from_text(text, nlp, custom_stopwords, titre=""):
    '''Renvoie un wordcloud à partir du texte
    '''
    stopwords = list(nlp.Defaults.stop_words)
    if custom_stopwords:
        stopwords += custom_stopwords

    wordcloud = WordCloud(background_color="white", stopwords=stopwords, max_words=50, scale=10).generate_from_text(text)

    return wordcloud


def simple_barplot(data):
    '''Renvoie un diagramme en batons à partir des données data
    '''
    fig, ax = plt.subplots()
    sns.barplot(data=data, ax=ax)
    return fig

def barplot_from_data(data, x, y):
    '''Renvoie un diagramme en batons à partir des données data
    x,y : chaines de caractères à afficher sur le dessin
    data : données en deux dimensions [x, y]
    '''
    fig, ax = plt.subplots()
    sns.barplot(x=y, y=x, data=data, ax=ax)
    return fig

def barplot_x_y(x, y):
    '''Renvoie un diagramme en batons à partir d'itérables x,y
    '''
    fig, ax = plt.subplots()
    sns.barplot(x=x, y=y, ax=ax)
    return fig

def get_tophashtags(tweets):
    '''Renvoie les hashtags les plus populaires
    '''
    hashtags = tweets.apply(lambda x: pd.value_counts(re.findall('(#\w+)', x.lower() ))).sum(axis=0).to_frame().reset_index().sort_values(by=0,ascending=False)
    hashtags.columns = ['hashtag','occurences']

    return hashtags


def get_topmentions(tweets):
    mentions = tweets.str.findall('(@[A-Za-z0-9]+)').apply(lambda x: pd.value_counts(x)).sum(axis=0).to_frame().reset_index().sort_values(by=0,ascending=False)
    mentions.columns = ['mention', 'occurences']

    return mentions

def get_locations_by_occurences(geoTweets):
    '''Renvoie un dataframe avec les colonnes [lat, lon] à partir d'un vecteur de chaines ['lat,lon,rayon']
    '''
    locations = pd.DataFrame([x.split(",") for x in geoTweets])
    locations.columns = ['lat', 'lon', 'rayon']
    locations = locations[['lat', 'lon']]
    locations.dropna(axis='columns', inplace=True)

    return (locations.groupby(['lat', 'lon'])).size().sort_values(ascending=False).reset_index(name='occ')

def map_from_locations(geoTweets):
    '''Crée une map à partir d'un vecteur de positions GPS ['lat,lon,rayon']
    '''
    grouped_locations = get_locations_by_occurences(geoTweets)
    m = folium.Map(location=[46.227638, 2.213749], zoom_start=5.5) # latitude et longitude du centre de la France
    mean = grouped_locations['occ'].mean() 


    for i in range(len(grouped_locations)):
        folium.Circle(
            radius=int(grouped_locations.loc[i].occ)*mean/2,
            location=[float(grouped_locations.loc[i].lat), float(grouped_locations.loc[i].lon)],
            color=generate_random_color(),
            fill=True,
        ).add_to(m)

    folium_static(m)


def generate_random_color():
    '''Génère une couleur aléatoire au format héxadécimal
    '''
    color = "%06x" % random.randint(0, 0xFFFFFF)
    return '#'+str(color)