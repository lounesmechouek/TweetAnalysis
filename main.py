# Imports
# ---------------------------
import pandas as pd
import numpy as np

import spacy
import datetime

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

import streamlit as st

from twitter_scrapping import scrapping_modules
from data_modules import preproc_modules
from data_modules import eda_modules
# ---------------------------

# Global variables
searchDone, outliers, valeurs_manquantes, normalisation = [False for i in range(4)]

# Chemin de sauvegarde des fichiers générés :
# L'exécution se fait depuis twtan/Scripts
SAVE_PATH = '../../outputs/'

# Variable temporaire regroupant quelques positions géographiques
# L'idée est d'améliorer cette approche en regroupant toutes les positions dans un fichier csv
allPositionsGeographiques = {
    'Peu importe' : None, 
    'Paris' : '48.8588897,2.320041,10km',
    'Lille' : '50.6365654,3.0635282,10km',
    'Marseille' : '43.2961743,5.3699525,10km',
    'Bretagne' : '48.2640845,-2.9202408,10km',
    'Lyon' : '45.7578137,4.8320114,10km',
    'Bordeaux' : '44.841225,-0.5800364,10km',
    'Nice' : '43.7009358,7.2683912,10km',
    'Toulouse' : '43.6044622,1.4442469,10km',
    'Dijon' : '47.3215806,5.0414701,10km',
    'Rennes' : '48.1113387,-1.6800198,10km',
    'Avignon' : '43.9492493,4.8059012,10km',
    'Nancy' : '48.6937223,6.1834097,10km',
    'Créteil' : '48.7771486,2.4530731,10km'
}

# Liste customisée de stop_words :
custom_stopwords = ["","ci-dessous","j'en","actuellement","etre","faire","voir","france","aller","oui","non","absolument","peut-etre","waw","mtn","trv","bcp","what","ptdrr","parcequ","ehh","allez","dsl","putain","merde","svp","ptn","jsuis","hahahahaha","ici","vraiment","fois","rien","mettre","mdr","bla","aujourd'hui"]

# 
nlp = spacy.load("fr_core_news_lg")

## Début application
st.title("Interface d'analyse de tweets")

# Containers
tweetsFetching = st.container()
dataExploration = st.container()
analyseAvancée = st.container()


# Partie I : Récupération des tweets à analyser
with tweetsFetching:
    st.subheader("Chargement des données")
    with st.expander("Recherche par mots clés"):
        searchValue = st.text_input('Mots à rechercher', placeholder='Exemple : Présidentielles 2022')
        
        colGauche, colDroite = st.columns(2)
        with colGauche:
            positionsGeographiques = st.multiselect(
                'Localités à prendre en compte',
                allPositionsGeographiques.keys(),
                help ='Si vous selectionnez "Peu importe", toutes les localités seront prises en compte.' +
                        "Attention, parfois à vouloir trop limiter la localisation, il se peut que la recherche renvoie peu de résultats."
                
            )

            dateDebut = st.date_input(
                label = 'A partir de quand ?',
                value = datetime.date.today()
            )

        with colDroite:
            max_tweets_position = st.number_input(
                label = "Nombre de tweets max à récupérer par localité",
                min_value = 20,
                max_value = 10000,   
            )

            dateFin = st.date_input(
                label = "Jusqu'à quand ?",
                value = datetime.date.today()
            )

        # On saute une ligne
        st.text("")
        searchDone = st.button(label = "Rechercher") 

        # searchDone est à true seulement si l'on a appuyé sur le bouton de recherche
        if searchDone:
            if searchValue != "" :
                try:
                    path = SAVE_PATH + "miningTwitter_{}.csv".format(searchValue)

                    #nest_asyncio.apply()

                    scrapping_modules.get_tweets(
                        max_tweets_position,
                        searchValue,
                        path,
                        dateDebut.strftime("%Y-%m-%d"),
                        dateFin.strftime("%Y-%m-%d"),
                        [allPositionsGeographiques[k] for k in positionsGeographiques]
                    )

                    st.success('Récupération de tweets terminée avec succès !')

                    # On lit les données récupérées
                    try:
                        input = pd.read_csv(path)
                        df = input.copy()
                        if "Unnamed: 0" in df.columns: df.drop("Unnamed: 0",inplace=True,axis=1)
                    except FileNotFoundError:
                        st.error('File not found.')

                except:
                    st.warning("Une erreur s'est produite lors de la récupération des tweets...")
            else:
                st.warning("Veuillez saisir un ou plusieurs mots clés valides !")

    with st.expander("Ou : Chargement depuis un fichier existant"):
        filename = st.file_uploader('Uploader le fichier')
        try:
            if filename is not None:
                input = pd.read_csv(filename)
                df = input.copy()
                if "Unnamed: 0" in df.columns: df.drop("Unnamed: 0",inplace=True,axis=1)
        except FileNotFoundError:
            st.error('File not found.')

# ===========================================
# Partie II : Nettoyage et exploration des données
with dataExploration :
        st.subheader("Exploration des données")
        #try:
            # nettoyage
        preproc_modules.language_selection(df)
        preproc_modules.basic_preproc(df, ['date', 'time', 'tweet', 'hashtags', 'username', 'name','retweet', 'geo'])
        tweet_tokens, vocab = preproc_modules.tokenization(tweets = df['tweet'], nlp = nlp)
        tweet_tokens = preproc_modules.remove_stopwords(tweet_tokens, nlp, custom_stopwords)

            # exploration
            # Affichage des mots les plus représentatifs du corpus
        st.text("Mots représentatifs du benchmark")

        # On regroupe les tokens sous forme d'une unique chaîne
        temp_words = []
        for tkn in tweet_tokens:
            temp_words.append(" ".join(tkn))

        words_asStr = " ".join(temp_words)

        # Word cloud
        wc = eda_modules.word_cloud_from_text(words_asStr, nlp, custom_stopwords)
        st.image(wc.to_array())

        # Hashtags
        hashtags = eda_modules.get_tophashtags(df['tweet'])
        hashtags.columns = ['hashtag','occurences']

        st.text("Hashtags les plus populaires")
        plot = eda_modules.barplot_from_data(hashtags.head(6), x='hashtag', y='occurences')
        st.pyplot(plot)

        # Map
        st.text("Répartition des discussions autour du sujet par localités")

        #st.map()



    #except:
        #st.error("Le document n'a pas pu être lu ou alors il est erroné")