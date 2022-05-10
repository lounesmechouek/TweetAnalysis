# Imports
# ---------------------------
from asyncio.windows_events import NULL
import pandas as pd
import numpy as np
import sys
import subprocess
import pickle

import spacy
import datetime

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import gensim

import streamlit as st

from twitter_scrapping import scrapping_modules
from data_modules import preproc_modules
from data_modules import eda_modules
from ressources import ressources_modules
from topic_modeling import tm_modules
# ---------------------------
# Chemin de sauvegarde des fichiers générés :
# L'exécution se fait depuis twtan/Scripts
SAVE_PATH = '../../outputs/'

# set page layout
st.set_page_config(
    page_title="Tweets Analysis",
    page_icon="📈",
    layout="centered",# or "wide" pour utiliser l'ecran tout entier .
    initial_sidebar_state="expanded"
)

# Variables globales
allPositionsGeographiques=None
tweet_tokens=None

# Variable regroupant quelques positions géographiques (les plus importantes communes de France)
try:
    allPositionsGeographiques = ressources_modules.load_coordinates_asDict()
except:
    st.error("Il y a eu une erreur lors de la récupération des coordonnées géographiques...")

# Liste customisée de stop_words :
custom_stopwords = preproc_modules.load_custom_stopwords()

# 
nlp = spacy.load("fr_core_news_lg")

## Début application
st.title("Interface d'analyse de tweets")

# Containers
tweetsFetching = st.container()
dataExploration = st.container()
analyseAvancee = st.container()


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
                    # Ajout du mot-clé de rechercha à la liste des stop words
                    custom_stopwords += searchValue
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
                        input = pd.read_csv(path, encoding = 'unicode_escape')
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
                input = pd.read_csv(filename, encoding = 'unicode_escape')
                df = input.copy()
                if "Unnamed: 0" in df.columns: df.drop("Unnamed: 0",inplace=True,axis=1)
        except FileNotFoundError:
            st.error('File not found.')

# ===========================================
# Partie II : Nettoyage et exploration des données
with dataExploration :
    st.subheader("Exploration des données")
    try:
        # Nettoyage
        preproc_modules.language_selection(df)
        preproc_modules.basic_preproc(df, ['date', 'time', 'tweet', 'hashtags', 'username', 'name','retweet', 'geo'])
        tweet_tokens, vocab = preproc_modules.tokenization(tweets = df['tweet'], nlp = nlp)

        #il faut mettre à jour le vocabulaire issu de la tokenisation en supprimant les stop words
        tweet_tokens,NewVocab = preproc_modules.remove_stopwords(tweet_tokens, nlp, custom_stopwords)

        st.session_state.TKNS = tweet_tokens

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
        st.text("Hashtags les plus populaires")

        hashtags = eda_modules.get_tophashtags(df['tweet'])
        plot = eda_modules.barplot_from_data(hashtags.head(6), x='hashtag', y='occurences')
        st.pyplot(plot)

        # Utilisateurs
        st.text("Utilisateurs les plus cités")
        mentions = eda_modules.get_topmentions(df['tweet'])

        plot = eda_modules.barplot_from_data(mentions.head(6), x='mention', y='occurences')
        st.pyplot(plt)

        # Map
        st.text("Répartition des discussions autour du sujet par localités")
        eda_modules.map_from_locations(df['geo'])
    except:
        st.error("Le document n'a pas pu être lu ou alors il est erroné. Veuillez le recharger ou refaire une recherche !")

# ===========================================
# Partie III : Analyse avancée
with analyseAvancee:
    st.subheader("Analyse Avancée")

    st.text("Extraction des sujets de discussion principaux")

    #try:
    model = None
    model_name = st.selectbox(
        'Choisissez un modèle',
        ('LDA', 'NMF')
    )

    colTopic, colCoherence = st.columns(2)
    with colTopic:
        nb_topics = st.number_input(
            label = "Nombre de sujets à détecter",
            min_value = 2,
            max_value = 20,   
        )

    if model_name=='LDA' :
        if tweet_tokens!=None :
            corpus, dictionary = tm_modules.create_dict(tweet_tokens)
            model = tm_modules.build_LDA_model(corpus, dictionary, nb_topics)
        else:
            st.error("Veuillez charger vos données afin de créer le modèle..")
    else :
        print("5") #NMF

    if model != None and tweet_tokens != None:
        with colCoherence:
            st.text("Score de cohérence obtenu")
            if model_name=='LDA' : 
                st.text("avec "+str(nb_topics)+" thèmes : "+ str(tm_modules.get_coherence_lda(model, tweet_tokens, dictionary, corpus)))
                print("Well done !")
            else :
                print("5") # NMF
        
        st.success("Les sujets ont été correctement détectés.")


        ## -- Choix du topic à analyser
        topic_dict = {i : 'Thème '+str(i+1) for i in range(nb_topics)}

        colChoose, colRename = st.columns(2)
        with colChoose:
            selected_topic = st.selectbox(
                "Veuillez choisir un thème à analyser",
                list(topic_dict.values())
            )

        with colRename:
            if selected_topic != '' and selected_topic != None :
                newName_topic = st.text_input(label = "Renommer le sujet ? Insérez un nouveau nom !")

                # On renomme le topic numéro "index"
                indx = list(topic_dict.keys())[list(topic_dict.values()).index(selected_topic)]
                topic_dict[indx] = newName_topic

                st.success("Nouvel alias : "+newName_topic+" pris en compte pour le thème "+str(indx))
        
        if model_name=='LDA' :
            st.text("Mots les plus significatifs pour le topic sélectionné")
            words_signifiance_dict = tm_modules.get_words_topic_from_ldamodel(model, indx)
            fig = eda_modules.barplot_x_y(list(words_signifiance_dict.keys()), [float(words_signifiance_dict[k]) for k in list(words_signifiance_dict.keys())])
            st.pyplot(fig)


        else:
            print('5') # NMF





    else :
        st.error("Le modèle n'a pas pu être construit avec les paramètres actuels...")
    
    #except





