# Imports
# ---------------------------
from asyncio.windows_events import NULL
from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np

import spacy
import datetime

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly
import seaborn as sns
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

import streamlit as st

from twitter_scrapping import scrapping_modules
from data_modules import preproc_modules
from data_modules import eda_modules
from ressources import ressources_modules
from topic_modeling import tm_modules
from PIL import Image

# ---------------------------

# Chemin de sauvegarde des fichiers générés :
# L'exécution se fait depuis twtan/Scripts
SAVE_PATH = '../../outputs/'

#définir le chemin du logo de twitter
curr_path = os.path.dirname(__file__)
logo_path = curr_path+'/twitter_logo.png'

# définition du layout de la page
st.set_page_config(
    page_title="Tweets Analysis",
    page_icon="📈",
    layout="centered",# or "wide" pour utiliser l'ecran tout entier .
    initial_sidebar_state="expanded"
)

#configuration de la taille de sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 700px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)

allPositionsGeographiques=None

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

#titleContainer,image=st.columns(2)
#with titleContainer:
st.title("Interface d'analyse de tweets")

#####
#with image:
#image = Image.open(logo_path)
    #st.image(image,width=50)


# Containers
#tweetsFetching = st.container()
dataExploration = st.container()
analyseAvancée = st.container()

charger=False
tweet_tokens=None

# Partie I : Récupération des tweets à analyser
#with tweetsFetching:
with st.sidebar:
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
                    print (max_tweets_position)
                    path = SAVE_PATH + "miningTwitter_{}.csv".format(searchValue)
                    #ajouter le mot-clé de rechercha à la liste des stop words
                    custom_stopwords.append(searchValue)
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
                        charger=True
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
    if(charger!=False):
        preproc_modules.language_selection(df)
        preproc_modules.basic_preproc(df, ['date', 'time', 'tweet', 'hashtags', 'username', 'name','retweet', 'geo'])
        tweet_tokens, vocab = preproc_modules.tokenization(tweets = df['tweet'], nlp = nlp)

        #il faut mettre à jour le vocabulaire issu de la tokenisation en supprimant les stop words
        tweet_tokens,NewVocab = preproc_modules.remove_stopwords(tweet_tokens, nlp, custom_stopwords)

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

        #except:
            #st.error("Le document n'a pas pu être lu ou alors il est erroné. Veuillez le recharger ou refaire une recherche !")
    else:
        st.info("Veuillez charger vos données afin de les visualiser")
# ===========================================
# Partie III : Analyse avancée
with st.sidebar:
    st.subheader("Analyse Avancée")

    gauche,droite=st.columns(2)

    with gauche:
        modele = st.selectbox('Modélisation par thématique avec :',('LDA', 'NMF'))
        #topics_button= st.button(label = "Découvrir les thématiques !") 
    with droite:
        nbr_topics = st.number_input('choisissez un nombre de thématiques',min_value=2, max_value=20)
    

with analyseAvancée:
    st.subheader("Découverte des thématiques")
    if(modele=='LDA'):
            if(tweet_tokens!=None):
                #on commence par créer le dictionnaire ainsi qu'une sorte de matrice documents-termes
                corpus,disct=tm_modules.create_freq_Doc_Term(tweet_tokens)
                #construire le modele LDA
                LDA_model=tm_modules.build_LDA_model(corpus,disct,nbr_topics)
                le=tm_modules.plot_top_words_topic(LDA_model,custom_stopwords,nbr_topics)
                st.pyplot(fig=le)
            else:
                st.info("Veuillez charger vos données pour la découverte de thématiques")




