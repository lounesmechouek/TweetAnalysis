# Imports
# ---------------------------
from asyncio.windows_events import NULL
from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np

import spacy
import datetime

import matplotlib.pyplot as plt
import os

import streamlit as st

from twitter_scrapping import scrapping_modules
from data_modules import preproc_modules
from data_modules import eda_modules
from ressources import ressources_modules
from topic_modeling import tm_modules
from sentiment_analysis import sa_modules
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


# Configuration de la taille de sidebar
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

# Variables globales
allPositionsGeographiques=None
charger=False
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
#titleContainer,image=st.columns(2)
#with titleContainer:
st.title("Interface d'analyse de tweets")
st.text("")

#####
#with image:
#image = Image.open(logo_path)
    #st.image(image,width=50)


# Containers
dataExploration = st.container()
topicModeling = st.container()
sentimentAnalysis= st.container()

# Partie I : Récupération des tweets à analyser

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
                    # Ajout du mot-clé de rechercha à la liste des stop words
                    custom_stopwords += searchValue
                    #nest_asyncio.apply()
                    if 'Peu importe' in positionsGeographiques:
                        scrapping_modules.get_tweets(
                            max_tweets_position,
                            searchValue,
                            path,
                            dateDebut.strftime("%Y-%m-%d"),
                            dateFin.strftime("%Y-%m-%d"),
                            [allPositionsGeographiques[k] for k in list(allPositionsGeographiques.keys()) if k != 'Peu importe']
                        )
                    else:
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
                input = pd.read_csv(filename, encoding = 'unicode_escape')
                df = input.copy()
                charger=True
                if "Unnamed: 0" in df.columns: df.drop("Unnamed: 0",inplace=True,axis=1)
        except FileNotFoundError:
            st.error('File not found.')

# ===========================================
# Partie II : Nettoyage et exploration des données
with dataExploration :
    st.subheader("Exploration des données")
    # nettoyage
    if(charger!=False):
        preproc_modules.language_selection(df)
        preproc_modules.basic_preproc(df, ['date', 'time', 'tweet', 'hashtags', 'username', 'name','retweet', 'geo'])
        tweet_tokens, vocab = preproc_modules.tokenization(tweets = df['tweet'], nlp = nlp)

        #il faut mettre à jour le vocabulaire issu de la tokenisation en supprimant les stop words
        tweet_tokens,NewVocab = preproc_modules.remove_stopwords(tweet_tokens, nlp, custom_stopwords)

        # exploration
        # Affichage des mots les plus représentatifs du corpus
        st.text("Mots les plus fréquents dans les tweets extraits")

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
        st.pyplot(plot)

        # Map
        st.text("Répartition des discussions autour du sujet par localités")
        eda_modules.map_from_locations(df['geo'])

        #except:
            #st.error("Le document n'a pas pu être lu ou alors il est erroné. Veuillez le recharger ou refaire une recherche !")
    else:
        st.info("Veuillez charger vos données afin de les visualiser !")
# ===========================================
# Partie III : Analyse avancée
with st.sidebar:
    st.subheader("Analyse Avancée")

    gauche,droite=st.columns(2)

   
    with droite:
        methode_nb_topic=st.selectbox('Comment déterminer le nombre de thèmes ?',('Score de cohérence optimal', 'Choix de l\'utilisateur'))

    with gauche:
        modele = st.selectbox('Modélisation par thématique avec :',('LDA', 'NMF'))
        nbr_topics = 2
        topic_dict = {}
        if(methode_nb_topic=='Choix de l\'utilisateur'):
            nbr_topics = st.number_input('Veuillez choisir le nombre de thématiques',min_value=2, max_value=20)
    
    

with topicModeling:
    st.subheader("Découverte des thématiques")
    if(modele=='LDA'):
            if(tweet_tokens!=None):
                #on commence par créer le dictionnaire ainsi qu'une sorte de matrice documents-termes
                corpus,disct=tm_modules.create_freq_Doc_Term(tweet_tokens)
                
                #vérifier la méthode choisi par l'utilisateur pour le choix du nombre de topics 
                if(methode_nb_topic=='Score de cohérence optimal'):
                    model_list, coherence_values=tm_modules.compute_coherence_values(disct, corpus, tweet_tokens, 8, start=2, step=1)
                    optimal_number_of_topics,optimal_score=tm_modules.find_optimal_number_of_topics(coherence_values)
                    topic_dict = {i : 'Thème '+str(i+1) for i in range(int(optimal_number_of_topics))}
                    st.write("Le meilleur nombre de thèmatiques est: ",optimal_number_of_topics)
                    st.write("Le score de cohérence correspondant est : ",optimal_score)
                    #construire le modele LDA
                    #LDA_model=tm_modules.build_LDA_model(corpus,disct,number)
                    LDA_model=model_list[optimal_number_of_topics-2]
                    le=tm_modules.plot_top_words_topic(LDA_model,custom_stopwords,optimal_number_of_topics)
                    st.pyplot(fig=le)
                else:
                    topic_dict = {i : 'Thème '+str(i+1) for i in range(nbr_topics)}
                    LDA_model=tm_modules.build_LDA_model(corpus,disct,nbr_topics)
                    coherence_score=tm_modules.calcul_coherence_score(LDA_model,tweet_tokens,disct)
                    st.write("Le score de cohérence correspondant est : ",coherence_score)
                    le=tm_modules.plot_top_words_topic(LDA_model,custom_stopwords,nbr_topics)
                    st.pyplot(fig=le)
            else:
                st.info("Veuillez charger vos données pour découvrir les sujets abordés!")
    elif(modele=='NMF'):
        st.text('comming soon')

with st.sidebar:
    st.text("Sentiments des utilisateurs")
    if(topic_dict):
        selected_topic = st.selectbox(
            "Veuillez choisir un thème à analyser",
            list(topic_dict.values())
        )
    else:
        st.info("Chargement des données...")

with sentimentAnalysis:
    st.subheader("Analyse de sentiments")

    polarity_dict = {
        0: 'négatif',
        1: 'positif'
    }

    if(modele=='LDA'):
        if(tweet_tokens!=None):
            term_doc = sa_modules.load_vectors()
            classifier = sa_modules.load_sa_model('Logistic_Regression')
            tokens_as_str = preproc_modules.get_tokens_as_listChar(tweet_tokens)
            new_vectors = sa_modules.adapt_to_model(term_doc, tokens_as_str)

            predicted_sentiments = classifier.predict(new_vectors)

            indx = list(topic_dict.keys())[list(topic_dict.values()).index(selected_topic)]
            st.text("Opinions glbales (pour tous les thèmes)")

            sr_preds = pd.Series(predicted_sentiments).value_counts()
            somme = sr_preds[0]+sr_preds[1]
            fig = eda_modules.simple_barplot(pd.DataFrame.from_dict([{polarity_dict[i]:(sr_preds[i]/somme)*100 for i in range(len(polarity_dict))}]))
            st.pyplot(fig)

            st.text("Opinions pour le "+topic_dict[indx])


            print(12*'+++++')
            print(len(predicted_sentiments))
            print(predicted_sentiments)
            print(12*'+++++')

        else:
            st.info("Veuillez charger vos données pour connaitre les sentiments des personnes !")
    elif(modele=='NMF'):
        st.text('comming soon')

