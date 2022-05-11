import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
from pprint import pprint
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import streamlit as st
from gensim.models import CoherenceModel


def create_freq_Doc_Term(data):
    #créer un dictionnaire
    id2word = corpora.Dictionary(data)
    #créer une sorte de bag of words
    corpus = [id2word.doc2bow(text) for text in data]
    return corpus,id2word



def build_LDA_model(corpus,id2word,nbr_topics):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=nbr_topics)
    
    pprint(lda_model.print_topics())
    #doc_lda = lda_model[corpus]
    #return doc_lda
    return lda_model


def plot_top_words_topic(LDA_model,custom_stopwords,nbr_topics):

    #liste de couleurs
    couleurs1 = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    couleurs2=[color for name, color in mcolors.TABLEAU_COLORS.items()]
    couleurs=couleurs1+couleurs2
    cloud = WordCloud(stopwords=custom_stopwords,
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    #colormap='tab10',
                    color_func=lambda *args, **kwargs: couleurs[i],#fonction qui définit la couleur des mots du wordcloud
                    prefer_horizontal=1.0)

    #topics liste dont chaque element est un tuple, le 1er element du tuple est l'id du topic
    #le 2eme element est une liste de tuple, tel que le 1 er element représente un mot représentatif du topic et le 2eme element son poids

    #num_topics paramétre tres important car la fonction show_topics retourne par défaut que 10 topics
    topics = LDA_model.show_topics(formatted=False,num_topics=nbr_topics)
    print("le nombre de topics",len(topics))
    print(topics)
    #calcul du nombre de ligne du subplot 
    if((nbr_topics%3)==0):
        nb_row=nbr_topics//3
    else:
        nb_row=(nbr_topics//3)+1

    print ("le nombre de lignes",nb_row)
    fig, axes = plt.subplots(nb_row, 3, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        print("**************************",i)
        if(i<=nbr_topics-1):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])#ca me fait une erreur car il crée 4 sous plot alors qu'il y a 3 topics du coup il trouve pas quoi mettre dans le dernier subplot
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')
        else:
            
            ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    #plt.show()
    return fig



@st.cache
def calcul_coherence_score(model,texts,dictionary):
    coherence_score = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_score.get_coherence()
    return coherence_lda

"""
fonction qui calcul le score de coherence pour différents nombre de topics 
et retourne une liste de modele ainsi qu'une liste contenant le score de cohérence de chaque modéle
"""""
@st.cache
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    
    coherence_values = []#liste pour stocker les différents score de cohérence
    model_list = []#liste pour garder les modéle généré avec différents nombre de topic
    for num_topics in range(start, limit, step):
        print(" ----> calcul d'un modéle LDA avec ",num_topics," topics")
        model=gensim.models.LdaMulticore(corpus=corpus,id2word=dictionary,num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



 
"""focntion qui renvoie le nombre de topic optimal en utilisant la méthode du score de coherence
ainsi que le score de cohérence correspondant
"""""
@st.cache
def find_optimal_number_of_topics(coherence_values):
    croissante=False #au début on suppose que la fonction du score de cohérence n'est pas croissante
    optimal_index=0 #on suppose que le nombre optimal de topic est 2
    optimal_score=coherence_values[0]

    for i in range(1,len(coherence_values)):
        if(croissante==False):
            if(coherence_values[i]>optimal_score):
                croissante=True
                optimal_index=i
                optimal_score=coherence_values[i]
        else:
            if(coherence_values[i]>optimal_score):
                optimal_index=i
                optimal_score=coherence_values[i]
            elif(coherence_values[i]<optimal_score):
                optimal_index=optimal_index+2
                return optimal_index,optimal_score
    
    optimal_index=optimal_index+2
    return optimal_index,optimal_score