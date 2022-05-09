import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
from pprint import pprint
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import streamlit as st


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
    couleurs = [color for name, color in mcolors.TABLEAU_COLORS.items()]

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
    topics = LDA_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        
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


#focntion qui renvoie le nombre de topic optimal en utilisant la méthode du score
