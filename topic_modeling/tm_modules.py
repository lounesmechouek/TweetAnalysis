import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
from pprint import pprint

def create_dict(data):
    '''Permet de créer une sorte de bag of words (mots, frequence d'apparition) mais qui n'est pas sparse 
    '''
    id2word = corpora.Dictionary(data)
    corpus = [id2word.doc2bow(text) for text in data]
    #print(corpus[:1][0][:30])
    #print(corpus)
    #ex:dans le 1er tweet le mot ayant l'id 0 apparait 1 fois
    return corpus,id2word


def build_LDA_model(corpus, id2word, nb_topics):
    '''Renvoie le modèle LDA construit
    '''
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=nb_topics,
        workers=1
    )

    # Print the Keyword in the 10 topics
    #pprint(lda_model.print_topics())
    #doc_lda = lda_model[corpus]
    return lda_model

def get_coherence_lda(lda_model, tokens, dictionary, bag_words, coherence='c_v'):
    '''Retourne le score de cohérence pour un seul modèle
    '''
    return CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, corpus=bag_words, coherence=coherence, processes=1).get_coherence()

def compute_coherence_values(dictionary, corpus, texts, max=10, min=2, saut=1):
    '''
    Construit plusieurs modèles et calcule leur scores de cohérence

    corpus : tokens du corpus
    dictionary : dictionnaire Gensim
    min, max, saut : valeurs min, max de topics à tester et de combien augmenter à chaque fois
    '''
    coherence_vals = []
    model_list = []
    for num_topics in range(min, max, saut):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model=gensim.models.LdaMulticore(corpus=corpus,id2word=dictionary,num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherencemodel.get_coherence())

    return model_list, coherence_vals

def get_words_topic_from_ldamodel(ldamodel, numtopic):
    '''Renvoie un dictionnaire dont les clés sont les mots du topic et les valeurs leur importance
    '''
    tpc = ldamodel.print_topic(numtopic)
    wds_as_list = tpc.split(" + ")
    return {elt.split("*")[1].replace('"',''):elt.split("*")[0] for elt in wds_as_list}







