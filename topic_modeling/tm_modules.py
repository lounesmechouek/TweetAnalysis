import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
from pprint import pprint




def create_dict(data):
    # permet de créer une sorte de bag of words (mots, frequence d'apparition) mais qui n'est pas sparse 
    id2word = corpora.Dictionary(data)
    corpus = [id2word.doc2bow(text) for text in data]
    #print(corpus[:1][0][:30])
    print(corpus)
    #ex:dans le 1er tweet le mot ayant l'id 0 apparait 1 fois
    return corpus,id2word



def build_LDA_model(corpus,id2word):
    # number of topics
    num_topics = 3
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    return doc_lda
