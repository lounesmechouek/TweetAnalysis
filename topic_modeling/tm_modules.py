import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora


def get_lda_parameters(tokens):
    id2word = corpora.Dictionary(tokens)
    frequencies = [id2word.doc2bow(text) for text in tokens]

    return id2word, frequencies

def get_coherence_lda(lda_model, tokens, dictionary, coherence='c_v'):
    return CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v').get_coherence()

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model=gensim.models.LdaMulticore(corpus=corpus,id2word=dictionary,num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values