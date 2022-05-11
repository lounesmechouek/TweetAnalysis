from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pickle

def load_vectors():
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open("../../sentiment_analysis/modeles/tfidf.sav", "rb"))) # chemin depuis twtan/Scripts
    return loaded_vec

def load_sa_model(model_name):
    '''Charge le modèle dont le nom est spécifié en paramètre
    Noms acceptés : 'Naive_Bayes' | 'Logistic_Regression' | 'KMeans'
    '''
    model = None

    if model_name=='Naive_Bayes' or model_name=='Logistic_Regression' or model_name=='KMeans':
        model = pickle.load(open('../../sentiment_analysis/modeles/'+model_name+'_Model.sav', "rb"))

    return model
    
def adapt_to_model(loaded_vectors, tokens):
    '''Créee une matrice terme-document adaptée au vocabulaire du model utilisé pour le training
    tokens : doit être un itérable de chaines de caractères ["","","",...,""]
    '''
    vct_adapter = CountVectorizer(vocabulary=loaded_vectors.get_feature_names_out())

    transformer = TfidfTransformer()
    fitted_data = transformer.fit_transform(vct_adapter.fit_transform(tokens))

    return fitted_data

### Comment les modèles ont été générés ?
### Voir le code dans le notebook : modeles/Modeles_Generation_Source_Code