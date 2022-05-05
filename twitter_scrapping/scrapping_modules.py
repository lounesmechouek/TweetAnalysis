from . import twintFetch


def get_tweets(nb_max_tweet, my_search, raw_corpus_filePath, date_Begin, date_End, geo=None):
    '''Récupère des tweets selon les paramètres spécifiés
    nb_max_tweet : limite le nombre de tweets à récupérer (par position géographique)
    my_search : mot clé à rechercher
    raw_corpus_filePath : chemin du fichier destination
    date_Begin, date_End : intervalle de temps auquel récupérer les tweets
    geo : liste de coordonnées géographiques à prendre en compte 'latitude,longitude,rayon' exemple : '48.8588897,2.320041,10km'
    '''
    

    runFetch=twintFetch.twintFetch(nb_limit=nb_max_tweet, out_file=raw_corpus_filePath, search_terms=my_search)

    for g in geo:
        # Les tweets sont insérés à la fin du fichier cible
        runFetch.fetch_tweets(since=date_Begin, until=date_End, geo=g)
    
    