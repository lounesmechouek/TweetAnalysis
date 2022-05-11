import twint
import os
import streamlit as st


class twintFetch:
  def __init__(self, nb_limit, out_file, search_terms):
    self.nb_limit=nb_limit 
    self.out_file=out_file 
    self.search_terms=search_terms 
    

  def fetch_tweets(self, since=None, until=None, geo=None):
    print("******Fetching tweets !******")
    c=twint.Config()
    c.Search=self.search_terms #le terme de recherche

    if since is not None:
      c.Since=since #extraire les tweets à partir de cette date
    if until is not None:
      c.Until=until #extraire les tweets jusqu'a cette date

    c.Store_csv= True #fichier de sortie est un fichier csv
    c.Output=self.out_file #le nom du fichier qui contientra le résultat (tweets)
    c.Limit=self.nb_limit #le nombre de tweets à extraire
    c.Geo = geo #condition sur coordonnées géographiques des tweets à extraire
    #c.Lang='fr'#NOT WORKING "extraire les tweets qui sont en francais uniquement"
    #c.Filter_retweets=True #ne pas extraire les retweets 
    c.Count=True #avoir le nombre de tweets extaits
    #c.Location=False en essayant ca ca ne récupére pas de tweets
    twint.run.Search(c)