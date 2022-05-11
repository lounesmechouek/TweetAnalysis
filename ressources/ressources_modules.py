import pandas as pd
import os
import streamlit as st


def load_coordinates_asDict(rayon='10km', default={"Peu importe": None}):
    '''Renvoie un dictionnaire du type {'Nom Ville': 'latitude,longitude,rayon'}
    rayon : rayon à prendre en compte lors de la recherche
    default : localisations par défaut à ajouter à la liste (notamment pour la recherche)
    '''
    try:
        coord = pd.read_csv('../../ressources/geoCord.csv', encoding = 'unicode_escape')
        
        coord_dict = {"Peu importe": None}
        coord_list = [(coord.loc[i].place, (str(coord.loc[i].lat)+','+str(coord.loc[i].lon)+','+rayon)) for i in range(len(coord))]

        coord_dict.update(dict(coord_list))

        return coord_dict
    except:
        raise


def load_coordinates_asDF():
    '''Fonction simple pour récupérer la liste des coordonnées en tant que dataframe
    '''
    try:
        coord = pd.read_csv('../../ressources/geoCord.csv', encoding = 'unicode_escape')
        return coord
    except:
        raise