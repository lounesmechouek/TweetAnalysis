import spacy
from spacy.tokens import Token
from spacy.attrs import ORTH

from itertools import chain
import re
from unidecode import unidecode

def language_selection(df, lang='fr'):
    '''Enlève les tweets qui ne sont pas dans les langues "lang"
    lang : langue des tweets à garder
    '''
    return df[df['language']==lang]


def basic_preproc(df, important_attr):
    """
    supprime les enregistrements ayant des valeurs manquantes, 
    sélectionne les colonnes intéressantes,
    supprime les doublons
    """
    df.dropna(axis='columns', inplace=True)
    df = df[important_attr]
    df.drop_duplicates(inplace=True, subset="tweet")


def clean(doc):
  return [
    unidecode(token.lemma_.lower()) for token in doc 
    if (not token.is_punct) 
    and (not token.is_space) 
    and (not token.like_url) 
    #and (len(token) > 1)
    and (len(token) > 2)
    and (not token._.like_handle)
    and (not token._.like_num)
    and(not token.is_stop)#si on l'ajoute pas ici il va ajouter les tokens comme .apres
    and (token.ent_type_ != "GPE") 
  ]

def tokenization(tweets, nlp):
    ''' Renvoie une liste de tokens pour chaque tweets ainsi que le vocabulaire global
    tweets : liste ou itérable de tweets
    '''
    # Ajout d'un pattern infixe pour découper les mots écrits en CamelCase
    default_infixes = list(nlp.Defaults.infixes)
    default_infixes.append('[A-Z][a-z0-9]+')
    infix_regex = spacy.util.compile_infix_regex(default_infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer

    # Ajout des mentions comme exceptions à ignorer lors de la recherche de patterns infixe
    #c.à.d:dire au tokenizer de ne pas tokeniser les mentions même si elles sont ecrites en CamelCase
    nlp.tokenizer.token_match = re.compile(r"@[\w\d_]+").match

    # Surcharge explicite de certaines exceptions de tokenisation
    # En gros : dire à spacy comment séparer certains mots particuliers en tokens
    nlp.tokenizer.add_special_case("passe-t-il", [{ORTH: "passe"}, {ORTH: "-"}, {ORTH: "t"}, {ORTH: "-"}, {ORTH: "il"}])
    nlp.tokenizer.add_special_case("est-ce", [{ORTH: "est"}, {ORTH: "-"}, {ORTH: "ce"}])
    nlp.tokenizer.add_special_case("n'est ce pas", [{ORTH: "n'"}, {ORTH: "est"},{ORTH: " "},{ORTH: "ce"},{ORTH: " "},{ORTH: "pas"}])
    nlp.tokenizer.add_special_case("qu'est-ce", [{ORTH: "qu"}, {ORTH: "'"},{ORTH: "est"},{ORTH: "-"},{ORTH: "ce"}])

    # Création de l'attribut personnalisé pour les mentions(détection de mentions)
    handle_regex = r"@[\w\d_]+"
    like_handle = lambda token: re.fullmatch(handle_regex, token.text)
    Token.set_extension("like_handle", getter=like_handle,force=True)

    #création d'un attribut qui vérifie si le token contient des chiffres
    #num_regex = r".*[0-9]+.*"
    num_regex = r"[0-9]+"
    like_num = lambda token: re.fullmatch(num_regex, token.text)
    Token.set_extension("like_num", getter=like_num,force=True)

    docs = nlp.pipe(tweets)
    tokens = [] #contiendra les tokens utiles
    for doc in docs:
        tokens.append(clean(doc))
    
    for i, tokenized_text in enumerate(tokens):
        print(f"TWEET {i + 1}:", tokenized_text, "\n")

    vocabulary=set(chain(*tokens))

    print("TAILLE DU VOCABULAIRE:", len(vocabulary))

    return tokens, vocabulary

def remove_stopwords(tkns, nlp, custom_stopwords=[]):
    '''Enlève, parmi les tokens, ceux qui sont des stop words
    tkns : Liste de tokens
    custom_stopwords : liste de stop words additionnelle (en plus de celle par défaut de spacy)
    '''
    ALL_STOP_WORDS = list(nlp.Defaults.stop_words)
    if custom_stopwords:
        ALL_STOP_WORDS += custom_stopwords
    
    ALL_STOP_WORDS = set(ALL_STOP_WORDS)

    final_tokens=[]
    for token in tkns:
        l=[]
        for t in token:
            if t not in ALL_STOP_WORDS:
                l.append(t)
        if(len(l)>0):
            final_tokens.append(l)
    
    return final_tokens 
