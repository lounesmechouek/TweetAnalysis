# Tweet Analysis Platform

## Requirements :
You should have a recent version of python installed on your computer.\
We recommand using Python 3.8.10 or higher, as it the version used by the authors for the configuration.
- Numpy >= 1.22.3
- Pandas >= 1.4.1
- Matplotlib >= 3.5.1

## Get the project running
From the root folder of the project :

**Step 1.** Create the virtual environment and activate it

```console
python -m venv twtan
.\twtan\Scripts\activate
```
**Step 2.** Add the virtual environment to the Python Kernel

From the folder **twtan/Scripts** : 

```console
python.exe -m pip install --upgrade pip
pip.exe install ipykernel 
python.exe -m ipykernel install --user --name=twtan
```

**Step 3.** Install the libraries we're going to use in our VE :

From the folder **twtan/Scripts** :

```console
pip.exe install spacy==3.2.3
python.exe -m spacy download fr_core_news_lg
pip.exe install pyLDAvis==3.3.1
pip.exe install scikit-learn==1.0.2
pip.exe install pandas-profiling==3.1.0
pip.exe install plotly==5.6.0
pip.exe install gensim==4.1.2
pip.exe install chart-studio==1.1.0
pip.exe install wordcloud==1.8.1
pip3.exe install --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
pip.exe install streamlit==1.8.1 
pip.exe install Unidecode==1.3.4
pip.exe install stopwordsiso==0.6.1
pip.exe install seaborn==0.11.2
pip.exe install folium==0.12.1.post1
pip.exe install streamlit-folium==0.6.7
pip.exe install nltk=3.7
```

**Step 3.** Run the project :

You need to first create a folder named **outputs** in the root folder of the project (this is where the mined tweets will be stored as csv files)

From the folder **twtan/Scripts** :

```console
streamlit.exe run ../../main.py
```