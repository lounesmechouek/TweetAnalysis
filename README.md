# Tweet Analysis Platform

## Requirements :
You should have a recent version of python installed on your computer.\
We recommand using Python 3.8.10 or higher, as it the version used by the authors for the configuration.\
- Numpy >= 1.22.3
- Pandas >= 1.4.1
- Matplotlib >= 3.5.1

## Get the project running
From the root folder of the project :\

**Step 1.** Create the virtual environment and activate it

```console
python -m venv twtan
.\twtan\Scripts\activate
```
**Step 2.** Add the virtual environment to the Python Kernel\

```console
python -m pip install --upgrade pip
pip install ipykernel 
python -m ipykernel install --user --name=twtan
```

**Step 3.** Install the libraries we're going to use in our VE :\

```console
pip install spacy==3.2.3
python -m spacy download en_core_web_lg
pip install pyLDAvis==3.3.1
pip install emoji==1.6.3
pip install pandas-profiling==3.1.0
pip install plotly==5.6.0
pip install gensim==4.1.2
pip install chart-studio==1.1.0
pip install wordcloud==1.8.1
```

