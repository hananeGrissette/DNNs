import pandas as pd
import numpy as np
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import multiprocessing
import re

import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
import string
from nltk import WordNetLemmatizer
from nltk import PorterStemmer


from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from time import time 
from collections import defaultdict

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
def display_closestwords_tsnescatterplot(model, word, size):
    
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model.similar_by_word(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
def preprocess(comments):
    data = []
    for text in str(comments).split(','):
        text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub('rt','',text)
        text = re.sub('xa0','',text)
        if text!= ' none ':
            data.append(text)
    return data

#remove punctuation
def no_punct(text):
    no_punctu = ''.join([c for c in text if c not in string.punctuation])
    return no_punctu
import string
#remove punctuation
def no_stopWords(text):
    no_stw = ''.join([c for c in text if c not in stopwords.words('english') ])
    return no_stw
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def word_lem(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text
def word_stem(text):
    stem_text=''.join([stemmer.stem(i) for i in text])
    return stem_text

def remove_stw(data):
    test = pd.Series(data).apply(lambda x:x.split(","))
    data = []
    for raw in test:
        data.append([c for c in raw if c not in stopwords.words('english')])
    return data
def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for raw in data:
#         print(raw)
        word_list = raw.split(" ")
        #word_list = list(word for word in word_list if len(word)>1)
#         print(word_list)
        for word in word_list:
            corpus.append(word)
        #corpus.append(word for word in word_list)
    
    while("" in corpus):
        corpus.remove("")
    return corpus
def remove_sw(data):
    corpora = []
    corpora.append([x for x in data if x not in stopwords.words('english')])
    return corpora
def item_in_list(data):
    proper_items = []
    for item in data:
        if item not in proper_items:
            proper_items.append(item)
    return proper_items