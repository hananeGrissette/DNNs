import os
import pandas
import pandas as pd
import numpy as np
import sent2vec
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re
import csv

#NLTK
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from scipy.spatial.distance import cosine as dist
from scipy.spatial.distance import euclidean

#gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer

from scipy.spatial import distance
file_path =os.path.abspath(os.path.join("", os.pardir))
print(file_path)
from gensim.models import KeyedVectors

import csv
import numpy as np
from numpy.random import RandomState
import os

def load_data_set():
    X = []
    Y = []
    with open(dataset_path,"rt", encoding= 'UTF8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            is_positive = line[1]=="1"
            text = line[3]
            X.append(text)
            Y.append(is_positive)
    return X,Y


def track_keyword_list(keyword,words):
    res = []
    for word in words:
        try:
            if word.find(keyword)!=-1:
                res.append(word)
        except:
            pass
    return res

def track_keyword(keyword,sentences,frequence):
    res = []
    frequencies= []
    for i in range(len(sentences)):
        try:
            if sentences[i].find(keyword)!=-1:
                res.append(sentences[i])
                frequencies.append(frequence[i])
        except:
            pass
    return res,frequencies
def track_sent(keywords,sentences,frequence):
    res = []
    frequencies= []
    for i in range(len(sentences)):
        if sentences[i].find(keywords)!=-1:
            return sentences[i],frequence[i]

def check_keyword(keyword,train_text):
    res = []
    for i in range(len(train_text)):
        if i!=1:
            if(keyword in train_text[i].split()):
                res.append(train_text[i])
    return res
def get_duplicate(AllTweets):
    result = pd.DataFrame()
    tweetChecklist = {}
    for current_tweet in AllTweets:
        # If tweet doesn't exist in the list
        if current_tweet not in tweetChecklist:
            j=0
            tweetChecklist[current_tweet]=j;
        else:
            tweetChecklist[current_tweet]= int(tweetChecklist[current_tweet]+1)
    result.append(tweetChecklist,ignore_index=True)
    return tweetChecklist

def get_Bio_sent2vec(file_path,sentences):
    window = 700
    vectors = np.zeros((len(sentences),window))
    model = sent2vec.Sent2vecModel()
    try:
        model.load_model(file_path)
    except Exception as e:
        print(e)
    i=0
    for s in sentences:
        vectors[i]=model.embed_sentence(s)
        i=i+1
    print('model successfully loaded')
    return vectors

def create_embedding_w2v(model,sentences,W2V_SIZE):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    vocab_size = len(tokenizer.word_index) + 1
#     vocab_size=len(words)
    print("Total words", vocab_size)
    words_ignored = []
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in model:
            embedding_matrix[i] = model[word]
        else:
            words_ignored.append(word)
    print(str(len(words_ignored))+" words ignored")
    print(embedding_matrix.shape)
    return embedding_matrix,words_ignored    
    
def load_Glove(file):
    embeddings_glove = {}
    with open(file, 'r',encoding="ISO-8859-1") as f:
        for line in f:
            values = line.split()
            word = ''.join(values[:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            embeddings_glove[word] = coefs
    f.close()
    return embeddings_glove

def create_embedding_glove(embeddings_glove,sentences,dimension):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    vocab_size = len(tokenizer.word_index) + 1
    print("Total words", vocab_size)
    words_ignored = []
    embedding_matrix = np.zeros((vocab_size, dimension))
    for word, i in tokenizer.word_index.items():
        if word in embeddings_glove.keys():
            embedding_matrix[i] = embeddings_glove[word]
        else:
            words_ignored.append(word)
    print(str(len(words_ignored))+" words ignored")
    print(embedding_matrix.shape)
    return embedding_matrix,words_ignored

def find_embeddings_glove(model,data_list):
    embeddings=[]
    ignored=[]
    words=[]
    for entry in data_list:
        if entry in model.keys():
            embeddings.append(model[entry])
            words.append(entry)
        else:
            ignored.append(entry)
    return embeddings,words,ignored

def find_embeddings(model,data_list):
    embeddings=[]
    words=[]
    ignored=[]
    for entry in data_list:
        if entry in model:
            embeddings.append(model[entry])
            words.append(entry)
        else:
            ignored.append(entry)
    return embeddings,words,ignored
        

def find_closest_embeddings(embeddings_dict,embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: euclidean(embeddings_dict[word], embedding))

def cosine(a, b):
    norm1 = np.linalg.norm(a)
    norm2 = np.linalg.norm(b)
    return np.dot(a, b) / (norm1 * norm2) 
