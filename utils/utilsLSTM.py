import sys
# from pathlib import Path
# print(Path)
# path = os.path.abspath(os.path.join('/data/h.grissette/SA/Paper', '..'))
# path = path+'/Paper/datasets'
# import sys,os
import pandas as pd
# import gensim
# from gensim.models import Word2Vec
# sys.path.append(os.path.abspath('/data/h.grissette/SA/Paper/datasets'))
import pandas as pd
import numpy as np
from fastai import *
# from fastai.text import *
# from fastai.vision import *
import seaborn as sns
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# import warnings
# warnings.filterwarnings('ignore')
import os,re
from collections import Counter
import logging
import time
import pickle
import itertools
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# # Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim
import pandas as pd
import numpy as np
import sys, urllib, re, json
import numba
from timeit import default_timer as timer
from numba import jit, njit
from numba import *
import string, unicodedata
import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from unidecode import unidecode
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import preprocessor as p
from  nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

def pretrained_embedding_layer(model,model2,model3, word_to_index,emb_dim_max):
    """
    Creates a Keras Embedding() layer and loads in pre-trained  vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    words_ignored = []
    vocab_len = len(word_to_index) + 1                  
    emb_matrix = np.zeros([vocab_len,emb_dim_max])
       
    print(' Total words would be processed : '+str(vocab_len))
    for word, idx in word_to_index.items():
        if word in model:
            emb_matrix[idx,:200] = model[word]
            emb_matrix[idx,200:] = 0
        if word in model2:
            emb_matrix[idx, :100] = model2[word]
            emb_matrix[idx, 100:] = 0
        if word in model3.keys():
            emb_matrix[idx,:] = model3[word]
        else:
            words_ignored.append(word)
    print(str(len(words_ignored))+" words ignored")
    print(emb_matrix.shape)   
        
        
    embedding_layer = Embedding(vocab_len,emb_dim_max,trainable = True)
  
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer,words_ignored 

def model_emb():
    model = Sequential()
    model.add(tf.keras.layers.Embedding(10, 6, input_length=10))
    # Now model.output_shape is (None, 10, 64), where `None` is the batch dimension(number of examples)  

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
#     embedding_layer, ignored_words = pretrained_embedding_layer(model,model2,model3,word_to_index,300)
     ######### Example of creating embeddings and predict by indices

    return model

def model(input_shape, model,model2,model3, word_to_index):
    """
    Function creating the S model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    sentence_indices = Input(input_shape,dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer, ignored_words = pretrained_embedding_layer(model,model2,model3,word_to_index,300)
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences.
    X = LSTM(units=128,input_shape=input_shape,return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(units=128,input_shape=input_shape,return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X through a Dense layer with 5 units
    X = Dense(units=num_classes)(X)
#     X = Dense(6, activation='softmax')
    # Add a softmax activation
#     print(X)
#     print(type(X))
#     print(X.shape)
#     print(sum(X))
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices,outputs=X)
    
    return model
# @numba.jit(nopython=True, forceobj=True)                  
def split(data):
    training_pos = []
    score_pos = []
    training_neg = []
    score_neg = []
    for index,item in data.iterrows():
        if item['Sentiment'] == 1:
            training_pos.append(item['Text'])
            score_pos.append(item['Sentiment'])
        if item['Sentiment'] == -1:
            training_neg.append(item['Text'])
            score_neg.append(item['Sentiment'])
    return [training_pos,score_pos,training_neg, score_neg]

# @numba.jit(nopython=True, forceobj=True)
def label_to_number(data):
    for index,item in data.iterrows(): 
        if item['Sentiment'] == 'positive':
            item['Sentiment'] == 1
        else:
            item['Sentiment'] == 0
    return data

def word2vec(sentences,W2V_SIZE = 300,W2V_WINDOW = 7, W2V_EPOCH =32, W2V_MIN_COUNT=10):
    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=8)
    documents = [str(_text).split() for _text in sentences]
    w2v_model.build_vocab(documents)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("Vocab size", vocab_size)
    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
    return w2v_model

def padding(sentences,tokenizer, maxlen = SEQUENCE_LENGTH):
    vocab_size = len(tokenizer.word_index) + 1
    print("Total words", vocab_size)
    x_train = pad_sequences(tokenizer.texts_to_sequences(sentences), maxlen)
    x_test = pad_sequences(tokenizer.texts_to_sequences(sentences), maxlen)
    return vocab_size,x_train,x_test


def Embeddings(w2v_model,tokenizer,vocab_size,input_length=SEQUENCE_LENGTH):
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    print(embedding_matrix.shape)
    embedding_layer = Embedding(vocab_size, W2V_SIZE,weights=[embedding_matrix],input_length=SEQUENCE_LENGTH,trainable=False)
    return embedding_layer