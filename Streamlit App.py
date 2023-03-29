#!/usr/bin/env python
# coding: utf-8

# Standard Packages
import pandas as pd
import numpy as np
import regex as re
import itertools
import streamlit as st
import pickle

# SKLearn Modules
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# NLTK modules
import nltk
from nltk import FreqDist, WordNetLemmatizer, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet

# load model
with open('best_svc_model.pkl', 'rb') as f:
    svc = pickle.load(f)

# load vectorizer
with open('tfidf_lemm.pkl', 'rb') as e:
    tfidf = pickle.load(e)

# define function
def preprocess_text(text):
    
    # make all characters lowercase
    text = text.lower()
    
    # remove URLs, twitter names, etc
    url_pattern = r'(?:http|https|pic\.twitter|www\.)\S+|@\S+|^@\S+|#\S+'
    string =  re.sub(url_pattern, '', text, flags=re.IGNORECASE)
    
    # remove Reuters
    text = text.replace('reuters', '')
    
    # tokenize text
    basic_token_pattern = r"(?u)\b([a-zA-Z]{3,}|\d{4,})\b"
    tokenizer = RegexpTokenizer(basic_token_pattern)
    stopwords_list = stopwords.words('english')
    text = tokenizer.tokenize(text)
    text = [token.lower() for token in text if token.lower() not in stopwords_list]
    
    #initialize lemmatizer
    wnl = WordNetLemmatizer()

    # helper function to change nltk's part of speech tagging to a wordnet format.
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None
        
    # creates list of tuples with tokens and POS tags in wordnet format
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tag(text))) 
    
    # lemmatizes each token based on part of speech in tuple
    doc = [wnl.lemmatize(token, pos) for token, pos in wordnet_tagged if pos is not None]
    
    return text

# display streamlit page title and prompt
st.title('Is this article fake news?')
st.write("Paste the text of an article below and we'll tell you if it's real news or not.")
news_input = st.text_area(label = 'Paste article text here:')

# preprocess input, vectorize, make a prediction, and display result
preprocess_text(news_input)
news_input = [news_input]
input_vec = tfidf.transform(news_input)
button = st.button('Real or Fake?')

if button:
    if svc.predict(input_vec)[0] == 0:
        st.write('This article appears to be real news.')
    elif svc.predict(input_vec)[0] == 1:
        st.write('This article appears to be fake news.')


