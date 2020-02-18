# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Name : Samujjwaal Dey
#
# ## CS 582 Information Retrieval : Homework 1

# ## Imports and Initializations

# + init_cell=true
# Importing dependancy libraries
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import numpy as np
import re
import math as m
from nltk.corpus import stopwords
stop_list = stopwords.words('english')


# + init_cell=true
# Declaring variables for file path
in_path = 'cranfieldDocs'
out_path = 'preprocessed_cranfieldDocs'

# Declaring variables for query files
query = 'queries.txt'
preproc_query = 'new_queries.txt'

relevance = 'relevance.txt'

# Checking if the preprocessed docs folder exists already
if not os.path.isdir(out_path):
    os.mkdir(out_path)

# Getting all filenames from the docs folder
filenames = os.listdir(in_path)  # To generate file path
# print(filenames)

# Initiallizing Porter Stemmer object
st = PorterStemmer()

# Initializing regex to remove words with one or two characters length
shortword = re.compile(r'\W*\b\w{1,2}\b')


# -

# ## Preprocessing the documents

def tokenize(data):
    """Preprocesses the string given as input. Converts to lower case,
    removes the punctuations and numbers, splits on whitespaces, 
    removes stopwords, performs stemming & removes words with one or 
    two characters length.

    Arguments:
        data {string} -- string to be tokenized

    Returns:
        string -- string of tokens generated
    """
    # converting to lower case
    lines = data.lower()
    # removing punctuations by using regular expression
    lines = re.sub('[^A-Za-z]+', ' ', lines)
    # splitting on whitespaces to generate tokens
    tokens = lines.split()
    # removing stop words from the tokens
    clean_tokens = [word for word in tokens if word not in stop_list]
    # stemming the tokens
    stem_tokens = [st.stem(word) for word in clean_tokens]
    # checking for stopwords again
    clean_stem_tokens = [word for word in stem_tokens if word not in stop_list]
    # converting list of tokens to string
    clean_stem_tokens = ' '.join(map(str,  clean_stem_tokens))
    # removing tokens with one or two characters length
    clean_stem_tokens = shortword.sub('', clean_stem_tokens)
    return clean_stem_tokens


def extractTokens(beautSoup, tag):
    """Extract tokens of the text between a specific SGML <tag>. The function
    calls tokenize() function to generate tokens from the text.
    
    Arguments:
        beautSoup {bs4.BeautifulSoup} -- soup bs object formed using text of a file
        tag {string} -- target SGML <tag>
    
    Returns:
        string -- string of tokens extracted from text between the target SGML <tag>
    """
    # extract text of a particular SGML <tag>
    textData = beautSoup.findAll(tag)
    # converting to string
    textData = ''.join(map(str, textData))
    textData = textData.replace(tag, '')
    # calling function to generate tokens from text
    textData = tokenize(textData)
    return textData


for fname in filenames:
    #generate filenames
    infilepath = in_path + '/' + fname
    outfilepath = out_path + '/' + fname
    with open(infilepath) as infile:
        with open(outfilepath, 'w') as outfile:
            fileData = infile.read()
            #creating BeautifulSoup object to extract text between SGML tags
            soup = BeautifulSoup(fileData)
            # extract tokens for <title>
            title = extractTokens(soup, 'title')
            # extract tokens for <text>
            text = extractTokens(soup, 'text')
            outfile.write(title)
            outfile.write(" ")
            outfile.write(text)
        outfile.close()
    infile.close()

# Pre processing the queries.txt file
q = open(query)
new_q = open(preproc_query, 'w')
text = q.readlines()
for line in text:
    # to avoid newline in the end of file
    if(line != text[-1]):
        query_tokens = tokenize(line)
        new_q.write(query_tokens + '\n')
    else:
        query_tokens = tokenize(line)
        new_q.write(query_tokens)

# +
# filepath = out_path + '/' + filenames[0]
# file = open(filepath)
# data = file.read()
# # len(data.split())
# # data.split()
# count = {}
# for w in data.split():
#     if w in count:
#         count[w] += 1
#     else:
#         count[w] = 1
        
# # print(count)
# len(count)
# +
all_docs = []

for fname in filenames:
    outfilepath = out_path + '/' + fname
    with open(outfilepath) as file:
        fileData = file.read()
        all_docs.append(fileData)
# -

no_of_docs = len(all_docs)
no_of_docs

# #### Calculating df values

# +
DF = {}

for i in range(no_of_docs):
    tokens = all_docs[i].split()
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

for i in DF:
    DF[i] = len(DF[i])
# -

# print(DF)


vocab_size = len(DF)
vocab_size

vocab = [term for term in DF]
# print(vocab)

# #### Calculating tf-idf values

# +
doc = 0

tf_idf = {}

for i in range(no_of_docs):
    
    tokens = all_docs[i].split()
    
    counter = Counter(tokens)
    words_count = len(tokens)
    
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = DF[token] if token in vocab else 0
        idf = np.log((no_of_docs+1)/(df+1))
        
        tf_idf[doc, token] = tf*idf

    doc += 1

# +
# tf_idf
#len(tf_idf)
# -

# #### Vectorizing tf-idf

D = np.zeros((no_of_docs, vocab_size))
for i in tf_idf:
    try:
        ind = vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass

# D
len(D)


def gen_vector(tokens):

    Q = np.zeros((len(vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = DF[token] if token in vocab else 0
        idf = m.log((no_of_docs+1)/(df+1))

        try:
            ind = vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


# +
def cosine_similarity(k, query):
#     print("Cosine Similarity")
    tokens = query.split()
    
#     print("\nQuery:", query)
#     print("")
#     print(tokens)
    
    d_cosines = []
    
    query_vector = gen_vector(tokens)
    
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))
        
    out = np.array(d_cosines).argsort()[-k:][::-1]
    
    
#     print("")
    
#     print(out)
    return out

# -

    cosine_similarity(10,'investig made wave system creat static pressur distribut liquid surfac')

query_file = open(preproc_query, 'r')
queries = query_file.readlines()
# type(queries)/
# queries[1].split()
# print(queries[1].split())

n = 10
for query in queries:
    print(cosine_similarity(n, query))


