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
# ## CS 582 Information Retrieval : Homework 2

# ## Imports and Initializations

# Importing dependancy libraries
import os
import pandas as pd
import numpy as np
import re
import math as m
from collections import Counter
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_list = stopwords.words('english')
# sorted(stop_list)


# +
# Declaring variables for file path
in_path = 'cranfieldDocs'
out_path = 'preprocessed_cranfieldDocs'

# Declaring variables for query files
query = 'queries.txt'
preproc_query = 'preprocessed_queries.txt'

# Declaring variable for file with query relevance values
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

# ## Preprocessing the documents and queries file

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
    # remove the SGML <tag> from text
    textData = textData.replace(tag, '')

    # calling function to generate tokens from text
    textData = tokenize(textData)

    return textData


# +
# Preprocessing all the documents in the cranfieldDocs directory

"""This cell might take about 20 seconds to run."""

for fname in filenames:

    # generate filenames
    infilepath = in_path + '/' + fname
    outfilepath = out_path + '/' + fname

    with open(infilepath) as infile:
        with open(outfilepath, 'w') as outfile:

            # read all text in a file
            fileData = infile.read()

            # creating BeautifulSoup object to extract text between SGML tags
            soup = BeautifulSoup(fileData)

            # extract tokens for <title>
            title = extractTokens(soup, 'title')

            # extract tokens for <text>
            text = extractTokens(soup, 'text')

            # write tokens for <title> into new file
            outfile.write(title)
            outfile.write(" ")

            # write tokens for <text> into new file
            outfile.write(text)
        outfile.close()
    infile.close()

# +
# Preprocessing the queries.txt file
q = open(query)

# opening new file to write preprocessed tokens into
new_q = open(preproc_query, 'w')

# read each line of file seperately
text = q.readlines()
for line in text:
    
    # if condition to avoid newline character in the end of file
    if(line != text[-1]):
        query_tokens = tokenize(line)
        new_q.write(query_tokens + '\n')
    else:
        query_tokens = tokenize(line)
        new_q.write(query_tokens)

# +
# Generate a single list of all preprocessed docs
all_docs = []

for fname in filenames:
    outfilepath = out_path + '/' + fname
    with open(outfilepath) as file:
        fileData = file.read()
        
        # append file data to the list
        all_docs.append(fileData)
# print(all_docs)

# -

# total number of documents is 1400
no_of_docs = len(all_docs)
no_of_docs

# # Task 1

# ## Calculating df values for each term in the vocabulary

# +
# create a dictionary of key-value pairs where tokens are keys and their occurence in the corpus the value
DF = {}

for i in range(no_of_docs):
    tokens = all_docs[i].split()
    for w in tokens:
        try:
            # add token as key and doc number as value is chained
            DF[w].add(i)
        except:
            # to handle when a new token is encountered
            DF[w] = {i}

for i in DF:
    # convert to number of occurences of the token from list of documents where token occurs
    DF[i] = len(DF[i])
# -

print(DF)

# count number of unique words in the corpus
vocab_size = len(DF)
print(vocab_size)

# create vocabulary list of all unique words
vocab = [term for term in DF]
print(vocab)

# ## Calculating tf-idf values for each term in the vocabulary

# +
doc = 0

# creating dictionary to store tf-idf values for each term in the vocabulary
tf_idf = {}

for i in range(no_of_docs):
    
    tokens = all_docs[i].split()
    
    # counter object to efficiently count number of occurence of a term in a particular document
    counter = Counter(tokens)
    words_count = len(tokens)
    
    for token in np.unique(tokens):
        
        # counting occurence of term in object using counter object
        tf = counter[token]/words_count
        # retrieving df values from DF dictionary
        df = DF[token] if token in vocab else 0
        
        # adding 1 to numerator & denominator to avoid divide by 0 error
        idf = np.log((no_of_docs+1)/(df+1))
        
        tf_idf[doc, token] = tf*idf

    doc += 1
# -

# print(tf_idf)
tf_idf

# ## Forming document vectors using the tf-idf values

# +
# initializing empty vector of vocabulary size
D = np.zeros((no_of_docs, vocab_size))

# creating vector of tf-idf values
for i in tf_idf:
    ind = vocab.index(i[1])
    D[i[0]][ind] = tf_idf[i]
# -

# len(D)
print(D)


def gen_vector(tokens):
    """To create a vector (with repsect to the vocabulary) of the tokens passed as input
    
    Arguments:
        tokens {list} -- list of tokens to be converted
    
    Returns:
        numpy.ndarray -- vector of tokens
    """
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


# # Task 2

# ## Calculating cosine similarity between each query vectors and document vectors

def cosine_sim(x, y):
    """To calculate cosine similarity between 2 vectors.
    
    Arguments:
        x {numpy.ndarray} -- vector 1
        y {numpy.ndarray} -- vector 2
    
    Returns:
        numpy.float64 -- cosine similarity between vector 1 & vector 2
    """
    cos_sim = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
    
    return cos_sim


def cosine_similarity(k, query):
    """To determine a ranked list of top k documents in descending order of their
    cosine similarity with the query
    
    Arguments:
        k {integer} -- top k documents to retrieve from 
        query {string} -- query whose cosine similarity is to be computed with the corpus
    
    Returns:
        numpy.ndarray -- list of top k cosine similarities between query and corpus of documents
    """

    tokens = query.split()
      
    d_cosines = []
    
    # vectorize the input query tokens
    query_vector = gen_vector(tokens)
    
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))
        
    if k == 0:
        # k=0 to retrieve all documents in descending order
        out = np.array(d_cosines).argsort()[::-1]
        
    else:
        # to retrieve the top k documents in descending order    
        out = np.array(d_cosines).argsort()[-k:][::-1]
    
    return out


# list of all queries from preprocessed queries file
query_file = open(preproc_query, 'r')
queries = query_file.readlines()
queries


def list_of_docs(k):
    """To generate a ranked list of k top documents in descending order of their cosine similarity 
    calculated against the queries. Output is a list of (query id, document id) pairs.
    
    If k=0 is given as input then list of all documnets in descending order is returned.
    
    Arguments:
        k {integer} -- number of top documents to be retrieved
    
    Returns:
        list -- list of documents in descending order of their cosine similarity
    """
    cos_sims = []
    for i in range(len(queries)):
        cs = [i, cosine_similarity(k, queries[i])]
        cos_sims.append(cs)
        
    return cos_sims



#to get list of all documents
no_of_top=0
list_of_docs(no_of_top)

# ## Calculating precision and recall values for each query

# retrieving relevance values from relevance.txt
colnames=['query', 'relevance'] 
rel = pd.read_csv(relevance, delim_whitespace=True, names=colnames, header=None)
rel.head(10)

# +
# list of relevant document numbers for a document
rel_list = []

# list of list of relevant document numbers for all documents
query_rel = []
for i in range(1,11):
    rel_list = rel[rel['query']==i]['relevance'].to_list()
    
    # append list rel_list to list of list query_list
    query_rel.append(rel_list)
print(query_rel)    


# -

def intersection(lst1, lst2): 
    """To count number of common items between 2 lists
    
    Arguments:
        lst1 {list} -- list 1
        lst2 {list} -- list 2
    
    Returns:
        integer -- number of common items between list 1 & list 2 
    """
    lst3 = [value for value in lst1 if value in lst2] 
    return len(lst3) 


# +
top = [10, 50, 100, 500]

# for top 100 docs
no_of_top = top[2]
no_of_top


# -

def calculate_recall(k):
    """To generate list of recall values for each query for given value of k
    
    Arguments:
        k {integer} -- number of top documents to be retrieved 
    
    Returns:
        list -- list of recall values for each query
    """
    
    recall = []
    for i in range(len(queries)):
        
        # Number of relevant documents retrieved
        a = intersection(list_of_docs(k)[i][1].tolist(), query_rel[i])
        
        # Total number of relevant documents
        b = len(query_rel[i])
        r = a / b
        recall.append(r)
    return recall   
# for top 100 docs
calculate_recall(no_of_top)


np.mean(calculate_recall(no_of_top))


# +
def calculate_precision(k):
    """To generate list of precision values for each query for given value of k
    
    Arguments:
        k {[type]} -- number of top documents to be retrieved
    
    Returns:
        list -- list of precision values for each query
    """
    precision = []
    for i in range(len(queries)):
        
        # Number of relevant documents retrieved
        a = intersection(list_of_docs(k)[i][1].tolist(), query_rel[i])
        
        # Total number of documents retrieved
        b = k
        p = a / b
        precision.append(p)
    return precision

# for top 100 docs
calculate_precision(no_of_top)

# -

np.mean(calculate_precision(no_of_top))

# +
# To print the output of answers to questions in Task 2
# Output saved to output.txt file

"""This cell might take about 40 seconds to run."""

for t in top:
    p=0
    r=0
    print("Top {0} documents in the rank list".format(t))
    p = calculate_precision(t)
    r = calculate_recall(t)
    for i in range(len(queries)):
        print("Query: {0} \t Pr: {1} \t Re: {2}\n".format(i+1,p[i],r[i]))
    print("Avg Precision: {0}\n".format(np.mean(p)))
    print("Avg Recall: {0}\n\n".format(np.mean(r)))
    

# -




