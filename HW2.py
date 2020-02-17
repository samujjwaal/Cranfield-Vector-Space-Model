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
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import re
from nltk.corpus import stopwords
stop_list = stopwords.words('english')


# + init_cell=true
# Declaring variables for file path
in_path = 'cranfieldDocs'
out_path = 'preprocessed_cranfieldDocs'

# Declaring variables for query files
query = 'queries.txt'
preproc_query = 'new_queries.txt'

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
# data
# -


