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


# -

# ## Preprocessing the documents

def tokenize(data):
    lines = data.lower()  # converting to lower case
    # removing punctuations by using regular expression
    lines = re.sub('[^A-Za-z]+', ' ', lines)
    tokens = lines.split()
    # removing stop words from the text
    clean_tokens = [word for word in tokens if word not in stop_list]
    stem_tokens = [st.stem(word)
                   for word in clean_tokens]  # stemming the words
    clean_stem_tokens = [word for word in stem_tokens if word not in stop_list]
    clean_stem_tokens = ' '.join(map(str,  clean_stem_tokens))
    return clean_stem_tokens


def extractTokens(beautSoup, tag):
    textData = beautSoup.findAll(tag)
    textData = ''.join(map(str, textData))
    textData = textData.replace(tag, '')
    textData = tokenize(textData)
    return textData


for fname in filenames:
    infilepath = in_path + '/' + fname
    outfilepath = out_path + '/' + fname
    with open(infilepath) as infile:
        with open(outfilepath, 'w') as outfile:
            fileData = infile.read()
            soup = BeautifulSoup(fileData)
            title = extractTokens(soup, 'title')
            text = extractTokens(soup, 'text')
            outfile.write(title)
            outfile.write(" ")
            outfile.write(text)
        outfile.close()
    infile.close()

q = open(query)
new_q = open(preproc_query, 'w')
text = q.readlines()
for line in text:
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


