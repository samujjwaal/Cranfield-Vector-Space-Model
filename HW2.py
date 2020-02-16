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

import os
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

#Importing the NLTK Stopwords
import nltk
nltk.download('stopwords')
stop_list = stopwords.words('english')
print(stop_list)

in_path = 'cranfieldDocs'
out_path = 'preprocessed_cranfieldDocs'
os.mkdir(out_path)
filenames = os.listdir(path)   #To generate file path
# print(filenames)

filepath = in_path + '/' + filenames[0]
file = open(filepath)
data = file.read()
file.close()

soup = BeautifulSoup(data)
title = soup.findAll('title')
text = soup.findAll('text')
# for item in title:
#     print(item.get_text())

# +
# print(soup.prettify())
# -

for fname in filenames:
    infilepath = in_path + '/' + fname
    outfilepath = out_path + '/' + fname
    with open(outfilepath, 'w') as outfile:    
        with open(infilepath) as infile:
            soup = BeautifulSoup(infile.read())
            title = soup.findAll('title')
            text = soup.findAll('text')
            for item in title:
                outfile.write(item.get_text())
            for item in text:
                 outfile.write(item.get_text())
        infile.close()
outfile.close()

filepath = out_path + '/' + filenames[0]
file = open(filepath)
data = file.read()
data


