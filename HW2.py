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
import collections

#Importing the NLTK Stopwords
import nltk
nltk.download('stopwords')
stop_list = stopwords.words('english')
print(stop_list)

path = 'cranfieldDocs'
filenames = os.listdir(path)   #To generate file path
# print(filenames)







