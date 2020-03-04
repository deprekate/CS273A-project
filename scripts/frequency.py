#!/usr/bin/python

import os
import sys
import zipfile
import csv
import codecs

import nltk
from nltk.corpus import stopwords
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "data"))

if len(sys.argv) < 2:
	print("usage: frequency.py INFILE")
	exit()

def tokenize(text):
	sr = stopwords.words('english')
	tokens = [t for t in text.lower().split()]
	clean_tokens = tokens[:]
	for token in tokens:
	    if token in stopwords.words('english'):
	        clean_tokens.remove(token)
	return clean_tokens


all_tokens = dict()
with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		reader = csv.reader(codecs.iterdecode(file_in, 'utf-8'))
		next(reader)
		for line in reader:
			clean_tokens = tokenize(line[1])
			freq = nltk.FreqDist(clean_tokens)
			for key,val in freq.items():
				all_tokens[key] = 1 #all_tokens.get(key,0) + 1

#print(all_tokens)

