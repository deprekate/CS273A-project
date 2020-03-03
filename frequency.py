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
	tokens = [t for t in text.split()]
	clean_tokens = tokens[:]
	for token in tokens:
	    if token in stopwords.words('english'):
	        clean_tokens.remove(token)
	return clean_tokens


with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		reader = csv.reader(codecs.iterdecode(file_in, 'utf-8'))
		next(reader)
		for line in reader:
			print(tokenize(line[1]))
			exit()



#freq = nltk.FreqDist(clean_tokens)
#for key,val in freq.items():
#    print(str(key) + ':' + str(val))freq.plot(20, cumulative=False)
