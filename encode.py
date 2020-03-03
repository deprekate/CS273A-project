#!/usr/bin/python

import os
import sys
import zipfile
import csv
import codecs

import nltk
from nltk.corpus import stopwords
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "data"))

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick


if len(sys.argv) < 2:
	print("usage: tokenize.py INFILE")
	exit()

def tokenize_cleaner(text):
	sr = stopwords.words('english')
	tokens = [t for t in text.lower().split()]
	clean_tokens = tokens[:]
	for token in tokens:
	    if token in stopwords.words('english'):
	        clean_tokens.remove(token)
	return clean_tokens

t = Tokenizer()
# fit the tokenizer on the documents

all_tokens = dict()
with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		reader = csv.reader(codecs.iterdecode(file_in, 'utf-8'))
		next(reader)
		for i, line in enumerate(reader):
			words = text_to_word_sequence(line[1])
			t.fit_on_texts(words)
			print(t.word_counts)
			#clean_tokens = tokenize(line[1])
			#freq = nltk.FreqDist(clean_tokens)
			#for key,val in freq.items():
			#	all_tokens[key] = 1 #all_tokens.get(key,0) + 1
		print(i)

# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
