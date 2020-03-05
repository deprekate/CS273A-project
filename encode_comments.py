#!/usr/bin/python

import os
import sys
import zipfile
import csv
import codecs
import pickle

import nltk
stemmer = nltk.PorterStemmer()
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

def remove_non_ascii(text):
	# damn strippping stupid unicode chars took forever
	# I tried all sorts of decode.encode etc
	# this is the only thing that worked
	return ''.join(i for i in text if ord(i)<128)

def swap_words(text):
	text = text.replace(" u ", " you ")
	return text

def destem(word):
	# the stemmer code breaks on really long words
	if len(word) < 50:
		return stemmer.stem(word)
	else:
		return word


word_to_token = pickle.load( open( "encodings.p", "rb" ) )
print(len(word_to_token))
ordered_words = []
for key, value in word_to_
	print(k,v)
	if int(v) > 1:
		ordered_words.append(k)
print(len(ordered_words))
exit()



with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		reader = csv.reader(codecs.iterdecode(file_in, 'utf-8'))
		next(reader)
		for i, line in enumerate(reader):
			all_words = []
			t = Tokenizer()
			line[1] = remove_non_ascii(line[1])
			words = text_to_word_sequence(line[1], filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
			for word in words:
				if (len(word) > 1) and (not word.isdigit()):
					word = destem(word)
					all_words.append(word)
			t.fit_on_texts(all_words)
			for word in ordered_words:
				print(t.word_counts.get(word, 0), end='')
			print()

