#!/usr/bin/python

import os
import sys
import zipfile
import csv
import codecs
import pickle

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

def remove_non_ascii(text):
	# damn strippping stupid unicode chars took forever
	# I tried all sorts of decode.encode etc
	# this is the only thing that worked
	return ''.join(i for i in text if ord(i)<128)

# fit the tokenizer on the documents
t = Tokenizer()

all_words = dict()
all_strings = ''

with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		reader = csv.reader(codecs.iterdecode(file_in, 'utf-8'))
		next(reader)
		for i, line in enumerate(reader):
			#line[1] = bytes(line[1], 'utf-8').decode('utf-8', 'ignore')
			line[1] = remove_non_ascii(line[1])
			all_strings += line[1]
			#t = Tokenizer(num_words=100000)
			#words = text_to_word_sequence(line[1])
			#t.fit_on_texts(words)
			#print(t.word_index)
			#clean_tokens = tokenize(line[1])
			#freq = nltk.FreqDist(clean_tokens)
			#for word in words:
			#	all_words[word] = all_words.get(word,0) + 1
			if not i % 100:
				print(i)


words = text_to_word_sequence(all_strings)
t.fit_on_texts(words)

#store mapping dict in a pickle
pickle.dump(t.word_index, open( "encodings.p", "wb" ) )


# summarize what was learned
#for k, v in sorted(all_words.items(), key=lambda item: item[1]):
#	print(k, v)
#print(t.word_counts)
#print(t.document_count)
#print(t.word_index)
#print(t.word_docs)
