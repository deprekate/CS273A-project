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
	print("This program reads in a zipped csv file of comment data, and")
	print("then returns the frequency of words for each comment")
	print("usage: create_encodings.py INFILE")
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
			line[1] = remove_non_ascii(line[1])
			all_strings += line[1]
			words = text_to_word_sequence(line[1], filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
			for word in words:
				if (len(word) > 1) and (not word.isdigit()):
					word = destem(word)
					all_words[word] = all_words.get(word,0) + 1

# here is where we can pick out only words we care about
good_words = []
for word, count in sorted(all_words.items(), key=lambda item: item[1], reverse=True):
	if count > 10:
		good_words.append(word)


with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		reader = csv.reader(codecs.iterdecode(file_in, 'utf-8'))
		next(reader)
		#if sys.argv[2]:
		#	for _ in range(int(sys.argv[2])):
		#		next(reader)
		for i, line in enumerate(reader):
			my_words = dict()
			line[1] = remove_non_ascii(line[1])
			all_strings += line[1]
			words = text_to_word_sequence(line[1], filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
			for word in words:
				if (len(word) > 1) and (not word.isdigit()):
					word = destem(word)
					my_words[word] = my_words.get(word,0) + 1
			for word in good_words:
				print(my_words.get(word,0), ',', sep='', end='')
			print()
			#if sys.argv[2] and i >= 1000:
			#	break
		




# summarize what was learned
#for k, v in sorted(all_words.items(), key=lambda item: item[1]):
#	print(k, v)

#print(t.word_counts)
#print(t.document_count)
#print(t.word_index)
#print(t.word_docs)
