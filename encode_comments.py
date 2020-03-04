#!/usr/bin/python

import os
import sys
import zipfile
import csv
import codecs
import pickle

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick


if len(sys.argv) < 2:
	print("usage: tokenize.py INFILE")
	exit()

def remove_non_ascii(text):
	# damn strippping stupid unicode chars took forever
	# I tried all sorts of decode.encode etc
	# this is the only thing that worked
	return ''.join(i for i in text if ord(i)<128)


word_to_token = pickle.load( open( "encodings.p", "rb" ) )

ordered_words = []
for k, v in sorted(word_to_token.items(), key=lambda item: item[1]):
	ordered_words.append(k)

with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		reader = csv.reader(codecs.iterdecode(file_in, 'utf-8'))
		next(reader)
		for i, line in enumerate(reader):
			line[1] = remove_non_ascii(line[1])
			t = Tokenizer()
			words = text_to_word_sequence(line[1])
			t.fit_on_texts(words)
			for word in ordered_words:
				print(t.word_counts.get(word, 0), end='')
			print()

