#!/usr/bin/python


import os
import sys
import numpy as np
import pandas as pd
import zipfile

import pickle

# language processing stuff
import nltk
from nltk.corpus import stopwords
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "data"))
stemmer = nltk.PorterStemmer()
stop_words = stopwords.words('english')


#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


if len(sys.argv) != 2:
	print("This program reads in a zipped csv file of comment data, and")
	print("then runs a TensorFlow neural network to classify the data")
	print()
	print("usage: train_neuralnet.py INFILE.CSV.ZIP")
	print()
	exit()

def remove_non_ascii(text):
	# damn strippping stupid unicode chars took forever
	# I tried all sorts of decode.encode etc
	# this is the only thing that worked
	return ''.join(i for i in text if ord(i)<128)

def destem(word):
        # the stemmer code breaks on really long words
        if len(word) < 50:
                return stemmer.stem(word)
        else:
                return word

def clean(text):
	# CHARACTER LEVEL CLEANING
	# get rid of weird characters
	text = remove_non_ascii(text)
	# get rid of punctation: using list from tensorflow plus single quote
	punctuation = '1234567890\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
	text = ''.join(c for c in text if c not in punctuation)
	# get rid of digits
	#text = ''.join(c for c in text if not c.isdigit())
	# get rid of extra spacing, newlines, etc
	text = " ".join(text.split())

	# WORD LEVEL CLEANING
	# remove stop-words
	text = ' '.join(w for w in text.split() if w not in stop_words)
	# drop words that are only one character
	text = ' '.join(w for w in text.split() if len(w) > 2)
	# de-stem the words
	text = ' '.join(destem(w) for w in text.split())

	# FINALIZING THE COMMENT
	# get rid of extra spacing from word deletions
	#text = " ".join(text.split())

	return text

def clean_list(a):
	b = []
	for i, text in enumerate(a):
		b.append(clean(text))
		if not i % 1000:
			sys.stderr.write(str(i) + "\n")

	return b

def create_model(input_dim):
    '''
    This creates and returns a new model
    '''
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = keras.Sequential([
        keras.layers.Dense(5000, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1000, activation='relu',),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(6, activation='sigmoid')
    ])
    model.compile(optimizer = sgd,
                  metrics=['accuracy'],
                  loss='binary_crossentropy'
    )
    return model


# ----------------------------TRAINING----------------------

num_words = 10000
t = Tokenizer(num_words, lower=True)

# open the training zip file
with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		# read in data
		tr = pd.read_csv(file_in)
		Xtr = tr['comment_text']
		Ytr = tr.loc[:,'toxic':'identity_hate'].values
		
		# comment or uncomment these as needed for speed
		#pickle.dump(clean_list(Xtr), open( "cleaned_comments.p", "wb" ) )
		Xtr = pickle.load( open( "cleaned_comments.p", "rb" ) )
		# tokenize data
		t.fit_on_texts(list(Xtr))
		Xtr = t.texts_to_matrix(Xtr, mode='count')

		# dont do these
		#Xtr = t.texts_to_sequences(Xtr)
		#Xtr = pad_sequences(Xtr, num_words)
		t.texts_to_sequences('asd')

# THIS IS TO DUMP WORD INFO
#print(t.word_counts)
#print(t.document_count)
#print(t.word_index)
#print(t.word_docs)

if 1:
	i = 1
	for word, count in sorted(t.word_counts.items(), key=lambda item: item[1], reverse=True):
		if i <= num_words:
			print(word, ',', sep='', end='')
		i += 1
	print()
	for row in Xtr:
		for col in row:
			print(col, ",", sep='', end='')
		print()
	

exit()

model = create_model(Xtr.shape[1])
model.fit(Xtr, Ytr, epochs=3, batch_size=2000) #, callbacks=[cp_callback])
#test_loss, test_acc = model.evaluate(Xtr,  Ytr, verbose=2)
#print('\nTest accuracy of', 'Adam', 'Model:', test_acc)\

Yhat = model.predict(Xtr)
for row in Yhat:
	print(np.round(row, 5))


# ----------------------------TESTING-----------------------
#te = pd.read_csv('test.csv')
#Xte = tr['comment_text']
#Xte = t.texts_to_sequences(Xte)
#Xte = pad_sequences(Xte, num_words)
