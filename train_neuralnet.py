#!/usr/bin/python


import os
import sys
import numpy as np
import pandas as pd
import zipfile

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD


if len(sys.argv) != 2:
	print("This program reads in a zipped csv file of comment data, and")
	print("then runs a TensorFlow neural network to classify the data")
	print()
	print("usage: create_encodings.py INFILE")
	print()
	exit()


def create_model(input_dim):
    '''
    This creates and returns a new model
    '''
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = keras.Sequential([
        keras.layers.Dense(5000, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(500, activation='relu',),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(6, activation='sigmoid')
    ])
    model.compile(optimizer = sgd,
                  metrics=['accuracy'],
                  loss='binary_crossentropy'
    )
    return model

num_words = 10000

# open the training zip file
with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		# read in data
		tr = pd.read_csv(file_in)
		Xtr = tr['comment_text']
		Ytr = tr.loc[:,'toxic':'identity_hate'].values
		# tokenize data
		t = Tokenizer(num_words, lower=True)
		t.fit_on_texts(list(Xtr))
		Xtr = t.texts_to_matrix(Xtr, mode='count')
		#Xtr = t.texts_to_sequences(Xtr)
		#Xtr = pad_sequences(Xtr, num_words)

#te = pd.read_csv('test.csv')
#Xte = tr['comment_text']
#Xte = t.texts_to_sequences(Xte)
#Xte = pad_sequences(Xte, num_words)

#

model = create_model(Xtr.shape[1])
model.fit(Xtr, Ytr, epochs=3, batch_size=2000) #, callbacks=[cp_callback])
#test_loss, test_acc = model.evaluate(Xtr,  Ytr, verbose=2)
#print('\nTest accuracy of', 'Adam', 'Model:', test_acc)\

Yhat = model.predict(Xtr)
for row in Yhat:
	print(row)

