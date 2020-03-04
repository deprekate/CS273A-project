import nltk

# I added something
stemmer = nltk.PorterStemmer()
stemmer.stem('running') # => run
stemmer.stem('shoes') # => shoe
stemmer.stem('expensive') # => expens
