import nltk

# I added something
stemmer = nltk.PorterStemmer()
print(stemmer.stem('ran')) # => 
print(stemmer.stem('running')) # => run
print(stemmer.stem('shoes')) # => shoe
print(stemmer.stem('expensive')) # => expens

print(stemmer.stem('y')) # => 
