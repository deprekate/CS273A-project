import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv('matrix/mat.csv.gz', compression='gzip', header=0)
Y = pd.read_csv('matrix/classes.csv.gz', compression='gzip', header=0)


clf = RandomForestClassifier(n_estimators = 1000, random_state = 0)
clf.fit(X, Y[0])

for name, importance in zip(X.columns, clf.feature_importances_):
	print(name, importance, sep='\t')


