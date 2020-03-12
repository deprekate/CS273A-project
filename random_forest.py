import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale

def class_feature_importance(X, Y, feature_importances):
    N, M = X.shape
    cols = X.columns
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            #zip(range(M), np.mean(X[Y==c, :], axis=0)*feature_importances)
            zip(cols, np.mean(X[Y==c, :], axis=0)*feature_importances)
        )

    return out


X = pd.read_csv('matrix/mat.csv.gz', compression='gzip', header=0)
Y = pd.read_csv('matrix/classes.csv.gz', compression='gzip', header=0)


clf = RandomForestClassifier(n_estimators = 1000, random_state = 0)
clf.fit(X, Y[sys.argv[1]])

#for name, importance in zip(X.columns, clf.feature_importances_):
#	print(name, importance, sep='\t')
labels = {
	1 : sys.argv[1],
	0 : "non_" + sys.argv[1]
}
result = class_feature_importance(X, Y[sys.argv[1]], clf.feature_importances_)
for cl in result:
	for w in result[cl]:
		print(labels[cl], w, result[cl][w], sep='\t')
