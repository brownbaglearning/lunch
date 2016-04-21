
from __future__ import print_function  # <-- this lets python2 use python3's print function

import sys, time, math, random

import numpy as np

from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import LinearSVC

stime = time.time()

X = []
y = []
for idx in range(11000):
    (xcoord,ycoord,zcoord) = ( random.random(), random.random(), random.random() )
    X.append([xcoord,ycoord,zcoord])
    if math.sqrt(xcoord*xcoord + ycoord*ycoord + zcoord*zcoord) <= 1.0:
        y.append(0)
    else:
        y.append(1)

classifier = LogisticRegression()
# classifier = SGDClassifier(n_iter=1000)
# classifier = RandomForestClassifier()
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier = LinearSVC()

classifier.fit(X,y)

print("joblib.dump-ing classifier", time.time()-stime)
joblib.dump(classifier,'ross_classifier.pkl')

print("total time:",time.time()-stime)
