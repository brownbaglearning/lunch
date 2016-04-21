from __future__ import print_function  # <-- this lets python2 use python3's print function

import sys, time, random

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
for idx in range(9000):
    val = random.randrange(0,3000)
    X.append([val])
    if val < 1000:
        y.append(0)
    elif val < 2000:
        y.append(1)
    else:
        y.append(2)

classifier = LogisticRegression()
# classifier = SGDClassifier(n_iter=1000)
# classifier = RandomForestClassifier()
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier = LinearSVC()
classifier = RandomForestClassifier()

classifier.fit(X,y)

ncorrect = 0
nchecked = 0
for idx in range(10):
    nchecked += 1
    val = random.randrange(0,3000)
    instance    = [val]
    instances   = [instance]
    predictions = classifier.predict(instances)
    prediction  = predictions[0]
#   prediction = classifier.predict([[val]]) [0]
    print("PRED",val,prediction)
    if val < 1000  and  prediction == 0:
        ncorrect += 1
    elif val >= 1000 and  val < 2000  and  prediction == 1:
        ncorrect += 1
    elif val >= 2000  and  val < 3000  and  prediction == 2:
        ncorrect += 1

print(float(ncorrect)/nchecked)    # <--- python2 must convert to float itself
print("total time:",time.time()-stime)
