
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

classifier = joblib.load('ross_classifier.pkl') 

ncorrect = 0
nchecked = 0
for idx in range(10):
    nchecked += 1
    instance = ( random.random(), random.random(), random.random() )  # coords to predict
    instances = [instance]
    predictions = classifier.predict(instances)
    prediction  = predictions[0]
    print("PRED",instance,prediction)
    (xcoord,ycoord,zcoord) = instance
    mag = math.sqrt( xcoord*xcoord + ycoord*ycoord + zcoord*zcoord )
    if mag <= 1.0  and  prediction == 0:
        ncorrect += 1
    elif mag > 1.0  and  prediction == 1:
        ncorrect += 1

print(float(ncorrect)/nchecked)    # <--- python2 must convert to float itself
print("total time:",time.time()-stime)
