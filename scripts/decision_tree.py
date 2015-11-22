import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('../data/train.csv')
(cv_train, cv_test) = cross_validation.train_test_split(train, train_size=0.75)
train_labels = cv_train['label']
train = cv_train.drop('label', 1)
test_labels = cv_test['label']
test = cv_test.drop('label', 1)
clf = DecisionTreeClassifier()
classifier = clf.fit(train, train_labels)
y = classifier.predict(test)
score = np.equal(test_labels, y).sum()/float(len(test))
print("Score: %f" % score)