import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('../data/train.csv')
(cv_train, cv_test) = cross_validation.train_test_split(train, train_size=0.75)
train_labels = normalize_labels(cv_train['label'])
train = cv_train.drop('label', 1)
test_labels = cv_test['label']
test = cv_test.drop('label', 1)

# Placeholder for input data
x = tf.placeholder("float", [None, 784])
# Linear regression coefficients
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# Model
y = tf.nn.softmax(tf.matmul(x,W) + b)
# Place holder for correct values
y_ = tf.placeholder("float", [None,10])
# Cost function (Cross entrophy)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# Minimize cost function using Gradient Descent (single step), with learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Init all variables
init = tf.initialize_all_variables()
# Create session and run variable initialization
sess = tf.Session()
sess.run(init)

print("Start learning")
# Do 1000 steps of training
for i in range(1000):
    if (i+1) % 100 == 0:
        print(i+1)
    sess.run(train_step, feed_dict={x: train[:100], y_: train_labels[:100]})

print('done.')
# score = np.equal(test_labels, y).sum()/float(len(test))
# print("Score: %f" % score)


def normalize_labels(labels):
    train[:size]