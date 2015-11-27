import pandas as pd
import tensorflow as tf
from sklearn import cross_validation

train = pd.read_csv('../data/train.csv')
(cv_train, cv_test) = cross_validation.train_test_split(train, train_size=0.75)
train_labels = pd.get_dummies(cv_train['label'])
train = cv_train.drop('label', 1) / 255.0
test_labels = pd.get_dummies(cv_test['label'])
test = cv_test.drop('label', 1) / 255.0

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
for i in range(100):
    if (i+1) % 100 == 0:
        print(i+1)
    (A, B, label_a, label_b) = cross_validation.train_test_split(train, train_labels, train_size=100)
    sess.run(train_step, feed_dict={x: A, y_: label_a})

# Get prediction and expecte value
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# Definie accuracy operation
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# check accuracy
print(sess.run(accuracy, feed_dict={x: test, y_: test_labels}))

print(sess.run(y, feed_dict={x: test}))
