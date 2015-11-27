import tensorflow as tf
import pandas as pd
from sklearn import cross_validation


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Prepare training and test set for cross validation
train = pd.read_csv('../data/train.csv')
(cv_train, cv_test) = cross_validation.train_test_split(train, train_size=0.75)
train_labels = pd.get_dummies(cv_train['label'])
train = cv_train.drop('label', 1) / 255.0
test_labels = pd.get_dummies(cv_test['label'])
test = cv_test.drop('label', 1) / 255.0

# Define graph of computation
# Placeholder for input data
x = tf.placeholder("float", [None, 784])
# Linear regression coefficients
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# Model
y = tf.nn.softmax(tf.matmul(x,W) + b)
# Place holder for correct values
y_ = tf.placeholder("float", [None,10])
# Cost function (Cross entropy)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# Minimize cost function using Gradient Descent (single step), with learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Init all variables
init = tf.initialize_all_variables()
# Create session and run variable initialization
sess = tf.Session()
sess.run(init)

# Do 1000 steps of training
print("Start learning")
for i in range(1000):
    if (i+1) % 100 == 0:
        print(i+1)
    (A, B, label_a, label_b) = cross_validation.train_test_split(train, train_labels, train_size=100)
    sess.run(train_step, feed_dict={x: A, y_: label_a})

# Get prediction and expected value
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# Define accuracy operation
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# check accuracy
print(sess.run(accuracy, feed_dict={x: test, y_: test_labels}))
