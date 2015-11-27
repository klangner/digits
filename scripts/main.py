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


# Placeholder for input data
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# First layer 32 features with 5x5 patch (kernel)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# Reshape x as tensor
x_image = tf.reshape(x, [-1,28,28,1])
# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer
# The second layer will have 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# To reduce overfitting, we will apply dropout before the readout layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Finally, we add a softmax layer, just like for the one layer softmax regression above.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and evaluate model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(20000):
    (A, B, label_a, label_b) = cross_validation.train_test_split(train, train_labels, train_size=100)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:A, y_: label_a, keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
    sess.run(train_step, feed_dict={x: A, y_: label_a, keep_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: test, y_: test_labels}))
