#
# Simple 3 layer network for MIST data
#
import pandas as pd
import tensorflow as tf
from sklearn import cross_validation

HIDDEN_NEURON_COUNT = 15


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def load_cross_validation():
    df = pd.read_csv('../data/train.csv')
    (cv_train, cv_test) = cross_validation.train_test_split(df, train_size=0.75)
    train_labels = pd.get_dummies(cv_train['label'])
    df = cv_train.drop('label', 1) / 255.0
    test_labels = pd.get_dummies(cv_test['label'])
    test = cv_test.drop('label', 1) / 255.0
    return df, train_labels, test, test_labels


def build_model(images):
    """ Create network with single hidden layer and softmax output layer """
    # Layer 2
    w2 = weight_variable([784, HIDDEN_NEURON_COUNT])
    b2 = bias_variable([HIDDEN_NEURON_COUNT])
    l2 = tf.nn.relu(tf.matmul(images, w2) + b2)
    # Layer 3
    w3 = weight_variable([HIDDEN_NEURON_COUNT, 10])
    b3 = bias_variable([10])
    l3 = tf.nn.softmax(tf.matmul(l2, w3) + b3)
    return l3


def calculate_score(expected, response):
    """ Function to calculate score. Score is the number of correctly classified images.
    Since this is not smooth function it can'b be used to train network """
    correct_prediction = tf.equal(tf.argmax(response, 1), tf.argmax(expected, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, "float"))


def train(session, model, image_placeholder, label_placeholder, df_train, df_labels):
    """ Train network on given data """
    cross_entropy = -tf.reduce_sum(label_placeholder * tf.log(model))
    accuracy = calculate_score(label_placeholder, model)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    for i in range(1000):
        (train_a, _, label_a, _) = cross_validation.train_test_split(df_train, df_labels, train_size=100)
        session.run(train_step, feed_dict={image_placeholder: train_a, label_placeholder: label_a})
        if (i+1) % 100 == 0:
            train_accuracy = session.run(accuracy,
                                         feed_dict={image_placeholder: train_a, label_placeholder: label_a})
            print "step %d, training accuracy %g" % (i+1, train_accuracy)
    return model


def main():
    image_placeholder = tf.placeholder("float", shape=[None, 784])
    label_placeholder = tf.placeholder("float", shape=[None, 10])
    df_train, train_labels, df_test, test_labels = load_cross_validation()
    model = build_model(image_placeholder)
    session = tf.Session()
    init = tf.initialize_all_variables()
    session.run(init)
    model = train(session, model, image_placeholder, label_placeholder, df_train, train_labels)
    accuracy = calculate_score(label_placeholder, model)
    score = session.run(accuracy, feed_dict={image_placeholder: df_test, label_placeholder: test_labels})
    print("Final score %f" % score)


main()
