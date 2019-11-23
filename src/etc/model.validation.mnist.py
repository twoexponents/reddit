from __future__ import division
import tensorflow as tf
import numpy as np
import sys
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
from operator import itemgetter

from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
epochs = 100
batch_size = 128 #128

input_dim = 28
seq_length = 28
hidden_size = 128 # 64 
n_class = 10
# new
keep_rate = 1.0#0.5

def main(argv):
    print ('learning_rate: %f, batch_size %d, epochs %d' %(learning_rate, batch_size, epochs))

    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

    tf.reset_default_graph()
    
    # 2. Run RNN
    X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
    Y = tf.placeholder(tf.float32, [None, n_class])

    is_training = tf.placeholder(tf.bool)
    #keep_prob = tf.placeholder(tf.float32)

    weights = {}
    biases = {}

    cells = []
    for _ in range(1):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                               state_is_tuple=True,
                                               activation=tf.nn.relu)
        #cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_size,
        #                                            activation=tf.nn.relu)
        #                                            #dropout_keep_prob=keep_prob) # Layer Normalization. num_units: ouput size

        cells.append(cell)

    cells = tf.nn.rnn_cell.MultiRNNCell(cells) # stackedRNN
    #cells = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.nn.relu)
    
    outputs, states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)

    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = outputs[-1]


    # three-level MLP
    key = 'fc_l1'
    weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
    biases[key] = tf.Variable(tf.random_normal([hidden_size]))

    key = 'fc_l2'
    weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
    biases[key] = tf.Variable(tf.random_normal([hidden_size]))

    key = 'fc_l3'
    weights[key] = tf.Variable(tf.random_normal([hidden_size, n_class]))
    biases[key] = tf.Variable(tf.random_normal([n_class]))
    
    optimizers = {}
    pred = []

    
    bn_output = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=is_training)
    
    l1_output = tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1']
    l1_bn_output = tf.contrib.layers.batch_norm(l1_output, center=True, scale=True, is_training=is_training)
    #l1_dropout = tf.layers.dropout(l1_output, rate=1-keep_prob, training=is_training)

    l2_output = tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
    l2_bn_output = tf.contrib.layers.batch_norm(l2_output, center=True, scale=True, is_training=is_training)
    #l2_dropout = tf.layers.dropout(l2_bn_output, rate=1-keep_prob, training=is_training)

    logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
    labels = Y

    #l1_output = tf.matmul(outputs, weights['fc_l1']) + biases['fc_l1']
    #l2_output = tf.matmul(outputs, weights['fc_l2']) + biases['fc_l2']
    #logits = tf.matmul(bn_output, weights['fc_l3']) + biases['fc_l3']
    #labels = Y

    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y)
    cost = tf.reduce_mean(loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizers = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizers = tf.group([optimizers, update_ops])

    #with tf.control_dependencies(update_ops):
    #    optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batch = int(mnist.train.num_examples / batch_size)

        print ('total_batch size: %d' % (total_batch))

        is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        test_batch_size = len(mnist.test.images)
        test_xs = mnist.test.images.reshape(test_batch_size, seq_length, input_dim)
        test_ys = mnist.test.labels

        for epoch in range(epochs):
            total_cost = 0

            for i in range(batch_size):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs = batch_xs.reshape((batch_size, seq_length, input_dim))

                _, cost_val = sess.run([optimizers, cost],
                        feed_dict={X: batch_xs, Y: batch_ys, is_training: True})
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1),
                'Avg. cost: {:.4}'.format(total_cost / total_batch))
            print('Accuracy:', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys, is_training: False}))

        print('done!')


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

