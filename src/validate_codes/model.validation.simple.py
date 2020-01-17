from __future__ import division
import tensorflow as tf
import numpy as np
import sys
import time

from tensorflow.keras import layers
from collections import Counter

output_dim = 10 # number of candidates
hidden_size = 100
batch_size = 100
epochs = 500
learning_rate = 0.01

# [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, ...]
def get_simple_sequence(n_vocab, repeat=5):
    data = []
    for i in range(repeat):
        for j in range(n_vocab):
            for k in range(j):
                data.append(j)
    return np.asarray(data, dtype=np.float32)

def main(argv):
    start_time = time.time()
    if len(sys.argv) <= 1:
        print "sequence len: 1"
        input_length = 1
    else:
        input_length = int(sys.argv[1])

    print 'learning_rate: %f, batch_size %d, epochs %d' %(learning_rate, batch_size, epochs)
    print 'features are loaded'

    for seq_length in range(input_length, input_length+1):
        data = get_simple_sequence(10)
        learn_instances = []
        for i in range(5):
            for j in range(len(data)-seq_length-1):
                learn_instances.append(data[j:j+seq_length+1])

        np.random.shuffle(learn_instances)
        
        learn_X = []; learn_Y = []
        for seq in learn_instances:
            learn_X.append(np.array(seq[:-1]))
            learn_Y.append((seq[-1]))

        print 'size of learn_Y: %d' % len(learn_Y)
        print '# learn_Y ', Counter(learn_Y) # shows the number of '0' and '1'

        learn_X = np.reshape(learn_X, [-1, seq_length, 1])
        learn_Y = map(lambda x:[x], learn_Y)

        test_instances = learn_instances[:5000]
        np.random.shuffle(test_instances)

        test_X = []; test_Y = []
        for seq in test_instances:
            test_X.append(np.array(seq[:-1]))
            test_Y.append((seq[-1]))

        test_X = np.reshape(test_X, [-1, seq_length, 1])
        test_Y = map(lambda x:[x], test_Y)
        
        print 'Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y))
        tf.reset_default_graph()

        # 2. Run RNN
        X = tf.placeholder(tf.float32, [None, seq_length, 1])
        Y = tf.placeholder(tf.int32, [None, 1])

        is_training = tf.placeholder(tf.bool)

        # sequence_length = seq_length
        keep_prob = tf.placeholder(tf.float32)

        weights = {}
        biases = {}

        cells = []
        for _ in range(1):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                                   state_is_tuple=True,
                                                   activation=tf.nn.relu)
            #cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_size,
            #                                            activation=tf.nn.relu,
            #                                            dropout_keep_prob=keep_prob) # Layer Normalization. num_units: ouput size
            cells.append(cell)

        cells = tf.nn.rnn_cell.MultiRNNCell(cells) # stackedRNN

        outputs, states = tf.nn.dynamic_rnn(cells, X,
                dtype=tf.float32) # called RNN driver

        outputs = outputs[:, -1]

        bn_output = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=is_training)

        # three-level MLP
        key = 'fc_l1'
        weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
        biases[key] = tf.Variable(tf.random_normal([hidden_size]))

        key = 'fc_l2'
        weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
        biases[key] = tf.Variable(tf.random_normal([hidden_size]))

        key = 'fc_l3'
        weights[key] = tf.Variable(tf.random_normal([hidden_size, output_dim]))
        biases[key] = tf.Variable(tf.random_normal([output_dim]))
        
        optimizers = {}
        pred = []

        l1_output = tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1']
        l1_bn_output = tf.contrib.layers.batch_norm(l1_output, center=True, scale=True, is_training=is_training)

        l2_output = tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
        l2_bn_output = tf.contrib.layers.batch_norm(l2_output, center=True, scale=True, is_training=is_training)

        logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
        labels = tf.one_hot(Y, depth=output_dim)
        
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        cost = tf.reduce_mean(loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #learning_rate = tf.train.exponential_decay(0.01, global_step, epochs, 0.96, staircase=True)
        with tf.control_dependencies(update_ops):
            optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        pred.append(tf.argmax(tf.nn.softmax(logits), 1))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            count = 0
            for e in range(epochs):
                #print 'epochs: %d'%(e)

                # train batch by batch
                batch_index_start = 0
                batch_index_end = batch_size

                for i in range(int(len(learn_X)/batch_size)):
                    X_train_batch = learn_X[batch_index_start:batch_index_end]
                    Y_train_batch = learn_Y[batch_index_start:batch_index_end]

                    opt, c, o, l, la = sess.run([optimizers, cost, outputs, logits, labels],
                            feed_dict={X: X_train_batch, Y: Y_train_batch, keep_prob:0.01, is_training:True})
                    if np.isnan(c):
                        print 'cost is nan'
                        sys.exit(1)
                    
                    batch_index_start += batch_size
                    batch_index_end += batch_size
                    count += 1

                if (e % 5 == 0 and e != 0):
                    print 'epochs: %d'%(e)
                    # TEST
                    rst, c, l = sess.run([pred, cost, logits], feed_dict={X: test_X, Y: test_Y, keep_prob:1.0, is_training:False})

                    out = np.vstack(rst).T

                    c1 = Counter(map(lambda x:x[0], out))
                    c2 = Counter(map(lambda x:x[0], test_Y))
                    print '# predict', c1.most_common()
                    print '# test', c2.most_common()
                    print 'iteration : %d, cost: %.8f'%(count, c)

                    predicts = []
                    test_Y = map(lambda x:x[0], test_Y)

                    for v1, v2 in zip(out, test_Y):
                        decision = False

                        if v1 == int(v2):
                            decision = True
                        predicts.append(decision)

                    print 'seq_length: %d, # predicts: %d, # corrects: %d, acc: %f' %(seq_length, len(predicts), len(filter(lambda x:x, predicts)), (len(filter(lambda x:x, predicts))/len(predicts)))
                    test_Y = map(lambda x:[x], test_Y)
            print 'work time: %s sec'%(time.time()-start_time)
            print '\n\n'

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

