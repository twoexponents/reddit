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

len_w2v_features = 300

input_dim = len_w2v_features

output_dim = 1 # (range 0 to 1)
hidden_size = 100
learning_rate = 0.01
batch_size = 100
epochs = 500

def main(argv):
    start_time = time.time()
    if len(sys.argv) <= 1:
        print ("sequence len: 1")
        input_length = 1
    else:
        input_length = int(sys.argv[1])

    print ('learning_rate: %f, batch_size %d, epochs %d' %(learning_rate, batch_size, epochs))

    # 1.1 load feature dataset
    #d_w2vfeatures = pickle.load(open('../data/contentfeatures.googlenews.posts.p', 'r'))
    #d_w2vfeatures = pickle.load(open('../data/contentfeatures.googlenews.p', 'r'))
    #d_w2vfeatures = pickle.load(open('/home/jhlim/data/contentfeatures.googlenews.nozero.p', 'rb'))
    #with open('/home/jhlim/data/contentfeatures.googlenews.p', 'rb') as f:
    with open('/home/jhlim/data/contentfeatures.googlenews.nozero.p', 'rb') as f:
        d_w2vfeatures = pickle.load(f)
    #d_w2vfeatures = pickle.load(open('/home/jhlim/data/contentfeatures.googlenews.nozero.p', 'rb'))
    #d_w2vfeatures = pickle.load(open('/home/jhlim/data/top5w2vnostop.p', 'r'))

    #d_userfeatures = pickle.load(open('../data/userfeatures.activity.p', 'r'))

    print ('features are loaded')

    #for seq_length in range(input_length, input_length+1):
    #for multiple test
    for _ in range(1):
        seq_length = input_length
        #f = open('../data/seq.learn.%d.csv'%(seq_length), 'r')
        f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
        learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close()

        np.random.shuffle(learn_instances)

        learn_X = []; learn_Y = []
        for seq in learn_instances:
            sub_x = []

            try:
                for element in seq[:-1]: # seq[-1] : Y. element: 't3_7dfvv'
                    w2v_features = []
                    if element in d_w2vfeatures:
                        w2v_features = d_w2vfeatures[element]['google.mean'][0]
                    else:
                        continue
                    
                    if w2v_features != []:
                        sub_x.append(np.array(w2v_features))

                if (len(sub_x) == seq_length):
                    learn_X.append(np.array(sub_x)) # feature list
                    learn_Y.append(float(seq[-1]))

            except Exception as e:
                continue

        print ('size of learn_Y: %d' % len(learn_Y))
        print (Counter(learn_Y)) # shows the number of '0' and '1'

        learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length*input_dim]) # row num = file's row num
        sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
        learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)
        learn_X = np.reshape(learn_X, [-1, seq_length, input_dim])

        matrix = []
        for v1, v2 in zip(learn_X, learn_Y):
            matrix.append([v1, v2])

        np.random.shuffle(matrix)
        learn_X = list(map(itemgetter(0), matrix))
        learn_Y = list(map(lambda x:[x], map(itemgetter(1), matrix)))

        print (Counter(list(map(lambda x:x[0], learn_Y))))

        #f = open('../data/seq.test.%d.csv'%(seq_length), 'r')
        f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
        test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close()

        np.random.shuffle(test_instances)

        test_X = []; test_Y = []

        for seq in test_instances:
            sub_x = []

            try:
                for element in seq[:-1]:
                    w2v_features = []
                    
                    if element in d_w2vfeatures:
                        w2v_features = d_w2vfeatures[element]['google.mean'][0]
                    else:
                        continue

                    if w2v_features != []:
                        sub_x.append(np.array(w2v_features))

                if (len(sub_x) == seq_length):
                    test_X.append(np.array(sub_x))
                    test_Y.append(float(seq[-1]))

            except Exception as e:
                continue

        test_Y = list(map(lambda x:[x], test_Y))
        
        print ('Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y)))
        tf.reset_default_graph()


        # 2. Run RNN
        X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
        Y = tf.placeholder(tf.float32, [None, 1])

        is_training = tf.placeholder(tf.bool)

        # sequence_length = seq_length
        keep_prob = tf.placeholder(tf.float32)

        weights = {}
        biases = {}

        cells = []
        for _ in range(2):
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
        weights[key] = tf.Variable(tf.random_normal([hidden_size, 1]))
        biases[key] = tf.Variable(tf.random_normal([1]))
        
        optimizers = {}
        pred = []

        # Using dropout -> Fail
        # Using only batch normalization (w/ Relu) -> Fail
        # Using only batch normalization (w/o relu) -> better

        #10_dropout = tf.layers.dropout(outputs, rate=1-keep_prob, training=is_training)

        l1_output = tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1']
        #l1_output = tf.nn.relu(tf.matmul(outputs, weights['fc_l1']) + biases['fc_l1']) # might move relu layer to the behind of bn
        l1_bn_output = tf.contrib.layers.batch_norm(l1_output, center=True, scale=True, is_training=is_training)
        #l1_dropout = tf.layers.dropout(l1_output, rate=1-keep_prob, training=is_training)

        l2_output = tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
        #l2_output = tf.nn.relu(tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
        l2_bn_output = tf.contrib.layers.batch_norm(l2_output, center=True, scale=True, is_training=is_training)

        #logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
        #labels = tf.one_hot(Y, depth=output_dim) # output_dim: 2
        
        logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
        labels = Y

        #logits = tf.contrib.layers.batch_norm(logits, center=True, scale=True, is_training=is_training)
        #labels = tf.one_hot(Y, depth=output_dim)

        #loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        #cost = tf.reduce_mean(loss)
        #optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        #hypothesis = tf.sigmoid(logits)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        cost = tf.reduce_mean(loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        #optimizers = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

        hypothesis = tf.sigmoid(logits)
        pred.append(tf.cast(hypothesis > 0.5, dtype=tf.float32))

        correct_pred = tf.equal(tf.round(hypothesis), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #pred.append(tf.argmax(tf.nn.softmax(logits), 1))

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

                    opt, c, o, h, l, acc = sess.run([optimizers, cost, outputs, hypothesis, logits, accuracy],
                            feed_dict={X: X_train_batch, Y: Y_train_batch, keep_prob:0.01, is_training:True})
                    
                    #print 'iteration : %d, cost: %.8f'%(count, c)
                    #print 'logits: '
                    #print l
                    #if i == 0:
                    #    print 'acc: ', acc

                    batch_index_start += batch_size
                    batch_index_end += batch_size
                    count += 1

                if (e != 0):
                    print ('epochs: %d'%(e))
                    # TEST
                    rst, c, h, l = sess.run([pred, cost, hypothesis, logits], feed_dict={X: test_X, Y: test_Y, keep_prob:1.0, is_training:True})

                    out = np.vstack(rst).T
                    out = out[0]

                    #print ('# predict', Counter(out))
                    #print ('# test', Counter(map(lambda x:x[0], test_Y)))

                    predicts = []
                    test_Y = list(map(lambda x:x[0], test_Y))

                    for v1, v2 in zip(out, test_Y):
                        decision = False

                        if v1 == int(v2):
                            decision = True
                        predicts.append(decision)

                    fpr, tpr, thresholds = roc_curve(list(map(int, test_Y)), out)
                    print ('seq_length: %d, # predicts: %d, # corrects: %d, acc: %f, auc: %f' %(seq_length, len(predicts), len(list(filter(lambda x:x, predicts))), (len(list(filter(lambda x:x, predicts)))/len(predicts)), auc(fpr,tpr)))
                    print (precision_recall_fscore_support(list(map(int, test_Y)), out))
                    test_Y = list(map(lambda x:[x], test_Y))
            print ('work time: %s sec'%(time.time()-start_time))
            print ('\n\n')


            print ('work time: %s sec'%(time.time()-start_time))

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])
